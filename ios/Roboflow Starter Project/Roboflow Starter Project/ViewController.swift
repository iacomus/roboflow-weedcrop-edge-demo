//
//  ViewController.swift
//  Roboflow Starter Project
//
//  Created by Nicholas Arner on 9/11/22.
//

import UIKit
import AVFoundation
import Vision
import CoreML
import Roboflow

var API_KEY = Secrets.apiKey
var MODEL = Secrets.model
var VERSION = Secrets.modelVersion

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    var bufferSize: CGSize = .zero
    var rootLayer: CALayer! = nil
    
    private var detectionOverlay: CALayer! = nil
    var currentPixelBuffer: CVPixelBuffer!
    
    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer! = nil
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    
    @IBOutlet weak private var previewView: UIView!
    @IBOutlet weak var fpsLabel: UILabel!
    
    // Roboflow SDK is kept only for the "Upload Incorrect Image" data-flywheel call.
    let rf = RoboflowMobile(apiKey: API_KEY)

    // On-device inference: our self-trained RF-DETR Small, bundled as CoreML.
    private var visionModel: VNCoreMLModel?
    private var visionRequest: VNCoreMLRequest?

    // Decode config. Class channels: 1 = crop, 2 = weed (0 = COCO supercategory, 3 = padding).
    private let confidenceThreshold: Float = 0.4
    private let classNames: [Int: String] = [1: "crop", 2: "weed"]

    override func viewDidLoad() {
        super.viewDidLoad()
        setupVisionModel()
        checkCameraAuthorization()
    }

    private func setupVisionModel() {
        guard let url = Bundle.main.url(forResource: "rfdetr_small", withExtension: "mlmodelc") else {
            print("RFDEMO: ❌ rfdetr_small.mlmodelc not in bundle — add rfdetr_small.mlpackage to the app target in Xcode.")
            return
        }
        do {
            let mlModel = try MLModel(contentsOf: url)
            let vnModel = try VNCoreMLModel(for: mlModel)
            let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
                self?.handleDetections(request: request, error: error)
            }
            // Match training preprocessing: stretch the full frame to 512x512 (no crop / no letterbox).
            request.imageCropAndScaleOption = .scaleFill
            self.visionModel = vnModel
            self.visionRequest = request
            print("RFDEMO: ✅ CoreML model loaded (rfdetr_small)")
        } catch {
            print("RFDEMO: ❌ failed to load CoreML model: \(error)")
        }
    }
    
    //--------------------------
    //MARK: Camera Session
    //--------------------------
    
    func checkCameraAuthorization() {
        let authStatus = AVCaptureDevice.authorizationStatus(for: AVMediaType.video)
        
        if authStatus == AVAuthorizationStatus.denied {
            // Denied access to camera
            // Explain that we need camera access and how to change it.
            let dialog = UIAlertController(title: "Unable to access the Camera", message: "To enable access, go to Settings > Privacy > Camera and turn on Camera access for this app.", preferredStyle: UIAlertController.Style.alert)
            let okAction = UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: nil)
            dialog.addAction(okAction)
            self.present(dialog, animated: true, completion: nil)
        } else if authStatus == AVAuthorizationStatus.notDetermined {
            // The user has not yet been presented with the option to grant access to the camera hardware.
            // Ask for it.
            AVCaptureDevice.requestAccess(for: AVMediaType.video, completionHandler: { [self] (granted) in
                if granted {
                    DispatchQueue.main.async { [self] in
                        //If we've been granted permission, start the camera session
                        setupAVCapture()
                    }
                }
            })
        } else {
            setupAVCapture()
        }
    }
    
    func setupAVCapture() {
        var deviceInput: AVCaptureDeviceInput!
        
        // Select a video device, make an input
        guard let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back).devices.first else {
            let alert = UIAlertController(
                title: "No Camera Found",
                message: "You must run this app on a physical device with a camera.", preferredStyle: UIAlertController.Style.alert)
            alert.addAction(UIAlertAction(title: "OK", style: .cancel, handler: { (_) in
            }))
            self.present(alert, animated: true, completion: nil)
            return
        }
        do {
            deviceInput = try AVCaptureDeviceInput(device: videoDevice)
        } catch {
            print("Could not create video device input: \(error)")
            return
        }
        
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1920x1080
        
        // Add a video input
        guard captureSession.canAddInput(deviceInput) else {
            print("Could not add video device input to the session")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addInput(deviceInput)
        
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
            // Add a video data output
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        } else {
            print("Could not add video data output to the session")
            captureSession.commitConfiguration()
            return
        }
        
        let captureConnection = videoDataOutput.connection(with: .video)
        // Always process the frames
        captureConnection?.isEnabled = true
        do {
            try  videoDevice.lockForConfiguration()
            let dimensions = CMVideoFormatDescriptionGetDimensions((videoDevice.activeFormat.formatDescription))
            bufferSize.width = CGFloat(dimensions.width)
            bufferSize.height = CGFloat(dimensions.height)
            videoDevice.unlockForConfiguration()
        } catch {
            print(error)
        }
        
        captureSession.commitConfiguration()
        
        DispatchQueue.main.async { [self] in
            previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
            rootLayer = previewView.layer
            previewLayer.frame = rootLayer.bounds
            rootLayer.addSublayer(previewLayer)
            
            setupLayers()
            updateLayerGeometry()
            startCaptureSession()
        }
    }
    
    func stopCaptureSession() {
        self.captureSession.stopRunning()

        if let inputs = captureSession.inputs as? [AVCaptureDeviceInput] {
            for input in inputs {
                self.captureSession.removeInput(input)
            }
        }
    }

    func startCaptureSession() {
        DispatchQueue.global(qos: .background).async { [self] in
            captureSession.startRunning()
        }
    }
    
    func captureOutput(_ captureOutput: AVCaptureOutput, didDrop didDropSampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        print("frame dropped")
    }
    
    //--------------------------
    //MARK: Model Inference
    //--------------------------
    
    // A single decoded detection in normalized [0,1] coordinates (model/512x512 space).
    struct Detection {
        let classId: Int
        let label: String
        let confidence: Float
        let rect: CGRect   // cxcywh converted to xywh, normalized, origin top-left
    }

    var inferenceStart: DispatchTime!
    var detecting: Bool = false

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        currentPixelBuffer = pixelBuffer

        guard !detecting, let request = visionRequest else { return }
        detecting = true
        inferenceStart = .now()

        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            // .right maps the back-camera landscape buffer to an upright frame for a
            // portrait-held phone. If on-device boxes look rotated/transposed, this is
            // the first knob to try (.up / .right / .down / .left).
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("RFDEMO: inference error \(error)")
                self?.detecting = false
            }
        }
    }

    private func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }

    // Vision completion: pull the `boxes`/`logits` MLMultiArrays, decode, draw.
    private func handleDetections(request: VNRequest, error: Error?) {
        defer { detecting = false }
        if let error = error { print("RFDEMO: \(error)"); return }
        guard let results = request.results as? [VNCoreMLFeatureValueObservation] else { return }
        var boxes: MLMultiArray?
        var logits: MLMultiArray?
        for obs in results {
            if obs.featureName == "boxes" { boxes = obs.featureValue.multiArrayValue }
            if obs.featureName == "logits" { logits = obs.featureValue.multiArrayValue }
        }
        guard let boxes = boxes, let logits = logits else { return }
        let detections = decode(boxes: boxes, logits: logits)

        DispatchQueue.main.async { [self] in
            drawBoundingBoxesFrom(detections: detections)
            if let d = inferenceStart.distance(to: .now()).toDouble(), d > 0 {
                fpsLabel.text = "\(Int(round(1 / d))) FPS"
            }
        }
    }

    // boxes: (1,300,4) cxcywh normalized; logits: (1,300,4) pre-sigmoid.
    // Subscript access is dtype-agnostic (model outputs may be fp16).
    private func decode(boxes: MLMultiArray, logits: MLMultiArray) -> [Detection] {
        let numQueries = boxes.shape[1].intValue
        var dets: [Detection] = []
        for q in 0..<numQueries {
            var bestId = -1
            var bestScore: Float = 0
            for c in classNames.keys {                 // only real classes (1=crop, 2=weed)
                let s = sigmoid(logits[[0, q, c] as [NSNumber]].floatValue)
                if s > bestScore { bestScore = s; bestId = c }
            }
            guard bestId >= 0, bestScore >= confidenceThreshold else { continue }
            let cx = boxes[[0, q, 0] as [NSNumber]].floatValue
            let cy = boxes[[0, q, 1] as [NSNumber]].floatValue
            let w  = boxes[[0, q, 2] as [NSNumber]].floatValue
            let h  = boxes[[0, q, 3] as [NSNumber]].floatValue
            let rect = CGRect(x: CGFloat(cx - w / 2), y: CGFloat(cy - h / 2),
                              width: CGFloat(w), height: CGFloat(h))
            dets.append(Detection(classId: bestId, label: classNames[bestId] ?? "?",
                                  confidence: bestScore, rect: rect))
        }
        return dets
    }
    
    //--------------------------
    //MARK: Bounding Boxes
    //--------------------------
    
    func setupLayers() {
        detectionOverlay = CALayer()
        detectionOverlay.name = "DetectionOverlay"
        detectionOverlay.frame = rootLayer.bounds
        rootLayer.addSublayer(detectionOverlay)
    }
    
    func drawBoundingBoxesFrom(detections: [Detection]) {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        detectionOverlay.sublayers = nil

        // The model saw the frame rotated .right -> an upright portrait image whose
        // dimensions are (rawHeight x rawWidth). The preview shows that same frame with
        // .resizeAspectFill (scale to cover the view, center, crop the overflow). Replicate
        // that transform so normalized model coords land on the right pixels.
        let viewW = detectionOverlay.bounds.width
        let viewH = detectionOverlay.bounds.height
        let frameW = bufferSize.height   // oriented (portrait) width  = raw buffer height
        let frameH = bufferSize.width    // oriented (portrait) height = raw buffer width
        guard frameW > 0, frameH > 0 else { CATransaction.commit(); return }
        let scale = max(viewW / frameW, viewH / frameH)
        let dW = frameW * scale, dH = frameH * scale
        let offX = (viewW - dW) / 2, offY = (viewH - dH) / 2

        for det in detections {
            let viewRect = CGRect(x: det.rect.minX * dW + offX,
                                  y: det.rect.minY * dH + offY,
                                  width: det.rect.width * dW,
                                  height: det.rect.height * dH)
            let color: UIColor = det.classId == 1 ? .systemGreen : .systemRed  // crop / weed
            let box = CAShapeLayer()
            box.frame = detectionOverlay.bounds
            box.path = UIBezierPath(roundedRect: viewRect, cornerRadius: 4).cgPath
            box.strokeColor = color.cgColor
            box.fillColor = color.withAlphaComponent(0.15).cgColor
            box.lineWidth = 2
            let label = CATextLayer()
            label.string = "\(det.label) \(Int(det.confidence * 100))%"
            label.fontSize = 12
            label.foregroundColor = UIColor.white.cgColor
            label.backgroundColor = color.withAlphaComponent(0.85).cgColor
            label.alignmentMode = .center
            label.contentsScale = UIScreen.main.scale
            label.frame = CGRect(x: viewRect.minX, y: max(0, viewRect.minY - 15), width: 64, height: 15)
            detectionOverlay.addSublayer(box)
            detectionOverlay.addSublayer(label)
        }
        CATransaction.commit()
    }
    
    //Create a bounding box and add it as a layer to the UI
    func drawBoundingBox(boundingBox: CGRect, color: UIColor, detectedValue: String, confidence: Double) {
        let shapeLayer = self.createRoundedRectLayerWithBounds(boundingBox, color: color)
        let textLayer = self.createTextSubLayerInBounds(boundingBox,
                                                        identifier: detectedValue,
                                                        confidence: VNConfidence(confidence))
        shapeLayer.addSublayer(textLayer)
        
        detectionOverlay.addSublayer(shapeLayer)
        self.updateLayerGeometry()
    }
    
    func drawPolygonBox(boundingBox: CGRect, polygon: [CGPoint], mask: [[UInt8]], color: UIColor, detectedValue: String, confidence: Double) {
            let shapeLayer = self.createPolygonLayerWithBounds(boundingBox, polygon: polygon, mask2D: mask, color: color)
            let textLayer = self.createTextSubLayerInBounds(boundingBox,
                                                            identifier: detectedValue,
                                                            confidence: VNConfidence(confidence))
            shapeLayer.addSublayer(textLayer)
            
            detectionOverlay.addSublayer(shapeLayer)
            self.updateLayerGeometry()
        }
    
    func createPolygonLayerWithBounds(_ bounds: CGRect,
                                          polygon: [CGPoint],
                                          mask2D: [[UInt8]],
                                          color: UIColor) -> CALayer {

            let container = CALayer()
            container.bounds = bounds
            container.position = CGPoint(x: bounds.origin.x, y: bounds.origin.y)
            container.name = "Found Object"
            
            

            container.borderColor  = color.withAlphaComponent(0.4).cgColor
            container.borderWidth  = 2
            container.cornerRadius = 7
            container.masksToBounds = true              // clip children to the box

            let bounds = rootLayer.bounds
            var scale: CGFloat
            let xScale: CGFloat = bounds.size.width / CGFloat(bufferSize.height)
            let yScale: CGFloat = bounds.size.height / CGFloat(bufferSize.width)
            scale = fmax(xScale, yScale)
            if scale.isInfinite {
                scale = 1.0
            }
        
            // ⬠ 3. polygon outline  ----------------------------------------------
            let shapeLayer = CAShapeLayer()
            shapeLayer.frame = detectionOverlay.bounds
            let path       = UIBezierPath()

            guard let first = polygon.first else { return container }
            path.move(to: first)
            polygon.dropFirst().forEach { path.addLine(to: $0) }
            path.close()

            shapeLayer.path        = path.cgPath
            shapeLayer.strokeColor = color.withAlphaComponent(0.4).cgColor
            shapeLayer.fillColor   = color.withAlphaComponent(0.4).cgColor         // fill is from bitmap
            shapeLayer.lineWidth   = 2
            shapeLayer.lineJoin    = .round
        
            

        
            detectionOverlay.addSublayer(shapeLayer)

            return container
        }
    
    //Create a layer displaying the classification result and it's confidence
    func createTextSubLayerInBounds(_ bounds: CGRect, identifier: String, confidence: VNConfidence) -> CATextLayer {
        let textLayer = CATextLayer()
        textLayer.name = "Object Label"
        let confidenceString: String = ("x: \(bounds.midX) y: \(bounds.midY)")//("Confidence: \(confidence)")
        
        let formattedString = NSMutableAttributedString(string: String(format: "\(identifier)\n\(confidenceString)"))
        let largeFont = UIFont(name: "Helvetica", size: 24.0)!
        
        formattedString.addAttributes([NSAttributedString.Key.font: largeFont], range: NSRange(location: 0, length: identifier.count))
        formattedString.addAttribute(NSAttributedString.Key.foregroundColor, value: UIColor.white, range: NSRange(location: 0, length: identifier.count + confidenceString.count + 1))
        
        textLayer.string = formattedString
        textLayer.bounds = CGRect(x: 0, y: 0, width: bounds.size.height - 10, height: bounds.size.width - 10)
        textLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        textLayer.shadowOffset = CGSize(width: 2, height: 2)
        textLayer.foregroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [0.0, 0.0, 0.0, 1.0])
        textLayer.contentsScale = 2.0 // retina rendering
        
        // Rotate the layer into screen orientation and scale and mirror
//        textLayer.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: -1.0, y: -1.0))
        return textLayer
    }
    
    //Creates the shape for bounding boxes to be displayed on the screen
    func createRoundedRectLayerWithBounds(_ bounds: CGRect, color: UIColor) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.origin.x, y: bounds.origin.y)
        shapeLayer.name = "Found Object"
        
        var colorComponents = color.cgColor.components
        colorComponents?.removeLast()
        colorComponents?.append(0.4)
        shapeLayer.backgroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: colorComponents!)
        shapeLayer.cornerRadius = 7
        return shapeLayer
    }
    
    // Keep the detection overlay aligned with the preview. Boxes are positioned in
    // screen coords via layerRectConverted, so no rotation transform is needed.
    func updateLayerGeometry() {
        detectionOverlay?.frame = rootLayer.bounds
    }
    
    //--------------------------
    //MARK: Image Uploading
    //--------------------------
    
    //Starts upload flow for if a user wants to upload the camera frame where an incorrect image classification occured
    @IBAction func uploadImage(_ sender: Any) {
        
        //Capture the current pixel buffer of the camera and convert it an image
        guard let pixelBuffer = currentPixelBuffer else {
            return
        }
        
        guard let capturedImage = UIImage(pixelBuffer: pixelBuffer) else {
            return
        }
        
        let rotatedImage = capturedImage.rotateImage(orientation: .down)
        
        let alert = UIAlertController(title: "Incorrect count?", message: "You've captured an image of this wrong count. Upload it to the open source dataset to improve this model.", preferredStyle: .alert)
        let imageView = UIImageView(frame: CGRect(x: 10, y: 100, width: 250, height: 230))
        imageView.image = rotatedImage
        alert.view.addSubview(imageView)
        let height = NSLayoutConstraint(item: alert.view!, attribute: .height, relatedBy: .equal, toItem: nil, attribute: .notAnAttribute, multiplier: 1, constant: 375)
        let width = NSLayoutConstraint(item: alert.view!, attribute: .width, relatedBy: .equal, toItem: nil, attribute: .notAnAttribute, multiplier: 1, constant: 250)
        alert.view.addConstraint(height)
        alert.view.addConstraint(width)
        
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { (_) in
        }))
        alert.addAction(UIAlertAction(title: "Upload", style: .default, handler: { [self] (_) in
            //Upload the captured image to your dataset
            upload(image: rotatedImage)
        }))
        
        self.present(alert, animated: true, completion: nil)
    }
    
    //Uploads the incorrect classification frame
    func upload(image: UIImage) {
        let project = "weed-crop-aerial-mbyst"
        
        rf.uploadImage(image: image, project: project) { result in
            var title: String!
            var message: String!
            
            switch result {
            case .Success:
                title = "Success!"
                message = " Your image has been uploaded to the open source training dataset for model improvement."
            case .Duplicate:
                title = "Duplicate"
                message = "You attempted to upload a duplicate image."
            case .Error:
                title = "Error"
                message = "An error occured while uploading your image."
            @unknown default:
                return
            }
            
            DispatchQueue.main.async {
                let alert = UIAlertController(title: title, message: message, preferredStyle: UIAlertController.Style.alert)
                alert.addAction(UIAlertAction(title: "OK", style: .cancel, handler: { (_) in
                }))
                self.present(alert, animated: true, completion: nil)
            }
        }
    }
    
}
