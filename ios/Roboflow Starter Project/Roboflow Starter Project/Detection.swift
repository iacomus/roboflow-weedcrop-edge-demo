import CoreGraphics

/// One decoded detection in normalized [0,1] coordinates, origin top-left, x/y/w/h.
/// Class channels: 1 = crop, 2 = weed.
struct Detection {
    let classId: Int
    let label: String
    let confidence: Float
    let rect: CGRect

    /// Per-frame crop/weed counts. Crop is the demo's hero (stand count);
    /// weeds are counted so they can be shown as the excluded confounder.
    static func counts(from detections: [Detection]) -> (crops: Int, weeds: Int) {
        var crops = 0
        var weeds = 0
        for d in detections {
            if d.classId == 1 { crops += 1 }
            else if d.classId == 2 { weeds += 1 }
        }
        return (crops, weeds)
    }
}
