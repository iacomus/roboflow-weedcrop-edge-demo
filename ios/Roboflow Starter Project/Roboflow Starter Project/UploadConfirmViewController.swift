import UIKit

/// Confirmation sheet for the "Upload Incorrect Image" data-flywheel flow.
/// Shows the captured frame, an explanation, and Cancel/Upload — laid out with
/// Auto Layout in a real view controller (UIAlertController doesn't support
/// custom subviews, which is why the old in-alert image hack overflowed).
final class UploadConfirmViewController: UIViewController {

    private let capturedImage: UIImage
    private let onConfirm: () -> Void

    init(image: UIImage, onConfirm: @escaping () -> Void) {
        self.capturedImage = image
        self.onConfirm = onConfirm
        super.init(nibName: nil, bundle: nil)
        modalPresentationStyle = .pageSheet
        if let sheet = sheetPresentationController {
            // Size the sheet to its content; allow expanding to full height for
            // large Dynamic Type.
            sheet.detents = [.custom { _ in 480 }, .large()]
            sheet.prefersGrabberVisible = true
            sheet.preferredCornerRadius = 24
        }
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground

        let titleLabel = UILabel()
        titleLabel.text = "Incorrect count?"
        titleLabel.font = .systemFont(ofSize: 22, weight: .bold)
        titleLabel.numberOfLines = 0

        let messageLabel = UILabel()
        messageLabel.text = "This frame shows the wrong count. Upload it to the open-source dataset to help improve the model."
        messageLabel.font = .systemFont(ofSize: 15)
        messageLabel.textColor = .secondaryLabel
        messageLabel.numberOfLines = 0

        let imageView = UIImageView(image: capturedImage)
        imageView.contentMode = .scaleAspectFill
        imageView.clipsToBounds = true
        imageView.layer.cornerRadius = 12
        imageView.heightAnchor.constraint(equalToConstant: 200).isActive = true

        var cancelCfg = UIButton.Configuration.gray()
        cancelCfg.title = "Cancel"
        let cancelButton = UIButton(configuration: cancelCfg)
        cancelButton.addAction(UIAction { [weak self] _ in
            self?.dismiss(animated: true)
        }, for: .touchUpInside)

        var uploadCfg = UIButton.Configuration.borderedProminent()
        uploadCfg.title = "Upload"
        let uploadButton = UIButton(configuration: uploadCfg)
        uploadButton.addAction(UIAction { [weak self] _ in
            guard let self else { return }
            let confirm = self.onConfirm
            self.dismiss(animated: true) { confirm() }
        }, for: .touchUpInside)

        let buttonRow = UIStackView(arrangedSubviews: [cancelButton, uploadButton])
        buttonRow.axis = .horizontal
        buttonRow.distribution = .fillEqually
        buttonRow.spacing = 12

        let stack = UIStackView(arrangedSubviews: [titleLabel, messageLabel, imageView, buttonRow])
        stack.axis = .vertical
        stack.spacing = 16
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 24),
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
        ])
    }
}
