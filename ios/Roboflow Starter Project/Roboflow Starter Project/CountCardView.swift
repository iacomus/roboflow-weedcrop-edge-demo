import UIKit

/// Bottom overlay card: crop stand count as hero, weed count shown as excluded.
final class CountCardView: UIView {
    let cropValueLabel = UILabel()
    let cropCaptionLabel = UILabel()
    let weedLabel = UILabel()

    override init(frame: CGRect) {
        super.init(frame: frame)
        setup()
    }
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }

    private func setup() {
        backgroundColor = UIColor.black.withAlphaComponent(0.55)
        layer.cornerRadius = 14

        cropValueLabel.font = .systemFont(ofSize: 34, weight: .bold)
        cropValueLabel.textColor = .systemGreen
        cropCaptionLabel.font = .systemFont(ofSize: 12, weight: .medium)
        cropCaptionLabel.textColor = .white
        cropCaptionLabel.text = "Crops detected"
        weedLabel.font = .systemFont(ofSize: 13, weight: .regular)
        weedLabel.textColor = UIColor.white.withAlphaComponent(0.7)

        let left = UIStackView(arrangedSubviews: [cropValueLabel, cropCaptionLabel])
        left.axis = .vertical
        left.spacing = 0

        let row = UIStackView(arrangedSubviews: [left, weedLabel])
        row.axis = .horizontal
        row.alignment = .center
        row.distribution = .equalSpacing
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)
        NSLayoutConstraint.activate([
            row.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            row.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            row.topAnchor.constraint(equalTo: topAnchor, constant: 10),
            row.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -10),
        ])
        showEmpty()
    }

    func update(crops: Int, weeds: Int) {
        cropValueLabel.text = "\(crops)"
        weedLabel.text = "Weeds \(weeds) · excluded"
    }

    /// Cloud-offline / no-data state.
    func showEmpty() {
        cropValueLabel.text = "—"
        weedLabel.text = "Weeds — · excluded"
    }
}
