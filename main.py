import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from data_exploration import DataExplorer
from preprocessor import MedicalImagePreprocessor
from feature_engineering import MedicalFeatureExtractor
from data_augmentation import *
from evaluation_metrics import MedicalEvaluator
from differ_arch import *
from grad import *

train_zip = "train.zip"
train_csv = "train_labels.csv"
test_zip = "test.zip"
output_dir = "outputs"
show_plots = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    """
    Main pipeline for histopathology demo.
    """

    explorer = DataExplorer(train_zip, train_csv, test_zip,
                            output_dir=output_dir, show_plots=show_plots)

    explorer.explore_dataset_statistics()
    # Total samples: 220025
    # Benign (0): 130,908 (59.5%)
    # Malignant (1): 89,117 (40.5%)

    # explorer.visualize_sample_images(n=12, seed=42, filename_prefix="samples_grid")
    # explorer.analyze_image_properties(n=300, filename_prefix="image_size_stats")

    # preprocessor
    preprocessor = MedicalImagePreprocessor(output_dir="preprocessed_results", show_plots=True)

    # make a fake H&E-like image
    sample_image = np.random.randint(100, 255, (96, 96, 3), dtype=np.uint8)
    sample_image[:, :, 0] = np.clip(sample_image[:, :, 0] * 0.8, 0, 255)  # less red
    sample_image[:, :, 2] = np.clip(sample_image[:, :, 2] * 1.2, 0, 255)  # more blue

    normalized_image = preprocessor.he_stain_separation(sample_image)

    # quick viz
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sample_image)
    axes[0].set_title('Original H&E Image')
    axes[0].axis('off')

    axes[1].imshow(normalized_image)
    axes[1].set_title('After Stain Normalization')
    axes[1].axis('off')

    diff = np.abs(sample_image.astype(np.float32) - normalized_image.astype(np.float32))
    axes[2].imshow(diff.astype(np.uint8))
    axes[2].set_title('Normalization Effect')
    axes[2].axis('off')

    plt.suptitle('H&E Stain Normalization')
    plt.tight_layout()
    preprocessor._savefig("he_normalization_demo.png")

    feature_extractor = MedicalFeatureExtractor()
    sample_features = feature_extractor.create_feature_vector(sample_image)

    print("features (first 10):")
    for feature, value in list(sample_features.items())[:10]:
        print(f"  {feature}: {value:.3f}")
    print(f"  ... {len(sample_features) - 10} more")

    train_transform, val_transform = create_medical_transforms()
    print("data aug set:")
    print("- rotations (0/90/180/270)")
    print("- color jitter (stain)")
    print("- spatial transforms")
    print("- normalize (pretrained)")

    # model
    model = MedicalResNet50(num_classes=2, pretrained=True)
    model = model.to(device).eval()
    print("model: ResNet50 (with attention)")
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # fake loaders
    synthetic_images = [torch.randn(3, 96, 96) for _ in range(200)]
    synthetic_targets = [np.random.choice([0, 1]) for _ in range(200)]

    train_imgs, val_imgs = synthetic_images[:160], synthetic_images[160:]
    train_targets, val_targets = synthetic_targets[:160], synthetic_targets[160:]

    train_dataset = MedicalImageDataset(train_imgs, train_targets, train_transform)
    val_dataset = MedicalImageDataset(val_imgs, val_targets, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # trainer
    trainer = MedicalTrainer(model, device)

    # simulate training (demo only)
    print("simulate training ...")

    simulated_epochs = 100
    simulated_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_metrics': []
    }

    for epoch in range(simulated_epochs):
        # simple trending numbers
        train_loss = 0.8 - (epoch * 0.04) + np.random.normal(0, 0.02)
        train_acc = 0.65 + (epoch * 0.02) + np.random.normal(0, 0.01)
        val_loss = 0.75 - (epoch * 0.035) + np.random.normal(0, 0.015)
        val_acc = 0.70 + (epoch * 0.018) + np.random.normal(0, 0.008)

        simulated_metrics = {
            'accuracy': val_acc,
            'sensitivity': 0.85 + (epoch * 0.01) + np.random.normal(0, 0.005),
            'specificity': 0.88 + (epoch * 0.008) + np.random.normal(0, 0.005),
            'precision': 0.86 + (epoch * 0.009) + np.random.normal(0, 0.005),
            'npv': 0.87 + (epoch * 0.007) + np.random.normal(0, 0.005),
            'f1_score': 0.85 + (epoch * 0.01) + np.random.normal(0, 0.005),
            'auc_roc': 0.88 + (epoch * 0.007) + np.random.normal(0, 0.003),
            'confusion_matrix': np.array([[35, 3], [2, 37]]),
            'tp': 37, 'tn': 35, 'fp': 3, 'fn': 2
        }

        simulated_history['train_loss'].append(max(0.1, train_loss))
        simulated_history['train_acc'].append(min(0.99, max(0.6, train_acc)))
        simulated_history['val_loss'].append(max(0.1, val_loss))
        simulated_history['val_acc'].append(min(0.98, max(0.65, val_acc)))
        simulated_history['val_metrics'].append(simulated_metrics)

        if epoch % 3 == 0:
            print(f"epoch {epoch + 1:02d} | train_loss={simulated_history['train_loss'][-1]:.4f} "
                  f"| val_auc={simulated_metrics['auc_roc']:.4f}")

    trainer.history = simulated_history
    trainer.plot_training_history(save_path=os.path.join(output_dir, "training_history.png"))

    final_metrics = simulated_history['val_metrics'][-1]

    evaluator = MedicalEvaluator()
    # build arrays roughly matching the cm above
    evaluator.all_targets = [0] * 35 + [1] * 37 + [0] * 3 + [1] * 2
    evaluator.all_predictions = [0] * 35 + [0] * 3 + [1] * 37 + [0] * 2
    evaluator.all_probabilities = (
        [0.1 + np.random.rand() * 0.3 for _ in range(38)] +
        [0.7 + np.random.rand() * 0.3 for _ in range(39)]
    )

    evaluator.plot_comprehensive_results(final_metrics, save_path="outputs/eval_report.png")

    # Grad-CAM
    target_layer = None
    for m in reversed(list(model.backbone.modules())):
        if isinstance(m, nn.Conv2d):
            target_layer = m
            break
    assert target_layer is not None, "Target convolutional layer not found"

    if target_layer is not None:
        gradcam = MedicalGradCAM(model, target_layer)

        sample_tensor = torch.randn(3, 96, 96, device=device)
        true_label = 1
        predicted_class = 1
        probability = 0.92

        print("make grad-cam example ...")

        os.makedirs(output_dir, exist_ok=True)
        heatmap, attention_stats = gradcam.visualize_medical_gradcam(
            sample_tensor, true_label, predicted_class, probability,
            save_path=os.path.join(output_dir, "gradcam_demo.png")
        )

        print("attention stats:")
        print(f"  center:    {attention_stats['center']:.3f}")
        print(f"  periphery: {attention_stats['periphery']:.3f}")
        print(f"  background:{attention_stats['background']:.3f}")

        gradcam.cleanup()

        # short, plain summary
        deployment_summary = (
            "summary:\n"
            f"- model: ResNet50 (attention)\n"
            f"- auc (val): {final_metrics['auc_roc']:.3f}\n"
            f"- sensitivity: {final_metrics['sensitivity']:.3f}\n"
            f"- specificity: {final_metrics['specificity']:.3f}\n"
            "- preprocessing: H&E normalization\n"
            "- explainability: Grad-CAM figure saved\n"
            "next:\n"
            "1) run on real data\n"
            "2) compare with pathologist labels\n"
            "3) check results across centers\n"
        )
        print(deployment_summary)
        print("pipeline done")     
        print("components:")
        print("preprocessing (H&E)")
        print("model + attention")
        print("eval metrics")
        print("grad-cam")
        print("simple summary")

if __name__ == "__main__":
    main()
