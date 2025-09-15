import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from evaluation_metrics import MedicalEvaluator


class MedicalGradCAM:
    """
    Grad-CAM for binary medical classification.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # hooks
        self.handles = []
        self.handles.append(target_layer.register_forward_hook(self._save_activation))
        # use full backward hook to avoid deprecation warnings
        self.handles.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, class_idx=None):
        """Generate CAM heatmap."""
        self.model.eval()
        output, _ = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        if self.gradients is not None and self.activations is not None:
            pooled_gradients = torch.mean(self.gradients, dim=[2, 3])
            # weight channels
            for i in range(self.activations.shape[1]):
                self.activations[:, i, :, :] *= pooled_gradients[:, i]

            heatmap = torch.mean(self.activations, dim=1).squeeze()
            heatmap = F.relu(heatmap)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
        else:
            # fallback
            heatmap = torch.rand(3, 3)

        return heatmap.cpu().numpy()

    def visualize_medical_gradcam(self, image_tensor, true_label, predicted_class,
                                  probability, save_path=None):
        """
        Plot original image, heatmap, overlay and simple notes.
        """
        heatmap = self.generate_cam(image_tensor.unsqueeze(0), predicted_class)

        # unnormalize (ImageNet mean/std)
        img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        # resize to patchcamelyon 96x96
        heatmap_resized = cv2.resize(heatmap, (96, 96))

        fig = plt.figure(figsize=(16, 10))

        # 1) original
        plt.subplot(2, 4, 1)
        plt.imshow(img_np)
        plt.title('Original', fontweight='bold')
        plt.axis('off')

        # 2) heatmap
        plt.subplot(2, 4, 2)
        im = plt.imshow(heatmap_resized, cmap='jet', alpha=0.8)
        plt.title('Grad-CAM', fontweight='bold')
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)

        # 3) overlay
        plt.subplot(2, 4, 3)
        plt.imshow(img_np)
        plt.imshow(heatmap_resized, cmap='jet', alpha=0.4)

        center = 48
        rect_size = 16
        rect = patches.Rectangle((center - rect_size, center - rect_size),
                                 rect_size * 2, rect_size * 2,
                                 linewidth=2, edgecolor='white',
                                 facecolor='none', linestyle='--')
        plt.gca().add_patch(rect)

        true_label_text = 'Benign' if true_label == 0 else 'Malignant'
        pred_label_text = 'Benign' if predicted_class == 0 else 'Malignant'

        plt.title(f'Overlay\nGT: {true_label_text} | Pred: {pred_label_text} ({probability:.3f})',
                  fontweight='bold')
        plt.axis('off')

        # 4) attention bars
        plt.subplot(2, 4, 4)
        attention_stats = self._analyze_attention_distribution(heatmap_resized)
        plt.bar(['Center(32x32)', 'Periphery', 'Background'],
                [attention_stats['center'], attention_stats['periphery'], attention_stats['background']],
                color=['red', 'orange', 'lightblue'])
        plt.title('Attention Split', fontweight='bold')
        plt.ylabel('Score')
        plt.ylim(0, 1)

        # 5) notes
        plt.subplot(2, 1, 2)
        plt.axis('off')

        if attention_stats['center'] > 0.6:
            focus_note = "Center focus: high"
        elif attention_stats['center'] > 0.3:
            focus_note = "Center focus: medium"
        else:
            focus_note = "Center focus: low"

        # keep the text short and plain
        text_block = (
            "Notes:\n"
            f"Center score: {attention_stats['center']:.3f} ({focus_note})\n"
            f"Confidence: {probability:.3f}\n"
            f"Pred: {pred_label_text} | GT: {true_label_text}\n\n"
        )

        plt.text(0.02, 0.98, text_block, transform=plt.gca().transAxes,
                 fontsize=11, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#eaf6ff", alpha=0.8, edgecolor='gray'))

        plt.suptitle('Grad-CAM (Binary Histopathology)', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return heatmap_resized, attention_stats

    def _analyze_attention_distribution(self, heatmap):
        """Compute simple region averages."""
        h, w = heatmap.shape
        center_h, center_w = h // 2, w // 2

        center_mask = np.zeros_like(heatmap, dtype=np.uint8)
        center_mask[center_h - 16:center_h + 16, center_w - 16:center_w + 16] = 1
        periphery_mask = 1 - center_mask

        center_attention = float(np.mean(heatmap[center_mask == 1])) if center_mask.sum() > 0 else 0.0
        periphery_attention = float(np.mean(heatmap[periphery_mask == 1])) if periphery_mask.sum() > 0 else 0.0
        background_attention = float(np.mean(heatmap[heatmap < 0.1])) if (heatmap < 0.1).any() else 0.0

        return {
            'center': center_attention,
            'periphery': periphery_attention,
            'background': background_attention
        }

    def cleanup(self):
        """Remove hooks."""
        for handle in self.handles:
            handle.remove()

class MedicalTrainer:
    """
    Basic training loop with validation.
    """

    def __init__(self, model, device, num_classes=2):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.evaluator = MedicalEvaluator()

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_metrics': []
        }

    def train_epoch(self, train_loader, criterion, optimizer):
        """Single epoch training."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output, _ = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            if batch_idx % 50 == 0:
                print(f'[train] batch {batch_idx} | loss {loss.item():.4f}')

        avg_loss = running_loss / max(len(train_loader), 1)
        acc = correct / max(total, 1)
        return avg_loss, acc

    def validate_epoch(self, val_loader, criterion):
        """Validation + metrics."""
        self.model.eval()
        self.evaluator.reset()
        running_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                loss = criterion(output, target)
                running_loss += loss.item()

                pred = output.argmax(dim=1)
                prob = F.softmax(output, dim=1)[:, 1]
                self.evaluator.update(pred, target, prob)

        metrics = self.evaluator.calculate_metrics()
        avg_loss = running_loss / max(len(val_loader), 1)
        return avg_loss, metrics

    def train_model(self, train_loader, val_loader, epochs=20, lr=1e-4, weight_decay=1e-5):
        """Main training loop."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        print("=" * 50)
        print("start training")
        print(f"epochs={epochs} lr={lr} wd={weight_decay} device={self.device}")
        print("=" * 50)

        best_auc = 0.0

        for epoch in range(epochs):
            print(f"\nepoch {epoch + 1}/{epochs}")

            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_metrics = self.validate_epoch(val_loader, criterion)

            scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_metrics'].append(val_metrics)

            print(f"[train] loss={train_loss:.4f} acc={train_acc:.4f}")
            print(f"[valid] loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} "
                  f"auc={val_metrics['auc_roc']:.4f} "
                  f"sens={val_metrics['sensitivity']:.4f} "
                  f"spec={val_metrics['specificity']:.4f}")

            if val_metrics['auc_roc'] > best_auc:
                best_auc = val_metrics['auc_roc']
                torch.save(self.model.state_dict(), 'best_medical_model.pth')
                print(f"[model] best auc updated to {best_auc:.4f} (saved)")

        print("\nfinished")
        print(f"best auc: {best_auc:.4f}")
        return self.history

    def plot_training_history(self, save_path=None):
        """Plot loss/acc and a few metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # loss
        axes[0, 0].plot(self.history['train_loss'], label='train', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='val', color='red')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # acc
        axes[0, 1].plot(self.history['train_acc'], label='train', color='blue')
        axes[0, 1].plot(self.history['val_acc'], label='val', color='red')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # metrics
        auc_scores = [m['auc_roc'] for m in self.history['val_metrics']]
        sensitivity_scores = [m['sensitivity'] for m in self.history['val_metrics']]
        specificity_scores = [m['specificity'] for m in self.history['val_metrics']]

        axes[1, 0].plot(auc_scores, label='AUC', color='green', linewidth=2)
        axes[1, 0].plot(sensitivity_scores, label='Sensitivity', color='orange', linewidth=2)
        axes[1, 0].plot(specificity_scores, label='Specificity', color='purple', linewidth=2)
        axes[1, 0].set_title('Validation Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # simple summary
        axes[1, 1].axis('off')
        final_metrics = self.history['val_metrics'][-1]

        summary_text = (
            "Final (val):\n"
            f"AUC:         {final_metrics['auc_roc']:.4f}\n"
            f"Sensitivity: {final_metrics['sensitivity']:.4f}\n"
            f"Specificity: {final_metrics['specificity']:.4f}\n"
            f"Precision:   {final_metrics['precision']:.4f}\n"
            f"F1:          {final_metrics['f1_score']:.4f}\n"
            f"Accuracy:    {final_metrics['accuracy']:.4f}\n"
            f"NPV:         {final_metrics['npv']:.4f}"
        )

        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, va='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e6ffe6", alpha=0.7))

        plt.suptitle('Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
