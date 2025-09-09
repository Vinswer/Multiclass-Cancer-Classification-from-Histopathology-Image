import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report,
                           roc_curve, auc, precision_score, recall_score, f1_score)
import seaborn as sns

class MedicalEvaluator:
    """
    Comprehensive medical evaluation metrics for clinical validation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []

    def update(self, predictions, targets, probabilities):
        """Update with batch results."""
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        self.all_probabilities.extend(probabilities.cpu().numpy())

    def calculate_metrics(self):
        """Calculate comprehensive medical metrics."""
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        y_prob = np.array(self.all_probabilities)

        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # PPV
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # NPV
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # F1 score and AUC
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_prob)

        # Clinical interpretation
        metrics = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'npv': npv,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }

        return metrics

    def plot_comprehensive_results(self, metrics, save_path=None):
        """Create comprehensive visualization of medical metrics."""
        fig = plt.figure(figsize=(16, 12))

        # Confusion Matrix
        plt.subplot(2, 3, 1)
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d',
                    cmap='Blues', cbar=True,
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Clinical Metrics
        plt.subplot(2, 3, 2)
        clinical_metrics = ['Sensitivity', 'Specificity', 'Precision', 'NPV']
        values = [metrics['sensitivity'], metrics['specificity'],
                  metrics['precision'], metrics['npv']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        bars = plt.bar(clinical_metrics, values, color=colors)
        plt.ylim(0, 1)
        plt.title('Clinical Performance Metrics')
        plt.xticks(rotation=45)

        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # ROC Curve
        plt.subplot(2, 3, 3)
        fpr, tpr, _ = roc_curve(self.all_targets, self.all_probabilities)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC Curve (AUC = {metrics["auc_roc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Overall Performance
        plt.subplot(2, 3, 4)
        overall_metrics = ['Accuracy', 'F1-Score', 'AUC-ROC']
        overall_values = [metrics['accuracy'], metrics['f1_score'], metrics['auc_roc']]
        colors = ['#9B59B6', '#E67E22', '#E74C3C']

        bars = plt.bar(overall_metrics, overall_values, color=colors)
        plt.ylim(0, 1)
        plt.title('Overall Model Performance')

        for bar, value in zip(bars, overall_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # Clinical Interpretation
        plt.subplot(2, 3, 5)
        plt.axis('off')

        # Performance assessment
        if metrics['auc_roc'] >= 0.95:
            performance_level = "EXCELLENT"
            color = "green"
        elif metrics['auc_roc'] >= 0.85:
            performance_level = "GOOD"
            color = "orange"
        else:
            performance_level = "MODERATE"
            color = "red"

        interpretation = f"""
Clinical Performance Summary:

• Sensitivity: {metrics['sensitivity']:.3f}
  (Malignant case detection rate)

• Specificity: {metrics['specificity']:.3f}
  (Benign case identification rate)

• Precision (PPV): {metrics['precision']:.3f}
  (Malignant prediction accuracy)

• AUC-ROC: {metrics['auc_roc']:.3f}
  (Overall discriminative ability)

• Performance Level: {performance_level}

Clinical Significance:
{performance_level.title()} performance for
pathologist-assisted diagnosis
        """

        plt.text(0.05, 0.95, interpretation, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

        # Diagnostic Statistics
        plt.subplot(2, 3, 6)
        plt.axis('off')

        stats_text = f"""
Diagnostic Statistics:

True Positives (TP): {metrics['tp']}
True Negatives (TN): {metrics['tn']}
False Positives (FP): {metrics['fp']}
False Negatives (FN): {metrics['fn']}

Total Samples: {sum([metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']])}

Error Analysis:
• Type I Error (FP Rate): {metrics['fp'] / (metrics['fp'] + metrics['tn']):.3f}
• Type II Error (FN Rate): {metrics['fn'] / (metrics['fn'] + metrics['tp']):.3f}

Clinical Impact:
• Missed Cancers: {metrics['fn']} cases
• False Alarms: {metrics['fp']} cases
        """

        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

        plt.suptitle('Medical AI Performance Evaluation Report',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


