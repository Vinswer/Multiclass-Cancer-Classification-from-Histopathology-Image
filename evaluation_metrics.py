import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve, f1_score
)
import seaborn as sns

class MedicalEvaluator:
    """
    Simple evaluator for binary medical classification.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []

    def update(self, predictions, targets, probabilities):
        """Add a batch of results."""
        # to numpy
        preds = predictions.detach().cpu().numpy()
        targs = targets.detach().cpu().numpy()
        probs = probabilities.detach().cpu().numpy()

        if probs.ndim == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]
        else:
            probs = probs.ravel()

        self.all_predictions.extend(preds)
        self.all_targets.extend(targs)
        self.all_probabilities.extend(probs)

    def calculate_metrics(self):
        """Compute common metrics."""
        y_true = np.asarray(self.all_targets).ravel()
        y_pred = np.asarray(self.all_predictions).ravel()
        y_prob = np.asarray(self.all_probabilities).ravel()

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            if cm.shape == (1, 1):
                if np.unique(y_true)[0] == 0:
                    tn = cm[0, 0]
                else:
                    tp = cm[0, 0]
            elif cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        denom = tp + tn + fp + fn
        accuracy = (tp + tn) / denom if denom > 0 else 0.0

        try:
            f1 = f1_score(y_true, y_pred)
        except Exception:
            f1 = 0.0

        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc_roc = 0.0

        metrics = {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "npv": npv,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "confusion_matrix": cm,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn
        }
        return metrics

    def plot_comprehensive_results(self, metrics, save_path=None):
        """Plot confusion matrix, bars, ROC and a short text summary."""
        fig = plt.figure(figsize=(16, 12))

        # 1) Confusion Matrix
        plt.subplot(2, 3, 1)
        sns.heatmap(
            metrics["confusion_matrix"], annot=True, fmt="d",
            cmap="Blues", cbar=True,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"]
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True")
        plt.xlabel("Pred")

        # 2) Clinical Metrics (bar)
        plt.subplot(2, 3, 2)
        names = ["Sensitivity", "Specificity", "Precision", "NPV"]
        vals = [
            metrics["sensitivity"], metrics["specificity"],
            metrics["precision"], metrics["npv"]
        ]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        bars = plt.bar(names, vals, color=colors)
        plt.ylim(0, 1)
        plt.title("Clinical Metrics")
        plt.xticks(rotation=30)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom")

        # 3) ROC Curve
        plt.subplot(2, 3, 3)
        y_true = np.asarray(self.all_targets).ravel()
        y_prob = np.asarray(self.all_probabilities).ravel()
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['auc_roc']:.3f}")
        except Exception:
            plt.plot([0, 1], [0, 1], lw=2, linestyle="--", alpha=0.5, label="ROC N/A")
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--", alpha=0.5)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("FPR (1 - Specificity)")
        plt.ylabel("TPR (Sensitivity)")
        plt.title("ROC")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # 4) Overall Metrics (bar)
        plt.subplot(2, 3, 4)
        overall_names = ["Accuracy", "F1", "AUC"]
        overall_vals = [
            metrics["accuracy"], metrics["f1_score"], metrics["auc_roc"]
        ]
        colors2 = ["#9B59B6", "#E67E22", "#E74C3C"]
        bars2 = plt.bar(overall_names, overall_vals, color=colors2)
        plt.ylim(0, 1)
        plt.title("Overall")
        for b, v in zip(bars2, overall_vals):
            plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom")

        # 5) Short Notes
        plt.subplot(2, 3, 5)
        plt.axis("off")
        auc_val = metrics["auc_roc"]
        if auc_val >= 0.95:
            level = "Excellent"
        elif auc_val >= 0.85:
            level = "Good"
        else:
            level = "OK"

        notes = (
            f"Notes:\n"
            f"- Sensitivity: {metrics['sensitivity']:.3f}\n"
            f"- Specificity: {metrics['specificity']:.3f}\n"
            f"- Precision:   {metrics['precision']:.3f}\n"
            f"- AUC:         {auc_val:.3f}\n"
            f"- Level:       {level}"
        )
        plt.text(0.05, 0.95, notes, transform=plt.gca().transAxes,
                 fontsize=10, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#eeeeee", alpha=0.8))

        # 6) Counts & Error Rates
        plt.subplot(2, 3, 6)
        plt.axis("off")
        tp, tn, fp, fn = metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]
        total = tp + tn + fp + fn
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        stats = (
            f"Counts:\n"
            f"TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}\n"
            f"Total: {total}\n\n"
            f"Error Rates:\n"
            f"FP Rate: {fp_rate:.3f}\n"
            f"FN Rate: {fn_rate:.3f}"
        )
        plt.text(0.05, 0.95, stats, transform=plt.gca().transAxes,
                 fontsize=10, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff6cc", alpha=0.8))

        plt.suptitle("Model Eval (Binary)", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
