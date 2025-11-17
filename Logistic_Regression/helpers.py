import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    return plt

def plot_roc_curves(y_true, y_proba, num_classes):
    plt.figure(figsize=(6,4))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {i}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve (OvR)")
    plt.legend()
    return plt

def plot_precision_recall(y_true, y_proba, num_classes):
    plt.figure(figsize=(6,4))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_proba[:, i])
        plt.plot(recall, precision, label=f"Class {i}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves")
    plt.legend()
    return plt

def plot_probability_distribution(y_proba, num_classes):
    plt.figure(figsize=(7,4))
    for i in range(num_classes):
        sns.kdeplot(y_proba[:, i], fill=True, label=f"Class {i}")
    plt.title("Class Probability Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend()
    return plt
