from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from Logistic_Regression.data_utils import load_data, split_data, scale_data

class LogisticRegressionModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = None
        self.scaler = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.feature_names = None
        self.target_names = None
        self.y_pred = None
        self.y_proba = None

    def load(self):
        X, y, self.feature_names, self.target_names = load_data(self.csv_path)
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train_scaled, X_test_scaled, y_train, y_test
        self.scaler = scaler

    def train(self, C=1.0, max_iter=1000):
        self.model = LogisticRegression(C=C, max_iter=max_iter)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.y_proba = self.model.predict_proba(self.X_test)

    def metrics(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred, target_names=self.target_names)
        return acc, report

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.set_xticks(np.arange(len(self.target_names)))
        ax.set_yticks(np.arange(len(self.target_names)))
        ax.set_xticklabels(self.target_names)
        ax.set_yticklabels(self.target_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        # Add text inside matrix
        for i in range(len(self.target_names)):
            for j in range(len(self.target_names)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
        return fig

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba[:,1])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0,1], [0,1], color='gray', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        return fig

    def plot_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba[:,1])
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='green')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        return fig

    def plot_probability_distribution(self):
        fig, ax = plt.subplots()
        ax.hist(self.y_proba[:,1], bins=10, alpha=0.7)
        ax.set_xlabel('Predicted Probability for Fire')
        ax.set_ylabel('Count')
        ax.set_title('Class Probability Distribution')
        return fig
