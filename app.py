import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay, roc_auc_score
import matplotlib.pyplot as plt

st.title("🔥 Algerian Forest Fire Prediction")

# Load dataset directly
df = pd.read_csv(".//Algerian_forest_fires_cleaned_dataset.csv")
df.columns = [c.strip() for c in df.columns]

# Drop Region column if exists
if "Region" in df.columns:
    df = df.drop(columns=["Region"])

# Target and features
target = "Classes"
X = df.drop(columns=[target]).values
y = df[target].str.strip().str.lower().map({"not fire": 0, "fire": 1}).values
feature_names = df.drop(columns=[target]).columns.tolist()

# Hyperparameters
st.write("### Hyperparameters")
C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
max_iter = st.slider("Max Iterations", 100, 1000, 500)

# Train Button
if st.button("Train Model"):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Logistic Regression
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    st.success("✅ Model Trained Successfully!")

    # Learned Coefficients
    st.write("### Learned Coefficients")
    coefs = dict(zip(feature_names, model.coef_[0]))
    st.json(coefs)

    # Predictions vs Actual
    results = pd.DataFrame({"Actual": y, "Predicted": y_pred})
    st.write("### Predictions vs Actual")
    st.dataframe(results.head())

    # RMSE
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    st.metric("RMSE", rmse)

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)

    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    st.write("Recall", recall)
    st.write("Precision", precision)

    # ROC Curve
    st.write("### ROC AUC Curve")
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = roc_auc_score(y, y_proba)
    fig, ax = plt.subplots()
    # RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)
    st.pyplot(fig)
    st.write(f"**ROC AUC Score:** {roc_auc:.4f}")

    # Precision-Recall Curve
    st.write("### Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y, y_proba)
    fig, ax = plt.subplots()
    PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax)
    st.pyplot(fig)

    # Probability Distribution
    st.write("### Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(y_proba, bins=20, color="skyblue", edgecolor="black")
    ax.set_xlabel("Predicted Probability of Fire")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

