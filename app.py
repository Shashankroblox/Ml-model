import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="XGBoost Classifier", page_icon="🤖", layout="centered")

st.title("🤖 XGBoost Classifier")
st.markdown("Upload your dataset, train an XGBoost model, and get results instantly.")

# --- Sidebar config ---
st.sidebar.header("Model Settings")
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
cv_folds = st.sidebar.slider("Cross-validation folds", 3, 15, 10)
random_state = st.sidebar.number_input("Random state", value=0, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**XGBoost Params**")
n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100, 50)
max_depth = st.sidebar.slider("max_depth", 2, 10, 6)
learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1, 0.01)

# --- File upload ---
st.subheader("1. Upload your CSV dataset")
st.caption("Last column is treated as the target (y). All other columns are features (X).")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    with st.expander("Preview dataset"):
        st.dataframe(df.head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 1)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    st.subheader("2. Train the model")
    if st.button("Train XGBoost", type="primary", use_container_width=True):
        with st.spinner("Training..."):

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state)
            )
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            # Train
            classifier = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=int(random_state)
            )
            classifier.fit(X_train, y_train)

            # Predict
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # Cross-val
            cv_scores = cross_val_score(
                estimator=classifier, X=X_train, y=y_train, cv=cv_folds
            )

        st.subheader("3. Results")

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Test Accuracy", f"{acc*100:.2f}%")
        m2.metric("CV Accuracy", f"{cv_scores.mean()*100:.2f}%")
        m3.metric("CV Std Dev", f"{cv_scores.std()*100:.2f}%")

        # Confusion matrix
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        fig.patch.set_alpha(0)
        st.pyplot(fig)

        # Feature importance
        st.markdown("**Feature Importance**")
        feature_names = df.columns[:-1].tolist()
        importances = classifier.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=True)

        fig2, ax2 = plt.subplots(figsize=(6, max(3, len(feature_names) * 0.4)))
        ax2.barh(fi_df["Feature"], fi_df["Importance"], color="#3B8BD4")
        ax2.set_xlabel("Importance Score")
        ax2.set_title("Feature Importance")
        fig2.patch.set_alpha(0)
        st.pyplot(fig2)

        # CV scores detail
        with st.expander("Cross-validation fold scores"):
            fold_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                "Accuracy": [f"{s*100:.2f}%" for s in cv_scores]
            })
            st.dataframe(fold_df, use_container_width=True)

        st.success("Done! Your model is trained and evaluated.")

else:
    st.info("Upload a CSV file above to get started.")
    st.markdown("**Example format:**")
    example = pd.DataFrame({
        "feature_1": [1.2, 3.4, 2.1],
        "feature_2": [0.5, 1.2, 0.9],
        "feature_3": [10, 20, 15],
        "target": [0, 1, 0]
    })
    st.dataframe(example)
