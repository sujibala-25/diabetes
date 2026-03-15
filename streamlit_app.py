import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_model(model_name: str, params: dict):
    if model_name == "Logistic Regression":
        return LogisticRegression(**params)
    if model_name == "KNN":
        return KNeighborsClassifier(**params)
    if model_name == "Random Forest":
        return RandomForestClassifier(**params)
    raise ValueError(f"Unknown model: {model_name}")


def train_and_store(data: pd.DataFrame, model_name: str, params: dict, test_size: float, random_state: int):
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(model_name, params)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.feature_columns = X.columns.tolist()
    st.session_state.test_metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report,
    }
    st.session_state.trained_model_name = model_name
    st.session_state.trained = True


def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="wide")
    st.title("Diabetes Prediction App")

    data_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
    data = load_data(data_path)

    st.markdown("### Dataset")
    st.dataframe(data.head())

    st.markdown("---")
    st.sidebar.header("Model selection")

    model_name = st.sidebar.selectbox(
        "Choose a model:",
        ["Logistic Regression", "KNN", "Random Forest"],
        index=0,
    )

    # Reset trained state if the user changes model selection
    if st.session_state.get("trained_model_name") != model_name:
        st.session_state.trained = False

    # Common train/test split settings
    test_size = st.sidebar.slider("Test set size", 0.05, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random state", value=42, step=1)

    # Model hyperparameters
    params = {}
    if model_name == "Logistic Regression":
        params["solver"] = st.sidebar.selectbox(
            "Solver", ["lbfgs", "liblinear", "saga"], index=0
        )
        params["max_iter"] = st.sidebar.number_input(
            "Max iterations", min_value=100, max_value=5000, value=1000, step=100
        )
        params["random_state"] = random_state

    elif model_name == "KNN":
        params["n_neighbors"] = st.sidebar.number_input(
            "n_neighbors", min_value=1, max_value=50, value=5, step=1
        )
        params["weights"] = st.sidebar.selectbox("Weights", ["uniform", "distance"])
        params["metric"] = st.sidebar.selectbox("Metric", ["minkowski", "euclidean", "manhattan"])

    elif model_name == "Random Forest":
        params["n_estimators"] = st.sidebar.number_input(
            "n_estimators", min_value=10, max_value=1000, value=100, step=10
        )
        max_depth_option = st.sidebar.selectbox(
            "max_depth", ["None", 5, 10, 20, 30, 50, 100], index=0
        )
        params["max_depth"] = None if max_depth_option == "None" else int(max_depth_option)
        params["min_samples_split"] = st.sidebar.number_input(
            "min_samples_split", min_value=2, max_value=20, value=2, step=1
        )
        params["random_state"] = random_state

    st.sidebar.markdown("---")
    st.sidebar.write("⚙️ **Training settings**")

    if st.sidebar.button("Train model"):
        with st.spinner("Training model..."):
            train_and_store(data, model_name, params, test_size, random_state)

    if st.session_state.get("trained"):
        metrics = st.session_state.test_metrics
        st.success(f"Training complete — accuracy: {metrics['accuracy']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion matrix")
            st.write(metrics["confusion_matrix"])
        with col2:
            st.subheader("Classification report")
            st.write(pd.DataFrame(metrics["report"]).transpose())

        st.markdown("---")
        st.subheader("Make a prediction")

        input_data = {
            col: st.number_input(
                col,
                float(data[col].min()),
                float(data[col].max()),
                float(data[col].median()),
            )
            for col in st.session_state.feature_columns
        }

        if st.button("Predict"):
            X_new = np.array([list(input_data.values())])
            X_new_scaled = st.session_state.scaler.transform(X_new)
            pred = st.session_state.model.predict(X_new_scaled)[0]
            st.write(f"**Predicted outcome:** {pred} (0 = no diabetes, 1 = diabetes)")

    else:
        st.info("Select model options above and click 'Train model' to begin.")


if __name__ == "__main__":
    main()
