# import basic library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

def load_data():
    """Load the breast cancer dataset"""
    return pd.read_csv('data.csv')

def prepare_data(data):
    """Prepare data for modeling"""
    # Drop ID column
    data = data.drop(['id'], axis=1)
    
    # Separate features and target
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    
    # Convert target to binary
    label_mapping = {"M": 0, "B": 1}
    y = y.map(label_mapping)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        })
    
    return pd.DataFrame(results), trained_models

def plot_target_distribution(data):
    """Create a pie chart for target distribution"""
    fig, ax = plt.subplots(figsize=(8, 6))
    target_counts = data['diagnosis'].value_counts()
    ax.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', 
           colors=sns.color_palette('Set2'))
    ax.set_title('Distribution of Target Cancer Types')
    return fig

def plot_pair_plot(data):
    """Create pair plot for mean values"""
    pair_plot_cols = ['diagnosis', 'x.radius_mean', 'x.texture_mean', 
                       'x.perimeter_mean', 'x.concave_pts_mean', 'x.radius_worst']
    fig = sns.pairplot(data[pair_plot_cols], hue='diagnosis')
    fig.fig.suptitle('Pair Plot for Mean Values', y=1.02)
    return fig

def plot_learning_curve(X_train, X_test, y_train, y_test):
    """Plot learning curve for Random Forest"""
    train_scores = []
    test_scores = []
    values = [i for i in range(1, 200, 10)]
    
    rf_model = RandomForestClassifier(random_state=42)
    
    for i in values:
        rf_model.set_params(n_estimators=i)
        rf_model.fit(X_train, y_train)
        
        train_yhat = rf_model.predict(X_train)
        train_acc = accuracy_score(y_train, train_yhat)
        
        test_yhat = rf_model.predict(X_test)
        test_acc = accuracy_score(y_test, test_yhat)
        
        train_scores.append(train_acc)
        test_scores.append(test_acc)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(values, train_scores, label="Training Accuracy", marker='o', color='blue')
    ax.plot(values, test_scores, label="Testing Accuracy", marker='o', color='green')
    ax.set_title("Random Forest: Training vs Testing Accuracy")
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Accuracy")
    ax.legend()
    return fig

def plot_model_comparison(results_df):
    """Create bar plot for model comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    bar_width = 0.2
    models = results_df["Model"].tolist()
    positions = np.arange(len(metrics))
    
    for i, (model, scores) in enumerate(zip(models, results_df[metrics].values)):
        ax.bar(positions + i * bar_width, scores, bar_width, label=model)
    
    ax.set_title("Performance Comparison of Models")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(positions + (len(models) - 1) * bar_width / 2)
    ax.set_xticklabels(metrics)
    ax.legend(title="Model", loc="lower center")
    
    # Add text annotations
    for i, model_scores in enumerate(results_df[metrics].values):
        for j, score in enumerate(model_scores):
            ax.text(positions[j] + i * bar_width, score + 0.02, f"{score:.2f}", 
                    ha='center', va='bottom')
    
    return fig

def main():
    # Set page configuration
    st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗", 
    layout="wide",
    initial_sidebar_state="expanded",
    )
    st.set_page_config(layout="wide", page_title="Breast Cancer Analysis")
    
    st.title("🏥 Breast Cancer Data Analysis")
    
    # Load data
    data = load_data()
    
    # Sidebar navigation
    menu = ["Dataset", "Visualizations", "Model Performance", "Prediction"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    if choice == "Dataset":
        st.header("Dataset Details")
        st.dataframe(data)
        st.write(f"Dataset Shape: {data.shape}")
    
    elif choice == "Visualizations":
        st.header("Data Visualizations")
        
        # Pie Chart for Target Distribution
        st.subheader("Target Distribution")
        fig_dist = plot_target_distribution(data)
        st.pyplot(fig_dist)
        
        # Pair Plot
        st.subheader("Pair Plot for Mean Values")
        fig_pair = plot_pair_plot(data)
        st.pyplot(fig_pair)
    
    elif choice == "Model Performance":
        st.header("Model Performance Analysis")
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler, X, y = prepare_data(data)
        
        # Train and evaluate models
        results_df, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Display model comparison results
        st.subheader("Model Performance Metrics")
        st.dataframe(results_df)
        
        # Model Comparison Bar Plot
        st.subheader("Model Comparison Graph")
        fig_model_comp = plot_model_comparison(results_df)
        st.pyplot(fig_model_comp)
        
        # Learning Curve for Random Forest
        st.subheader("Random Forest Learning Curve")
        fig_learning = plot_learning_curve(X_train, X_test, y_train, y_test)
        st.pyplot(fig_learning)
    
    elif choice == "Prediction":
        st.header("Cancer Type Prediction")
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler, X, y = prepare_data(data)
        
        # Train models
        _, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # User input for features
        st.subheader("Enter Patient Features")
        feature_cols = data.drop(columns=['diagnosis', 'id']).columns
        
        # Create input fields for each feature
        feature_inputs = {}
        for col in feature_cols:
            feature_inputs[col] = st.number_input(
                f"Enter {col}", 
                value=float(data[col].mean()), 
                step=0.01
            )

        
        # Predict button
        if st.button("Predict"):
            # Convert input to numpy array and scale
            input_features = np.array(list(feature_inputs.values())).reshape(1, -1)
            input_features_scaled = scaler.transform(input_features)
            
            # Make predictions
            predictions = {}
            for name, model in models.items():
                pred = model.predict(input_features_scaled)
                predictions[name] = "Malignant" if pred[0] == 0 else "Benign"
            
            # Display predictions
            st.subheader("Prediction Results")
            for model, pred in predictions.items():
                st.write(f"{model}: {pred}")

if __name__ == "__main__":
    main()
