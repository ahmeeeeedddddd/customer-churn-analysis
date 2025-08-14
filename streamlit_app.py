import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('default')

# Streamlit configuration
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")
st.title("🔍 Customer Churn Analysis")
st.write("Upload your CSV file and run the complete analysis pipeline")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Your original code starts here with minimal modifications

    st.write("## Loading the dataset")

    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Successfully loaded: {uploaded_file.name}")

    st.write("Dataset Shape:", df.shape)
    st.write("**First 5 rows:**")
    st.dataframe(df.head())

    with st.expander("Dataset Info"):
        buffer = st.empty()
        st.text(str(df.info()))

    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    st.write("**Target Variable Distribution:**")
    st.write(df['Churn'].value_counts())

    # Data preprocessing
    st.write("## Data preprocessing")

    df_processed = df.copy()

    st.write("Original TotalCharges data type:", df_processed['TotalCharges'].dtype)
    st.write("Sample TotalCharges values:", df_processed['TotalCharges'].head(10).tolist())

    # Handle TotalCharges column (it's object type but should be numeric)
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')

    # Check for missing values after conversion
    st.write("**Missing values after TotalCharges conversion:**")
    missing_after_conversion = df_processed.isnull().sum()
    st.write(missing_after_conversion[missing_after_conversion > 0])

    # Handle missing values in TotalCharges (fill with median)
    if df_processed['TotalCharges'].isnull().sum() > 0:
        median_total_charges = df_processed['TotalCharges'].median()
        df_processed['TotalCharges'].fillna(median_total_charges, inplace=True)
        st.write(
            f"Filled {df_processed['TotalCharges'].isnull().sum()} missing TotalCharges values with median: {median_total_charges}")

    # Remove customerID as it's not useful for prediction
    df_processed = df_processed.drop('customerID', axis=1)

    st.success("Data preprocessing completed!")
    st.write("Final dataset shape:", df_processed.shape)
    st.write("Final data types:")
    st.write(df_processed.dtypes)

    # Data splitting
    st.write("## Data splitting")

    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']

    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    st.success("Data splitting completed!")
    st.write(f"Training set shape: {X_train.shape}")
    st.write(f"Test set shape: {X_test.shape}")
    st.write(f"Training target distribution:\n{y_train.value_counts()}")
    st.write(f"Test target distribution:\n{y_test.value_counts()}")

    # Encoding
    st.write("## Encoding")

    # Define the exact categorical and numerical columns based on your dataset
    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    numerical_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    st.write("Categorical columns:", categorical_columns)
    st.write("Numerical columns:", numerical_columns)
    st.write(f"Total categorical: {len(categorical_columns)}, Total numerical: {len(numerical_columns)}")

    # Verify all columns are accounted for
    all_feature_columns = categorical_columns + numerical_columns
    st.write("All feature columns accounted for:", set(all_feature_columns) == set(X_train.columns))

    # Initialize encoders
    label_encoders = {}

    # Apply Label Encoding to categorical variables
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    st.write("**Encoding categorical variables:**")
    for column in categorical_columns:
        st.write(f"Encoding {column}...")
        le = LabelEncoder()
        X_train_encoded[column] = le.fit_transform(X_train_encoded[column].astype(str))
        X_test_encoded[column] = le.transform(X_test_encoded[column].astype(str))
        label_encoders[column] = le

        # Show encoding mapping for first few categories
        unique_values = X_train[column].unique()[:5]  # Show first 5 unique values
        encoded_values = le.transform(unique_values.astype(str))
        st.write(f"  Sample mapping: {dict(zip(unique_values, encoded_values))}")

    # Encode target variable
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_test_encoded = le_target.transform(y_test)

    st.write(f"Target encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
    st.success("Encoding completed!")
    st.write("Encoded training set shape:", X_train_encoded.shape)

    # Resampling to handle class imbalance
    st.write("## Resampling to handle class imbalance")

    # Check class distribution before resampling
    st.write("Class distribution before resampling:")
    unique, counts = np.unique(y_train_encoded, return_counts=True)
    class_distribution_before = dict(zip(unique, counts))
    st.write(f"Class 0 (No): {class_distribution_before.get(0, 0)}")
    st.write(f"Class 1 (Yes): {class_distribution_before.get(1, 0)}")
    st.write(f"Imbalance ratio: {class_distribution_before.get(0, 0) / class_distribution_before.get(1, 0):.2f}:1")

    # Apply SMOTE for oversampling
    st.write("**Applying SMOTE resampling...**")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train_encoded)

    st.write("**Class distribution after SMOTE resampling:**")
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    class_distribution_after = dict(zip(unique, counts))
    st.write(f"Class 0 (No): {class_distribution_after.get(0, 0)}")
    st.write(f"Class 1 (Yes): {class_distribution_after.get(1, 0)}")
    st.write(f"Balanced ratio: {class_distribution_after.get(0, 0) / class_distribution_after.get(1, 0):.2f}:1")
    st.write(f"Resampled training set shape: {X_train_resampled.shape}")

    # Rescaling
    st.write("## Rescaling")

    # Apply StandardScaler (recommended for most ML algorithms)
    st.write("Applying StandardScaler...")
    scaler_standard = StandardScaler()
    X_train_scaled_std = scaler_standard.fit_transform(X_train_resampled)
    X_test_scaled_std = scaler_standard.transform(X_test_encoded)

    # Apply MinMaxScaler (alternative approach)
    st.write("Applying MinMaxScaler...")
    scaler_minmax = MinMaxScaler()
    X_train_scaled_mm = scaler_minmax.fit_transform(X_train_resampled)
    X_test_scaled_mm = scaler_minmax.transform(X_test_encoded)

    st.success("Rescaling completed!")
    st.write(f"StandardScaler - Training set shape: {X_train_scaled_std.shape}")
    st.write(f"MinMaxScaler - Training set shape: {X_train_scaled_mm.shape}")

    # Check scaling results for numerical features
    st.write("## SCALING VERIFICATION")

    # Convert to DataFrame for easier analysis
    column_names = X_train_encoded.columns
    df_before = pd.DataFrame(X_train_resampled, columns=column_names)
    df_std = pd.DataFrame(X_train_scaled_std, columns=column_names)
    df_mm = pd.DataFrame(X_train_scaled_mm, columns=column_names)

    # Focus on numerical columns for scaling comparison
    numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    st.write("Original data statistics (numerical features):")
    st.dataframe(df_before[numerical_cols].describe().round(2))

    st.write("After StandardScaling (numerical features):")
    st.dataframe(df_std[numerical_cols].describe().round(2))

    st.write("After MinMaxScaling (numerical features):")
    st.dataframe(df_mm[numerical_cols].describe().round(2))

    # PCA
    st.write("## PCA")

    st.write("Applying PCA analysis...")

    # Apply PCA with different strategies
    # Strategy 1: Keep 95% of variance
    pca_95 = PCA(n_components=0.95, random_state=42)
    X_train_pca_95 = pca_95.fit_transform(X_train_scaled_std)
    X_test_pca_95 = pca_95.transform(X_test_scaled_std)

    # Strategy 2: Keep specific number of components
    pca_10 = PCA(n_components=10, random_state=42)
    X_train_pca_10 = pca_10.fit_transform(X_train_scaled_std)
    X_test_pca_10 = pca_10.transform(X_test_scaled_std)

    # Strategy 3: Keep components that explain at least 1% variance each
    # Find optimal number of components
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train_scaled_std)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_90 = np.argmax(cumsum_var >= 0.90) + 1
    n_components_99 = np.argmax(cumsum_var >= 0.99) + 1

    pca_90 = PCA(n_components=n_components_90, random_state=42)
    X_train_pca_90 = pca_90.fit_transform(X_train_scaled_std)
    X_test_pca_90 = pca_90.transform(X_test_scaled_std)

    st.success("PCA completed!")
    st.write(f"Original features: {X_train_scaled_std.shape[1]}")
    st.write(f"PCA (95% variance): {X_train_pca_95.shape[1]} components")
    st.write(f"PCA (90% variance): {X_train_pca_90.shape[1]} components")
    st.write(f"PCA (10 components): {X_train_pca_10.shape[1]} components")
    st.write(f"Explained variance ratios:")
    st.write(f"  95% PCA: {pca_95.explained_variance_ratio_.sum():.4f}")
    st.write(f"  90% PCA: {pca_90.explained_variance_ratio_.sum():.4f}")
    st.write(f"  10 PCA: {pca_10.explained_variance_ratio_.sum():.4f}")

    # Visualization and summary
    st.write("## Visualization and summary")

    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))

    # Plot 1: PCA Explained Variance
    axes[0, 0].plot(range(1, len(pca_95.explained_variance_ratio_) + 1),
                    np.cumsum(pca_95.explained_variance_ratio_), 'b-o', markersize=4)
    axes[0, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
    axes[0, 0].axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    axes[0, 0].set_xlabel('Number of Components')
    axes[0, 0].set_ylabel('Cumulative Explained Variance')
    axes[0, 0].set_title('PCA - Cumulative Explained Variance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Individual Component Variance (Top 15)
    n_show = min(15, len(pca_full.explained_variance_ratio_))
    axes[0, 1].bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show])
    axes[0, 1].set_xlabel('Principal Component')
    axes[0, 1].set_ylabel('Explained Variance Ratio')
    axes[0, 1].set_title(f'Individual Component Variance (Top {n_show})')
    axes[0, 1].set_xticks(range(1, n_show + 1))
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Original vs PCA dimensions comparison
    methods = ['Original', '99% Var', '95% Var', '90% Var', '10 Comp']
    dimensions = [
        X_train_scaled_std.shape[1],
        n_components_99,
        pca_95.n_components_,
        pca_90.n_components_,
        10
    ]
    colors = ['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral']
    bars = axes[0, 2].bar(methods, dimensions, color=colors)
    axes[0, 2].set_ylabel('Number of Features')
    axes[0, 2].set_title('Dimensionality Comparison')
    axes[0, 2].tick_params(axis='x', rotation=45)
    for bar, dim in zip(bars, dimensions):
        axes[0, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(dim), ha='center', va='bottom')

    # Plot 4: Class distribution before/after resampling
    labels = ['Before SMOTE', 'After SMOTE']
    no_churn = [class_distribution_before.get(0, 0), class_distribution_after.get(0, 0)]
    yes_churn = [class_distribution_before.get(1, 0), class_distribution_after.get(1, 0)]

    x = np.arange(len(labels))
    width = 0.35
    axes[1, 0].bar(x - width / 2, no_churn, width, label='No Churn', color='lightblue')
    axes[1, 0].bar(x + width / 2, yes_churn, width, label='Yes Churn', color='salmon')
    axes[1, 0].set_xlabel('Resampling Stage')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Class Distribution: Before vs After SMOTE')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].legend()

    # Plot 5: Feature scaling comparison (numerical features only)
    numerical_idx = [list(X_train_encoded.columns).index(col) for col in numerical_columns]
    original_std = np.std(X_train_resampled, axis=0)[numerical_idx]
    scaled_std = np.std(X_train_scaled_std, axis=0)[numerical_idx]

    x_pos = np.arange(len(numerical_columns))
    axes[1, 1].bar(x_pos - 0.2, original_std, 0.4, label='Original', color='lightcoral')
    axes[1, 1].bar(x_pos + 0.2, scaled_std, 0.4, label='Scaled', color='lightblue')
    axes[1, 1].set_xlabel('Numerical Features')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('Feature Scaling Effect')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([col[:8] + '...' if len(col) > 8 else col for col in numerical_columns], rotation=45)
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')  # Log scale due to large differences

    # Plot 6: Top feature importance in first PC
    feature_names = X_train_encoded.columns
    component_importance = np.abs(pca_95.components_[0])
    top_n = 10
    top_features_idx = component_importance.argsort()[-top_n:][::-1]
    top_features_names = [feature_names[idx][:10] + '...' if len(feature_names[idx]) > 10
                          else feature_names[idx] for idx in top_features_idx]
    top_importance_values = component_importance[top_features_idx]

    axes[1, 2].barh(range(top_n), top_importance_values, color='lightgreen')
    axes[1, 2].set_yticks(range(top_n))
    axes[1, 2].set_yticklabels(top_features_names)
    axes[1, 2].set_xlabel('Absolute Importance')
    axes[1, 2].set_title(f'Top {top_n} Features in First Principal Component')
    axes[1, 2].invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig)

    # Models to train
    st.write("## Models to train")

    if st.button("🚀 Train All Models"):
        from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier

        # Models to train
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "SVM": SVC(kernel='rbf', random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
            "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }

        results = {}
        conf_matrices = {}

        # Train and evaluate
        progress_bar = st.progress(0)
        for i, (name, model) in enumerate(models.items()):
            st.write(f"Training {name}...")
            model.fit(X_train_pca_95, y_train_resampled)
            y_pred = model.predict(X_test_pca_95)

            acc = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='macro')
            recall = recall_score(y_test_encoded, y_pred, average='macro')

            results[name] = {"Accuracy": acc, "Macro_F1": f1, "Recall": recall}
            conf_matrices[name] = confusion_matrix(y_test_encoded, y_pred)

            st.write(f"**{name}**")
            st.write(f"Accuracy: {acc:.4f}")
            st.write(f"Macro F1-score: {f1:.4f}")
            st.write(f"Macro Recall: {recall:.4f}")

            progress_bar.progress((i + 1) / len(models))

        # Best model
        best_recall_model = max(results, key=lambda x: results[x]["Recall"])
        st.success(
            f"🏆 Model with highest recall: {best_recall_model} (Recall={results[best_recall_model]['Recall']:.4f})")

        # Display results table
        results_df = pd.DataFrame(results).T.round(4)
        st.write("**Model Performance Summary:**")
        st.dataframe(results_df)

        # Plot results
        model_names = list(results.keys())
        acc_values = [results[m]["Accuracy"] for m in model_names]
        f1_values = [results[m]["Macro_F1"] for m in model_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.bar(model_names, acc_values, color='skyblue')
        ax1.set_title("Model Accuracy")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)

        ax2.bar(model_names, f1_values, color='lightcoral')
        ax2.set_title("Model Macro F1-score")
        ax2.set_ylabel("Macro F1-score")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

        # Confusion matrices
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for ax, name in zip(axes, model_names):
            sns.heatmap(conf_matrices[name], annot=True, fmt="d", cmap="Blues",
                        xticklabels=le_target.classes_, yticklabels=le_target.classes_, ax=ax, cbar=False)
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        plt.tight_layout()
        st.pyplot(fig)

else:

    st.info("👆 Please upload your CSV file to start the analysis")
