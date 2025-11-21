from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Projects paths
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW_DATASET_CSV_NAME = "creditcard.csv"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "models" / "mlruns"

# Ensure directories exist
for directory in [DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL, MODELS_DIR, MLFLOW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def setup_mlflow(experiment_name: str = "fraud_detection") -> None:
    """
    Initialize MLflow experiment tracking.

    Args:
        experiment_name: Name of the MLflow experiment
    """
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment(experiment_name)
    print("✅ MLflow tracking initialized")
    print(f"📊 Experiment: {experiment_name}")
    print(f"📁 Tracking URI: {MLFLOW_DIR}")


def load_fraud_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load credit card fraud dataset with validation.

    Args:
        filepath: Path to CSV file. If None, searches data/raw directory.

    Returns:
        DataFrame with fraud transaction data

    Raises:
        FileNotFoundError: If dataset not found
    """
    if filepath is None:
        # If the dataset is not present in the raw dataset look for possible file names
        possible_names = [
            "creditcard.csv",
            "creditcard_data.csv",
            "creditcard_fraud_data.csv",
            "fraud_data.csv",
        ]
        for name in possible_names:
            potential_path = DATA_RAW / name
            if potential_path.exists():
                filepath = potential_path
                break

        # If dataset doesn't exists in raw data directory raise an error
        if filepath is None:
            raise FileNotFoundError(
                f"Dataset not found in {DATA_RAW}\n"
                f"Expected filenames: {possible_names}"
                f"Please download from: https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud\\"
                f"And place it in {DATA_RAW}"
            )

    # Load dataset
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset from: {Path(filepath).name}")
    print(f"📊 Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    return df


def quick_data_summary(df: pd.DataFrame) -> None:
    """
    Print comphrehensive data summary with key statistics.

    Args:
        df: DataFrame to summarize
    """
    print("=" * 60)
    print("📋 QUICK DATA SUMMARY")
    print("=" * 60)

    # Basic info
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Column types
    print("\nColumn Types:")
    print(df.dtypes.value_counts())

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️ Missing Values:")
        print(f"{missing[missing > 0]}")
    else:
        print("\n✅ No missing values")

    # Target variable (if Class column exists)
    if "Class" in df.columns:
        print("\n🎯 Target Distribution:")
        target_counts = df["Class"].value_counts().sort_index()
        for class_val, count in target_counts.items():
            pct = count / len(df) * 100
            label = "Legitimate" if class_val == 0 else "Fraud"
            print(f"\tClass {class_val} ({label}): {count:,} ({pct:.2f}%)")

        # Imbalance ratio
        if len(target_counts) == 2:
            imbalance_ratio = target_counts.min() / target_counts.max()
            print(f"\tImbalance Ratio: {imbalance_ratio:.4f} ({1/imbalance_ratio:.1f}:1)")


def plot_target_distribution(df: pd.DataFrame, target_col: str = "Class") -> None:
    """
    Visualize target variable distribution with both count and percentage.

    Args:
        df: DataFrame containing target column
        target_col: Name of the target column
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    target_counts = df[target_col].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"] if target_col == "Class" else target_counts.index

    axes[0].bar(range(len(target_counts)), target_counts.values, color=["#2ecc71", "#e74c3c"])
    axes[0].set_xlabel("Class", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title(f"{target_col} Distribution (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xticks(range(len(target_counts)))
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.3)

    # Add count labels
    for i, count in enumerate(target_counts.values):
        axes[0].text(i, count, f"{count:,}", ha="center", va="bottom", fontweight="bold")

    # Percentage plot
    percentages = (target_counts / len(df) * 100).values
    axes[1].bar(range(len(target_counts)), percentages, color=["#2ecc71", "#e74c3c"])
    axes[1].set_xlabel("Class", fontsize=12)
    axes[1].set_ylabel("Percentage (%)", fontsize=12)
    axes[1].set_title(f"{target_col} Distribution (Percentage)", fontsize=14, fontweight="bold")
    axes[1].set_xticks(range(len(target_counts)))
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.3)

    # Add percentage labels
    for i, pct in enumerate(percentages):
        axes[1].text(i, pct, f"{pct:.3f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.show()

    # Print imbalance assessment
    if len(target_counts) == 2:
        imbalance_ratio = target_counts.min() / target_counts.max()
        print("\n📊 Class Imbalance Assessment:")
        print(f"\tRatio: {imbalance_ratio:.4f} ({1/imbalance_ratio:.1f}:1)")

        if imbalance_ratio < 0.01:
            severity = "🔴 EXTREME imbalance - specialized techniques required"
        elif imbalance_ratio < 0.1:
            severity = "🟠 SEVERE imbalance - resampling/class weights essential"
        elif imbalance_ratio < 0.5:
            severity = "🟡MODERATE imbalance - consider class weights"
        else:
            severity = "🟢 BALANCED classes"

        print(f"\tSeverity: {severity}")


def save_dataframe(df: pd.DataFrame, filename: str, description: str = "") -> Path:
    """
    Save DataFrame to processed data directory with logging.

    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of the file
        description (str, optional): Optional description for logging. Defaults to "".

    Returns:
        Path: Path to saved file
    """
    filepath = DATA_PROCESSED / filename
    df.to_csv(filepath, index=False)

    print(f"💾 Saved: {filename}")
    if description:
        print(f"\tDescription: {description}")

    print(f"\tShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\tLocation: {filepath}")

    return filepath


# Print import confirmation
print("📦 Fraud Detection Utils Loaded")
print(f"📁 Project Root: {PROJECT_ROOT}")
print("\n✅ Utility module ready for import")
