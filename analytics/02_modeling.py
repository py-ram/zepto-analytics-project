import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

import seaborn as sns
import joblib


def load_data():

    print("Loading Titanic data...")

    df = pd.read_csv("titanic_clean.csv")

    # Keep the columns used for modelling
    columns = [
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]

    df = df[columns]

    print("Dataset shape:", df.shape)

    return df


def prepare_data(df):

    X = df.drop("survived", axis=1)
    y = df["survived"]

    numeric_features = [
        "age",
        "fare",
        "sibsp",
        "parch"
    ]

    categorical_features = [
        "sex",
        "embarked"
    ]

    # Fill missing values and prepare categorical columns
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            drop="first",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
        ("pclass", "passthrough", ["pclass"])
    ])

    return X, y, preprocessor


def evaluate_model(model, X_test, y_test, name):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"ROC AUC  : {auc:.3f}")

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "probabilities": probabilities
    }


def plot_confusion_matrix(model, X_test, y_test, name):

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    file_name = name.lower().replace(" ", "_") + "_confusion_matrix.png"

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


def plot_roc_curves(results, y_test):

    plt.figure(figsize=(8, 6))

    for result in results:

        fpr, tpr, _ = roc_curve(
            y_test,
            result["probabilities"]
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{result['model']} (AUC = {result['auc']:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()

    plt.tight_layout()
    plt.savefig("roc_curves.png")
    plt.show()


def save_decision_tree(tree_model, preprocessor):

    # Get the transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    plt.figure(figsize=(18, 10))

    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=["Not Survived", "Survived"],
        filled=True,
        max_depth=5,
        fontsize=7
    )

    plt.title("Decision Tree Visualization")

    plt.tight_layout()
    plt.savefig("decision_tree.png")
    plt.show()


def run_models():

    df = load_data()

    X, y, preprocessor = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining data:", X_train.shape)
    print("Testing data :", X_test.shape)

    models = {}

    # Logistic Regression
    models["Logistic Regression"] = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    # Decision Tree
    models["Decision Tree"] = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ))
    ])

    # Random Forest
    models["Random Forest"] = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            max_depth=10,
            max_features=None,
            oob_score=True,
            random_state=42
        ))
    ])

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        result = evaluate_model(
            model,
            X_test,
            y_test,
            name
        )

        results.append(result)

        plot_confusion_matrix(
            model,
            X_test,
            y_test,
            name
        )

    # ROC curves
    plot_roc_curves(results, y_test)

    # Save decision tree plot
    tree_model = models["Decision Tree"].named_steps["classifier"]
    tree_preprocessor = models["Decision Tree"].named_steps["preprocessor"]

    save_decision_tree(
        tree_model,
        tree_preprocessor
    )

    # Compare model results
    comparison = pd.DataFrame([
        {
            "Model": r["model"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1 Score": r["f1_score"],
            "ROC AUC": r["auc"]
        }
        for r in results
    ])

    print("\nModel Comparison")
    print(comparison.round(3))

    comparison.to_csv(
        "model_results.csv",
        index=False
    )

    # Save the Random Forest pipeline
    joblib.dump(
        models["Random Forest"],
        "best_pipeline.joblib"
    )

    print("\nSaved Random Forest model to best_pipeline.joblib")

    return comparison


if __name__ == "__main__":

    results = run_models()

    print("\nModeling completed successfully.")