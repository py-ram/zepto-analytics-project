import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def load_data():

    print("Loading Titanic dataset...")

    # Use the same dataset as the EDA file
    df = sns.load_dataset("titanic")

    # Remove columns that are not useful for the model
    df = df.drop(columns=["deck", "alive", "who", "adult_male", "embark_town"])

    # Create a simple family size feature
    df["family_size"] = df["sibsp"] + df["parch"] + 1

    print("Dataset shape:", df.shape)

    return df


def prepare_data(df):

    # Target variable
    X = df.drop("survived", axis=1)
    y = df["survived"]

    # Columns based on data type
    numeric_cols = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "family_size"
    ]

    categorical_cols = [
        "sex",
        "embarked",
        "class"
    ]

    # Fill missing values and encode categorical columns
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])

    return X, y, preprocessor


def evaluate_model(model, X_test, y_test, name):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")

    # Confusion matrix
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
    plt.tight_layout()

    file_name = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(file_name)
    plt.show()

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


def train_models():

    df = load_data()

    X, y, preprocessor = prepare_data(df)

    # Keep the same split for both models
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining data:", X_train.shape)
    print("Testing data :", X_test.shape)

    # Logistic Regression
    logistic_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    logistic_model.fit(X_train, y_train)

    logistic_result = evaluate_model(
        logistic_model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    # Random Forest
    random_forest = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    random_forest.fit(X_train, y_train)

    rf_result = evaluate_model(
        random_forest,
        X_test,
        y_test,
        "Random Forest"
    )

    # Compare the two models
    results = pd.DataFrame([
        logistic_result,
        rf_result
    ])

    print("\nModel Comparison")
    print(results)

    results.to_csv("model_results.csv", index=False)

    best_model = results.loc[
        results["F1 Score"].idxmax(),
        "Model"
    ]

    print(f"\nBest model based on F1 Score: {best_model}")

    return results


if __name__ == "__main__":

    results = train_models()

    print("\nModeling completed successfully.")
