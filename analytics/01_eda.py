import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_and_clean_data():

    print("Loading Titanic dataset...")

    # Load dataset from seaborn
    df = sns.load_dataset("titanic")

    # Save a copy for reference
    df.to_csv("titanic.csv", index=False)

    print("\nDataset Shape")
    print(df.shape)

    print("\nDataset Information")
    df.info()

    print("\nSummary Statistics")
    print(df.describe())

    # Check missing values
    print("\nMissing Values")

    missing = (df.isnull().sum() / len(df)) * 100

    for col in missing.index:
        if missing[col] > 0:
            print(f"{col}: {missing[col]:.2f}%")

    # Fill missing age values with median
    df["age"].fillna(df["age"].median(), inplace=True)

    # Remove deck because most values are missing
    df.drop(columns=["deck"], inplace=True)

    # Remove rows where embark information is missing
    df.dropna(subset=["embarked", "embark_town"], inplace=True)

    # Save cleaned dataset
    df.to_csv("titanic_clean.csv", index=False)

    print("\nData cleaning completed.")

    return df

def univariate_analysis(df):

    print("\nUnivariate Analysis")

    # Create histograms and box plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    df["age"].hist(ax=axes[0, 0], bins=30)
    axes[0, 0].set_title("Age Distribution")

    df["fare"].hist(ax=axes[0, 1], bins=30)
    axes[0, 1].set_title("Fare Distribution")

    df.boxplot(column="age", ax=axes[1, 0])
    axes[1, 0].set_title("Age Box Plot")

    df.boxplot(column="fare", ax=axes[1, 1])
    axes[1, 1].set_title("Fare Box Plot")

    plt.tight_layout()
    plt.savefig("univariate_plots.png")
    plt.show()

    print("\nOutlier Analysis (IQR Method)")

    for col in ["age", "fare"]:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        print(f"\n{col.capitalize()}")

        print(f"Q1: {q1:.2f}")
        print(f"Q3: {q3:.2f}")
        print(f"IQR: {iqr:.2f}")

        print(f"Lower Limit: {lower:.2f}")
        print(f"Upper Limit: {upper:.2f}")

        print(f"Outliers Found: {len(outliers)}")

    print("\nFare Statistics")

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode()[0]

    print(f"Mean   : {fare_mean:.2f}")
    print(f"Median : {fare_median:.2f}")
    print(f"Mode   : {fare_mode:.2f}")

    if fare_mean > fare_median:
        print("Observation: Fare distribution is positively skewed.")
    elif fare_mean < fare_median:
        print("Observation: Fare distribution is negatively skewed.")
    else:
        print("Observation: Fare distribution is approximately symmetric.")

def bivariate_analysis(df):

    print("\nBivariate Analysis")

    # Survival rate by gender
    survival_by_sex = df.groupby("sex")["survived"].mean() * 100

    print("\nSurvival Rate by Gender")
    print(survival_by_sex)

    # Survival rate by passenger class
    survival_by_class = df.groupby("pclass")["survived"].mean() * 100

    print("\nSurvival Rate by Class")
    print(survival_by_class)

    # Survival based on both gender and class
    survival_table = df.pivot_table(
        values="survived",
        index="sex",
        columns="pclass",
        aggfunc="mean"
    ) * 100

    print("\nSurvival by Gender and Class")
    print(survival_table)

    # Correlation matrix
    columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    corr_matrix = df[columns].corr()

    print("\nCorrelation Matrix")
    print(corr_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.show()

    # Find strongest correlations
    corr_list = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            corr_list.append(
                (
                    columns[i],
                    columns[j],
                    abs(corr_matrix.iloc[i, j])
                )
            )

    corr_list.sort(key=lambda x: x[2], reverse=True)

    print("\nTop Correlations")

    first = corr_list[0]
    second = corr_list[1]

    print(
        f"1. {first[0]} and {first[1]} : "
        f"{corr_matrix.loc[first[0], first[1]]:.3f}"
    )

    print("Observation: Passenger class and fare have a strong relationship.")

    print(
        f"2. {second[0]} and {second[1]} : "
        f"{corr_matrix.loc[second[0], second[1]]:.3f}"
    )

    print("Observation: Passenger class also affects survival.")

def multivariate_analysis(df):

    print("\nMultivariate Analysis")

    # Create family size feature
    df["family_size"] = df["sibsp"] + df["parch"] + 1

    # Chart 1 - Survival by class and gender
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="pclass",
        y="survived",
        hue="sex",
        errorbar=None
    )

    plt.title("Survival Rate by Class and Gender")
    plt.ylabel("Survival Rate")

    plt.savefig("chart1_class_sex_survival.png")
    plt.show()

    print("Observation: Female passengers had higher survival rates across all passenger classes.")

    # Chart 2 - Age distribution
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="age",
        hue="survived",
        multiple="stack",
        bins=30
    )

    plt.title("Age Distribution by Survival")

    plt.savefig("chart2_age_survival.png")
    plt.show()

    print("Observation: Children had a better survival rate than most adults.")

    # Chart 3 - Fare by passenger class
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="pclass",
        y="fare",
        hue="survived"
    )

    plt.title("Fare Distribution by Passenger Class")

    plt.savefig("chart3_fare_class_survival.png")
    plt.show()

    print("Observation: Higher fare passengers generally had better survival.")

    # Chart 4 - Family size
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="family_size",
        y="survived",
        errorbar=None
    )

    plt.title("Survival Rate by Family Size")

    plt.savefig("chart4_family_survival.png")
    plt.show()

    print("Observation: Small families had better survival than passengers travelling alone.")

    return df

def standardization_check(df):

    print("\nStandardization Check")

    # Standardize age and fare using z-score
    for col in ["age", "fare"]:

        mean = df[col].mean()
        std = df[col].std()

        df[f"{col}_zscore"] = (df[col] - mean) / std

        print(f"\n{col.capitalize()}")

        print(f"Before Standardization")
        print(f"Mean : {mean:.2f}")
        print(f"Std  : {std:.2f}")

        print(f"After Standardization")
        print(f"Mean : {df[f'{col}_zscore'].mean():.4f}")
        print(f"Std  : {df[f'{col}_zscore'].std():.4f}")

    # Compare original and standardized distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    df["age"].hist(ax=axes[0, 0], bins=30)
    axes[0, 0].set_title("Age - Original")

    df["age_zscore"].hist(ax=axes[0, 1], bins=30)
    axes[0, 1].set_title("Age - Standardized")

    df["fare"].hist(ax=axes[1, 0], bins=30)
    axes[1, 0].set_title("Fare - Original")

    df["fare_zscore"].hist(ax=axes[1, 1], bins=30)
    axes[1, 1].set_title("Fare - Standardized")

    plt.tight_layout()
    plt.savefig("standardization_comparison.png")
    plt.show()

    print("\nObservation: Standardization changes the scale but keeps the distribution shape.")

    return df


if __name__ == "__main__":
    df = load_and_clean_data()
    univariate_analysis(df)
    bivariate_analysis(df)
    df = multivariate_analysis(df)
    df = standardization_check(df)
    print("\nEDA completed successfully.")