import pandas as pd
import sqlite3
import numpy as np
import os
import sys


def clean_and_setup_database():

    # Make sure the scraped file is available
    if not os.path.exists("books_raw.csv"):
        print("books_raw.csv not found.")
        print("Run scrape_books.py before creating the database.")
        sys.exit(1)

    # Load data
    df = pd.read_csv("books_raw.csv")

    # Clean price column
    df["price_gbp"] = (
        df["price"]
        .str.replace("£", "")
        .str.replace(",", "")
        .astype(float)
    )

    # Convert ratings into numbers
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    df["rating"] = df["star_rating"].map(rating_map)

    # Create stock status column
    df["in_stock"] = df["availability"].str.contains(
        "In stock",
        case=False,
        na=False
    )

    # Handle missing values if any
    if df["price_gbp"].isna().any():
        median_price = df["price_gbp"].median()
        df["price_gbp"].fillna(median_price, inplace=True)
        print("Missing prices filled with median value.")

    if df["rating"].isna().any():
        df = df.dropna(subset=["rating"])
        print("Rows with missing ratings removed.")

    # Convert GBP to INR
    conversion_rate = 105.50
    df["price_inr"] = df["price_gbp"] * conversion_rate

    conn = sqlite3.connect("books.db")
    cursor = conn.cursor()

    # Recreate tables every time the script runs
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
    """)

    # Store categories
    category_ids = {}

    for category in df["category"].unique():
        cursor.execute(
            "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
            (category,)
        )

        cursor.execute(
            "SELECT category_id FROM categories WHERE category_name = ?",
            (category,)
        )

        category_ids[category] = cursor.fetchone()[0]

    # Store books
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO books
            (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            int(row["rating"]),
            int(row["in_stock"]),
            category_ids[row["category"]]
        ))

    conn.commit()

    # Save cleaned dataset
    clean_df = df[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    ]

    clean_df.to_csv("books_clean.csv", index=False)

    print("\nDatabase created successfully.")
    print(f"Total books: {len(df)}")
    print(f"Categories: {list(df['category'].unique())}")
    print("\nBooks in each category:")
    print(df["category"].value_counts())

    print(
        f"\nPrice range: £{df['price_gbp'].min():.2f} "
        f"to £{df['price_gbp'].max():.2f}"
    )
    print(f"Average rating: {df['rating'].mean():.2f}")

    return conn


if __name__ == "__main__":
    connection = clean_and_setup_database()
    connection.close()