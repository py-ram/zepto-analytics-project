import sqlite3
import pandas as pd


def run_queries():

    conn = sqlite3.connect("books.db")

    # SQL queries used in this assignment
    queries = {
        "select_where": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE rating >= 4 AND in_stock = 1
            LIMIT 5
        """,

        "order_by": """
            SELECT title, price_inr
            FROM books
            ORDER BY price_inr DESC
            LIMIT 10
        """,

        "distinct": """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating
        """,

        "in_between": """
            SELECT title, price_gbp
            FROM books
            WHERE rating IN (4, 5)
            AND price_gbp BETWEEN 20 AND 40
            LIMIT 5
        """,

        "join": """
            SELECT c.category_name,
                   COUNT(b.book_id) AS book_count,
                   AVG(b.rating) AS avg_rating,
                   AVG(b.price_gbp) AS avg_price
            FROM books b
            JOIN categories c
            ON b.category_id = c.category_id
            GROUP BY c.category_name
            ORDER BY avg_rating DESC
        """
    }

    results = {}

    # Execute each query
    for name, query in queries.items():
        print(f"\n{name.upper()}")
        df = pd.read_sql(query, conn)
        print(df)
        results[name] = df

    # Load tables into pandas
    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)

    # Merge and calculate summary
    merged_df = books_df.merge(categories_df, on="category_id")

    summary = (
        merged_df.groupby("category_name")
        .agg({
            "book_id": "count",
            "rating": "mean",
            "price_gbp": "mean"
        })
        .reset_index()
    )

    summary.columns = [
        "category_name",
        "book_count",
        "avg_rating",
        "avg_price"
    ]

    print("\nPandas Summary")
    print(summary)

    # Compare SQL and pandas results
    sql_result = results["join"].sort_values("category_name").reset_index(drop=True)
    pandas_result = summary.sort_values("category_name").reset_index(drop=True)

    print("\nVerification")
    print("Results match:", sql_result.equals(pandas_result))

    conn.close()

    return results


if __name__ == "__main__":
    results = run_queries()