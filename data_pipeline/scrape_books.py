import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys


def scrape_category(base_url, category_slug):

    books_list = []
    page_num = 1

    while True:

        if page_num == 1:
            page_url = f"{base_url}/catalogue/category/books/{category_slug}/index.html"
        else:
            page_url = f"{base_url}/catalogue/category/books/{category_slug}/page-{page_num}.html"

        print(f"Scraping page {page_num}...")

        try:
            response = requests.get(page_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Unable to open page: {e}")
            break

        soup = BeautifulSoup(response.content, "html.parser")

        books = soup.select("article.product_pod")

        if not books:
            break

        # Read category name from the page
        if page_num == 1:
            heading = soup.select_one("div.page-header h1")
            if heading:
                category_name = heading.text.strip()
            else:
                category_name = category_slug

        # Extract book details
        for book in books:
            try:
                title = book.select_one("h3 a")["title"]
                price = book.select_one("p.price_color").text.strip()
                rating = book.select_one("p.star-rating")["class"][1]
                availability = (
                    book.select_one("p.instock.availability")
                    .text.strip()
                )

                books_list.append({
                    "title": title,
                    "price": price,
                    "star_rating": rating,
                    "availability": availability,
                    "category": category_name
                })

            except Exception as e:
                print("Skipping one book:", e)

        # Stop if there are no more pages
        if soup.select_one("li.next a"):
            page_num += 1
            time.sleep(1)
        else:
            break

    return books_list


def scrape_books():

    base_url = "http://books.toscrape.com"

    categories = {
        "travel_2": "Travel",
        "mystery_3": "Mystery",
        "historical-fiction_4": "Historical Fiction",
        "science-fiction_16": "Science Fiction"
    }

    all_books = []

    print("Starting scraping...")

    for slug, name in categories.items():

        print(f"\nCategory: {name}")

        books = scrape_category(base_url, slug)

        all_books.extend(books)

        print(f"Collected {len(books)} books")

        # Small delay between categories
        time.sleep(2)

    if len(all_books) == 0:
        print("No data was scraped.")
        sys.exit(1)

    df = pd.DataFrame(all_books)

    # Save raw data
    df.to_csv("books_raw.csv", index=False)

    print("\nScraping completed.")
    print(f"Total books: {len(df)}")

    print("\nBooks in each category:")
    print(df["category"].value_counts())

    print("\nData saved to books_raw.csv")

    return df


if __name__ == "__main__":

    try:
        df = scrape_books()

        print("\nFirst few records:")
        print(df.head())

    except Exception as e:
        print("Error:", e)
        sys.exit(1)