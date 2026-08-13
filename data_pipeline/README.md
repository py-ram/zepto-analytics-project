# Data Pipeline

## Overview

This module is used to scrape book data from the **Books to Scrape** website. The scraped data is cleaned using pandas, converted from GBP to INR, and then stored in a SQLite database for further analysis.

## Conversion Rate

For this assignment, a fixed conversion rate is used:

**1 GBP = 105.50 INR**

## Project Files

- `scrape_books.py` - Scrapes book information from the website and saves it as a CSV file.
- `database_setup.py` - Cleans the data and creates the SQLite database.
- `queries.py` - Runs SQL queries and shows the results using both SQL and pandas.

## How to Run

Install the required packages:

```bash
pip install requests beautifulsoup4 pandas
```

Run the scripts in the following order:

```bash
python scrape_books.py
python database_setup.py
python queries.py
```

## Output Files

After running the project, the following files will be created:

- `books_raw.csv` - Raw scraped data
- `books_clean.csv` - Cleaned dataset
- `books.db` - SQLite database