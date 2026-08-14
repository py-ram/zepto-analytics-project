# Zepto Analytics Project

This repository contains three modules for Zepto's data science needs:

1. **Data Pipeline** (`/data_pipeline`) - Web scraping, data cleaning and SQLite database management
2. **Analytics** (`/analytics`) - Exploratory data analysis and predictive modelling
3. **Support Assistant** (`/support_assistant`) - Retrieval-based support assistant for Zepto policy documents

## Setup

### Requirements

Each module has its own requirements file. The root `requirements.txt` contains the main project dependencies.

Install the dependencies with:

bash
pip install -r requirements.txt


**##1. Data Pipeline**

#The data pipeline collects book information from Books to Scrape, cleans the data and stores it in a SQLite database.

Go to the module:

cd data_pipeline

Run:

python scrape_books.py
python database_setup.py
python queries.py


The pipeline uses the project-defined conversion rate:

1 GBP = 105.50 INR
Main files
scrape_books.py - Scrapes book information
database_setup.py - Cleans the data and creates the SQLite database
queries.py - Runs SQL queries and compares SQL results with Pandas

**##2. Analytics**

The analytics module uses the Titanic dataset for exploratory analysis and classification modelling.

The analysis includes:

Data cleaning
Missing value analysis
Univariate analysis
Bivariate analysis
Outlier detection
Correlation analysis
Multivariate analysis
Feature standardization
Logistic Regression
Decision Tree
Random Forest
Model evaluation

Go to the module:

cd analytics

Run the analysis:

python 01_eda.py
python 02_modeling.py

The analysis generates cleaned data, model results and visualizations.

**3. Support Assistant**
T#he support assistant provides a simple API for answering questions related to Zepto policy documents.

The current implementation uses:

Sentence-transformer embeddings
Cosine similarity
Basic intent classification
In-memory document retrieval
FastAPI

Go to the module:

cd support_assistant

Install its dependencies:

pip install -r requirements.txt

Start the API:

python main.py

The API documentation is available at:

http://127.0.0.1:7860/docs
Testing

Test the assistant:

python test_assistant.py

Test the API:

python test_api.py
└── support_assistant/




