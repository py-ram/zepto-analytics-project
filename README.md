# Zepto Analytics Project

This repository contains three modules developed for the Zepto data science assignment.

## Project Modules

### 1. Data Pipeline

The `data_pipeline` module scrapes book data from Books to Scrape, cleans the data, converts prices from GBP to INR, and loads the data into SQLite.

### 2. Analytics

The `analytics` module performs exploratory data analysis and predictive modelling using the Titanic dataset.

The analysis includes:

- Data cleaning
- Missing value analysis
- Exploratory data analysis
- Correlation analysis
- Feature standardization
- Classification modelling
- Model evaluation

Models used:

- Logistic Regression
- Decision Tree
- Random Forest

### 3. Support Assistant

The `support_assistant` module implements a retrieval-based Zepto policy assistant using:

- `all-MiniLM-L6-v2`
- Sentence Transformers
- ChromaDB
- LangGraph `StateGraph`
- Pydantic
- FastAPI

The default execution uses `MOCK_LLM=1`, so an external LLM API is not required.

---

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Pipeline

```bash
cd data_pipeline
python scrape_books.py
python database_setup.py
python queries.py
```

---

## Analytics

```bash
cd analytics
python 01_eda.py
python 02_modeling.py
```

The modelling workflow compares Logistic Regression, Decision Tree and Random Forest models using accuracy, precision, recall, F1 score and ROC AUC.

---

## Support Assistant

```bash
cd support_assistant
pip install -r requirements.txt
python test_assistant.py
```

The test covers both:

1. Policy question → retrieval route
2. General question → direct-answer route

Expected result:

```text
All support assistant tests passed.
```

Start the API:

```bash
python main.py
```

API documentation:

```text
http://127.0.0.1:7860/docs
```

---

## API Example

### Policy Question

```json
{
  "query": "What is the delivery fee for orders below INR 149?"
}
```

### General Question

```json
{
  "query": "What is the weather today?"
}
```

---

## Docker

From the `support_assistant` directory:

```bash
docker build -t zepto-support .
```

Run:

```bash
docker run -p 7860:7860 zepto-support
```

Then open:

```text
http://127.0.0.1:7860/docs
```

---

## Notes

The ChromaDB database is generated locally at runtime.

The default `MOCK_LLM=1` mode does not require an external LLM API.

The optional real-LLM path can be enabled using:

```text
MOCK_LLM=0
```

with the required API key configured in the environment.
