# Spam Classifier API

A simple spam classification API built with FastAPI and scikit-learn.

## Features

- Text classification (spam/ham)
- REST API endpoint for predictions
- TF-IDF vectorization
- Logistic Regression model

## Setup

1. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Unix
# or
venv\Scripts\activate  # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
uvicorn main:app --reload
```

## API Usage

Send POST request to `/predict`:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "YOUR TEXT HERE"}'
```

## Project Structure

```
spam-classifier/
├── data/           # Dataset
├── src/            # Source code
├── trained_models/ # Saved models
└── notebooks/      # Jupyter notebooks
```

## Technologies

- FastAPI
- scikit-learn
- TF-IDF
- Pandas

Sources
