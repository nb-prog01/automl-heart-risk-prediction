# automl-heart-risk-prediction
AutoML Pipeline for Heart Disease Prediction

## Overview

This project provides a machine learning pipeline to predict the risk of heart disease using the Framingham dataset. It includes data preprocessing, model training, and a FastAPI-based REST API for serving predictions.

## Features

- Data imputation using KNNImputer
- Logistic Regression model with class balancing
- Model serialization to `model.pkl`
- REST API for prediction using FastAPI
- Unit tests for model and API

## Project Structure

```
.
├── api.py                # FastAPI app for prediction
├── model.py              # Data processing and ML model code
├── model.pkl             # Saved trained model
├── model.ipynb           # Jupyter notebook for EDA and prototyping
├── data/
│   ├── framingham.csv    # Raw dataset
│   └── framingham_imputed.csv # Imputed dataset
├── tests/
│   ├── test_api.py       # API endpoint tests
│   └── test_model.py     # Model training and prediction tests
├── requirements.txt      # Python dependencies
├── Dockerfile            # Containerization setup
└── README.md             # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:
    ```sh
    git clone <repo-url>
    cd automl-heart-risk-prediction
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Training the Model

Run the following to train and save the model:
```sh
python -c "from model import train_model; train_model()"
```

### Running the API

Start the FastAPI server:
```sh
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Making Predictions

Send a POST request to `/predict` with patient features:
```json
{
  "features": [1.0, 39.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 195.0, 106.0, 70.0, 26.97, 80.0, 77.0]
}
```
Example using `curl`:
```sh
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.0, 39.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 195.0, 106.0, 70.0, 26.97, 80.0, 77.0]}'
```

### Running Tests

```sh
pytest tests/
```

## Docker Usage

Build and run the container:
```sh
docker build -t heart-risk-api .
docker run -p 8000:8000 heart-risk-api
```

## API Reference

- `GET /`  
  Returns a welcome message.

- `POST /predict`  
  Request body:  
  ```json
  { "features": [ ... ] }
  ```
  Response:  
  ```json
  { "heart_disease_probability": 0.42 }
  ```

## License

MIT License

---

**Dataset:** Framingham Heart Study  
**Model:** Logistic Regression  
**Imputation:** KNNImputer

For more details, see `model.py` and `api.py`.
