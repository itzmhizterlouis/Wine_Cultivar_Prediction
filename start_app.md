# Start App - Vigneron AI

Precision Enological Classification System for determining wine provenance based on chemical analysis.

## Prerequisites

- Python 3.8+
- Required libraries: FastAPI, Uvicorn, Scikit-learn, joblib, Matplotlib, Seaborn, Pandas

## Installation

1. Navigate to the project directory:
   ```bash
   cd "Wine-Cultivar-Prediction"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the model (`best_svm.pkl`) and scaler (`scaler.pkl`) files are present.

## Running the Application

Execute the following command south to start the server:

```bash
uvicorn app:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

## Features

- **Chemical Spectrum Analysis**: Input 13 chemical markers including alcohol, flavonoids, and color intensity.
- **SVM Classification**: High-accuracy Support Vector Machine model for cultivar identification.
- **Domain Insights**: Embedded exploratory data analysis (EDA) visualizations.
- **Premium Design**: Vineyard-themed interface with sophisticated enological aesthetics.
