import os
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse

# Use Agg backend for matplotlib
matplotlib.use("Agg")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vigneron AI | Precision Cultivar Analysis")

# Assets
os.makedirs('static', exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Constants
MODEL_PATH = 'best_svm.pkl'
SCALER_PATH = 'scaler.pkl'
TARGET_NAMES = ['Class 0 (High Alcohol/Phenols)', 'Class 1 (Moderate)', 'Class 2 (Low)']
FEATURE_NAMES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Model and Scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        logger.error(f"Failed to load assets: {e}")
        return None, None

model, scaler = load_assets()

# Generate EDA images
def generate_eda():
    wine_data = load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    df['target'] = wine_data.target
    
    eda_path = os.path.join('static', 'eda_scatter.png')
    dist_path = os.path.join('static', 'class_distribution.png')
    
    if not os.path.exists(eda_path):
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x='alcohol', y='flavanoids', hue='target', data=df, palette='viridis')
        plt.title('Alcohol vs Flavanoids Spectrum')
        plt.tight_layout()
        plt.savefig(eda_path)
        plt.close()
        logger.info(f"Generated {eda_path}")

    if not os.path.exists(dist_path):
        plt.figure(figsize=(6, 4))
        sns.countplot(x='target', data=df, palette='magma')
        plt.title('Cultivar Distribution Density')
        plt.tight_layout()
        plt.savefig(dist_path)
        plt.close()
        logger.info(f"Generated {dist_path}")

generate_eda()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": FEATURE_NAMES,
            "result": False,
            "eda_scatter": "/static/eda_scatter.png",
            "class_dist": "/static/class_distribution.png"
        }
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()
    errors = []
    input_values = []

    for feature in FEATURE_NAMES:
        val = form_data.get(feature)
        if not val or val.strip() == "":
            errors.append(f"{feature.replace('_', ' ').capitalize()} is missing.")
        else:
            try:
                input_values.append(float(val))
            except ValueError:
                errors.append(f"{feature.replace('_', ' ').capitalize()} must be numeric.")

    if errors or model is None:
        if model is None: errors.append("Analysis engine is currently offline.")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "features": FEATURE_NAMES,
                "result": False,
                "errors": errors,
                "eda_scatter": "/static/eda_scatter.png",
                "class_dist": "/static/class_distribution.png"
            }
        )

    try:
        logger.info(f"Analysis requested for profile: {input_values}")
        input_array = np.array(input_values).reshape(1, -1)
        
        # Scaling if necessary (original code had scaler loaded but not used in prediction block?)
        # For SVM, scaling is usually required.
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]
        cultivar = TARGET_NAMES[prediction]
        
        logger.info(f"Analysis result: {cultivar}")

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "features": FEATURE_NAMES,
                "result": True,
                "cultivar": cultivar,
                "eda_scatter": "/static/eda_scatter.png",
                "class_dist": "/static/class_distribution.png"
            }
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "features": FEATURE_NAMES,
                "result": False,
                "errors": [f"Synthesis failure: {str(e)}"],
                "eda_scatter": "/static/eda_scatter.png",
                "class_dist": "/static/class_distribution.png"
            }
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
