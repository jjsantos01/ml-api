# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()
########################################################################
# 1. Carga del pipeline entrenado (incluye preprocesamiento + modelo)  #
########################################################################
MODEL_PATH = os.environ["MODEL_PATH"]
pipe = joblib.load(MODEL_PATH)

###############################################
# 2. Definición del esquema de entrada (JSON) #
###############################################
class PenguinFeatures(BaseModel):
    species: str = Field(..., example="Adelie")          # Adelie, Gentoo, Chinstrap
    island:  str = Field(..., example="Torgersen")       # Torgersen, Biscoe, Dream
    sex:     str = Field(..., example="male")            # male, female
    bill_length_mm:    float = Field(..., example=39.1)
    bill_depth_mm:     float = Field(..., example=18.7)
    flipper_length_mm: float = Field(..., example=181)

###########################################
# 3. Inicializa la aplicación FastAPI     #
###########################################
app = FastAPI(
    title="Penguin Body-Mass Predictor",
    version="1.0.0",
    description="API que predice el peso (en gramos) de pingüinos del conjunto Palmer Penguins."
)

###########################################
# 4. Endpoint de salud opcional           #
###########################################
@app.get("/")
def read_root():
    return {"message": "Alive and ready to predict 🐧"}

###########################################
# 5. Endpoint de predicción               #
###########################################
@app.post("/predict")
def predict(penguins: List[PenguinFeatures]):
    """
    Devuelve la predicción de `body_mass_g` para uno o varios pingüinos.
    """
    # Convertimos lista de objetos Pydantic → DataFrame
    X = pd.DataFrame([p.dict() for p in penguins])

    # Realizamos predicciones
    preds = pipe.predict(X)

    # Retornamos cada resultado redondeado a 2 decimales
    return {
        "predictions": [round(float(p), 2) for p in preds]
    }
