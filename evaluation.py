import pandas as pd
import os
import sys
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

os.makedirs('logs', exist_ok=True)
log_file = os.path.join('logs', 'evaluation.log')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) != 3:
        print("Uso: python evaluation.py <ruta_a_modelo> <ruta_a_test_csv>")
        sys.exit(1)
    model_path = sys.argv[1]
    test_path = sys.argv[2]
    if not os.path.exists(model_path):
        logger.error(f"Modelo no encontrado: {model_path}")
        sys.exit(1)
    if not os.path.exists(test_path):
        logger.error(f"Archivo de test no encontrado: {test_path}")
        sys.exit(1)
    logger.info(f"Cargando modelo desde {model_path}")
    pipe = joblib.load(model_path)
    logger.info(f"Leyendo datos de prueba desde {test_path}")
    df = pd.read_csv(test_path)
    X_test = df.drop(columns=['body_mass_g'])
    y_test = df['body_mass_g']
    logger.info("Evaluando modelo en datos de prueba")
    y_pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Test RMSE : {rmse:,.0f} g")
    logger.info(f"Test MAE  : {mae:,.0f} g")
    logger.info(f"Test R²   : {r2:.3f}")

if __name__ == "__main__":
    main()
