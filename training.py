import pandas as pd
import os
import sys
import logging
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib

MAX_RMSE_THRESHOLD = 500
os.makedirs('logs', exist_ok=True)

log_file = os.path.join('logs', 'training.log')
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
        print("Uso: python training.py <ruta_a_train_csv> <ruta_output_modelo>")
        sys.exit(1)
    train_path = sys.argv[1]
    model_path = sys.argv[2]
    if not os.path.exists(train_path):
        logger.error(f"Archivo no encontrado: {train_path}")
        sys.exit(1)
    out_dir = os.path.dirname(model_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Leyendo datos de entrenamiento desde {train_path}")
    df = pd.read_csv(train_path)
    X = df.drop(columns=['body_mass_g'])
    y = df['body_mass_g']
    num_attribs = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
    cat_attribs = ['species', 'island', 'sex']
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_attribs),
        ('cat', categorical_pipeline, cat_attribs)
    ])
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline([
        ('prep', preprocessor),
        ('rf', model)
    ])
    logger.info("Realizando validación cruzada (5-fold)")
    cv_rmse = -cross_val_score(pipe, X, y, scoring='neg_root_mean_squared_error', cv=5)
    logger.info(f"CV RMSEs: {cv_rmse}")
    logger.info(f"CV RMSE promedio (5-fold): {cv_rmse.mean():,.0f} g")
    if cv_rmse.mean() > MAX_RMSE_THRESHOLD:
        logger.warning(f"RMSE promedio alto ({cv_rmse.mean():,.0f} g), revisar los datos o el modelo. El entrenamiento no continuará.")
        return
    logger.info("Entrenando modelo final")
    pipe.fit(X, y)
    joblib.dump(pipe, model_path)
    logger.info(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    main()
