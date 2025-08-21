import pandas as pd
import os
import sys
import logging
from sklearn.model_selection import train_test_split

os.makedirs('logs', exist_ok=True)
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)
log_file = os.path.join('logs', 'read_data.log')
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
    if len(sys.argv) != 2:
        print("Uso: python read_data.py <ruta_a_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        logger.error(f"Archivo no encontrado: {csv_path}")
        sys.exit(1)
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    logger.info(f"Leyendo datos desde {csv_path}")
    df = pd.read_csv(csv_path)
    if 'body_mass_g' not in df.columns:
        logger.error("Columna 'body_mass_g' no encontrada en el dataset.")
        sys.exit(1)
    df = df.dropna(subset=['body_mass_g'])
    logger.info("Dividiendo datos en entrenamiento y prueba")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['species'])
    train_path = os.path.join('data', 'train', f'{filename}_train.csv')
    test_path = os.path.join('data', 'test', f'{filename}_test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Datos de entrenamiento guardados en {train_path}")
    logger.info(f"Datos de prueba guardados en {test_path}")

if __name__ == "__main__":
    main()
