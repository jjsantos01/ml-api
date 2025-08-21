# Entrenamiento, Evaluación y API de Penguins

## Requisitos
- Python 3.8+
- Instalar dependencias:
  ```sh
  pip install -r requirements.txt
  ```


## Variables de entorno
- `PYTHONIOENCODING=utf-8` (recomendado para evitar problemas de encoding en logs)
- `MODEL_PATH` = # Ruta de donde se cargará el modelo entrenado para la API

## Estructura de carpetas
- `data/raw/` — Archivos CSV originales
- `data/` — Archivos generados de train/test
- `models/` — Modelos entrenados
- `logs/` — Logs de cada script
- `training/` — Scripts de preparación, entrenamiento y evaluación
- `api/` — Aplicación FastAPI para predicción

## Uso de scripts (carpeta `training/`)

### 1. Separar datos en train/test
```sh
python training/read_data.py <ruta_al_csv_original>
```
- Ejemplo:
  ```sh
  python training/read_data.py data/raw/penguins_20250101.csv
  ```
- Genera: `data/penguins_20250101_train.csv` y `data/penguins_20250101_test.csv`

### 2. Entrenar modelo
```sh
python training/training.py <ruta_train_csv> <ruta_output_modelo>
```
- Ejemplo:
  ```sh
  python training/training.py data/penguins_20250101_train.csv models/penguins_model_v1.joblib
  ```

### 3. Evaluar modelo
```sh
python training/evaluation.py <ruta_modelo> <ruta_test_csv>
```
- Ejemplo:
  ```sh
  python training/evaluation.py models/penguins_model_v1.joblib data/penguins_20250101_test.csv
  ```

## Ejecutar la API (FastAPI)

- Asegúrate de tener un modelo entrenado (por ejemplo `models/penguins_model_v2.joblib`).
- Configura la variable de entorno `MODEL_PATH` (puedes usar `.env`).

```sh
# Opción A: exportar en shell
export MODEL_PATH=models/penguins_model_v2.joblib

# Opción B: archivo .env en la raíz
echo "MODEL_PATH=models/penguins_model_v2.joblib" > .env
```

- Instala dependencias y arranca el servidor con Uvicorn:

```sh
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Probar endpoints:

```sh
# Salud
curl http://localhost:8000/

# Predicción (ejemplo con un registro)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{
    "species": "Adelie",
    "island": "Torgersen",
    "sex": "male",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181
  }]'
```

## Logs
- Cada script genera su propio log en la carpeta `logs/`.

## Notas
- Todos los scripts deben ejecutarse desde la raíz del proyecto.
- Los nombres de archivos pueden tener cualquier formato, pero deben ser consistentes entre train/test/modelo.
- La API carga el modelo desde `MODEL_PATH` al iniciar; si cambia el path, reinicia el servidor.
