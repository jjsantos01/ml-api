# Entrenamiento y Evaluación de Modelos de Penguins

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

## Uso de scripts

### 1. Separar datos en train/test
```sh
python read_data.py <ruta_al_csv_original>
```
- Ejemplo:
  ```sh
  python read_data.py data/raw/penguins_20250101.csv
  ```
- Genera: `data/penguins_20250101_train.csv` y `data/penguins_20250101_test.csv`

### 2. Entrenar modelo
```sh
python training.py <ruta_train_csv> <ruta_output_modelo>
```
- Ejemplo:
  ```sh
  python training.py data/penguins_20250101_train.csv models/penguins_model_v1.joblib
  ```

### 3. Evaluar modelo
```sh
python evaluation.py <ruta_modelo> <ruta_test_csv>
```
- Ejemplo:
  ```sh
  python evaluation.py models/penguins_model_v1.joblib data/penguins_20250101_test.csv
  ```

## Logs
- Cada script genera su propio log en la carpeta `logs/`.

## Notas
- Todos los scripts deben ejecutarse desde la raíz del proyecto.
- Los nombres de archivos pueden tener cualquier formato, pero deben ser consistentes entre train/test/modelo.
