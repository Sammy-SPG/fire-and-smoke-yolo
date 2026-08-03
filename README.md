# YOLO26 Entrenamiento para Detección de Fuego y Humo

Este proyecto tiene como objetivo el entrenamiento específico del modelo YOLO26 para la detección de humo y fuego. A partir de varios datasets, se estructura la información en el formato físico requerido por YOLO y se procede con el entrenamiento y posterior testeo del modelo resultante.

## Origen de los Datos (Datasets)

Los archivos utilizados para el entrenamiento provienen de los siguientes datasets de Ultralytics Hub:
- [dfire (por xuwei-pro)](https://platform.ultralytics.com/xuwei-pro/datasets/dfire)
- [fire-and-smoke-dataset (por dong)](https://platform.ultralytics.com/dong/datasets/fire-and-smoke-dataset)

## Estructura y Flujo del Proyecto

El proceso está dividido en tres etapas principales: preparación del dataset, entrenamiento y testeo.

### 1. Construcción del Dataset (`build-dataset.py`)

A partir de los datasets crudos, este script se encarga de construir la estructura de carpetas que YOLO requiere para el entrenamiento. La estructura resultante separa físicamente:
- **Imágenes** (`images/`)
- **Etiquetas** (`labels/`)

Ambas carpetas están divididas en los subconjuntos `train` (entrenamiento), `val` (validación) y `test` (prueba). Cada imagen va acompañada de un archivo de texto (`.txt`) con las coordenadas de las cajas delimitadoras y se genera además un archivo `.yaml` de configuración para indicarle al modelo dónde encontrar los datos y sus clases.

### 2. Entrenamiento (`train-cpu.py` o `train-gpu.py`)

Este script utiliza los archivos estructurados y generados en el paso anterior junto con el archivo de configuración `.yaml` para realizar el entrenamiento del modelo YOLO.

### 3. Pruebas y Testeo (`image-test.py` y `camera-test.py`)

Una vez que el modelo ha sido entrenado, se utilizan estos scripts para comprobar su eficacia:
- **`image-test.py`**: Prueba el modelo entrenado realizando predicciones sobre una imagen estática.
- **`camera-test.py`**: Realiza detección de humo y fuego en tiempo real utilizando la cámara web.

## Configuración del Entorno y Dependencias

Para ejecutar este proyecto, es recomendable utilizar un entorno virtual para instalar las dependencias de forma aislada.

1. **Crear el entorno virtual:**
   Abre una terminal en la raíz del proyecto y ejecuta:
   ```bash
   python3 -m venv venv
   ```

2. **Activar el entorno virtual:**
   - En Linux:
     ```bash
     source venv/bin/activate
     ```
   - En Windows:
     ```bash
     .\venv\Scripts\activate
     ```

3. **Instalar dependencias:**
   Este proyecto requiere las dependencias especificadas. Puedes instalarlas mediante:
   ```bash
   pip install -r requirements.txt
   ```
