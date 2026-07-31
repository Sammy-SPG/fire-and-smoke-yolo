import json
import os
import yaml
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ESTANDAR_CLASES = {
    "smoke": 0,
    "fire": 1
}

def setup_directories(base_path):
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(f"{base_path}/images/{split}", exist_ok=True)
        os.makedirs(f"{base_path}/labels/{split}", exist_ok=True)

def download_and_process(data, output_dir, original_classes):
    if data.get("type") != "image":
        return
        
    split = data.get("split", "train")
    if split not in ["train", "val", "test"]:
        split = "train"
        
    img_name = data["file"]
    img_path = f"{output_dir}/images/{split}/{img_name}"
    label_name = Path(img_name).stem + ".txt"
    label_path = f"{output_dir}/labels/{split}/{label_name}"

    if not os.path.exists(img_path):
        try:
            response = requests.get(data["url"], timeout=10)
            if response.status_code == 200:
                with open(img_path, "wb") as img_file:
                    img_file.write(response.content)
            else:
                return
        except Exception as e:
            print(f"Error {img_name}: {e}")
            return

    if "annotations" in data and "boxes" in data["annotations"]:
        with open(label_path, "w") as label_file:
            for box in data["annotations"]["boxes"]:
                original_id, x, y, w, h = box
                
                # Traducir el ID original al nombre en texto (ej. de 0 a "fire")
                class_name = original_classes.get(str(int(original_id)))
                
                # Mapear el nombre en texto a nuestro ID unificado (ej. de "fire" a 1)
                if class_name in ESTANDAR_CLASES:
                    new_id = ESTANDAR_CLASES[class_name]
                    label_file.write(f"{new_id} {x} {y} {w} {h}\n")

def procesar_datasets_multiples(archivos_ndjson, output_dir, max_workers=20):
    setup_directories(output_dir)

    yaml_data = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {"0": "smoke", "1": "fire"}
    }
            
    with open(f"{output_dir}/fire-and-smoke.yaml", "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file)

    tareas = []
    
    for archivo in archivos_ndjson:
        with open(archivo, 'r') as f:
            primera_linea = json.loads(f.readline())
            clases_originales = primera_linea.get("class_names", {})
            
            for linea in f:
                data = json.loads(linea)
                tareas.append((data, clases_originales))

    print(f"Descargando y procesando {len(tareas)} imágenes. Esto puede tomar un momento...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for data, clases_orig in tareas:
            executor.submit(download_and_process, data, output_dir, clases_orig)

lista_archivos = [
    "fire-and-smoke0.ndjson",
    "fire-and-smoke1.ndjson",
    "fire-and-smoke2.ndjson"
]

procesar_datasets_multiples(lista_archivos, "result_build_dataset")