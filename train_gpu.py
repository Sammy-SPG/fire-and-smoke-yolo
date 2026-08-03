import sys
import torch
from ultralytics import YOLO

def check_gpu():
    print("=" * 50)
    print("Verificando disponibilidad de GPU (CUDA)...")
    if not torch.cuda.is_available():
        print("[ERROR] CUDA no está disponible en este entorno PyTorch.")
        print("\nPara activar la GPU en PyTorch, ejecuta en tu terminal:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        print("=" * 50)
        sys.exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[OK] GPU detectada: {device_name}")
    print(f"[OK] VRAM total: {vram:.2f} GB")
    print("=" * 50)

def main():
    check_gpu()

    model = YOLO("yolo26n.pt")

    print("\nIniciando entrenamiento optimizado para GPU RTX 5060...")
    
    results = model.train(
        data="result_build_dataset/fire-and-smoke.yaml",
        epochs=60,             # Número de épocas para entrenamiento completo
        imgsz=640,             # Tamaño de imagen estándar
        device=0,              # Fuerza el uso del ID de GPU 0
        batch=-1,              # AutoBatch: calcula automáticamente el batch máximo según la VRAM
        workers=8,             # Procesamiento en paralelo para cargar datos a la GPU
        amp=True,              # Precisión mixta automática (FP16/BF16) para máxima velocidad
        plots=True             # Genera gráficas de rendimiento (loss, mAP, etc.)
    )

    print("\n¡Entrenamiento completado!")
    print("El modelo guardado y las métricas se encuentran en la carpeta 'runs/detect/train'.")

if __name__ == "__main__":
    main()
