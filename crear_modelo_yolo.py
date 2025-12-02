import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURACIÓN ---
DATASET_PATH = './recursos/dataset'
NOMBRE_PROYECTO = 'experimentos_yolo'

# Configuraciones de entrenamiento
EPOCHS = 40
LR_INICIAL = 0.0005

# Nombre dinámico basado en hiperparámetros
NOMBRE_EXPERIMENTO = f'epochs_{EPOCHS}_lr_{LR_INICIAL}'.replace('.', '-')

def entrenar_y_evaluar():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: No existe el dataset en '{DATASET_PATH}'.")
        print("   Asegúrate de haber ejecutado el script 'preparar_dataset_tres_conjuntos.py' antes.")
        return
    print(f"🚀 INICIANDO ENTRENAMIENTO: {NOMBRE_EXPERIMENTO}")
    print(f"   Datos: {DATASET_PATH}")

    # 1. Cargar Modelo
    model = YOLO('yolov8m-cls.pt') 

    # 2. Entrenar
    # YOLO usará automáticamente las carpetas 'train' y 'val' dentro de DATASET_PATH
    results = model.train(
        data=DATASET_PATH,
        project=NOMBRE_PROYECTO,
        name=NOMBRE_EXPERIMENTO,
        
        epochs=EPOCHS,
        imgsz=224,
        batch=8, 
        
        optimizer='AdamW',
        lr0=LR_INICIAL,
        patience=10,
        dropout=0.1,
        
        device=0, # GPU
        workers=8,
        amp=False, 

        mosaic=0.0,
        erasing=0.0,
        fliplr=0.0,
        
        plots=True,
        save=True
    )
    
    save_dir = results.save_dir
    print(f"\n✅ Entrenamiento finalizado. Resultados base en: {save_dir}")

    print("\n📊 Generando Métricas Detalladas sobre el conjunto de TEST...")
    # --- A. Gráficos de Evolución (Loss / Accuracy) ---
    csv_path = os.path.join(save_dir, 'results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        # Graficar Loss
        plt.figure(figsize=(10, 5))
        plt.plot(df['train/loss'], label='Train Loss', color='blue')
        plt.plot(df['val/loss'], label='Val Loss', color='orange')
        plt.title('Evolución de Pérdida (Loss)')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'grafico_evolucion_loss.png'))
        plt.close()

        # Graficar Accuracy (Top-1)
        acc_col = [c for c in df.columns if 'top1' in c]
        if acc_col:
            plt.figure(figsize=(10, 5))
            plt.plot(df[acc_col[0]], label='Val Accuracy (Top-1)', color='green')
            plt.title('Evolución de Precisión (Accuracy)')
            plt.xlabel('Épocas')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'grafico_evolucion_accuracy.png'))
            plt.close()
            
        print(f"   -> Gráficos de evolución guardados en: {save_dir}")
    
    # --- B. Métricas Detalladas Finales (F1, Precision, Recall por clase) ---
    # Cargamos el mejor modelo para hacer la validación final
    best_model = YOLO(os.path.join(save_dir, 'weights', 'best.pt'))
    test_dir = os.path.join(DATASET_PATH, 'test')
    if not os.path.exists(test_dir):
        print("⚠️ No se encontró carpeta 'test'. Verifica tu dataset.")
    else:
        clases = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        
        y_true = []
        y_pred = []
        
        print("   -> Calculando matriz de confusión y F1-Score sobre TEST...")
        
        for idx, clase_nombre in enumerate(clases):
            path_clase = os.path.join(test_dir, clase_nombre)
            imagenes = [os.path.join(path_clase, f) for f in os.listdir(path_clase) if f.lower().endswith(('.jpg', '.png'))]
            
            # Inferencia lote por lote
            batch_size_inf = 8
            for i in range(0, len(imagenes), batch_size_inf):
                batch_imgs = imagenes[i:i+batch_size_inf]
                if not batch_imgs: continue
                
                results_inf = best_model(batch_imgs, verbose=False)
                
                for r in results_inf:
                    y_true.append(idx)
                    y_pred.append(r.probs.top1)

        # Generar Reporte de Texto
        reporte = classification_report(y_true, y_pred, target_names=clases)
        print("\n" + "="*50)
        print("REPORTE DE CLASIFICACIÓN FINAL (TEST SET)")
        print("="*50)
        print(reporte)
        
        with open(os.path.join(save_dir, 'reporte_metricas_test.txt'), 'w') as f:
            f.write(reporte)

        # Generar Matriz de Confusión
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
        plt.xlabel('Predicción')
        plt.ylabel('Realidad')
        plt.title('Matriz de Confusión (Test)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'matriz_confusion_test.png'))
        plt.close()

    print(f"\n🏆 PROCESO COMPLETO.")
    print(f"   Revisa la carpeta '{save_dir}' para ver todos los archivos.")

if __name__ == "__main__":
    entrenar_y_evaluar()