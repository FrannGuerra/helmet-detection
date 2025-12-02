# helmet-detection

## 📋 Descripción del Proyecto

Sistema de detección y clasificación de cascos en motociclistas.

---

## 📁 Estructura del Proyecto

```
helmet-detection/
├── app.py                              # Aplicación Flask (API + interfaz web)
├── crear_modelo_yolo.py                # Script para entrenar el modelo
├── recursos/
│   ├── dataset/                        # Dataset (comprimido)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── imagenes_prueba_app/            # Imágenes para probar la app final (comprimido)
├── static/
│   ├── uploads/                        # Imágenes cargadas por el usuario
│   └── results/                        # Resultados procesados
├── templates/
│   └── index.html                      # Interfaz web
└── README.md                           # Este archivo
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- GPU NVIDIA (recomendado para YOLO, opcional)

### Archivos a descargar:
- Modelo YOLO de clasificación entrenado si no se desea entrenar
- Imágenes de dataset y de prueba de aplicación

## 📦 Instalación de Dependencias

#### **Para `crear_modelo_yolo.py`**
```bash
pip install ultralytics pandas matplotlib seaborn scikit-learn
```

#### **Para `app.py` (Flask)**
```bash
pip install flask ultralytics opencv-python pillow numpy
```

---

## 🔧 Cómo Usar

### Paso 1: Entrenar el Modelo

```bash
python crear_modelo_yolo.py
```

**Esto:**
- Entrena un modelo YOLOv8M para clasificación
- Genera gráficos de evolución (loss, accuracy)
- Crea matriz de confusión
- Guarda el mejor modelo en `experimentos_yolo/`
- Exporta métricas detalladas (F1-Score, Precision, Recall)

Ya hay un modelo .pt en el repositorio, para probar directamente la aplicación, se encuentra en `experimentos_yolo/epochs_40_lr_0-0005/weights/best.pt`

### Paso 2: Ejecutar la Aplicación Web

```bash
python app.py
```

**Luego acceder a:**
```
http://localhost:5000
```

**La aplicación permite:**
- Subir modelo entrenado (.pt)
- Procesar imágenes y ver los resultados de la detección y clasificación


---