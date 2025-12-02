import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO
import uuid

# --- CONFIGURACIÓN ---
app = Flask(__name__)

# Directorios
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
MODEL_FILENAME = "custom_helmet_model.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Variables globales
detector = None
classifier = None
CLASS_NAMES = ['sin_casco', 'con_casco']
TARGET_SIZE = (224, 224)

def redimensionar_con_padding(img, target_size):
    """
    Aplica Letterbox Resize con Lanczos y Padding Negro.
    """
    img = img.convert("RGB")
    original_w, original_h = img.size
    target_w, target_h = target_size
    
    ratio = min(target_w / original_w, target_h / original_h)
    new_size = (int(original_w * ratio), int(original_h * ratio))
    
    if hasattr(Image, 'Resampling'):
        resample_method = Image.Resampling.LANCZOS
    else:
        resample_method = Image.LANCZOS
        
    img = img.resize(new_size, resample_method)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    
    paste_x = (target_w - new_size[0]) // 2
    paste_y = (target_h - new_size[1]) // 2
    
    new_img.paste(img, (paste_x, paste_y))
    return new_img

def calcular_area_superposicion(person_box, moto_box):
    """Calcula: (Área Intersección) / (Área Persona)"""
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = moto_box

    inter_x1 = max(px1, mx1)
    inter_y1 = max(py1, my1)
    inter_x2 = min(px2, mx2)
    inter_y2 = min(py2, my2)

    inter_width = inter_x2 - inter_x1
    inter_height = inter_y2 - inter_y1

    if inter_width <= 0 or inter_height <= 0:
        return 0.0

    intersection_area = inter_width * inter_height
    person_area = (px2 - px1) * (py2 - py1)

    if person_area == 0: return 0.0
        
    return intersection_area / person_area

def obtener_caja_union(moto_box, personas_en_esta_moto):
    """Calcula la caja que engloba a la moto Y a todos sus ocupantes."""
    ux1, uy1, ux2, uy2 = moto_box

    for p_box in personas_en_esta_moto:
        px1, py1, px2, py2 = p_box
        ux1 = min(ux1, px1)
        uy1 = min(uy1, py1)
        ux2 = max(ux2, px2)
        uy2 = max(uy2, py2)
    
    return int(ux1), int(uy1), int(ux2), int(uy2)


def load_detector():
    global detector
    if detector is None:
        print("Cargando YOLOv8x (Extra Large)...")
        detector = YOLO('yolov8x.pt') 

def load_classifier():
    global classifier
    if os.path.exists(MODEL_FILENAME):
        try:
            print(f"Cargando Clasificador: {MODEL_FILENAME}")
            classifier = YOLO(MODEL_FILENAME)
            return True
        except Exception as e:
            print(f"Error cargando clasificador: {e}")
            return False
    return False


# --- RUTAS FLASK ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    global classifier
    file = request.files.get('model')
    if not file or not file.filename.endswith('.pt'):
        return jsonify({"error": "Sube un archivo .pt válido"}), 400
    
    try:
        file.save(MODEL_FILENAME)
        classifier = YOLO(MODEL_FILENAME)
        return jsonify({"message": "Modelo actualizado."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    load_detector()
    if classifier is None:
        if not load_classifier():
            return jsonify({"error": "Sube tu modelo .pt primero"}), 400

    file = request.files.get('media')
    if not file: return jsonify({"error": "Falta archivo"}), 400

    # Guardar archivo original
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    output_filename = f"processed_{filename}"
    output_path = os.path.join(RESULTS_FOLDER, output_filename)

    try:
        # Leer y procesar imagen
        frame = cv2.imread(filepath)
        if frame is None:
            return jsonify({"error": "No se pudo leer la imagen"}), 400

        processed_frame, detections = process_single_frame(frame)
        
        # Guardar resultado
        cv2.imwrite(output_path, processed_frame)

        return jsonify({
            "output_url": f"/{RESULTS_FOLDER}/{output_filename}",
            "detections": detections
        })

    except Exception as e:
        print(e)
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

def process_single_frame(frame):
    detections_info = []
    
    # 1. DETECCIÓN (YOLOv8x) - Solo Personas(0) y Motos(3)
    results = detector(frame, classes=[0, 3], conf=0.15, verbose=False)
    
    if not results: return frame, []

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    
    person_boxes = []
    motorcycle_boxes = []

    for box, cls in zip(boxes, classes):
        if int(cls) == 0: 
            person_boxes.append(tuple(map(int, box)))
        elif int(cls) == 3: 
            motorcycle_boxes.append(tuple(map(int, box)))

    if not motorcycle_boxes: return frame, []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    min_overlap_ratio = 0.1

    # 2. PROCESAMIENTO
    for moto_box in motorcycle_boxes:
        riders_detected = []
        
        # Buscar ocupantes para esta moto
        for person_box in person_boxes:
            overlap = calcular_area_superposicion(person_box, moto_box)
            if overlap >= min_overlap_ratio:
                riders_detected.append(person_box)
        
        # Si tiene ocupantes, crear Unión
        if len(riders_detected) > 0:
            ux1, uy1, ux2, uy2 = obtener_caja_union(moto_box, riders_detected)
            
            # Limpiar coordenadas (evitar salir del margen)
            ux1, uy1 = max(0, ux1), max(0, uy1)
            ux2, uy2 = min(frame.shape[1], ux2), min(frame.shape[0], uy2)
            
            if ux2 <= ux1 or uy2 <= uy1: continue

            # A. RECORTE
            crop = pil_img.crop((ux1, uy1, ux2, uy2))
            
            # B. PREPROCESAMIENTO (Padding Negro)
            processed_crop = redimensionar_con_padding(crop, TARGET_SIZE)
            
            # C. CLASIFICACIÓN
            cls_results = classifier(processed_crop, verbose=False)
            top1 = int(cls_results[0].probs.top1)
            conf = float(cls_results[0].probs.top1conf)
            label = CLASS_NAMES[top1]
            
            # Generar Base64 para el frontend
            buffered = io.BytesIO()
            crop.save(buffered, format="JPEG") 
            b64_str = import_base64(buffered)
            
            detections_info.append({
                "label": label,
                "confidence": round(conf, 2),
                "image": b64_str
            })

            # D. DIBUJAR EN IMAGEN
            color = (0, 255, 0) if top1 == 1 else (0, 0, 255)
            # Caja
            cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), color, 3)
            # Etiqueta
            label_text = f"{label} ({conf:.2f})"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (ux1, uy1 - 25), (ux1 + w, uy1), color, -1)
            cv2.putText(frame, label_text, (ux1, uy1 - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, detections_info

def import_base64(buffered):
    import base64
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)