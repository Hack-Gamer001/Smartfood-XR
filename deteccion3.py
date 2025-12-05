import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
from collections import defaultdict, deque
import tkinter as tk

class AdaptiveFruitVegetableTracker:
    def __init__(self, model_path=None):
        print("🍎 Inicializando Detector Adaptable de Frutas y Verduras...")

        if model_path is None:
            model_path = 'yolov8n.pt'

        self.model = YOLO(model_path)
        print("✅ Modelo cargado exitosamente!")

        # Obtener dimensiones de pantalla
        self.get_screen_dimensions()

        # Diccionario expandido con información nutricional
        self.produce_items = {
            'apple': {
                'color': (0, 0, 255),
                'type': 'fruta',
                'name': 'Manzana',
                'nutrition': {
                    'calorias': '52 kcal/100g',
                    'vitaminas': 'Vitamina C (4.6mg), Vitamina K (2.2μg)',
                    'minerales': 'Potasio (107mg), Calcio (6mg)',
                    'fibra': '2.4g/100g',
                    'carbohidratos': '13.8g/100g',
                    'proteinas': '0.3g/100g',
                    'beneficios': 'Antioxidante, Digestivo, Anti-inflamatorio'
                }
            },
            'banana': {
                'color': (0, 255, 255),
                'type': 'fruta',
                'name': 'Plátano',
                'nutrition': {
                    'calorias': '89 kcal/100g',
                    'vitaminas': 'Vitamina B6 (0.4mg), Vitamina C (8.7mg)',
                    'minerales': 'Potasio (358mg), Magnesio (27mg)',
                    'fibra': '2.6g/100g',
                    'carbohidratos': '22.8g/100g',
                    'proteinas': '1.1g/100g',
                    'beneficios': 'Energético, Cardiovascular, Digestivo'
                }
            },
            'orange': {
                'color': (0, 165, 255),
                'type': 'fruta',
                'name': 'Naranja',
                'nutrition': {
                    'calorias': '47 kcal/100g',
                    'vitaminas': 'Vitamina C (53.2mg), Folato (40μg)',
                    'minerales': 'Calcio (40mg), Potasio (181mg)',
                    'fibra': '2.4g/100g',
                    'carbohidratos': '11.8g/100g',
                    'proteinas': '0.9g/100g',
                    'beneficios': 'Inmunológico, Antioxidante, Digestivo'
                }
            },
            'broccoli': {
                'color': (0, 255, 0),
                'type': 'verdura',
                'name': 'Brócoli',
                'nutrition': {
                    'calorias': '34 kcal/100g',
                    'vitaminas': 'Vitamina C (89.2mg), Vitamina K (101.6μg)',
                    'minerales': 'Calcio (47mg), Hierro (0.7mg)',
                    'fibra': '2.6g/100g',
                    'carbohidratos': '6.6g/100g',
                    'proteinas': '2.8g/100g',
                    'beneficios': 'Anticancerígeno, Detox, Inmunológico'
                }
            },
            'carrot': {
                'color': (0, 165, 255),
                'type': 'verdura',
                'name': 'Zanahoria',
                'nutrition': {
                    'calorias': '41 kcal/100g',
                    'vitaminas': 'Vitamina A (835μg), Vitamina K (13.2μg)',
                    'minerales': 'Potasio (320mg), Calcio (33mg)',
                    'fibra': '2.8g/100g',
                    'carbohidratos': '9.6g/100g',
                    'proteinas': '0.9g/100g',
                    'beneficios': 'Visión, Antioxidante, Piel saludable'
                }
            },
            'bottle': {'color': (100, 255, 100), 'type': 'objeto', 'name': 'Botella'},
            'cup': {'color': (150, 255, 150), 'type': 'objeto', 'name': 'Vaso'},
            'bowl': {'color': (200, 255, 200), 'type': 'objeto', 'name': 'Bowl'},
        }

        # Sistema de seguimiento persistente
        self.tracked_objects = {}
        self.next_id = 1
        self.max_disappeared = 30

        # Distancia máxima adaptable según resolución
        self.max_distance = min(self.screen_width, self.screen_height) * 0.08

        # Historial de posiciones para suavizado
        self.position_history = defaultdict(lambda: deque(maxlen=5))

        # Configuración
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

        # Estadísticas
        self.detection_stats = defaultdict(int)
        self.tracking_stats = {
            'total_tracked': 0,
            'currently_tracking': 0,
            'avg_tracking_time': 0
        }

        # Configuración adaptable de interfaz
        self.setup_adaptive_ui()

    def get_screen_dimensions(self):
        try:
            root = tk.Tk()
            self.screen_width = root.winfo_screenwidth()
            self.screen_height = root.winfo_screenheight()
            root.destroy()
            print(f"📐 Dimensiones de pantalla: {self.screen_width}x{self.screen_height}")
        except:
            self.screen_width = 1920
            self.screen_height = 1080
            print("📐 Usando dimensiones por defecto: 1920x1080")

    def setup_adaptive_ui(self):
        base_resolution = 1920
        self.scale_factor = min(self.screen_width / base_resolution, 1.5)

        self.font_scale_small = max(0.4, 0.5 * self.scale_factor)
        self.font_scale_medium = max(0.6, 0.7 * self.scale_factor)
        self.font_scale_large = max(0.8, 1.0 * self.scale_factor)

        self.thickness_thin = max(1, int(2 * self.scale_factor))
        self.thickness_medium = max(2, int(3 * self.scale_factor))
        self.thickness_thick = max(3, int(4 * self.scale_factor))

        self.info_panel_width = int(450 * self.scale_factor)
        self.stats_panel_width = int(350 * self.scale_factor)
        self.panel_margin = int(10 * self.scale_factor)

        print(f"🎨 Factor de escala: {self.scale_factor:.2f}")

    def is_produce_item(self, class_name):
        return class_name.lower() in self.produce_items

    def calculate_distance(self, center1, center2):
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def smooth_position(self, object_id, new_center):
        self.position_history[object_id].append(new_center)
        positions = list(self.position_history[object_id])

        if len(positions) == 1:
            return new_center

        weights = [i + 1 for i in range(len(positions))]
        total_weight = sum(weights)

        smooth_x = sum(pos[0] * weight for pos, weight in zip(positions, weights)) / total_weight
        smooth_y = sum(pos[1] * weight for pos, weight in zip(positions, weights)) / total_weight

        return (int(smooth_x), int(smooth_y))

    def update_tracking(self, detections):
        current_detections = []

        for det in detections:
            center = ((det['bbox'][0] + det['bbox'][2]) // 2,
                     (det['bbox'][1] + det['bbox'][3]) // 2)
            current_detections.append({
                'center': center,
                'bbox': det['bbox'],
                'class_name': det['name'],
                'confidence': det['confidence'],
                'area': det['area']
            })

        used_detections = set()
        updated_objects = set()

        for obj_id, obj_data in list(self.tracked_objects.items()):
            best_match = None
            best_distance = float('inf')
            best_idx = -1

            for idx, detection in enumerate(current_detections):
                if idx in used_detections:
                    continue

                if detection['class_name'] == obj_data['class_name']:
                    distance = self.calculate_distance(obj_data['center'], detection['center'])

                    if distance < self.max_distance and distance < best_distance:
                        best_match = detection
                        best_distance = distance
                        best_idx = idx

            if best_match:
                smooth_center = self.smooth_position(obj_id, best_match['center'])

                self.tracked_objects[obj_id].update({
                    'center': smooth_center,
                    'bbox': best_match['bbox'],
                    'confidence': best_match['confidence'],
                    'area': best_match['area'],
                    'disappeared_count': 0,
                    'last_seen': time.time(),
                    'tracking_duration': time.time() - obj_data['first_seen']
                })

                used_detections.add(best_idx)
                updated_objects.add(obj_id)
            else:
                self.tracked_objects[obj_id]['disappeared_count'] += 1

        objects_to_remove = []
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_data['disappeared_count'] > self.max_disappeared:
                objects_to_remove.append(obj_id)

        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
            if obj_id in self.position_history:
                del self.position_history[obj_id]

        for idx, detection in enumerate(current_detections):
            if idx not in used_detections:
                obj_id = self.next_id
                self.next_id += 1

                smooth_center = self.smooth_position(obj_id, detection['center'])

                self.tracked_objects[obj_id] = {
                    'id': obj_id,
                    'center': smooth_center,
                    'bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'area': detection['area'],
                    'disappeared_count': 0,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'tracking_duration': 0,
                    'color': self.produce_items[detection['class_name']]['color']
                }

                self.detection_stats[detection['class_name']] += 1
                self.tracking_stats['total_tracked'] += 1

        self.tracking_stats['currently_tracking'] = len(self.tracked_objects)

    def draw_nutritional_info(self, frame, obj_data, frame_width, frame_height):
        class_name = obj_data['class_name']
        item_info = self.produce_items.get(class_name, {})

        if 'nutrition' not in item_info:
            return frame

        nutrition = item_info['nutrition']
        x1, y1, x2, y2 = obj_data['bbox']

        panel_width = self.info_panel_width
        panel_padding = int(15 * self.scale_factor)
        line_height = int(22 * self.scale_factor)

        info_lines = [
            f"{item_info['name'].upper()} - ID:{obj_data['id']}",
            f"Calorías: {nutrition['calorias']}",
            f"Vitaminas: {nutrition['vitaminas']}",
            f"Minerales: {nutrition['minerales']}",
            f"Fibra: {nutrition['fibra']}",
            f"Carbohidratos: {nutrition['carbohidratos']}",
            f"Proteínas: {nutrition['proteinas']}",
            f"Beneficios: {nutrition['beneficios']}"
        ]

        max_width = 0
        for line in info_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                      self.font_scale_small, self.thickness_thin)[0]
            max_width = max(max_width, text_size[0])

        panel_width = min(panel_width, max_width + panel_padding * 2)
        panel_height = len(info_lines) * line_height + panel_padding * 2

        panel_x = x2 + self.panel_margin
        if panel_x + panel_width > frame_width:
            panel_x = max(self.panel_margin, x1 - panel_width - self.panel_margin)

        panel_y = y1
        if panel_y + panel_height > frame_height:
            panel_y = frame_height - panel_height - self.panel_margin
        panel_y = max(self.panel_margin, panel_y)

        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        cv2.rectangle(frame,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     obj_data['color'], self.thickness_medium)

        for i, line in enumerate(info_lines):
            y_pos = panel_y + panel_padding + (i + 1) * line_height

            if i == 0:
                color = obj_data['color']
                font_scale = self.font_scale_medium
                thickness = self.thickness_medium
            elif 'Calorías' in line:
                color = (0, 255, 255)
                font_scale = self.font_scale_small
                thickness = self.thickness_thin
            elif 'Vitaminas' in line or 'Minerales' in line:
                color = (255, 255, 0)
                font_scale = self.font_scale_small
                thickness = self.thickness_thin
            elif 'Beneficios' in line:
                color = (255, 0, 255)
                font_scale = self.font_scale_small
                thickness = self.thickness_thin
            else:
                color = (255, 255, 255)
                font_scale = self.font_scale_small
                thickness = self.thickness_thin

            cv2.putText(frame, line, (panel_x + panel_padding, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        return frame

    def draw_persistent_tracking(self, frame):
        height, width = frame.shape[:2]

        for obj_id, obj_data in self.tracked_objects.items():
            x1, y1, x2, y2 = obj_data['bbox']
            center = obj_data['center']
            color = obj_data['color']
            class_name = obj_data['class_name']
            confidence = obj_data['confidence']

            is_active = obj_data['disappeared_count'] == 0

            if is_active:
                thickness = max(self.thickness_medium, int(confidence * 5 * self.scale_factor))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                corner_length = int(25 * self.scale_factor)
                corner_thickness = int(4 * self.scale_factor)
                pulse = int(10 * math.sin(time.time() * 3)) + 15
                corner_length += pulse // 3

                cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
                cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
                cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
                cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)

                ring_radius = int((15 + 5 * math.sin(time.time() * 4)) * self.scale_factor)
                center_radius = int(6 * self.scale_factor)
                cv2.circle(frame, center, ring_radius, color, self.thickness_thin)
                cv2.circle(frame, center, center_radius, color, -1)
                cv2.circle(frame, center, center_radius + 2, (255, 255, 255), self.thickness_thin)

                frame = self.draw_nutritional_info(frame, obj_data, width, height)
            else:
                self.draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color,
                                         self.thickness_thin, int(10 * self.scale_factor))
                cv2.circle(frame, center, int(5 * self.scale_factor), color, self.thickness_thin)

        return frame

    def draw_dashed_rectangle(self, frame, pt1, pt2, color, thickness, dash_length):
        x1, y1 = pt1
        x2, y2 = pt2

        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)

        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)

    def draw_enhanced_statistics(self, frame):
        height, width = frame.shape[:2]

        panel_width = self.stats_panel_width
        panel_height = min(int(300 * self.scale_factor), height - 100)
        panel_x = width - panel_width - self.panel_margin
        panel_y = 50

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 255, 0), self.thickness_medium)

        cv2.putText(frame, "INFORMACIÓN NUTRICIONAL",
                   (panel_x + self.panel_margin, panel_y + int(25 * self.scale_factor)),
                   cv2.FONT_HERSHEY_DUPLEX, self.font_scale_medium, (0, 255, 0), self.thickness_medium)

        y_offset = int(50 * self.scale_factor)
        line_height = int(20 * self.scale_factor)

        stats_info = [
            f"Objetos Activos: {self.tracking_stats['currently_tracking']}",
            f"Total Detectados: {self.tracking_stats['total_tracked']}",
            "",
            "DETECCIONES:"
        ]

        for info in stats_info:
            if info == "":
                y_offset += int(10 * self.scale_factor)
                continue
            color = (255, 255, 255) if not info.startswith("DETECCIONES") else (255, 255, 0)
            cv2.putText(frame, info, (panel_x + self.panel_margin, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small, color, self.thickness_thin)
            y_offset += line_height

        for item, count in sorted(self.detection_stats.items(), key=lambda x: x[1], reverse=True):
            if y_offset > panel_height - int(30 * self.scale_factor):
                break

            item_info = self.produce_items.get(item, {'name': item.upper(), 'color': (255, 255, 255)})
            text = f"  {item_info['name']}: {count}"
            cv2.putText(frame, text, (panel_x + self.panel_margin, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small, item_info['color'], self.thickness_thin)
            y_offset += int(18 * self.scale_factor)

        return frame

    def process_detections(self, results):
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]

                    if (confidence > self.confidence_threshold and
                        self.is_produce_item(class_name)):

                        area = (x2 - x1) * (y2 - y1)
                        detections.append({
                            'name': class_name,
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2),
                            'area': area
                        })

        return detections

    def run_adaptive_detection(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("❌ Error: No se pudo abrir la cámara")
            return

        optimal_width = min(1280, int(self.screen_width * 0.7))
        optimal_height = min(720, int(self.screen_height * 0.7))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, optimal_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, optimal_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("🎥 Detector Adaptativo con Información Nutricional iniciado")
        print(f"📺 Resolución optimizada: {optimal_width}x{optimal_height}")
        print("📋 Detecta: Manzanas, Plátanos, Naranjas, Brócoli, Zanahorias")
        print("⌨️ Controles:")
        print("   'q' - Salir")
        print("   's' - Guardar captura")
        print("   'r' - Reiniciar seguimiento")
        print("   'i' - Alternar panel de estadísticas")
        print("   'n' - Alternar información nutricional")

        frame_count = 0
        start_time = time.time()
        show_stats = True
        show_nutrition = True

        cv2.namedWindow('Detector Nutricional Adaptativo', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detector Nutricional Adaptativo', optimal_width, optimal_height)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error leyendo cámara")
                break

            results = self.model(frame,
                               conf=self.confidence_threshold,
                               iou=self.nms_threshold,
                               verbose=False)

            detections = self.process_detections(results)
            self.update_tracking(detections)

            if show_nutrition:
                frame_with_tracking = self.draw_persistent_tracking(frame)
            else:
                frame_with_tracking = frame.copy()
                for obj_data in self.tracked_objects.values():
                    if obj_data['disappeared_count'] == 0:
                        x1, y1, x2, y2 = obj_data['bbox']
                        cv2.rectangle(frame_with_tracking, (x1, y1), (x2, y2),
                                    obj_data['color'], self.thickness_medium)

            if show_stats:
                frame_with_tracking = self.draw_enhanced_statistics(frame_with_tracking)

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

                fps_panel_width = int(300 * self.scale_factor)
                fps_panel_height = int(40 * self.scale_factor)

                cv2.rectangle(frame_with_tracking, (self.panel_margin, self.panel_margin),
                             (fps_panel_width, fps_panel_height), (0, 0, 0), -1)
                cv2.rectangle(frame_with_tracking, (self.panel_margin, self.panel_margin),
                             (fps_panel_width, fps_panel_height), (0, 255, 0), 2)
                cv2.putText(frame_with_tracking, f"FPS: {fps:.1f} | Tracking: {len(self.tracked_objects)}",
                           (self.panel_margin + 5, self.panel_margin + 25),
                           cv2.FONT_HERSHEY_DUPLEX, self.font_scale_small, (0, 255, 0), self.thickness_thin)

            cv2.imshow('Detector Nutricional Adaptativo', frame_with_tracking)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"tracking_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_tracking)
                print(f"📸 Captura guardada: {filename}")
            elif key == ord('r'):
                self.tracked_objects.clear()
                self.position_history.clear()
                self.next_id = 1
                print("🔄 Sistema de seguimiento reiniciado")
            elif key == ord('i'):
                show_stats = not show_stats
                print(f"ℹ️ Panel de estadísticas: {'ON' if show_stats else 'OFF'}")
            elif key == ord('n'):
                show_nutrition = not show_nutrition
                print(f"ℹ️ Información nutricional: {'ON' if show_nutrition else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()

def main():
    print("=" * 70)
    print("🍎🥕 DETECTOR ADAPTATIVO DE FRUTAS Y VERDURAS CON INFORMACIÓN NUTRICIONAL 🥬🍌")
    print("=" * 70)
    print("• Seguimiento continuo y persistente de objetos")
    print("• Los recuadros se mantienen mientras el objeto esté visible")
    print("• Seguimiento suave con predicción de movimiento")
    print("• Identificación única para cada objeto detectado")
    print("-" * 70)

    try:
        tracker = AdaptiveFruitVegetableTracker()
        tracker.run_adaptive_detection(camera_index=0)

    except KeyboardInterrupt:
        print("\n👋 Detector cerrado por el usuario")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Verifica:")
        print("• Cámara conectada y funcionando")
        print("• Librerías instaladas correctamente")
        print("• Permisos de cámara habilitados")

if __name__ == "__main__":
    main()
