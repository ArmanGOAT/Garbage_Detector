import cv2
import numpy as np
import torch
import os
import time
from datetime import datetime


class GarbageDumpingDetection:
    def __init__(self,
                 model_path='best.pt',  # Путь к обученной модели YOLO
                 camera_id=0,
                 confidence_threshold=0.35,
                 action_threshold=0.5,
                 save_dir='detected_events'):
        """
        Инициализация системы обнаружения выброса мусора с YOLO

        Args:
            model_path: Путь к обученной модели YOLO
            camera_id: ID камеры или путь к видеофайлу
            confidence_threshold: Порог уверенности для детекции
            action_threshold: Порог для распознавания действия
            save_dir: Директория для сохранения обнаруженных событий
        """
        self.confidence_threshold = confidence_threshold
        self.action_threshold = action_threshold
        self.save_dir = save_dir
        self.model_path = model_path

        # Создаем директорию сохранения, если она не существует
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Инициализируем детектор движения для фильтрации кадров
        print("Инициализация детектора движения")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=30, detectShadows=False)

        # Загружаем модель YOLO
        self.load_yolo_model()

        # Инициализируем захват видео
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть камеру {camera_id}")

        # Устанавливаем меньшее разрешение для увеличения производительности
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Инициализируем трекер объектов
        self.tracker = cv2.legacy.TrackerMOSSE_create()  # Быстрый и легкий трекер

        # Буфер кадров для обнаружения действий
        self.frame_buffer = []
        self.buffer_size = 10

        # Состояние детекции
        self.tracking_object = False
        self.track_bbox = None
        self.garbage_detected = False
        self.person_detected = False
        self.event_frames = []
        self.last_detections = []

        # История движения для более стабильного обнаружения
        self.motion_history = None

        # Журнал инцидентов
        self.incident_log = []

        # Названия классов YOLO
        self.class_names = ["person", "garbage"]  # Должны соответствовать обученной модели

        print("Система обнаружения выброса мусора инициализирована")

    def load_yolo_model(self):
        """Загрузка модели YOLO"""
        try:
            if os.path.exists(self.model_path):
                print(f"Загрузка модели YOLO из {self.model_path}")
                # Используем модель YOLO
                self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                            path=self.model_path, force_reload=True)

                # Настраиваем для CPU
                self.model.cpu()
                # Отключаем вывод логов
                self.model.conf = self.confidence_threshold
                self.model.iou = 0.45
                print("Модель YOLO успешно загружена")
                self.yolo_available = True
            else:
                print(f"Модель {self.model_path} не найдена!")
                self.yolo_available = False
        except Exception as e:
            print(f"Ошибка загрузки модели YOLO: {e}")
            self.yolo_available = False

    def detect_objects(self, frame):
        """Обнаружение объектов с помощью YOLO"""
        detections = []
        self.person_detected = False
        self.garbage_detected = False

        # Обнаружение с помощью YOLO
        if self.yolo_available:
            try:
                # Инференс YOLO
                start_time = time.time()
                results = self.model(frame)
                inference_time = time.time() - start_time

                # Обработка результатов
                predictions = results.pandas().xyxy[0]

                for _, det in predictions.iterrows():
                    x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    conf = float(det['confidence'])
                    cls = det['class']
                    name = det['name']

                    if conf >= self.confidence_threshold:
                        if name == 'person':
                            detections.append({
                                'bbox': (x1, y1, x2 - x1, y2 - y1),
                                'class_id': cls,
                                'confidence': conf,
                                'type': 'person'
                            })
                            self.person_detected = True
                        elif name == 'garbage':
                            detections.append({
                                'bbox': (x1, y1, x2 - x1, y2 - y1),
                                'class_id': cls,
                                'confidence': conf,
                                'type': 'garbage'
                            })
                            self.garbage_detected = True

                # Если YOLO не обнаружил ничего, используем обнаружение движения
                if not detections:
                    motion_detections = self._detect_with_motion(frame)
                    detections.extend(motion_detections)
            except Exception as e:
                print(f"Ошибка при детекции YOLO: {e}")
                # Используем детекцию движения как запасной вариант
                detections = self._detect_with_motion(frame)
        else:
            # Если YOLO недоступен, используем только детекцию движения
            detections = self._detect_with_motion(frame)

        # Обновляем флаги обнаружения
        self.person_detected = any(d['type'] == 'person' for d in detections)
        self.garbage_detected = any(d['type'] == 'garbage' for d in detections)

        return detections

    def _detect_with_motion(self, frame):
        """Обнаружение с помощью детекции движения (резервный метод)"""
        detections = []

        try:
            # Преобразуем кадр в оттенки серого и размываем его
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (11, 11), 0)

            # Применяем вычитание фона
            fg_mask = self.bg_subtractor.apply(gray)

            # Добавляем временную согласованность с помощью истории движения
            if self.motion_history is None:
                self.motion_history = np.zeros_like(fg_mask)

            # Обновляем историю движения (затухание старого движения и добавление нового)
            self.motion_history = np.maximum(self.motion_history * 0.8, fg_mask)

            # Применяем порог для получения бинарной маски
            _, thresh = cv2.threshold(self.motion_history, 50, 255, cv2.THRESH_BINARY)

            # Применяем морфологические операции для улучшения областей движения
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Находим контуры на маске
            contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Обрабатываем контуры
            for contour in contours:
                # Фильтруем по площади контура
                area = cv2.contourArea(contour)
                if area > 300:  # Нижний порог для обнаружения
                    x, y, w, h = cv2.boundingRect(contour)

                    # Используем соотношение сторон и площадь для классификации
                    aspect_ratio = w / float(h) if h > 0 else 0

                    # Эвристика для человека и мусора
                    if area > 5000 and 0.3 < aspect_ratio < 1.0:
                        detections.append({
                            'bbox': (x, y, w, h),
                            'class_id': 0,  # Человек
                            'confidence': min(0.6, area / 20000),  # Масштабируем уверенность по площади
                            'type': 'person'
                        })
                    elif 300 < area < 8000:
                        detections.append({
                            'bbox': (x, y, w, h),
                            'class_id': 1,  # Мусор
                            'confidence': min(0.5, area / 5000),
                            'type': 'garbage'
                        })
        except Exception as e:
            print(f"Ошибка детекции движения: {e}")

        return detections

    def recognize_action(self):
        """Распознавание действия выброса мусора из буфера кадров"""
        if len(self.frame_buffer) < self.buffer_size:
            return False, 0.0

        try:
            # Проверяем пространственные отношения между человеком и мусором
            if self.person_detected and self.garbage_detected and self.last_detections:
                person_dets = [d for d in self.last_detections if d['type'] == 'person']
                garbage_dets = [d for d in self.last_detections if d['type'] == 'garbage']

                if person_dets and garbage_dets:
                    # Вычисляем оценку близости
                    proximity_score = 0.0
                    person_bbox = person_dets[0]['bbox']

                    # Центр человека
                    px = person_bbox[0] + person_bbox[2] / 2
                    py = person_bbox[1] + person_bbox[3] / 2

                    for garbage in garbage_dets:
                        garbage_bbox = garbage['bbox']
                        # Центр мусора
                        gx = garbage_bbox[0] + garbage_bbox[2] / 2
                        gy = garbage_bbox[1] + garbage_bbox[3] / 2

                        # Вычисляем расстояние
                        distance = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)

                        # Нормализуем по диагонали
                        frame_diagonal = np.sqrt(640 ** 2 + 480 ** 2)
                        norm_distance = distance / frame_diagonal

                        # Обратное отношение - более близкие объекты имеют более высокий балл
                        current_score = max(0, 1 - norm_distance * 2)
                        proximity_score = max(proximity_score, current_score)

                    # Проверяем движение человека
                    person_movement = self._calculate_movement(person_dets)

                    # Итоговая оценка сочетает близость и движение
                    dumping_score = 0.7 * proximity_score + 0.3 * person_movement

                    return dumping_score > self.action_threshold, dumping_score

            return False, 0.0

        except Exception as e:
            print(f"Ошибка в распознавании действия: {e}")
            return False, 0.0

    def _calculate_movement(self, person_detections):
        """Вычисление величины движения для обнаруженного человека"""
        if len(self.frame_buffer) < 3 or not person_detections:
            return 0.0

        try:
            # Используем маску движения для определения движения в области человека
            current_person = person_detections[0]['bbox']
            x, y, w, h = current_person

            # Получаем движение в области человека
            if self.motion_history is not None:
                person_roi = self.motion_history[y:y + h, x:x + w]
                if person_roi.size > 0:
                    motion_percentage = np.sum(person_roi > 50) / person_roi.size
                    return min(1.0, motion_percentage * 3)  # Масштабируем для лучшей чувствительности

            return 0.0
        except Exception as e:
            print(f"Ошибка вычисления движения: {e}")
            return 0.0

    def save_event(self):
        """Сохранение обнаруженного события выброса мусора"""
        if not self.event_frames:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            event_dir = os.path.join(self.save_dir, f"event_{timestamp}")
            os.makedirs(event_dir, exist_ok=True)

            # Сохраняем кадры как изображения
            for i, frame in enumerate(self.event_frames):
                cv2.imwrite(os.path.join(event_dir, f"frame_{i:03d}.jpg"), frame)

            # Создаем видео из кадров
            height, width = self.event_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = os.path.join(event_dir, "event.avi")
            video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))

            for frame in self.event_frames:
                video.write(frame)

            video.release()

            # Логируем инцидент
            self.incident_log.append({
                'timestamp': timestamp,
                'location': 'Camera 1',
                'video_path': video_path
            })

            print(f"Событие сохранено в {event_dir}")

            # Сбрасываем кадры события
            self.event_frames = []
        except Exception as e:
            print(f"Ошибка сохранения события: {e}")
            self.event_frames = []

    def annotate_frame(self, frame, detections, is_dumping, dumping_score):
        """Добавление визуальных аннотаций к кадру"""
        annotated = frame.copy()

        # Рисуем рамки обнаружения
        for det in detections:
            x, y, w, h = det['bbox']
            color = (0, 255, 0) if det['type'] == 'person' else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"{det['type']} {det['confidence']:.2f}"
            cv2.putText(annotated, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Рисуем рамку отслеживания, если активна
        if self.tracking_object and self.track_bbox is not None:
            x, y, w, h = [int(v) for v in self.track_bbox]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(annotated, "Tracking", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Показываем результат распознавания действия
        if is_dumping:
            cv2.putText(annotated, f"ВЫБРОС МУСОРА! ({dumping_score:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(annotated, f"Нет выброса ({dumping_score:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Добавляем временную метку
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, timestamp, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Добавляем информацию о FPS
        fps = 1.0 / (time.time() - self.last_time) if hasattr(self, 'last_time') else 0
        self.last_time = time.time()
        cv2.putText(annotated, f"FPS: {fps:.1f}", (annotated.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def run(self):
        """Запуск системы обнаружения"""
        print("Запуск обнаружения...")
        event_in_progress = False
        frames_since_detection = 0
        max_frames_to_keep = 30  # Храним ~1 секунду при 30 fps
        skip_frames = 1  # Обрабатываем каждый n-й кадр для лучшей производительности
        current_frame = 0
        self.last_time = time.time()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Не удалось захватить кадр. Выход...")
                    break

                current_frame += 1

                # Пропускаем кадры для улучшения производительности
                if current_frame % skip_frames != 0:
                    continue

                # Добавляем в буфер кадров
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) > self.buffer_size * 2:
                    self.frame_buffer.pop(0)

                # Обновляем трекер, если активен
                if self.tracking_object and self.track_bbox is not None:
                    success, self.track_bbox = self.tracker.update(frame)
                    if not success:
                        self.tracking_object = False

                # Обнаруживаем объекты
                if not self.tracking_object or frames_since_detection % 5 == 0:
                    detections = self.detect_objects(frame)
                    self.last_detections = detections
                    frames_since_detection = 0

                    # Начинаем отслеживание человека, если обнаружен
                    if self.person_detected and not self.tracking_object:
                        person_dets = [d for d in detections if d['type'] == 'person']
                        if person_dets:
                            largest_person = max(person_dets, key=lambda x: x['bbox'][2] * x['bbox'][3])
                            self.track_bbox = largest_person['bbox']
                            self.tracker = cv2.legacy.TrackerMOSSE_create()  # Более быстрый трекер
                            self.tracker.init(frame, tuple(self.track_bbox))
                            self.tracking_object = True
                else:
                    detections = []
                    frames_since_detection += 1

                # Распознаем действие
                is_dumping, dumping_score = self.recognize_action()

                # Обрабатываем обнаружение события
                if is_dumping and not event_in_progress:
                    event_in_progress = True
                    self.event_frames.extend(self.frame_buffer[-min(len(self.frame_buffer), max_frames_to_keep):])
                    print("Обнаружен возможный выброс мусора!")

                if event_in_progress:
                    self.event_frames.append(frame)

                    if not is_dumping:
                        frames_since_detection += 1
                        if frames_since_detection > 15:  # Более короткий период для отзывчивости
                            print("Событие завершено")
                            self.save_event()
                            event_in_progress = False
                            frames_since_detection = 0

                # Аннотируем и отображаем кадр
                annotated_frame = self.annotate_frame(frame, detections, is_dumping, dumping_score)
                cv2.imshow('Обнаружение выброса мусора', annotated_frame)

                # Проверяем ввод пользователя
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Ручное сохранение события
                    print("Ручное сохранение текущего события")
                    if not self.event_frames:
                        self.event_frames = self.frame_buffer.copy()
                    self.save_event()
                    event_in_progress = False

            # Освобождаем ресурсы
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Ошибка в основном цикле обнаружения: {e}")
            self.cap.release()
            cv2.destroyAllWindows()


def run_detection_system():
    try:
        detector = GarbageDumpingDetection(
            model_path='best.pt',  # Путь к обученной модели YOLO
            camera_id=0,
            confidence_threshold=0.35,
            action_threshold=0.5,
            save_dir='detected_dumping_events'
        )

        detector.run()
    except Exception as e:
        print(f"Фатальная ошибка в системе обнаружения: {e}")
        import traceback
        traceback.print_exc()

    import requests

    API_URL = "http://127.0.0.1:8000/api/alerts/"

    def send_detection_alert(category, confidence, image_path):
        data = {"category": category, "confidence": confidence}
        files = {"image": open(image_path, "rb")} if image_path else None
        response = requests.post(API_URL, data=data, files=files)
        print(response.json())

    def detect_objects(self, frame):
        detections = []
        self.person_detected = False
        self.garbage_detected = False

        if self.yolo_available:
            try:
                results = self.model(frame)
                predictions = results.pandas().xyxy[0]

                for _, det in predictions.iterrows():
                    x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    conf = float(det['confidence'])
                    cls = det['class']
                    name = det['name']

                    if conf >= self.confidence_threshold:
                        detections.append({
                            'bbox': (x1, y1, x2 - x1, y2 - y1),
                            'class_id': cls,
                            'confidence': conf,
                            'type': name
                        })

                for detection in detections:
                    send_detection_alert(detection["type"], detection["confidence"], "detected_frame.jpg")

            except Exception as e:
                print(f"Ошибка при детекции YOLO: {e}")

        return detections


if __name__ == "__main__":
    # Необходимые пакеты:
    # pip install torch torchvision opencv-python numpy

    # Запуск системы
    run_detection_system()
