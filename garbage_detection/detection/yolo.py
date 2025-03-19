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
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'class_id': cls,
                        'confidence': conf,
                        'type': name
                    })

            for detection in detections:
                send_detection_alert(detection["type"], detection["confidence"], "detected_frame.jpg")

        except Exception as e:
            print(f"Ошибка при детекции YOLO: {e}")

    return detections