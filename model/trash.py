import cv2
import time
from ultralytics import YOLO
import numpy as np

# Загрузка модели
model = YOLO('yolov8n.pt')
print("Модель успешно загружена")

# Открытие камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: Невозможно подключиться к камере")
    exit()

# Получение размеров кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Размер кадра: {width}x{height}")

# Явно задаем размер окна
cv2.namedWindow('Обнаружение объектов', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Обнаружение объектов', 1280, 720)  # Большое окно

# Считываем тестовый кадр
ret, test_frame = cap.read()
if ret:
    print(f"Тестовый кадр успешно получен, размер: {test_frame.shape}")
    # Сохраняем тестовый кадр
    cv2.imwrite('test_frame.jpg', test_frame)
    print("Тестовый кадр сохранен в файл test_frame.jpg")
else:
    print("Не удалось получить тестовый кадр с камеры!")

# Основной цикл
print("Начало основного цикла обработки...")
frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка получения кадра")
            break
            
        frame_count += 1
        
        # Проверка размера кадра
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            print(f"Ошибка: некорректный размер кадра {frame.shape}")
            continue
            
        # Детекция объектов
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Информация на кадре
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        text = f"FPS: {fps:.1f}"
        cv2.putText(annotated_frame, text, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Отображаем кадр
        cv2.imshow('Обнаружение объектов', annotated_frame)
        
        # Каждые 5 секунд сохраняем кадр (для отладки)
        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
            cv2.imwrite(f'frame_{int(elapsed)}.jpg', annotated_frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC или q для выхода
            print("Выход по команде пользователя")
            break
            
except Exception as e:
    print(f"Ошибка: {e}")
finally:
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
    print(f"Обработано {frame_count} кадров за {time.time()-start_time:.1f} секунд")