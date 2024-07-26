import cv2
import numpy as np
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO('lastM.pt')

# Загрузка видео
video_path = 'fish.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Параметры зоны (координаты вершин многоугольника)
zone_points = np.array([(650, 1000), (205, 500), (900, 150), (1800, 300), (1450, 1000)])

# Функция для проверки нахождения bbox в зоне
def is_in_zone(bbox, zone_points):
    x1, y1, x2, y2 = bbox
    bbox_points = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    intersection_points = cv2.intersectConvexConvex(np.float32(zone_points), np.float32(bbox_points))[1]
    if intersection_points is not None:
        intersection_area = cv2.contourArea(intersection_points)
        bbox_area = (x2 - x1) * (y2 - y1)
        if intersection_area / bbox_area > 0.5:
            return True
    return False

# Параметры для подсчета времени
frames_per_second_real = fps / 6
seconds_working = 0
seconds_absent = 0
no_objects_start_frame = None

# Выходное видео
output_path = 'output_fish.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Переменная для хранения предыдущего кадра
prev_frame_zone = None

# Переменные для отслеживания состояния клиентов
motion_value_history = []
client_status = "No"
client_status_changed = False

# Обработка видео
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция объектов с порогом уверенности 0.3
    results = model(frame, conf=0.3)

    objects_in_zone = 0

    if results[0].boxes is not None:
        bboxes = results[0].boxes.xyxy.cpu().numpy()  # Получение bbox
        confidences = results[0].boxes.conf.cpu().numpy()  # Получение уверенности детекций

        for bbox, conf in zip(bboxes, confidences):
            x1, y1, x2, y2 = bbox[:4]
            if is_in_zone((x1, y1, x2, y2), zone_points):
                objects_in_zone += 1
                # Отображение bbox и точности
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2 + 10)), (0, 255, 0), 2)  # Увеличение прямоугольника вниз на 10 пикселей
                cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение зоны
    if objects_in_zone > 0:
        cv2.polylines(frame, [zone_points], isClosed=True, color=(0, 255, 0), thickness=3)
    else:
        cv2.polylines(frame, [zone_points], isClosed=True, color=(0, 0, 255), thickness=3)

    # Детекция движения
    # Создаем маску зоны
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [zone_points], (255, 255, 255))
    frame_zone = cv2.bitwise_and(frame, mask)

    if prev_frame_zone is not None:
        # Вычисляем разницу между текущим и предыдущим кадрами в зоне
        diff = cv2.absdiff(frame_zone, prev_frame_zone)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_thresh = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        motion_value = np.sum(diff_thresh) / 255
    else:
        motion_value = 0

    # Обновляем предыдущий кадр
    prev_frame_zone = frame_zone.copy()

    # Обновляем историю значений motion_value
    motion_value_history.append(motion_value)
    if len(motion_value_history) > 10:
        motion_value_history.pop(0)

    # Определяем статус клиента
    if len(motion_value_history) == 10:
        if all(value > 100 for value in motion_value_history):
            if client_status != "Yes":
                client_status_changed = True
            client_status = "Yes"
        elif all(value <= 100 for value in motion_value_history):
            if client_status != "No":
                client_status_changed = True
            client_status = "No"

    # Обновляем время работы и отсутствия
    if client_status == "Yes":
        if objects_in_zone > 0:
            seconds_working += 1 / frames_per_second_real
        else:
            seconds_absent += 1 / frames_per_second_real

    # Отображение таймеров и значений
    # Белая зона для таймеров
    cv2.rectangle(frame, (5, 10), (505, 210), (255, 255, 255), -1)

    # Форматирование времени
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f'{hours:02}:{minutes:02}:{secs:02}'

    # Общее время, когда объекты находятся в зоне
    cv2.putText(frame, f'Working: {format_time(seconds_working)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Общее время отсутствия объектов (Absent)
    cv2.putText(frame, f'Absent: {format_time(seconds_absent)}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Отображение статуса клиентов
    cv2.putText(frame, f'Clients: {client_status}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Запись кадра в выходное видео
    out.write(frame)
    frame_idx += 1

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()
