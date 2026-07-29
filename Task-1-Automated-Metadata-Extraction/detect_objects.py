from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(image_path):
    results = model(image_path)

    objects = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            if name not in objects:
                objects.append(name)

    return objects