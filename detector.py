from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]

        frame_height = frame.shape[0]
        detections = []
        closest_tier = "none"

        for box in results.boxes:
            label = self.model.names[int(box.cls[0])]

            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            box_height = y2 - y1
            ratio = box_height / frame_height

            if ratio > 0.4:
                tier = "close"
            elif ratio > 0.2:
                tier = "medium"
            else:
                tier = "far"

            if tier == "close":
                closest_tier = "close"
            elif tier == "medium" and closest_tier != "close":
                closest_tier = "medium"

            detections.append({
                "label": label,
                "tier": tier,
                "box": (x1, y1, x2, y2)
            })

        return {
            "detections": detections,
            "closest_tier": closest_tier
        }