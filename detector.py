from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    frame_height = frame.shape[0]
    closest_tier = "none"

    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0]
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

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"{label} {tier}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    print(closest_tier)

    cv2.imshow("detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()