import cv2
from detector import ObjectDetector


detector = ObjectDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = detector.detect(frame)

    for detection in result["detections"]:
        label = detection["label"]
        tier = detection["tier"]
        x1, y1, x2, y2 = detection["box"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{label} {tier}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    print("Closest:", result["closest_tier"])

    cv2.imshow("SafeWalk", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if cv2.getWindowProperty("SafeWalk", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()