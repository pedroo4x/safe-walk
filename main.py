import cv2
from detector import ObjectDetector


detector = ObjectDetector()
self.depth_model = YOLO("yolo26n-depth.pt")

cap = cv2.VideoCapture(0)

window_name = "SafeWalk"

cv2.namedWindow(window_name)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    result = detector.detect(frame)

    # Draw detections
    for detection in result["detections"]:

        x1, y1, x2, y2 = detection["box"]

        label = detection["label"]
        track_id = detection["id"]

        proximity = detection["proximity"]
        status = detection["status"]
        direction = detection["direction"]

        movement = detection["movement"]

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display information
        text = (
            f"{label} "
            f"ID:{track_id} "
            f"{proximity} "
            f"{status}"
            f"{direction}"
        )

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # Draw center point
        center_x, center_y = detection["center"]

        cv2.circle(
            frame,
            (int(center_x), int(center_y)),
            5,
            (0, 0, 255),
            -1
        )

    # Display overall proximity
    cv2.putText(
        frame,
        f"Closest: {result['closest_tier']}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(window_name, frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Close window
    if cv2.getWindowProperty(
        window_name,
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break


cap.release()
cv2.destroyAllWindows()