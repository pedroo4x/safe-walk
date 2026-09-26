from ultralytics import YOLO
import cv2
import numpy as np

# Load models
object_model = YOLO("yolov8n.pt")
depth_model = YOLO("yolo26n-depth.pt")

# Open webcam
cap = cv2.VideoCapture(0)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

HORIZONTAL_FOV = 70
VERTICAL_FOV = 55

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # -------------------------
    # OBJECT DETECTION
    # -------------------------

    object_results = object_model(
        frame,
        verbose=False
    )[0]

    # -------------------------
    # DEPTH ESTIMATION
    # -------------------------

    depth_result = depth_model(
        frame,
        verbose=False
    )[0]

    depth_map = depth_result.depth.data.cpu().numpy()

    # Remove extra dimensions if necessary
    depth_map = np.squeeze(depth_map)

    # -------------------------
    # OBJECTS
    # -------------------------

    for box in object_results.boxes:

        label = object_model.names[int(box.cls[0])]

        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Calculate center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Make sure coordinates are inside depth map
        center_x = np.clip(
            center_x,
            0,
            depth_map.shape[1] - 1
        )

        center_y = np.clip(
            center_y,
            0,
            depth_map.shape[0] - 1
        )

        # Get estimated depth
        # Get a small region around the object's center
        region_size = 10

        x_start = max(0, center_x - region_size)
        x_end = min(depth_map.shape[1], center_x + region_size)

        y_start = max(0, center_y - region_size)
        y_end = min(depth_map.shape[0], center_y + region_size)

        depth_region = depth_map[
            y_start:y_end,
            x_start:x_end
        ]

        # Use the median to reduce noise
        depth = np.median(depth_region)

        # -------------------------
        # 3D CAMERA COORDINATES
        # -------------------------

        Z = depth

        # Distance from image center
        pixel_x = center_x - FRAME_WIDTH / 2
        pixel_y = center_y - FRAME_HEIGHT / 2

        # Convert FOV from degrees to radians
        horizontal_fov = np.radians(HORIZONTAL_FOV)
        vertical_fov = np.radians(VERTICAL_FOV)

        # Calculate approximate physical X/Y position
        X = Z * np.tan(horizontal_fov / 2) * (
                pixel_x / (FRAME_WIDTH / 2)
        )

        Y = Z * np.tan(vertical_fov / 2) * (
                pixel_y / (FRAME_HEIGHT / 2)
        )

        magnitude = np.sqrt(
            X ** 2 +
            Y ** 2 +
            Z ** 2
        )

        print(
            f"{label}: "
            f"X={X:.2f}, "
            f"Y={Y:.2f}, "
            f"Z={Z:.2f}, "
            f"Magnitude={magnitude:.2f}"
        )

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display depth
        text = f"{label}: {depth:.2f}m"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Draw center point
        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

    cv2.imshow(
        "SafeWalk Depth",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()