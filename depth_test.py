from ultralytics import YOLO
import cv2
import numpy as np

from movement_analysis import MovementAnalyzer


# -------------------------
# LOAD MODELS
# -------------------------

object_model = YOLO("yolov8n.pt")
depth_model = YOLO("yolo26n-depth.pt")


# -------------------------
# OPEN WEBCAM
# -------------------------

cap = cv2.VideoCapture(0)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# -------------------------
# CAMERA FOV
# -------------------------

# Approximate webcam field of view
HORIZONTAL_FOV = 70
VERTICAL_FOV = 55


# -------------------------
# MOVEMENT ANALYZER
# -------------------------

movement_analyzer = MovementAnalyzer()


# -------------------------
# MAIN LOOP
# -------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # -------------------------
    # OBJECT DETECTION + TRACKING
    # -------------------------

    object_results = object_model.track(
        frame,
        persist=True,
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
    # PROCESS EACH OBJECT
    # -------------------------

    for box in object_results.boxes:

        # -------------------------
        # OBJECT LABEL
        # -------------------------

        label = object_model.names[
            int(box.cls[0])
        ]


        # -------------------------
        # TRACKING ID
        # -------------------------

        if box.id is not None:
            track_id = int(box.id[0])
        else:
            track_id = -1


        # -------------------------
        # BOUNDING BOX
        # -------------------------

        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # -------------------------
        # OBJECT CENTER
        # -------------------------

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )


        # -------------------------
        # KEEP CENTER INSIDE DEPTH MAP
        # -------------------------

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


        # -------------------------
        # DEPTH REGION
        # -------------------------

        region_size = 10

        x_start = max(
            0,
            center_x - region_size
        )

        x_end = min(
            depth_map.shape[1],
            center_x + region_size
        )

        y_start = max(
            0,
            center_y - region_size
        )

        y_end = min(
            depth_map.shape[0],
            center_y + region_size
        )


        depth_region = depth_map[
            y_start:y_end,
            x_start:x_end
        ]


        # -------------------------
        # MEDIAN DEPTH
        # -------------------------

        depth = np.median(
            depth_region
        )


        # -------------------------
        # MOVEMENT ANALYSIS
        # -------------------------

        movement = movement_analyzer.analyze(
            track_id,
            center_x,
            center_y,
            depth
        )


        movement_status = movement[
            "movement_status"
        ]

        movement_direction = movement[
            "movement_direction"
        ]

        movement_distance = movement[
            "movement_distance"
        ]

        depth_status = movement[
            "depth_status"
        ]

        depth_change = movement[
            "depth_change"
        ]


        # -------------------------
        # 3D CAMERA COORDINATES
        # -------------------------

        Z = depth


        # Distance from image center
        pixel_x = (
            center_x -
            FRAME_WIDTH / 2
        )

        pixel_y = (
            center_y -
            FRAME_HEIGHT / 2
        )


        # Convert FOV to radians
        horizontal_fov = np.radians(
            HORIZONTAL_FOV
        )

        vertical_fov = np.radians(
            VERTICAL_FOV
        )


        # -------------------------
        # X COORDINATE
        # -------------------------

        X = (
            Z *
            np.tan(horizontal_fov / 2) *
            (pixel_x / (FRAME_WIDTH / 2))
        )


        # -------------------------
        # Y COORDINATE
        # -------------------------

        Y = (
            Z *
            np.tan(vertical_fov / 2) *
            (pixel_y / (FRAME_HEIGHT / 2))
        )


        # -------------------------
        # 3D MAGNITUDE
        # -------------------------

        magnitude = np.sqrt(
            X ** 2 +
            Y ** 2 +
            Z ** 2
        )


        # -------------------------
        # TERMINAL OUTPUT
        # -------------------------

        print(
            f"{label} "
            f"ID={track_id}: "
            f"X={X:.2f}m, "
            f"Y={Y:.2f}m, "
            f"Z={Z:.2f}m, "
            f"Distance={magnitude:.2f}m, "
            f"Movement={movement_status}, "
            f"Direction={movement_direction}, "
            f"Depth={depth_status}"
        )


        # -------------------------
        # DRAW BOUNDING BOX
        # -------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # -------------------------
        # DRAW CENTER POINT
        # -------------------------

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )


        # -------------------------
        # DISPLAY MOVEMENT
        # -------------------------

        movement_text = (
            f"{label} "
            f"ID:{track_id} "
            f"{movement_status} "
            f"{movement_direction}"
        )

        cv2.putText(
            frame,
            movement_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        # -------------------------
        # DISPLAY DEPTH
        # -------------------------

        depth_text = (
            f"Z:{Z:.2f}m "
            f"{depth_status}"
        )

        cv2.putText(
            frame,
            depth_text,
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


    # -------------------------
    # SHOW CAMERA
    # -------------------------

    cv2.imshow(
        "SafeWalk Depth",
        frame
    )


    # -------------------------
    # QUIT
    # -------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -------------------------
# CLEANUP
# -------------------------

cap.release()

cv2.destroyAllWindows()