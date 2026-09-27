import cv2
import numpy as np

from ultralytics import YOLO
from movement_analysis import MovementAnalyzer
from movement_history import MovementHistory

class SafeWalk:

    def __init__(self):

        print("1. Starting SafeWalk...")

        # Models
        print("2. Loading object model...")
        self.object_model = YOLO("yolov8n.pt")

        print("3. Loading depth model...")
        self.depth_model = YOLO("yolo26n-depth.pt")

        print("4. Creating movement analyzer...")
        self.movement_analyzer = MovementAnalyzer()
        self.movement_history = MovementHistory()

        print("5. Opening webcam...")
        self.cap = cv2.VideoCapture(0)

        self.frame_width = 640
        self.frame_height = 480

        self.horizontal_fov = 70
        self.vertical_fov = 55

        print("6. SafeWalk initialized.")

    def run(self):

        print("7. Starting camera loop...")

        while True:

            ret, frame = self.cap.read()

            if not ret:
                print("ERROR: Could not read frame.")
                break

            print("Frame received.")

            # Object detection + tracking
            print("Running object detection...")

            object_results = self.object_model.track(
                frame,
                persist=True,
                verbose=False
            )[0]

            print("Object detection finished.")

            # Depth estimation
            print("Running depth estimation...")

            depth_result = self.depth_model(
                frame,
                verbose=False
            )[0]

            print("Depth estimation finished.")

            depth_map = depth_result.depth.data.cpu().numpy()
            depth_map = np.squeeze(depth_map)

            print("Depth map created.")

            # Process detected objects
            for box in object_results.boxes:

                label = self.object_model.names[
                    int(box.cls[0])
                ]

                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = -1

                x1, y1, x2, y2 = box.xyxy[0]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

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

                depth = np.median(depth_region)

                print(
                    f"Detected {label}, "
                    f"ID={track_id}, "
                    f"depth={depth:.2f}"
                )

                # Movement analysis
                movement = self.movement_analyzer.analyze(
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

                depth_status = movement[
                    "depth_status"
                ]

                # --------------------------------
                # 3D COORDINATES
                # --------------------------------

                Z = depth

                pixel_x = (
                        center_x -
                        self.frame_width / 2
                )

                pixel_y = (
                        center_y -
                        self.frame_height / 2
                )

                horizontal_fov = np.radians(
                    self.horizontal_fov
                )

                vertical_fov = np.radians(
                    self.vertical_fov
                )

                X = (
                        Z *
                        np.tan(horizontal_fov / 2) *
                        (
                                pixel_x /
                                (self.frame_width / 2)
                        )
                )

                Y = (
                        Z *
                        np.tan(vertical_fov / 2) *
                        (
                                pixel_y /
                                (self.frame_height / 2)
                        )
                )

                magnitude = np.sqrt(
                    X ** 2 +
                    Y ** 2 +
                    Z ** 2
                )

                # --------------------------------
                # MOVEMENT HISTORY
                # --------------------------------

                self.movement_history.add_record(
                    track_id,
                    X,
                    Y,
                    Z,
                    magnitude,
                    movement_status,
                    movement_direction,
                    depth_status
                )

                approaching_duration = (
                    self.movement_history.get_duration(
                        track_id,
                        "APPROACHING"
                    )
                )

                moving_duration = (
                    self.movement_history.get_duration(
                        track_id,
                        "MOVING"
                    )
                )
                print(
                    f"History saved: "
                    f"ID={track_id}, "
                    f"X={X:.2f}, "
                    f"Y={Y:.2f}, "
                    f"Z={Z:.2f}, "
                    f"Distance={magnitude:.2f}"
                )

                # Draw object
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1
                )

                movement_text = (
                    f"{label} "
                    f"ID:{track_id} "
                    f"{movement_status} "
                    f"{movement_direction}"
                )

                cv2.putText(
                    frame,
                    movement_text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            print("Showing frame...")

            cv2.imshow(
                "SafeWalk",
                frame
            )

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    app = SafeWalk()
    app.run()