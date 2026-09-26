from ultralytics import YOLO
import math


class ObjectDetector:

    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

        # Store previous center positions for tracked objects
        self.previous_positions = {}

        # Objects that are important for SafeWalk
        self.important_objects = {
            "person",
            "car",
            "truck",
            "bus",
            "bicycle",
            "motorcycle"
        }

        # Movement threshold in pixels
        self.movement_threshold = 10

    def estimate_proximity(self, box_height, frame_height):
        """
        Estimate how visually close an object is
        based on its bounding-box height.

        This is an approximation, not a real distance measurement.
        """

        ratio = box_height / frame_height

        if ratio > 0.4:
            return "CLOSE"

        elif ratio > 0.2:
            return "MEDIUM"

        else:
            return "FAR"

    def calculate_movement(self, track_id, center):
        """
        Compare the current center position with the
        previous position of the same tracked object.
        """

        current_x, current_y = center

        # First time seeing this object
        if track_id not in self.previous_positions:
            self.previous_positions[track_id] = center

            return {
                "distance": 0,
                "status": "STATIONARY",
                "direction": "NONE"
            }

        previous_x, previous_y = self.previous_positions[track_id]

        # Calculate movement
        dx = current_x - previous_x
        dy = current_y - previous_y

        distance = math.sqrt(
            dx ** 2 +
            dy ** 2
        )

        # Determine whether object moved
        if distance <= self.movement_threshold:

            status = "STATIONARY"
            direction = "NONE"

        else:

            status = "MOVING"

            # Determine primary direction
            if abs(dx) > abs(dy):

                if dx > 0:
                    direction = "RIGHT"
                else:
                    direction = "LEFT"

            else:

                if dy > 0:
                    direction = "DOWN"
                else:
                    direction = "UP"

        # Save current position
        self.previous_positions[track_id] = center

        return {
            "distance": distance,
            "status": status,
            "direction": direction
        }

    def detect(self, frame):

        # Use YOLO tracking instead of normal detection
        results = self.model.track(
            frame,
            persist=True,
            verbose=False
        )[0]

        frame_height = frame.shape[0]

        detections = []

        closest_tier = "none"

        if results.boxes is None:
            return {
                "detections": [],
                "closest_tier": "none"
            }

        for box in results.boxes:

            # Get object label
            label = self.model.names[int(box.cls[0])]

            # Ignore objects that aren't important to SafeWalk
            if label not in self.important_objects:
                continue

            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Bounding box height
            box_height = y2 - y1

            # -------------------------
            # CENTER
            # -------------------------

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            center = (center_x, center_y)

            # -------------------------
            # PROXIMITY
            # -------------------------

            proximity = self.estimate_proximity(
                box_height,
                frame_height
            )

            # Track closest object
            if proximity == "CLOSE":

                closest_tier = "close"

            elif proximity == "MEDIUM" and closest_tier != "close":

                closest_tier = "medium"

            # -------------------------
            # TRACKING ID
            # -------------------------

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            # -------------------------
            # MOVEMENT
            # -------------------------

            movement = {
                "distance": 0,
                "status": "STATIONARY",
                "direction": "NONE"
            }

            if track_id is not None:

                movement = self.calculate_movement(
                    track_id,
                    center
                )

            # -------------------------
            # STORE DETECTION
            # -------------------------

            detections.append({

                "id": track_id,

                "label": label,

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "center": center,

                "proximity": proximity,

                "movement": movement["distance"],

                "status": movement["status"]
            })

        return {
            "detections": detections,
            "closest_tier": closest_tier
        }