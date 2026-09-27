import cv2
import numpy as np
import time

from ultralytics import YOLO

from movement_analysis import MovementAnalyzer
from movement_history import MovementHistory
from safety_events import SafetyEvents
from event_logger import EventLogger
from audio_alerts import AudioAlerts
from event_data import EventData


class SafeWalk:

    def __init__(self):

        # =========================
        # AI MODELS
        # =========================

        self.object_model = YOLO("yolov8n.pt")
        self.depth_model = YOLO("yolo26n-depth.pt")

        self.safety_objects = {
            "person"
        }

        # =========================
        # ANALYSIS COMPONENTS
        # =========================

        self.movement_analyzer = MovementAnalyzer()
        self.movement_history = MovementHistory()

        self.safety_events = SafetyEvents()
        self.event_logger = EventLogger()
        self.audio_alerts = AudioAlerts()

        # =========================
        # EVENT STATE
        # =========================

        self.recent_events = []
        self.max_recent_events = 20

        self.latest_event = "NONE"
        self.latest_event_time = 0
        self.event_display_duration = 5

        # =========================
        # CAMERA
        # =========================

        self.cap = None

        self.frame_width = 640
        self.frame_height = 480

        self.horizontal_fov = 70
        self.vertical_fov = 55

        # =========================
        # UI STATE
        # =========================

        self.monitoring = False
        self.audio_enabled = True

    # ============================================================
    # MONITORING CONTROLS
    # ============================================================

    def start_monitoring(self):

        if self.monitoring:
            return True

        print("Starting SafeWalk monitoring...")

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("ERROR: Could not open camera.")
            self.cap = None
            self.monitoring = False
            return False

        # Try to keep camera resolution consistent
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.frame_width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.frame_height
        )

        self.monitoring = True

        print("SafeWalk monitoring started.")

        return True

    def stop_monitoring(self):

        print("Stopping SafeWalk monitoring...")

        self.monitoring = False

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        print("SafeWalk monitoring stopped.")

        return True

    def set_audio_enabled(self, enabled):

        self.audio_enabled = bool(enabled)

        print(
            "Audio alerts:",
            "ON" if self.audio_enabled else "OFF"
        )

        return self.audio_enabled

    # ============================================================
    # CAMERA FRAME
    # ============================================================

    def process_camera_frame(self):

        if not self.monitoring:
            return None

        if self.cap is None:
            return None

        ret, frame = self.cap.read()

        if not ret:

            print("ERROR: Could not read camera frame.")

            return None

        return self.process_frame(frame)

    # ============================================================
    # PROCESS ONE FRAME
    # ============================================================

    def process_frame(self, frame):

        # Make sure the frame is valid
        if frame is None:
            return None

        # Get actual dimensions
        frame_height, frame_width = frame.shape[:2]

        # =========================
        # OBJECT DETECTION
        # =========================

        object_results = self.object_model.track(
            frame,
            persist=True,
            verbose=False
        )[0]

        # =========================
        # DEPTH ESTIMATION
        # =========================

        depth_result = self.depth_model(
            frame,
            verbose=False
        )[0]

        depth_map = (
            depth_result.depth.data
            .cpu()
            .numpy()
        )

        depth_map = np.squeeze(depth_map)

        # =========================
        # INITIAL STATE
        # =========================

        overall_status = "SAFE"

        safety_object_count = 0

        current_track_ids = set()

        object_details = []

        # ========================================================
        # PROCESS DETECTED OBJECTS
        # ========================================================

        for box in object_results.boxes:

            label = self.object_model.names[
                int(box.cls[0])
            ]

            # Only process people
            if label not in self.safety_objects:
                continue

            safety_object_count += 1

            # =========================
            # TRACK ID
            # =========================

            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            if track_id != -1:
                current_track_ids.add(track_id)

            # =========================
            # BOUNDING BOX
            # =========================

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            # =========================
            # DEPTH
            # =========================

            region_size = 10

            y_start = max(
                0,
                center_y - region_size
            )

            y_end = min(
                depth_map.shape[0],
                center_y + region_size
            )

            x_start = max(
                0,
                center_x - region_size
            )

            x_end = min(
                depth_map.shape[1],
                center_x + region_size
            )

            depth_region = depth_map[
                y_start:y_end,
                x_start:x_end
            ]

            if depth_region.size > 0:

                depth = float(
                    np.median(depth_region)
                )

            else:

                depth = 0.0

            # =========================
            # MOVEMENT ANALYSIS
            # =========================

            movement = (
                self.movement_analyzer.analyze(
                    track_id,
                    center_x,
                    center_y,
                    depth
                )
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

            # =========================
            # 3D COORDINATES
            # =========================

            Z = depth

            pixel_x = (
                center_x -
                frame_width / 2
            )

            pixel_y = (
                center_y -
                frame_height / 2
            )

            horizontal_fov = np.radians(
                self.horizontal_fov
            )

            vertical_fov = np.radians(
                self.vertical_fov
            )

            X = (
                Z
                * np.tan(horizontal_fov / 2)
                * (
                    pixel_x /
                    (frame_width / 2)
                )
            )

            Y = (
                Z
                * np.tan(vertical_fov / 2)
                * (
                    pixel_y /
                    (frame_height / 2)
                )
            )

            magnitude = np.sqrt(
                X ** 2 +
                Y ** 2 +
                Z ** 2
            )

            # =========================
            # MOVEMENT HISTORY
            # =========================

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

            # =========================
            # DURATIONS
            # =========================

            approaching_duration = (
                self.movement_history.get_duration(
                    track_id,
                    "APPROACHING"
                )
            )

            stationary_duration = (
                self.movement_history.get_duration(
                    track_id,
                    "STATIONARY"
                )
            )

            # =========================
            # SAFETY ANALYSIS
            # =========================

            safety = self.safety_events.analyze(
                track_id,
                magnitude,
                movement_status,
                depth_status,
                approaching_duration,
                stationary_duration
            )

            safety_events = safety["events"]

            zone = safety["zone"]

            # =========================
            # OVERALL STATUS
            # =========================

            if zone == "DANGER":

                overall_status = "DANGER"

            elif (
                zone == "WARNING"
                and overall_status != "DANGER"
            ):

                overall_status = "WARNING"

            # =========================
            # SAFETY EVENTS
            # =========================

            for event in safety_events:

                self.latest_event = event

                self.latest_event_time = time.time()

                event_data = EventData(
                    track_id=track_id,
                    object_type=label,
                    event=event,
                    zone=zone,
                    distance=magnitude,
                    x=X,
                    y=Y,
                    z=Z,
                    movement_status=movement_status,
                    movement_direction=movement_direction,
                    depth_status=depth_status,
                    approaching_duration=approaching_duration,
                    stationary_duration=stationary_duration
                )

                event_dict = (
                    event_data.to_dict()
                )

                self.recent_events.append(
                    event_dict
                )

                if (
                    len(self.recent_events)
                    > self.max_recent_events
                ):

                    self.recent_events.pop(0)

                print(
                    f"SAFETY EVENT: "
                    f"{event_dict}"
                )

                # Log event
                self.event_logger.log_event(
                    track_id,
                    event,
                    zone,
                    magnitude,
                    X,
                    Y,
                    Z,
                    movement_status,
                    movement_direction,
                    depth_status
                )

                # Audio
                if self.audio_enabled:

                    if event == "PERSON_APPROACHING":

                        self.audio_alerts.speak(
                            "Object approaching"
                        )

                    elif event == "ENTERED_WARNING_ZONE":

                        self.audio_alerts.speak(
                            "Warning. Object nearby"
                        )

                    elif event == "ENTERED_DANGER_ZONE":

                        self.audio_alerts.speak(
                            "Danger. Object very close"
                        )

            # =========================
            # BOUNDING BOX COLOR
            # =========================

            if zone == "SAFE":

                safety_color = (
                    0,
                    255,
                    0
                )

            elif zone == "WARNING":

                safety_color = (
                    0,
                    255,
                    255
                )

            else:

                safety_color = (
                    0,
                    0,
                    255
                )

            # =========================
            # DRAW BOX
            # =========================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                safety_color,
                2
            )

            # Center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # =========================
            # DRAW LABEL
            # =========================

            movement_text = (
                f"{label} "
                f"ID:{track_id} "
                f"ZONE:{zone}"
            )

            detail_text = (
                f"{movement_status} "
                f"{movement_direction} "
                f"{depth_status}"
            )

            cv2.putText(
                frame,
                movement_text,
                (
                    x1,
                    max(y1 - 30, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                safety_color,
                2
            )

            cv2.putText(
                frame,
                detail_text,
                (
                    x1,
                    max(y1 - 10, 40)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            # =========================
            # OBJECT DATA FOR UI
            # =========================

            object_details.append({
                "track_id": track_id,
                "object_type": label,
                "zone": zone,
                "distance": magnitude,
                "x": X,
                "y": Y,
                "z": Z,
                "movement_status": movement_status,
                "movement_direction": movement_direction,
                "depth_status": depth_status,
                "approaching_duration":
                    approaching_duration,
                "stationary_duration":
                    stationary_duration,
                "bounding_box":
                    (x1, y1, x2, y2),
                "center":
                    (center_x, center_y)
            })

        # ========================================================
        # CLEAN UP OLD TRACKS
        # ========================================================

        tracked_ids = set(
            self.movement_history.history.keys()
        )

        for track_id in tracked_ids:

            if track_id not in current_track_ids:

                self.movement_analyzer.remove_track(
                    track_id
                )

                self.movement_history.remove_track(
                    track_id
                )

                self.safety_events.remove_track(
                    track_id
                )

        # ========================================================
        # LATEST EVENT
        # ========================================================

        if (
            time.time() -
            self.latest_event_time
            <= self.event_display_duration
        ):

            latest_event = self.latest_event

        else:

            latest_event = "NONE"

        # ========================================================
        # RETURN DATA TO UI
        # ========================================================

        return {
            "frame": frame,
            "objects": safety_object_count,
            "status": overall_status,
            "events": list(
                self.recent_events
            ),
            "object_details": object_details,
            "latest_event": latest_event
        }

    # ============================================================
    # RELEASE
    # ============================================================

    def release(self):

        self.monitoring = False

        if self.cap is not None:

            self.cap.release()

            self.cap = None

        cv2.destroyAllWindows()

    # ============================================================
    # OLD OPENCV TEST MODE
    # ============================================================

    def run(self):

        if not self.start_monitoring():
            return

        try:

            while self.monitoring:

                data = (
                    self.process_camera_frame()
                )

                if data is None:
                    break

                cv2.imshow(
                    "SafeWalk",
                    data["frame"]
                )

                key = (
                    cv2.waitKey(30)
                    & 0xFF
                )

                if key == ord("q"):
                    break

        finally:

            self.stop_monitoring()
            self.release()


if __name__ == "__main__":

    app = SafeWalk()

    app.run()