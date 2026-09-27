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

        self.object_model = YOLO("yolov8n.pt")
        self.depth_model = YOLO("yolo26n-depth.pt")

        self.safety_objects = {
            "person"
        }


        self.movement_analyzer = MovementAnalyzer()
        self.movement_history = MovementHistory()

        self.safety_events = SafetyEvents()
        self.event_logger = EventLogger()
        self.audio_alerts = AudioAlerts()

        self.recent_events = []
        self.max_recent_events = 20

        self.latest_event = "NONE"
        self.latest_event_time = 0
        self.event_display_duration = 5



        self.cap = cv2.VideoCapture(0)

        self.frame_width = 640
        self.frame_height = 480

        self.horizontal_fov = 70
        self.vertical_fov = 55



    def run(self):

        while True:

            ret, frame = self.cap.read()

            if not ret:
                print("ERROR: Could not read frame.")
                break

            # Run object detection and tracking
            object_results = self.object_model.track(
                frame,
                persist=True,
                verbose=False
            )[0]

            # Run depth estimation
            depth_result = self.depth_model(
                frame,
                verbose=False
            )[0]

            depth_map = depth_result.depth.data.cpu().numpy()
            depth_map = np.squeeze(depth_map)

            # Determine overall safety status
            overall_status = "SAFE"

            # Count only safety-relevant objects
            safety_object_count = 0

            # Track IDs visible in the current frame
            current_track_ids = set()

            # ============================================================
            # PROCESS DETECTED OBJECTS
            # ============================================================

            for box in object_results.boxes:

                label = self.object_model.names[int(box.cls[0])]

                # Ignore objects that are not safety-relevant
                if label not in self.safety_objects:
                    continue

                safety_object_count += 1

                # Get tracking ID
                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = -1

                # Remember that this track is currently visible
                if track_id != -1:
                    current_track_ids.add(track_id)

                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calculate center of bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # --------------------------------------------------------
                # Get depth around object center
                # --------------------------------------------------------

                region_size = 10

                y_start = max(0, center_y - region_size)
                y_end = min(depth_map.shape[0], center_y + region_size)

                x_start = max(0, center_x - region_size)
                x_end = min(depth_map.shape[1], center_x + region_size)

                depth_region = depth_map[
                    y_start:y_end,
                    x_start:x_end
                ]

                if depth_region.size > 0:
                    depth = float(np.median(depth_region))
                else:
                    depth = 0.0

                # --------------------------------------------------------
                # Analyze movement
                # --------------------------------------------------------

                movement = self.movement_analyzer.analyze(
                    track_id,
                    center_x,
                    center_y,
                    depth
                )

                movement_status = movement["movement_status"]
                movement_direction = movement["movement_direction"]
                depth_status = movement["depth_status"]

                # --------------------------------------------------------
                # Calculate approximate 3D coordinates
                # --------------------------------------------------------

                Z = depth

                pixel_x = center_x - self.frame_width / 2
                pixel_y = center_y - self.frame_height / 2

                horizontal_fov = np.radians(
                    self.horizontal_fov
                )

                vertical_fov = np.radians(
                    self.vertical_fov
                )

                X = (
                        Z
                        * np.tan(horizontal_fov / 2)
                        * (pixel_x / (self.frame_width / 2))
                )

                Y = (
                        Z
                        * np.tan(vertical_fov / 2)
                        * (pixel_y / (self.frame_height / 2))
                )

                magnitude = np.sqrt(
                    X ** 2 +
                    Y ** 2 +
                    Z ** 2
                )

                # --------------------------------------------------------
                # Save movement history
                # --------------------------------------------------------

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

                # --------------------------------------------------------
                # Calculate durations
                # --------------------------------------------------------

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

                # --------------------------------------------------------
                # Analyze safety
                # --------------------------------------------------------

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

                # --------------------------------------------------------
                # Determine overall safety status
                # --------------------------------------------------------

                if zone == "DANGER":
                    overall_status = "DANGER"

                elif (
                        zone == "WARNING"
                        and overall_status != "DANGER"
                ):
                    overall_status = "WARNING"

                # --------------------------------------------------------
                # Process safety events
                # --------------------------------------------------------

                for event in safety_events:

                    self.latest_event = event
                    self.latest_event_time = time.time()

                    # Create standardized event
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

                    event_dict = event_data.to_dict()

                    self.recent_events.append(event_dict)

                    if len(self.recent_events) > self.max_recent_events:
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

                    if event == "PERSON_APPROACHING":
                        self.audio_alerts.speak("Object approaching")

                    elif event == "ENTERED_WARNING_ZONE":
                        self.audio_alerts.speak("Warning. Object nearby")

                    elif event == "ENTERED_DANGER_ZONE":
                        self.audio_alerts.speak("Danger. Object very close")
                # --------------------------------------------------------
                # Determine bounding box color
                # --------------------------------------------------------

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

                # --------------------------------------------------------
                # Draw bounding box
                # --------------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    safety_color,
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

                # --------------------------------------------------------
                # Draw object information
                # --------------------------------------------------------

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

            # ============================================================
            # CLEAN UP DISAPPEARED TRACKS
            # ============================================================

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

            # ============================================================
            # DASHBOARD
            # ============================================================

            object_count = safety_object_count

            if overall_status == "SAFE":

                dashboard_color = (
                    0,
                    255,
                    0
                )

            elif overall_status == "WARNING":

                dashboard_color = (
                    0,
                    255,
                    255
                )

            else:

                dashboard_color = (
                    0,
                    0,
                    255
                )

            # Dashboard background
            cv2.rectangle(
                frame,
                (10, 10),
                (400, 155),
                (0, 0, 0),
                -1
            )

            # Title
            cv2.putText(
                frame,
                "SAFEWALK",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Object count
            cv2.putText(
                frame,
                f"Objects: {object_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Overall safety status
            cv2.putText(
                frame,
                f"STATUS: {overall_status}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                dashboard_color,
                2
            )

            # ============================================================
            # LATEST EVENT
            # ============================================================

            if (
                    time.time() - self.latest_event_time
                    <= self.event_display_duration
            ):

                dashboard_event = self.latest_event

            else:

                dashboard_event = "NONE"

            cv2.putText(
                frame,
                f"EVENT: {dashboard_event}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            # ============================================================
            # DISPLAY
            # ============================================================

            print("Showing frame...")

            cv2.imshow(
                "SafeWalk",
                frame
            )

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break

            # Allow X button to close the window
            try:

                if (
                        cv2.getWindowProperty(
                            "SafeWalk",
                            cv2.WND_PROP_VISIBLE
                        ) < 1
                ):
                    break

            except cv2.error:

                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    app = SafeWalk()
    app.run()