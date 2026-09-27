import cv2
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Settings
MODEL_PATH = "pose_landmarker_heavy.task"
MAX_PEOPLE = 5
MOVEMENT_THRESHOLD = 0.0057

# Fall settings
FALL_ANGLE_THRESHOLD = 55.0
FALL_HORIZONTAL_ANGLE = 75.0
FALL_HISTORY_FRAMES = 8
FALL_ANGLE_CHANGE_THRESHOLD = 20.0
FALL_ANGLE_VELOCITY_THRESHOLD = 70.0
FALL_DROP_VELOCITY_THRESHOLD = 0.20
FALL_CONFIRMATION_FRAMES = 2
FALL_ALERT_DURATION = 5.0
FALL_COOLDOWN = 2.0

# Tracking settings
MAX_MATCH_DISTANCE = 0.25
MAX_MISSED_FRAMES = 120

# MediaPipe setup
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=MAX_PEOPLE
)

detector = vision.PoseLandmarker.create_from_options(options)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Tracks
tracks = {}

# Important landmarks
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def get_available_person_id():
    for person_id in range(1, MAX_PEOPLE + 1):
        if person_id not in tracks:
            return person_id

    oldest_id = max(
        tracks,
        key=lambda pid: tracks[pid]["missed"]
    )

    return oldest_id


def get_landmark_position(landmark):
    return landmark.x, landmark.y


def get_center(landmarks, index1, index2):
    x = (landmarks[index1].x + landmarks[index2].x) / 2
    y = (landmarks[index1].y + landmarks[index2].y) / 2
    return x, y


def get_torso_angle(landmarks):
    shoulder_x, shoulder_y = get_center(
        landmarks,
        LEFT_SHOULDER,
        RIGHT_SHOULDER
    )

    hip_x, hip_y = get_center(
        landmarks,
        LEFT_HIP,
        RIGHT_HIP
    )

    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y

    angle = math.degrees(math.atan2(abs(dx), abs(dy)))

    return angle


def get_body_size(landmarks):
    shoulder_x, shoulder_y = get_center(
        landmarks,
        LEFT_SHOULDER,
        RIGHT_SHOULDER
    )

    hip_x, hip_y = get_center(
        landmarks,
        LEFT_HIP,
        RIGHT_HIP
    )

    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y

    return math.sqrt(dx * dx + dy * dy)


def get_movement(current_position, previous_position):
    if previous_position is None:
        return 0.0

    dx = current_position[0] - previous_position[0]
    dy = current_position[1] - previous_position[1]

    return math.sqrt(dx * dx + dy * dy)


def detect_fall(track, angle, hip_center, body_size, current_time):
    angle_history = track["angle_history"]
    hip_y_history = track["hip_y_history"]
    time_history = track["time_history"]

    previous_angle = track["previous_angle"]
    previous_time = track["previous_time"]

    angle_history.append(angle)
    hip_y_history.append(hip_center[1])
    time_history.append(current_time)

    if len(angle_history) > FALL_HISTORY_FRAMES:
        angle_history.pop(0)
        hip_y_history.pop(0)
        time_history.pop(0)

    angle_change = 0.0
    angle_velocity = 0.0
    frame_angle_velocity = 0.0
    drop_velocity = 0.0

    if len(angle_history) >= 2:
        angle_change = (
            angle_history[-1] - angle_history[0]
        )

        history_time = (
            time_history[-1] - time_history[0]
        )

        if history_time > 0:
            angle_velocity = abs(angle_change) / history_time

    if previous_angle is not None and previous_time is not None:
        frame_time = current_time - previous_time

        if frame_time > 0:
            frame_angle_velocity = abs(
                angle - previous_angle
            ) / frame_time

    if len(hip_y_history) >= 2:
        history_time = (
            time_history[-1] - time_history[0]
        )

        if history_time > 0:
            drop_velocity = (
                hip_y_history[-1] - hip_y_history[0]
            ) / history_time

    # Adjust drop threshold based on distance
    if body_size < 0.15:
        drop_threshold = FALL_DROP_VELOCITY_THRESHOLD * 0.35
    elif body_size < 0.25:
        drop_threshold = FALL_DROP_VELOCITY_THRESHOLD * 0.50
    elif body_size < 0.40:
        drop_threshold = FALL_DROP_VELOCITY_THRESHOLD * 0.70
    else:
        drop_threshold = FALL_DROP_VELOCITY_THRESHOLD

    fall_score = 0

    tilted = angle >= FALL_ANGLE_THRESHOLD
    horizontal = angle >= FALL_HORIZONTAL_ANGLE
    rapid_rotation = (
        angle_velocity >= FALL_ANGLE_VELOCITY_THRESHOLD
        or frame_angle_velocity >= FALL_ANGLE_VELOCITY_THRESHOLD
    )
    rapid_drop = drop_velocity >= drop_threshold

    if tilted:
        fall_score += 1

    if rapid_rotation:
        fall_score += 2

    if rapid_drop:
        fall_score += 2

    if horizontal:
        fall_score += 2

    if tilted and rapid_rotation:
        fall_score += 1

    if tilted and rapid_drop:
        fall_score += 1

    possible_fall = fall_score >= 3

    if possible_fall:
        track["fall_frames"] += 1
    else:
        track["fall_frames"] = max(
            0,
            track["fall_frames"] - 1
        )

    if (
        track["fall_frames"] >= FALL_CONFIRMATION_FRAMES
        and current_time >= track["fall_cooldown_until"]
    ):
        track["fall_alert_until"] = (
            current_time + FALL_ALERT_DURATION
        )

        track["fall_cooldown_until"] = (
            current_time + FALL_COOLDOWN
        )

        track["fall_frames"] = 0

    track["previous_angle"] = angle
    track["previous_time"] = current_time

    # Save debug values internally
    track["debug_angle_velocity"] = angle_velocity
    track["debug_frame_angle_velocity"] = frame_angle_velocity
    track["debug_angle_change"] = angle_change
    track["debug_drop_velocity"] = drop_velocity
    track["debug_drop_threshold"] = drop_threshold
    track["debug_fall_score"] = fall_score

    return current_time < track["fall_alert_until"]


# Main loop
while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    # Convert frame
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int(current_time * 1000)

    # Run pose detection
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    detections = []

    # Process detected people
    for pose_landmarks in result.pose_landmarks:

        if len(pose_landmarks) < 29:
            continue

        landmarks = pose_landmarks

        hip_center = get_center(
            landmarks,
            LEFT_HIP,
            RIGHT_HIP
        )

        shoulder_center = get_center(
            landmarks,
            LEFT_SHOULDER,
            RIGHT_SHOULDER
        )

        body_size = get_body_size(landmarks)

        detections.append({
            "landmarks": landmarks,
            "hip_center": hip_center,
            "shoulder_center": shoulder_center,
            "body_size": body_size
        })

    # Mark tracks as unmatched
    for track in tracks.values():
        track["matched"] = False

    # Match people to existing tracks
    used_tracks = set()

    for detection in detections:

        current_position = detection["hip_center"]

        best_id = None
        best_distance = MAX_MATCH_DISTANCE

        for person_id, track in tracks.items():

            if person_id in used_tracks:
                continue

            previous_position = track["position"]

            dx = (
                current_position[0]
                - previous_position[0]
            )

            dy = (
                current_position[1]
                - previous_position[1]
            )

            distance = math.sqrt(
                dx * dx + dy * dy
            )

            if distance < best_distance:
                best_distance = distance
                best_id = person_id

        # Create new person
        if best_id is None:

            new_id = get_available_person_id()

            tracks[new_id] = {
                "position": current_position,
                "matched": True,
                "missed": 0,
                "fall_frames": 0,
                "previous_angle": None,
                "previous_time": None,
                "angle_history": [],
                "hip_y_history": [],
                "time_history": [],
                "fall_alert_until": 0,
                "fall_cooldown_until": 0,
                "debug_angle_velocity": 0.0,
                "debug_frame_angle_velocity": 0.0,
                "debug_angle_change": 0.0,
                "debug_drop_velocity": 0.0,
                "debug_drop_threshold": 0.0,
                "debug_fall_score": 0
            }

            best_id = new_id

        # Update existing person
        track = tracks[best_id]

        track["position"] = current_position
        track["matched"] = True
        track["missed"] = 0

        used_tracks.add(best_id)

        detection["person_id"] = best_id

    # Update missed tracks
    for person_id, track in list(tracks.items()):

        if not track["matched"]:
            track["missed"] += 1

        if track["missed"] > MAX_MISSED_FRAMES:
            del tracks[person_id]

    # Draw each person
    for detection in detections:

        person_id = detection["person_id"]
        landmarks = detection["landmarks"]
        hip_center = detection["hip_center"]
        shoulder_center = detection["shoulder_center"]
        body_size = detection["body_size"]

        track = tracks[person_id]

        # Movement
        movement = get_movement(
            hip_center,
            track.get("previous_position")
        )

        if movement > MOVEMENT_THRESHOLD:
            movement_status = "MOVING"
        else:
            movement_status = "STATIONARY"

        track["previous_position"] = hip_center

        # Fall detection
        angle = get_torso_angle(landmarks)

        fall_active = detect_fall(
            track,
            angle,
            hip_center,
            body_size,
            current_time
        )

        # Status
        if fall_active:
            status = "POSSIBLE FALL"
            text_color = (0, 0, 255)
        else:
            status = movement_status
            text_color = (0, 255, 0)

        # Person label
        label_x = int(
            shoulder_center[0] * frame.shape[1]
        )

        label_y = int(
            shoulder_center[1] * frame.shape[0]
        ) - 20

        cv2.putText(
            frame,
            f"Person {person_id}: {status}",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            text_color,
            2
        )

        # Draw important landmarks
        important_points = [
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP,
            LEFT_ELBOW,
            RIGHT_ELBOW,
            LEFT_WRIST,
            RIGHT_WRIST,
            LEFT_KNEE,
            RIGHT_KNEE,
            LEFT_ANKLE,
            RIGHT_ANKLE
        ]

        for index in important_points:

            landmark = landmarks[index]

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                text_color,
                -1
            )

    # Fall warning
    fall_warning = any(
        current_time < track["fall_alert_until"]
        for track in tracks.values()
    )

    if fall_warning:
        cv2.putText(
            frame,
            "!!! FALL ALERT !!!",
            (150, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    # Ppl count
    people_detected = len(detections)

    cv2.putText(
        frame,
        f"People Detected: {people_detected}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Show frame
    cv2.imshow(
        "SafeWalk",
        frame
    )

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()