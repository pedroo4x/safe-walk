from datetime import datetime


class EventData:
    def __init__(
        self,
        track_id,
        object_type,
        event,
        zone,
        distance,
        x,
        y,
        z,
        movement_status,
        movement_direction,
        depth_status,
        approaching_duration,
        stationary_duration
    ):
        self.timestamp = datetime.now().isoformat(
            timespec="seconds"
        )

        self.track_id = track_id
        self.object_type = object_type
        self.event = event
        self.zone = zone

        self.distance = round(distance, 2)
        self.x = round(x, 2)
        self.y = round(y, 2)
        self.z = round(z, 2)

        self.movement_status = movement_status
        self.movement_direction = movement_direction
        self.depth_status = depth_status

        self.approaching_duration = round(
            approaching_duration,
            2
        )

        self.stationary_duration = round(
            stationary_duration,
            2
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "track_id": self.track_id,
            "object_type": self.object_type,
            "event": self.event,
            "zone": self.zone,
            "distance": self.distance,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "movement_status": self.movement_status,
            "movement_direction": self.movement_direction,
            "depth_status": self.depth_status,
            "approaching_duration": self.approaching_duration,
            "stationary_duration": self.stationary_duration
        }