import math


class MovementAnalyzer:

    def __init__(
        self,
        movement_threshold=10,
        depth_threshold=0.05
    ):
        self.movement_threshold = movement_threshold
        self.depth_threshold = depth_threshold

        self.previous_positions = {}
        self.previous_depths = {}

    def analyze(
        self,
        track_id,
        center_x,
        center_y,
        depth
    ):

        movement_distance = 0
        movement_status = "STATIONARY"
        movement_direction = "NONE"

        depth_change = 0
        depth_status = "UNKNOWN"

        current_position = (
            center_x,
            center_y
        )

        # MOVEMENT
        if track_id != -1:

            if track_id in self.previous_positions:

                previous_x, previous_y = (
                    self.previous_positions[track_id]
                )

                dx = center_x - previous_x
                dy = center_y - previous_y

                movement_distance = math.sqrt(
                    dx ** 2 +
                    dy ** 2
                )

                if movement_distance > self.movement_threshold:

                    movement_status = "MOVING"

                    if abs(dx) > abs(dy):

                        if dx > 0:
                            movement_direction = "RIGHT"
                        else:
                            movement_direction = "LEFT"

                    else:

                        if dy > 0:
                            movement_direction = "DOWN"
                        else:
                            movement_direction = "UP"

            self.previous_positions[
                track_id
            ] = current_position

        # DEPTH
        if track_id != -1:

            if track_id in self.previous_depths:

                previous_depth = (
                    self.previous_depths[track_id]
                )

                depth_change = (
                    depth -
                    previous_depth
                )

                if depth_change < -self.depth_threshold:

                    depth_status = "APPROACHING"

                elif depth_change > self.depth_threshold:

                    depth_status = "MOVING AWAY"

                else:

                    depth_status = "DEPTH STABLE"

            self.previous_depths[
                track_id
            ] = depth

        return {
            "movement_distance": movement_distance,
            "movement_status": movement_status,
            "movement_direction": movement_direction,
            "depth_change": depth_change,
            "depth_status": depth_status,
            "events": []
        }

    def remove_track(self, track_id):

        if track_id in self.previous_positions:
            del self.previous_positions[track_id]

        if track_id in self.previous_depths:
            del self.previous_depths[track_id]

    def reset(self):

        self.previous_positions.clear()
        self.previous_depths.clear()