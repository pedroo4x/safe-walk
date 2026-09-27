from datetime import datetime


class MovementHistory:

    def __init__(self, max_history=300):

        self.max_history = max_history

        self.history = {}

    def add_record(
        self,
        track_id,
        x,
        y,
        z,
        distance,
        movement_status,
        movement_direction,
        depth_status
    ):

        if track_id == -1:
            return

        if track_id not in self.history:
            self.history[track_id] = []

        record = {
            "timestamp": datetime.now(),
            "x": x,
            "y": y,
            "z": z,
            "distance": distance,
            "movement_status": movement_status,
            "movement_direction": movement_direction,
            "depth_status": depth_status
        }

        self.history[track_id].append(record)

        if len(self.history[track_id]) > self.max_history:

            self.history[track_id] = (
                self.history[track_id][
                    -self.max_history:
                ]
            )

    def get_history(self, track_id):

        return self.history.get(
            track_id,
            []
        )

    def get_latest(self, track_id):

        history = self.get_history(track_id)

        if not history:
            return None

        return history[-1]

    def get_duration(
        self,
        track_id,
        status
    ):

        history = self.get_history(track_id)

        if not history:
            return 0

        duration = 0

        # Start from the most recent record
        for i in range(
            len(history) - 1,
            -1,
            -1
        ):

            record = history[i]

            if (
                record["movement_status"] == status
                or
                record["depth_status"] == status
            ):

                if i == 0:
                    duration = (
                        history[-1]["timestamp"]
                        -
                        history[i]["timestamp"]
                    ).total_seconds()

                else:
                    duration = (
                        history[-1]["timestamp"]
                        -
                        history[i]["timestamp"]
                    ).total_seconds()

            else:
                break

        return duration

    def remove_track(self, track_id):

        if track_id in self.history:
            del self.history[track_id]

    def reset(self):

        self.history.clear()