import pyttsx3
import threading
import time


class AudioAlerts:
    def __init__(self):
        self.is_speaking = False
        self.last_alert_time = 0
        self.cooldown = 3

    def speak(self, message):
        current_time = time.time()

        if self.is_speaking:
            print(f"Audio skipped — already speaking: {message}")
            return

        if current_time - self.last_alert_time < self.cooldown:
            print(f"Audio skipped — cooldown: {message}")
            return

        print(f"AUDIO ALERT: {message}")

        self.last_alert_time = current_time

        thread = threading.Thread(
            target=self._speak,
            args=(message,),
            daemon=True
        )

        thread.start()

    def _speak(self, message):
        self.is_speaking = True

        try:
            engine = pyttsx3.init()

            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)

            engine.say(message)
            engine.runAndWait()
            engine.stop()

        except Exception as e:
            print(f"Audio error: {e}")

        finally:
            self.is_speaking = False