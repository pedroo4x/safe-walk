import sys

from PySide6.QtWidgets import QApplication

from safewalk_backend import SafeWalk
from safewalk_ui import SafeWalkUI


def main():
    app = QApplication(sys.argv)

    # Create the backend
    backend = SafeWalk()

    # Create the UI and give it access to the backend
    window = SafeWalkUI(backend)

    # Show the application
    window.show()

    # Start Qt event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()