import cv2

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QScrollArea,
    QSizePolicy,
)


# ============================================================
# CAMERA WORKER
# ============================================================

class CameraWorker(QThread):
    frame_ready = Signal(object)
    error = Signal(str)

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.running = False

    def run(self):
        self.running = True

        while self.running:
            try:
                data = self.backend.process_camera_frame()

                if data is not None:
                    self.frame_ready.emit(data)

            except Exception as e:
                self.error.emit(str(e))
                break

            self.msleep(30)

    def stop(self):
        self.running = False


# ============================================================
# SAFEWALK UI
# ============================================================

class SafeWalkUI(QMainWindow):

    def __init__(self, backend):
        super().__init__()

        self.backend = backend

        self.worker = None
        self.monitoring = False
        self.current_status = "SAFE"

        self.setWindowTitle("SafeWalk")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 850)

        self.setup_ui()
        self.apply_styles()

    # ========================================================
    # MAIN UI
    # ========================================================

    def setup_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ----------------------------------------------------
        # SIDEBAR
        # ----------------------------------------------------

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(235)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 24, 18, 20)
        sidebar_layout.setSpacing(8)

        # Logo
        logo = QLabel("SafeWalk")
        logo.setObjectName("logo")
        sidebar_layout.addWidget(logo)

        subtitle = QLabel("Personal Safety Monitor")
        subtitle.setObjectName("sidebarSubtitle")
        sidebar_layout.addWidget(subtitle)

        sidebar_layout.addSpacing(30)

        section_label = QLabel("MONITOR")
        section_label.setObjectName("sidebarSectionTitle")
        sidebar_layout.addWidget(section_label)

        # Navigation buttons
        self.overview_button = QPushButton("Overview")
        self.events_button = QPushButton("Events")
        self.objects_button = QPushButton("Tracked Objects")

        for button in [
            self.overview_button,
            self.events_button,
            self.objects_button,
        ]:
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            sidebar_layout.addWidget(button)

        self.overview_button.setChecked(True)

        sidebar_layout.addStretch()

        # Audio button
        self.audio_button = QPushButton("Audio Alerts: ON")
        self.audio_button.setCheckable(True)
        self.audio_button.setChecked(True)
        self.audio_button.setCursor(Qt.PointingHandCursor)

        sidebar_layout.addWidget(self.audio_button)

        # Monitor button
        self.camera_button = QPushButton("Start Monitoring")
        self.camera_button.setObjectName("monitorButton")
        self.camera_button.setCursor(Qt.PointingHandCursor)

        sidebar_layout.addWidget(self.camera_button)

        main_layout.addWidget(sidebar)

        # ----------------------------------------------------
        # MAIN CONTENT
        # ----------------------------------------------------

        self.content_widget = QWidget()
        self.content_widget.setObjectName("content")

        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(30, 25, 30, 25)
        content_layout.setSpacing(20)

        main_layout.addWidget(self.content_widget, 1)

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header_layout = QHBoxLayout()

        title_container = QVBoxLayout()
        title_container.setSpacing(4)

        self.page_title = QLabel("Safety Overview")
        self.page_title.setObjectName("pageTitle")

        self.page_subtitle = QLabel(
            "Real-time monitoring and environmental awareness"
        )
        self.page_subtitle.setObjectName("pageSubtitle")

        title_container.addWidget(self.page_title)
        title_container.addWidget(self.page_subtitle)

        header_layout.addLayout(title_container)
        header_layout.addStretch()

        # Connection status
        connection_container = QHBoxLayout()
        connection_container.setSpacing(8)

        self.connection_dot = QLabel()
        self.connection_dot.setObjectName("connectionDot")
        self.connection_dot.setFixedSize(10, 10)

        self.connection_status = QLabel("Camera Offline")
        self.connection_status.setObjectName("connectionStatus")

        connection_container.addWidget(self.connection_dot)
        connection_container.addWidget(self.connection_status)

        header_layout.addLayout(connection_container)

        content_layout.addLayout(header_layout)

        # ----------------------------------------------------
        # METRIC CARDS
        # ----------------------------------------------------

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)

        self.status_card = self.create_metric_card(
            "SAFETY STATUS",
            "SAFE",
            "status"
        )

        self.objects_card = self.create_metric_card(
            "TRACKED OBJECTS",
            "0",
            "objects"
        )

        self.events_card = self.create_metric_card(
            "RECENT EVENTS",
            "0",
            "events"
        )

        metrics_layout.addWidget(self.status_card)
        metrics_layout.addWidget(self.objects_card)
        metrics_layout.addWidget(self.events_card)

        content_layout.addLayout(metrics_layout)

        # ----------------------------------------------------
        # CAMERA CARD
        # ----------------------------------------------------

        self.camera_card = self.create_camera_card()

        content_layout.addWidget(
            self.camera_card,
            stretch=1
        )

        # ----------------------------------------------------
        # LOWER PANELS
        # ----------------------------------------------------

        lower_layout = QHBoxLayout()
        lower_layout.setSpacing(15)

        # Tracked objects
        self.objects_panel = self.create_objects_panel()

        # Events
        self.events_panel = self.create_events_panel()

        lower_layout.addWidget(
            self.objects_panel,
            stretch=1
        )

        lower_layout.addWidget(
            self.events_panel,
            stretch=1
        )

        content_layout.addLayout(lower_layout)

        # ----------------------------------------------------
        # SIGNAL CONNECTIONS
        # ----------------------------------------------------

        self.camera_button.clicked.connect(
            self.toggle_camera
        )

        self.audio_button.clicked.connect(
            self.toggle_audio
        )

        self.overview_button.clicked.connect(
            self.show_overview
        )

        self.events_button.clicked.connect(
            self.show_events
        )

        self.objects_button.clicked.connect(
            self.show_objects
        )

    # ========================================================
    # METRIC CARD
    # ========================================================

    def create_metric_card(self, title, value, card_type):

        card = QFrame()
        card.setObjectName("statCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(5)

        title_label = QLabel(title)
        title_label.setObjectName("statTitle")

        value_label = QLabel(value)
        value_label.setObjectName("statValue")

        if card_type == "status":
            value_label.setProperty("status", "safe")

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        # Save references
        if card_type == "status":
            self.status_value = value_label
        elif card_type == "objects":
            self.objects_value = value_label
        elif card_type == "events":
            self.events_value = value_label

        return card

    # ========================================================
    # CAMERA CARD
    # ========================================================

    def create_camera_card(self):

        card = QFrame()
        card.setObjectName("cameraCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("cameraHeader")

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)

        title = QLabel("Live Camera")
        title.setObjectName("cameraTitle")

        self.camera_connection = QLabel("OFFLINE")
        self.camera_connection.setObjectName("cameraOffline")

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.camera_connection)

        layout.addWidget(header)

        # Video area
        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")

        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_label = QLabel("Camera Offline")
        self.camera_label.setObjectName("cameraView")

        self.camera_label.setAlignment(
            Qt.AlignCenter
        )

        self.camera_label.setMinimumHeight(350)

        self.camera_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        video_layout.addWidget(self.camera_label)

        layout.addWidget(
            video_frame,
            stretch=1
        )

        # Footer
        footer = QFrame()
        footer.setObjectName("cameraFooter")

        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(18, 12, 18, 12)

        self.live_indicator = QLabel("● LIVE")
        self.live_indicator.setObjectName("liveIndicator")

        self.camera_objects = QLabel("0 objects")
        self.camera_objects.setObjectName("cameraInfo")

        resolution = QLabel("640 × 480")
        resolution.setObjectName("cameraInfo")

        footer_layout.addWidget(
            self.live_indicator
        )

        footer_layout.addSpacing(15)

        footer_layout.addWidget(
            self.camera_objects
        )

        footer_layout.addStretch()

        footer_layout.addWidget(
            resolution
        )

        layout.addWidget(footer)

        return card

    # ========================================================
    # OBJECTS PANEL
    # ========================================================

    def create_objects_panel(self):

        panel = QFrame()
        panel.setObjectName("panel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 16)

        title = QLabel("Tracked Objects")
        title.setObjectName("panelTitle")

        layout.addWidget(title)

        self.objects_scroll = QScrollArea()
        self.objects_scroll.setWidgetResizable(True)
        self.objects_scroll.setFrameShape(QFrame.NoFrame)

        self.objects_container = QWidget()

        self.object_layout = QVBoxLayout(
            self.objects_container
        )

        self.object_layout.setContentsMargins(
            0, 5, 0, 5
        )

        self.object_layout.setSpacing(8)

        self.objects_scroll.setWidget(
            self.objects_container
        )

        layout.addWidget(
            self.objects_scroll,
            stretch=1
        )

        self.empty_objects_label = QLabel(
            "No objects currently tracked"
        )

        self.empty_objects_label.setObjectName(
            "emptyLabel"
        )

        self.object_layout.addWidget(
            self.empty_objects_label
        )

        return panel

    # ========================================================
    # EVENTS PANEL
    # ========================================================

    def create_events_panel(self):

        panel = QFrame()
        panel.setObjectName("panel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 16)

        title = QLabel("Recent Events")
        title.setObjectName("panelTitle")

        layout.addWidget(title)

        self.events_scroll = QScrollArea()
        self.events_scroll.setWidgetResizable(True)
        self.events_scroll.setFrameShape(QFrame.NoFrame)

        self.events_container = QWidget()

        self.event_layout = QVBoxLayout(
            self.events_container
        )

        self.event_layout.setContentsMargins(
            0, 5, 0, 5
        )

        self.event_layout.setSpacing(8)

        self.events_scroll.setWidget(
            self.events_container
        )

        layout.addWidget(
            self.events_scroll,
            stretch=1
        )

        self.empty_events_label = QLabel(
            "No recent safety events"
        )

        self.empty_events_label.setObjectName(
            "emptyLabel"
        )

        self.event_layout.addWidget(
            self.empty_events_label
        )

        return panel

    # ========================================================
    # CAMERA CONTROL
    # ========================================================

    def toggle_camera(self):

        if self.monitoring:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):

        print("UI: Starting camera...")

        success = self.backend.start_monitoring()

        if not success:
            self.connection_status.setText(
                "Camera Error"
            )

            self.camera_connection.setText(
                "ERROR"
            )

            self.status_value.setText(
                "ERROR"
            )

            return

        self.monitoring = True

        self.camera_button.setText(
            "Stop Monitoring"
        )

        self.connection_status.setText(
            "Camera Connected"
        )

        self.camera_connection.setText(
            "LIVE"
        )

        self.live_indicator.setText(
            "● LIVE"
        )

        self.worker = CameraWorker(
            self.backend
        )

        self.worker.frame_ready.connect(
            self.update_dashboard
        )

        self.worker.error.connect(
            self.camera_error
        )

        self.worker.start()

        print("UI: Camera worker started.")

    def stop_camera(self):

        print("UI: Stopping camera...")

        self.monitoring = False

        if self.worker is not None:

            self.worker.stop()

            self.worker.wait()

            self.worker = None

        self.backend.stop_monitoring()

        self.camera_button.setText(
            "Start Monitoring"
        )

        self.connection_status.setText(
            "Camera Offline"
        )

        self.camera_connection.setText(
            "OFFLINE"
        )

        self.live_indicator.setText(
            "● OFFLINE"
        )

        self.camera_label.clear()

        self.camera_label.setText(
            "Camera Offline"
        )

        print("UI: Camera stopped.")

    # ========================================================
    # DASHBOARD UPDATE
    # ========================================================

    def update_dashboard(self, data):

        if data is None:
            return

        frame = data.get(
            "frame"
        )

        status = data.get(
            "status",
            "SAFE"
        )

        objects = data.get(
            "objects",
            0
        )

        events = data.get(
            "events",
            []
        )

        object_details = data.get(
            "object_details",
            []
        )

        # Camera
        if frame is not None:
            self.update_camera(
                frame
            )

        # Status
        self.update_status(
            status
        )

        # Objects
        self.update_objects(
            object_details
        )

        # Events
        self.update_events(
            events
        )

        # Metrics
        self.objects_value.setText(
            str(objects)
        )

        self.events_value.setText(
            str(len(events))
        )

        self.camera_objects.setText(
            f"{objects} object"
            + ("" if objects == 1 else "s")
        )

    # ========================================================
    # CAMERA DISPLAY
    # ========================================================

    def update_camera(self, frame):

        try:

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            height, width, channels = (
                rgb_frame.shape
            )

            bytes_per_line = (
                channels * width
            )

            image = QImage(
                rgb_frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(
                image
            )

            scaled_pixmap = pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.camera_label.setPixmap(
                scaled_pixmap
            )

        except Exception as e:

            print(
                "Camera display error:",
                e
            )

    # ========================================================
    # STATUS UPDATE
    # ========================================================

    def update_status(self, status):

        status = str(status).upper()

        self.current_status = status

        self.status_value.setText(
            status
        )

        self.status_value.setProperty(
            "status",
            status.lower()
        )

        self.status_value.style().unpolish(
            self.status_value
        )

        self.status_value.style().polish(
            self.status_value
        )

        if status == "DANGER":

            self.status_value.setText(
                "DANGER"
            )

        elif status == "WARNING":

            self.status_value.setText(
                "WARNING"
            )

        else:

            self.status_value.setText(
                "SAFE"
            )

    # ========================================================
    # OBJECT UPDATE
    # ========================================================

    def update_objects(self, objects):

        # Remove old widgets
        while self.object_layout.count():

            item = self.object_layout.takeAt(0)

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

        if not objects:

            self.empty_objects_label = QLabel(
                "No objects currently tracked"
            )

            self.empty_objects_label.setObjectName(
                "emptyLabel"
            )

            self.object_layout.addWidget(
                self.empty_objects_label
            )

            return

        for obj in objects:

            widget = QFrame()
            widget.setObjectName(
                "objectWidget"
            )

            layout = QVBoxLayout(widget)

            layout.setContentsMargins(
                12, 10, 12, 10
            )

            track_id = obj.get(
                "track_id",
                "?"
            )

            object_type = obj.get(
                "object_type",
                "person"
            )

            zone = obj.get(
                "zone",
                "UNKNOWN"
            )

            distance = obj.get(
                "distance",
                0
            )

            movement = obj.get(
                "movement_status",
                "UNKNOWN"
            )

            direction = obj.get(
                "movement_direction",
                "UNKNOWN"
            )

            title = QLabel(
                f"{object_type.title()} #{track_id}"
            )

            title.setObjectName(
                "objectTitle"
            )

            details = QLabel(
                f"Zone: {zone}   |   "
                f"Distance: {distance:.2f}   |   "
                f"Movement: {movement}"
            )

            details.setObjectName(
                "objectDetails"
            )

            direction_label = QLabel(
                f"Direction: {direction}"
            )

            direction_label.setObjectName(
                "objectDetails"
            )

            layout.addWidget(title)
            layout.addWidget(details)
            layout.addWidget(direction_label)

            self.object_layout.addWidget(
                widget
            )

        self.object_layout.addStretch()

    # ========================================================
    # EVENT UPDATE
    # ========================================================

    def update_events(self, events):

        while self.event_layout.count():

            item = self.event_layout.takeAt(0)

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

        if not events:

            empty = QLabel(
                "No recent safety events"
            )

            empty.setObjectName(
                "emptyLabel"
            )

            self.event_layout.addWidget(
                empty
            )

            return

        # Newest first
        for event in reversed(events[-10:]):

            widget = QFrame()
            widget.setObjectName(
                "eventWidget"
            )

            layout = QVBoxLayout(widget)

            layout.setContentsMargins(
                12, 10, 12, 10
            )

            event_type = event.get(
                "event_type",
                event.get(
                    "event",
                    "Safety Event"
                )
            )

            track_id = event.get(
                "track_id",
                "?"
            )

            timestamp = event.get(
                "timestamp",
                ""
            )

            title = QLabel(
                str(event_type)
            )

            title.setObjectName(
                "eventTitle"
            )

            details = QLabel(
                f"Object #{track_id}"
            )

            details.setObjectName(
                "eventDetails"
            )

            layout.addWidget(
                title
            )

            layout.addWidget(
                details
            )

            if timestamp:

                time_label = QLabel(
                    str(timestamp)
                )

                time_label.setObjectName(
                    "eventDetails"
                )

                layout.addWidget(
                    time_label
                )

            self.event_layout.addWidget(
                widget
            )

        self.event_layout.addStretch()

    # ========================================================
    # AUDIO
    # ========================================================

    def toggle_audio(self):

        enabled = (
            self.audio_button.isChecked()
        )

        self.backend.set_audio_enabled(
            enabled
        )

        if enabled:

            self.audio_button.setText(
                "Audio Alerts: ON"
            )

        else:

            self.audio_button.setText(
                "Audio Alerts: OFF"
            )

    # ========================================================
    # NAVIGATION
    # ========================================================

    def show_overview(self):

        self.overview_button.setChecked(
            True
        )

        self.events_button.setChecked(
            False
        )

        self.objects_button.setChecked(
            False
        )

        self.page_title.setText(
            "Safety Overview"
        )

        self.page_subtitle.setText(
            "Real-time monitoring and environmental awareness"
        )

        self.camera_card.show()
        self.objects_panel.show()
        self.events_panel.show()

    def show_events(self):

        self.overview_button.setChecked(
            False
        )

        self.events_button.setChecked(
            True
        )

        self.objects_button.setChecked(
            False
        )

        self.page_title.setText(
            "Safety Events"
        )

        self.page_subtitle.setText(
            "Recent safety activity detected by SafeWalk"
        )

        self.camera_card.hide()
        self.objects_panel.hide()
        self.events_panel.show()

    def show_objects(self):

        self.overview_button.setChecked(
            False
        )

        self.events_button.setChecked(
            False
        )

        self.objects_button.setChecked(
            True
        )

        self.page_title.setText(
            "Tracked Objects"
        )

        self.page_subtitle.setText(
            "Objects currently detected by the safety system"
        )

        self.camera_card.hide()
        self.events_panel.hide()
        self.objects_panel.show()

    # ========================================================
    # CAMERA ERROR
    # ========================================================

    def camera_error(self, message):

        print(
            "Camera worker error:",
            message
        )

        self.connection_status.setText(
            "Camera Error"
        )

        self.camera_connection.setText(
            "ERROR"
        )

        self.status_value.setText(
            "ERROR"
        )

        self.status_value.setProperty(
            "status",
            "danger"
        )

        self.status_value.style().unpolish(
            self.status_value
        )

        self.status_value.style().polish(
            self.status_value
        )

    # ========================================================
    # STYLING
    # ========================================================

    def apply_styles(self):

        self.setStyleSheet("""

        QMainWindow {
            background: #0f1117;
        }

        QWidget {
            font-family: "Segoe UI";
            color: #f5f5f5;
        }

        /* SIDEBAR */

        #sidebar {
            background: #11151d;
            border-right: 1px solid #242a34;
        }

        #logo {
            color: #ffffff;
            font-size: 23px;
            font-weight: 700;
        }

        #sidebarSubtitle {
            color: #697586;
            font-size: 11px;
        }

        #sidebarSectionTitle {
            color: #596474;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
            padding-top: 5px;
            padding-bottom: 5px;
        }

        QPushButton {
            background: transparent;
            border: none;
            border-radius: 8px;
            color: #8993a3;
            text-align: left;
            padding: 11px 13px;
            font-size: 13px;
        }

        QPushButton:hover {
            background: #1b222d;
            color: #ffffff;
        }

        QPushButton:checked {
            background: #202a37;
            color: #ffffff;
            font-weight: 600;
        }

        #monitorButton {
            background: #42d878;
            color: #07130b;
            border-radius: 9px;
            padding: 12px;
            font-weight: 700;
            text-align: center;
        }

        #monitorButton:hover {
            background: #55e587;
        }

        /* HEADER */

        #pageTitle {
            color: #f8fafc;
            font-size: 25px;
            font-weight: 700;
        }

        #pageSubtitle {
            color: #687586;
            font-size: 12px;
        }

        #connectionDot {
            background: #596474;
            border-radius: 5px;
        }

        #connectionStatus {
            color: #7b8797;
            font-size: 12px;
        }

        /* METRICS */

        #statCard {
            background: #151a22;
            border: 1px solid #252c36;
            border-radius: 12px;
        }

        #statTitle {
            color: #687586;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.7px;
        }

        #statValue {
            color: #42d878;
            font-size: 23px;
            font-weight: 700;
        }

        #statValue[status="safe"] {
            color: #42d878;
        }

        #statValue[status="warning"] {
            color: #f4c95d;
        }

        #statValue[status="danger"] {
            color: #ff6577;
        }

        /* CAMERA */

        #cameraCard {
            background: #111820;
            border: 1px solid #26313d;
            border-radius: 18px;
        }

        #cameraHeader {
            background: #151e27;
            border-top-left-radius: 18px;
            border-top-right-radius: 18px;
            border-bottom: 1px solid #222c36;
        }

        #cameraTitle {
            color: #f1f5f9;
            font-size: 16px;
            font-weight: 600;
        }

        #cameraOffline {
            color: #64748b;
            font-size: 11px;
            font-weight: 700;
        }

        #videoFrame {
            background: #080c11;
        }

        #cameraView {
            background: #080c11;
            color: #64748b;
            font-size: 15px;
            font-weight: 500;
        }

        #cameraFooter {
            background: #151e27;
            border-bottom-left-radius: 18px;
            border-bottom-right-radius: 18px;
            border-top: 1px solid #222c36;
        }

        #liveIndicator {
            color: #42d878;
            font-size: 11px;
            font-weight: 700;
        }

        #cameraInfo {
            color: #94a3b8;
            font-size: 11px;
        }

        /* PANELS */

        #panel {
            background: #151a22;
            border: 1px solid #252c36;
            border-radius: 12px;
        }

        #panelTitle {
            color: #f1f5f9;
            font-size: 14px;
            font-weight: 600;
        }

        /* OBJECTS */

        #objectWidget {
            background: #1a212b;
            border: 1px solid #28323e;
            border-radius: 8px;
        }

        #objectTitle {
            color: #e7edf5;
            font-size: 12px;
            font-weight: 600;
        }

        #objectDetails {
            color: #778496;
            font-size: 10px;
        }

        /* EVENTS */

        #eventWidget {
            background: #1a212b;
            border: 1px solid #28323e;
            border-radius: 8px;
        }

        #eventTitle {
            color: #e7edf5;
            font-size: 12px;
            font-weight: 600;
        }

        #eventDetails {
            color: #778496;
            font-size: 10px;
        }

        #emptyLabel {
            color: #566273;
            font-size: 11px;
            padding: 15px;
        }

        QScrollArea {
            background: transparent;
            border: none;
        }

        QScrollBar:vertical {
            background: #11151d;
            width: 6px;
            border-radius: 3px;
        }

        QScrollBar::handle:vertical {
            background: #343d4a;
            border-radius: 3px;
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }

        """)

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):

        print("Closing SafeWalk...")

        self.stop_camera()

        self.backend.release()

        event.accept()