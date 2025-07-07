import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSlot
# from made import TimeViewFeature
# from new import TimeViewFeature
# from gain import TimeViewFeature
# from hello import TimeViewFeature
# from updated import TimeViewFeature
# from freq import TimeViewFeature
# from modern import TimeViewFeature
from julyed import TimeViewFeature
# from okayed import TimeViewFeature
from mqtt_handler import MQTTHandler
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Filter Design")
        self.resize(1200, 800)  # Increased size for better plot visibility

        # Central widget with layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Initialize TimeViewFeature with model_name="model1"
        self.time_view = TimeViewFeature(self, model_name="model1")
        layout.addWidget(self.time_view.get_widget())



        # Initialize MQTTHandler
        try:
            self.mqtt_handler = MQTTHandler(broker="192.168.1.232")
            self.mqtt_handler.data_received.connect(self.time_view.on_data_received)
            self.mqtt_handler.connection_status.connect(self.on_mqtt_connection_status)
            self.mqtt_handler.start()
            # self.append_to_console("MQTT Handler initialized")
        except Exception as e:
            logging.error(f"Failed to initialize MQTT Handler: {str(e)}")
            # self.append_to_console(f"Error initializing MQTT Handler: {str(e)}")

    @pyqtSlot(str)
    def on_mqtt_connection_status(self, status):
        """Handle MQTT connection status updates."""
        # self.append_to_console(status)
        logging.info(status)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())