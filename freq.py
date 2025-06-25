import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
from pyqtgraph import PlotWidget, mkPen, AxisItem
from datetime import datetime
import time
import logging
import paho.mqtt.client as mqtt
import json
import struct
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeAxisItem(AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(v).strftime('%Y-%m-%d\n%H:%M:%S') for v in values]

class MQTTHandler(QObject):
    data_received = pyqtSignal(str, str, list, int)
    connection_status = pyqtSignal(str)

    def __init__(self, broker="192.168.1.232", port=1883):
        super().__init__()
        self.broker = broker
        self.port = port
        self.client = None
        self.connected = False
        self.topic = "sarayu/d1/topic1"
        self.model_name = "model1"
        logging.debug(f"Initializing MQTTHandler with broker={self.broker}, topic={self.topic}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            self.connection_status.emit("Connected to MQTT Broker")
            logging.info("Connected to MQTT Broker")
            QTimer.singleShot(0, self.subscribe_to_topic)
        else:
            self.connected = False
            self.connection_status.emit(f"Connection failed with code {rc}")
            logging.error(f"Failed to connect to MQTT Broker with code {rc}")

    def on_disconnect(self, client, userdata, _):
        self.connected = False
        self.connection_status.emit("Disconnected from MQTT Broker")
        logging.info("Disconnected from MQTT Broker")

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            if topic != self.topic:
                logging.debug(f"Ignoring message for topic {topic}, expected {self.topic}")
                return

            payload = msg.payload
            try:
                payload_str = payload.decode('utf-8')
                data = json.loads(payload_str)
                values = data.get("values", [])
                sample_rate = data.get("sample_rate", 4096)
                if not isinstance(values, list) or not values:
                    logging.warning(f"Invalid JSON payload format: {payload_str}")
                    return
                logging.debug(f"Parsed JSON payload: {len(values)} channels")
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload_length = len(payload)
                if payload_length < 2:
                    logging.warning(f"Invalid payload length: {payload_length} bytes")
                    return

                num_samples = payload_length // 2
                try:
                    values = struct.unpack(f">{num_samples}H", payload)
                except struct.error as e:
                    logging.error(f"Failed to unpack payload of {num_samples} uint16: {str(e)}")
                    return

                samples_per_channel = 4096
                num_channels = 4
                tacho_samples = 4096
                expected_total = samples_per_channel * num_channels + tacho_samples * 2

                if len(values) != expected_total:
                    logging.warning(f"Unexpected data length: got {len(values)}, expected {expected_total}")
                    return

                channel_data = [[] for _ in range(num_channels)]
                for i in range(0, samples_per_channel * num_channels, num_channels):
                    for ch in range(num_channels):
                        channel_data[ch].append(values[i + ch])
                tacho_freq_data = values[samples_per_channel * num_channels:samples_per_channel * num_channels + tacho_samples]
                tacho_trigger_data = values[samples_per_channel * num_channels + tacho_samples:]

                values = [[float(v) for v in ch] for ch in channel_data]
                values.append([float(v) for v in tacho_freq_data])
                values.append([float(v) for v in tacho_trigger_data])
                sample_rate = 4096

                logging.debug(f"Parsed binary payload: {num_channels} channels, {len(channel_data[0])} samples/channel")

            self.data_received.emit(self.topic, self.model_name, values, sample_rate)
            logging.debug(f"Emitted data for {self.topic}/{self.model_name}: {len(values)} channels, sample_rate={sample_rate}")

        except Exception as e:
            logging.error(f"Error processing MQTT message: {str(e)}")

    def subscribe_to_topic(self):
        try:
            self.client.subscribe(self.topic)
            logging.info(f"Subscribed to topic: {self.topic}")
        except Exception as e:
            logging.error(f"Error subscribing to topic: {str(e)}")
            self.connection_status.emit(f"Failed to subscribe to topic: {str(e)}")

    def start(self):
        try:
            self.client = mqtt.Client()
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            self.client.connect_async(self.broker, self.port, 60)
            self.client.loop_start()
            logging.info("MQTT client started")
        except Exception as e:
            logging.error(f"Failed to start MQTT client: {str(e)}")
            self.connection_status.emit(f"Failed to start MQTT: {str(e)}")

    def stop(self):
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
                logging.info("MQTT client stopped")
        except Exception as e:
            logging.error(f"Error stopping MQTT client: {str(e)}")

class DualRangeSlider(QWidget):
    range_changed = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.minimum = 0
        self.maximum = 4096
        self.lower_value = 0
        self.upper_value = 4096
        self.active_handle = None
        self.setMouseTracking(True)
        self.lower_label = QLabel("0")
        self.upper_label = QLabel("4096")
        layout = QHBoxLayout()
        layout.addWidget(self.lower_label)
        layout.addStretch()
        layout.addWidget(self.upper_label)
        self.setLayout(layout)
        self.setStyleSheet("QLabel { font-size: 12px; color: #2f3640; }")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        groove_y = self.height() // 2
        groove_x_start = 10
        groove_x_end = self.width() - 10
        groove_height = 8

        painter.setPen(QPen(QColor(187, 187, 187), 1))
        painter.setBrush(QColor(240, 240, 240))
        painter.drawRoundedRect(groove_x_start, groove_y - groove_height // 2, groove_x_end - groove_x_start, groove_height, 4, 4)

        lower_pos = self.value_to_position(self.lower_value)
        upper_pos = self.value_to_position(self.upper_value)
        painter.setBrush(QBrush(QColor(76, 175, 80)))
        painter.drawRect(lower_pos, groove_y - groove_height // 2, upper_pos - lower_pos, groove_height)

        painter.setBrush(QBrush(QColor(33, 150, 243)))
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.drawEllipse(QPoint(lower_pos, groove_y), 8, 8)
        painter.drawEllipse(QPoint(upper_pos, groove_y), 8, 8)

    def value_to_position(self, value):
        range_width = self.width() - 20
        return 10 + int((value - self.minimum) / (self.maximum - self.minimum) * range_width)

    def position_to_value(self, pos):
        range_width = self.width() - 20
        value = self.minimum + (pos - 10) / range_width * (self.maximum - self.minimum)
        return int(max(self.minimum, min(self.maximum, value)))

    def mousePressEvent(self, event):
        pos = event.pos().x()
        lower_pos = self.value_to_position(self.lower_value)
        upper_pos = self.value_to_position(self.upper_value)

        if abs(pos - lower_pos) < abs(pos - upper_pos):
            self.active_handle = 'lower'
        else:
            self.active_handle = 'upper'

        self.update_slider(event.pos().x())

    def mouseMoveEvent(self, event):
        if self.active_handle:
            self.update_slider(event.pos().x())

    def mouseReleaseEvent(self, event):
        self.active_handle = None

    def update_slider(self, pos):
        value = self.position_to_value(pos)
        if self.active_handle == 'lower':
            if value < self.upper_value - 100:
                self.lower_value = value
            else:
                self.lower_value = self.upper_value - 100
        elif self.active_handle == 'upper':
            if value > self.lower_value + 100:
                self.upper_value = value
            else:
                self.upper_value = self.lower_value + 100

        self.lower_label.setText(str(self.lower_value))
        self.upper_label.setText(str(self.upper_value))
        self.update()
        self.range_changed.emit(self.lower_value, self.upper_value)

class TimeViewFeature(QObject):
    def __init__(self, parent, channel=None, model_name="model1", console=None):
        super().__init__()
        self.parent = parent
        self.channel = channel
        self.model_name = model_name
        self.console = console
        self.topic = "sarayu/d1/topic1"
        self.widget = None
        self.plot_widgets = []
        self.plots = []
        self.data = [[] for _ in range(3)]  # Initialize for 3 plots
        self.channel_times = []
        self.sample_rate = 4096
        self.num_channels = 2
        self.scaling_factor = 3.3 / 65535
        self.num_plots = 3
        self.channel_samples = 4096
        self.tacho_samples = 4096
        self.vrms_label = None
        self.frequency_label = None
        self.start_button = None
        self.stop_button = None
        self.clear_gain_button = None
        self.is_plotting = True
        self.gain_vs_freq_data = {'gain': [], 'input_freq': []}
        self.y_range_fixed = True
        self.input_freq_range = [0, 1000]
        self.y_range_3v_button = None
        self.y_range_auto_button = None
        self.default_button = None
        self.data_range_start = 0
        self.data_range_end = 4096
        self.range_slider = None
        self.freq_buttons = []

        self.initUI()
        self.connect_buttons()
        logging.debug("TimeViewFeature initialized with plotting enabled")

    def initUI(self):
        self.widget = QWidget()
        self.widget.setStyleSheet("background-color: #f5f6fa;")
        main_layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none; background: transparent;")
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)

        colors = ['r', 'g', 'b']
        for i in range(self.num_plots):
            axis_items = {'bottom': TimeAxisItem(orientation='bottom') if i < 2 else AxisItem(orientation='bottom')}
            plot_widget = PlotWidget(axisItems=axis_items, background='w')
            plot_widget.setFixedHeight(250)
            plot_widget.setMinimumWidth(600)
            plot_widget.setStyleSheet("border: 1px solid #dfe4ea; border-radius: 8px; padding: 5px;")
            if i < self.num_channels:
                plot_widget.setLabel('left', f'CH{i+1} Value (V)')
                plot_widget.setYRange(0, 3.0, padding=0)
            elif i == 2:
                plot_widget.setLabel('left', 'Gain (dB)')
                plot_widget.setLabel('bottom', 'Input Frequency (Hz)')
                plot_widget.setYRange(10, -85, padding=0)
                plot_widget.setXRange(0, 1000, padding=0)
            plot_widget.showGrid(x=True, y=True)
            plot_widget.addLegend()
            pen = mkPen(color=colors[i % len(colors)], width=2)
            plot = plot_widget.plot([], [], pen=pen, name=f'Plot {i+1}')
            self.plots.append(plot)
            self.plot_widgets.append(plot_widget)
            scroll_layout.addWidget(plot_widget)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        bottom_layout = QVBoxLayout()

        self.range_slider = DualRangeSlider()
        range_label = QLabel("Data Range Selection:")
        range_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2f3640;")
        bottom_layout.addWidget(range_label)
        bottom_layout.addWidget(self.range_slider)

        freq_layout = QHBoxLayout()
        freq_layout.setSpacing(8)
        freq_label = QLabel("Frequency Range (Hz):")
        freq_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2f3640;")
        freq_layout.addWidget(freq_label)
        for freq in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            btn = QPushButton(f"{freq} Hz")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF5722;
                    color: white;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #E64A19; }
                QPushButton:pressed { background-color: #D84315; }
            """)
            freq_layout.addWidget(btn)
            self.freq_buttons.append(btn)
        bottom_layout.addLayout(freq_layout)

        self.vrms_label = QLabel("Channel 1 Vrms: N/A | Channel 2 Vrms: N/A")
        self.vrms_label.setStyleSheet("font-size: 14px; color: #2f3640; background-color: #dfe4ea; padding: 8px; border-radius: 6px;")
        self.frequency_label = QLabel("Tacho Frequency: N/A")
        self.frequency_label.setStyleSheet("font-size: 14px; color: #2f3640; background-color: #dfe4ea; padding: 8px; border-radius: 6px;")
        bottom_layout.addWidget(self.vrms_label)
        bottom_layout.addWidget(self.frequency_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.setSpacing(10)
        self.start_button = QPushButton("Start MQTT")
        self.stop_button = QPushButton("Stop MQTT")
        self.clear_gain_button = QPushButton("Clear Gain Plot")
        self.y_range_3v_button = QPushButton("3V Range")
        self.y_range_auto_button = QPushButton("Auto Y Range")
        self.default_button = QPushButton("Default Settings")

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.clear_gain_button.setEnabled(True)
        self.y_range_auto_button.setEnabled(False)

        button_style = """
            QPushButton {
                background-color: %s;
                color: %s;
                padding: 10px 15px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover { background-color: %s; }
            QPushButton:pressed { background-color: %s; }
            QPushButton:disabled { background-color: #dcdde1; color: #7f8c8d; }
        """
        self.start_button.setStyleSheet(button_style % ("#4CAF50", "white", "#45a049", "#3d8b40"))
        self.stop_button.setStyleSheet(button_style % ("#f44336", "white", "#e53935", "#d32f2f"))
        self.clear_gain_button.setStyleSheet(button_style % ("#2196F3", "white", "#1e88e5", "#1976d2"))
        self.y_range_3v_button.setStyleSheet(button_style % ("#FFC107", "black", "#FFB300", "#FFA000"))
        self.y_range_auto_button.setStyleSheet(button_style % ("#FFC107", "black", "#FFB300", "#FFA000"))
        self.default_button.setStyleSheet(button_style % ("#9C27B0", "white", "#8e24aa", "#7b1fa2"))

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.clear_gain_button)
        button_layout.addWidget(self.y_range_3v_button)
        button_layout.addWidget(self.y_range_auto_button)
        button_layout.addWidget(self.default_button)

        bottom_layout.addLayout(button_layout)
        main_layout.addLayout(bottom_layout)
        self.widget.setLayout(main_layout)

        if not self.model_name and self.console:
            self.console.append_to_console("No model selected in TimeViewFeature.")
        if not self.channel and self.console:
            self.console.append_to_console("No channel selected in TimeViewFeature.")

    def connect_buttons(self):
        self.start_button.clicked.connect(self.start_plotting)
        self.stop_button.clicked.connect(self.stop_plotting)
        self.clear_gain_button.clicked.connect(self.clear_gain_vs_freq_plot)
        self.y_range_3v_button.clicked.connect(self.set_3v_range)
        self.y_range_auto_button.clicked.connect(self.set_auto_y_range)
        self.default_button.clicked.connect(self.set_default_settings)
        self.range_slider.range_changed.connect(self.update_data_range)
        for idx, btn in enumerate(self.freq_buttons):
            freq = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000][idx]
            btn.clicked.connect(lambda checked, f=freq: self.set_input_freq_range(f))
        logging.debug("Button signals connected")

    def start_plotting(self):
        self.is_plotting = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.clear_gain_button.setEnabled(True)
        logging.info("Started plotting MQTT data")
        if self.console:
            self.console.append_to_console("Started plotting MQTT data")

    def stop_plotting(self):
        self.is_plotting = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.clear_gain_button.setEnabled(True)
        logging.info("Stopped plotting MQTT data")
        if self.console:
            self.console.append_to_console("Stopped plotting MQTT data")

    def clear_gain_vs_freq_plot(self):
        self.gain_vs_freq_data['gain'] = []
        self.gain_vs_freq_data['input_freq'] = []
        self.plots[2].setData([], [])
        self.plot_widgets[2].setXRange(self.input_freq_range[0], self.input_freq_range[1], padding=0)
        self.plot_widgets[2].setYRange(10, -85, padding=0)
        logging.info("Cleared Gain vs Input Frequency plot")
        if self.console:
            self.console.append_to_console("Cleared Gain vs Input Frequency plot")

    def set_3v_range(self):
        self.y_range_fixed = True
        self.y_range_3v_button.setEnabled(False)
        self.y_range_auto_button.setEnabled(True)
        for i in range(self.num_channels):
            self.plot_widgets[i].setYRange(0, 3.0, padding=0)
        self.plot_widgets[2].setYRange(10, -85, padding=0)
        logging.info("Set Y-axis to 0-3V for Channel 1 and 2")
        if self.console:
            self.console.append_to_console("Set Y-axis to 0-3V for Channel 1 and 2")

    def set_auto_y_range(self):
        self.y_range_fixed = False
        self.y_range_3v_button.setEnabled(True)
        self.y_range_auto_button.setEnabled(False)
        for i in range(self.num_channels):
            self.plot_widgets[i].enableAutoRange(axis='y')
        self.plot_widgets[2].enableAutoRange(axis='y')
        logging.info("Enabled auto Y-axis range for Channel 1 and 2")
        if self.console:
            self.console.append_to_console("Enabled auto Y-axis range for Channel 1 and 2")

    def set_input_freq_range(self, freq):
        self.input_freq_range = [0, freq]
        self.plot_widgets[2].setXRange(0, freq, padding=0)
        self.apply_ranges()
        logging.info(f"Set Input Frequency range to 0-{freq}Hz")
        if self.console:
            self.console.append_to_console(f"Set Input Frequency range to 0-{freq}Hz")

    def update_data_range(self, start, end):
        self.data_range_start = start
        self.data_range_end = end
        self.apply_data_range()
        logging.info(f"Data range updated to {start}-{end}")
        if self.console:
            self.console.append_to_console(f"Data range updated to {start}-{end}")

    def apply_data_range(self):
        for ch in range(self.num_channels):
            if len(self.data[ch]) > 0:
                sliced_data = self.data[ch][self.data_range_start:self.data_range_end]
                sliced_times = self.channel_times[self.data_range_start:self.data_range_end]
                self.plots[ch].setData(sliced_times, sliced_data)
                if len(sliced_times) > 0:
                    self.plot_widgets[ch].setXRange(sliced_times[0], sliced_times[-1], padding=0)
        if len(self.data[0]) > 0 and len(self.data[1]) > 0 and len(self.data[2]) > 0:
            sliced_data_ch1 = self.data[0][self.data_range_start:self.data_range_end]
            sliced_data_ch2 = self.data[1][self.data_range_start:self.data_range_end]
            if len(sliced_data_ch1) > 0 and len(sliced_data_ch2) > 0:
                vrms_ch1 = self.calculate_vrms(sliced_data_ch1)
                vrms_ch2 = self.calculate_vrms(sliced_data_ch2)
                tacho_freq = self.calculate_frequency(self.data[2])  # Use tacho frequency
                self.vrms_label.setText(f"Channel 1 Vrms: {vrms_ch1:.2f} V | Channel 2 Vrms: {vrms_ch2:.2f} V")
                self.frequency_label.setText(f"Tacho Frequency: {tacho_freq:.2f}")
                gain_db = self.calculate_gain(vrms_ch1, vrms_ch2)
                if tacho_freq > 0 and self.input_freq_range[0] <= tacho_freq <= self.input_freq_range[1]:
                    if len(self.gain_vs_freq_data['input_freq']) == 0 or self.gain_vs_freq_data['input_freq'][-1] != tacho_freq:
                        self.gain_vs_freq_data['gain'].append(gain_db)
                        self.gain_vs_freq_data['input_freq'].append(tacho_freq)
                        logging.debug(f"Appended gain: {gain_db:.2f} dB at {tacho_freq:.2f} Hz")
                    self.apply_ranges()

    def apply_ranges(self):
        filtered_gain = []
        filtered_freq = []
        for g, freq in zip(self.gain_vs_freq_data['gain'], self.gain_vs_freq_data['input_freq']):
            if self.input_freq_range[0] <= freq <= self.input_freq_range[1]:
                filtered_gain.append(g)
                filtered_freq.append(freq)
        if len(filtered_freq) > 0:
            self.plots[2].setData(filtered_freq, filtered_gain)
        else:
            self.plots[2].setData([], [])
        self.plot_widgets[2].setYRange(10, -85, padding=0)
        self.plot_widgets[2].setXRange(self.input_freq_range[0], self.input_freq_range[1], padding=0)

        logging.debug(f"Applied ranges: Input Freq {self.input_freq_range}, {len(filtered_freq)} points plotted")
        if self.console:
            self.console.append_to_console(f"Applied ranges: Input Freq {self.input_freq_range}")

    def set_default_settings(self):
        self.y_range_fixed = True
        self.y_range_3v_button.setEnabled(False)
        self.y_range_auto_button.setEnabled(True)
        for i in range(self.num_channels):
            self.plot_widgets[i].setYRange(0, 3.0, padding=0)
        self.plot_widgets[2].setYRange(10, -85, padding=0)
        self.input_freq_range = [0, 1000]
        self.data_range_start = 0
        self.data_range_end = 4096
        self.range_slider.lower_value = 0
        self.range_slider.upper_value = 4096
        self.range_slider.lower_label.setText("0")
        self.range_slider.upper_label.setText("4096")
        self.gain_vs_freq_data['gain'] = []
        self.gain_vs_freq_data['input_freq'] = []
        self.apply_ranges()
        self.apply_data_range()
        logging.info("Default settings applied")
        if self.console:
            self.console.append_to_console("Default settings applied")

    def get_widget(self):
        return self.widget

    def calculate_vrms(self, data):
        if len(data) == 0:
            return 0.0
        vmax = np.max(data)
        vmin = np.min(data)
        return vmax - vmin

    def calculate_frequency(self, data):
        if len(data) == 0:
            return 0.0
        # Return the mean of tacho frequency data after scaling
        return np.mean(data) * 100.0  # Reverse the /100 scaling applied earlier

    def calculate_gain(self, vrms_ch1, vrms_ch2):
        if vrms_ch1 == 0 or vrms_ch2 == 0:
            return 0.0
        gain_db = 20 * np.log10(vrms_ch2 / vrms_ch1)
        return gain_db

    def on_data_received(self, tag_name, model_name, values, sample_rate):
        if not self.is_plotting:
            logging.debug("Plotting disabled, ignoring MQTT data")
            return
        if tag_name != self.topic:
            logging.debug(f"Ignoring data for topic {tag_name}, expected {self.topic}")
            return
        if self.model_name != model_name:
            logging.debug(f"Ignoring data for model {model_name}, expected {self.model_name}")
            return

        logging.info(f"Processing MQTT data: {len(values)} channels, sample_rate={sample_rate}")
        try:
            if not values or len(values) != 6:
                logging.error(f"Received incorrect number of sublists: {len(values)}, expected 6")
                if self.console:
                    self.console.append_to_console(f"Error: Received incorrect number of values: {len(values)}")
                return

            self.sample_rate = sample_rate
            self.channel_samples = 4096
            self.tacho_samples = 4096

            for ch in range(4):
                if len(values[ch]) != self.channel_samples:
                    logging.error(f"Channel {ch+1} has {len(values[ch])} samples, expected {self.channel_samples}")
                    if self.console:
                        self.console.append_to_console(f"Error: Channel {ch+1} samples {len(values[ch])}")
                    return
            if len(values[4]) != self.tacho_samples or len(values[5]) != self.tacho_samples:
                logging.error(f"Tacho data length mismatch: freq={len(values[4])}, trigger={len(values[5])}, expected={self.tacho_samples}")
                if self.console:
                    self.console.append_to_console(f"Error: Tacho data length mismatch: freq={len(values[4])}, trigger={len(values[5])}")
                return

            current_time = time.time()
            channel_time_step = 1.0 / self.sample_rate
            self.channel_times = np.array([current_time - (self.channel_samples - 1 - i) * channel_time_step for i in range(self.channel_samples)])

            self.data[0] = np.array(values[0][:self.channel_samples]) * self.scaling_factor
            self.data[1] = np.array(values[1][:self.channel_samples]) * self.scaling_factor
            self.data[2] = np.array(values[4][:self.tacho_samples]) / 100.0

            self.apply_data_range()

            logging.info(f"Updated {self.num_plots} plots with {self.channel_samples} samples")
            if self.console:
                self.console.append_to_console(f"Time View ({self.model_name}): Updated {self.num_plots} plots with {self.channel_samples} samples")

        except Exception as e:
            logging.error(f"Error updating plots: {str(e)}")
            if self.console:
                self.console.append_to_console(f"Error updating plots: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    time_view = TimeViewFeature(None)
    main_window.setCentralWidget(time_view.get_widget())
    mqtt_handler = MQTTHandler()
    mqtt_handler.data_received.connect(time_view.on_data_received)
    mqtt_handler.start()
    main_window.show()
    sys.exit(app.exec_())