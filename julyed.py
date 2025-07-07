import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QPushButton, QSlider, QLineEdit
from PyQt5.QtGui import QFont

from PyQt5.QtCore import QObject, Qt, pyqtSignal, QTimer
from pyqtgraph import PlotWidget, mkPen, AxisItem
from datetime import datetime
import time
import logging
import paho.mqtt.client as mqtt
import json
import struct
import sys
import uuid

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeAxisItem(AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(v).strftime('%Y-%m-%d\n%H:%M:%S') for v in values]

class MQTTHandler(QObject):
    data_received = pyqtSignal(str, str, list, int, int)
    connection_status = pyqtSignal(str)

    def __init__(self, broker="192.168.1.232", port=1883):
        super().__init__()
        self.broker = broker
        self.port = port
        self.client = None
        self.connected = False
        self.topic = "sarayu/d1/topic1"
        self.model_name = "model1"
        logging.debug(f"Initializing MQTTHandler with broker: {broker}, topic: {self.topic}")

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

    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        self.connection_status.emit("Disconnected from MQTT Broker")
        logging.info("Disconnected from MQTT Broker")

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload

            if topic != self.topic:
                logging.debug(f"Ignoring message for topic {topic}, expected {self.topic}")
                return

            try:
                payload_str = payload.decode('utf-8')
                data = json.loads(payload_str)
                values = data.get("values", [])
                sample_rate = data.get("sample_rate", 0)
                samples_per_channel = data.get("samples_per_channel", 0)
                if not isinstance(values, list) or not values or sample_rate <= 0 or samples_per_channel <= 0:
                    logging.warning(f"Invalid JSON payload format: {payload_str}")
                    return
                logging.debug(f"Parsed JSON payload: {len(values)} channels, sample_rate={sample_rate}, samples_per_channel={samples_per_channel}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload_length = len(payload)
                if payload_length < 20 or payload_length % 2 != 0:
                    logging.warning(f"Invalid payload length: {payload_length} bytes")
                    return

                num_samples = payload_length // 2
                try:
                    values = struct.unpack(f"<{num_samples}H", payload)
                except struct.error as e:
                    logging.error(f"Failed to unpack payload of {num_samples} int16_t: {str(e)}")
                    return

                if len(values) < 100:
                    logging.warning(f"Payload too short: {len(values)} samples")
                    return

                header = values[:100]
                total_values = values[100:]

                num_channels = header[2] if len(header) > 2 and header[2] > 0 else 4
                sample_rate = header[3] if len(header) > 3 and header[3] > 0 else 1000
                samples_per_channel = header[4] if len(header) > 4 and header[4] > 0 else 1000
                num_tacho_channels = header[6] if len(header) > 6 and header[6] > 0 else 2

                expected_main = samples_per_channel * num_channels
                expected_tacho = samples_per_channel * num_tacho_channels
                expected_total = expected_main + expected_tacho

                if len(total_values) != expected_total:
                    logging.warning(f"Unexpected data length: got {len(total_values)}, expected {expected_total}")
                    return

                main_data = total_values[:expected_main]
                tacho_freq_data = total_values[expected_main:expected_main + samples_per_channel]
                tacho_trigger_data = total_values[expected_main + samples_per_channel:expected_main + 2 * samples_per_channel]

                channel_data = [[] for _ in range(num_channels)]
                for i in range(0, len(main_data), num_channels):
                    for ch in range(num_channels):
                        channel_data[ch].append(main_data[i + ch])

                values = [[float(v) for v in ch] for ch in channel_data]
                values.append([float(v) for v in tacho_freq_data])
                values.append([float(v) for v in tacho_trigger_data])

                logging.debug(f"Parsed binary payload: {num_channels} channels, {len(channel_data[0])} samples/channel")
                logging.debug(f"Tacho freq (first 5): {tacho_freq_data[:5]}")
                logging.debug(f"Tacho trigger (first 20): {tacho_trigger_data[:20]}")

            self.data_received.emit(self.topic, self.model_name, values, sample_rate, samples_per_channel)
            logging.debug(f"Emitted data for {self.topic}/{self.model_name}: {len(values)} channels, sample_rate={sample_rate}, samples_per_channel={samples_per_channel}")

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
            self.client = mqtt.Client(client_id=f"client-{uuid.uuid4()}")
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

class RangeSlider(QWidget):
    range_changed = pyqtSignal(int, int)

    def __init__(self, max_samples=1000, parent=None):
        super().__init__(parent)
        self.max_samples = max_samples
        self.layout = QHBoxLayout()
        self.min_slider = QSlider(Qt.Horizontal)
        self.max_slider = QSlider(Qt.Horizontal)

        for slider in [self.min_slider, self.max_slider]:
            slider.setMinimum(0)
            slider.setMaximum(max_samples)
            slider.setMinimumWidth(200)

        self.min_slider.setValue(0)
        self.max_slider.setValue(max_samples)
        self.min_value_label = QLabel(f"{self.min_slider.value()}")
        self.max_value_label = QLabel(f"{self.max_slider.value()}")

        label=QLabel("Min::")
        font = QFont()
        font.setPointSize(10)  # This uses points, not pixels
        label.setFont(font)
        self.layout.addWidget(label)
        self.layout.addWidget(self.min_slider)
        self.layout.addWidget(self.min_value_label)
        # self.layout.addWidget(QLabel("Max:"))
        label=QLabel("Max::")
        font = QFont()
        font.setPointSize(10)  # This uses points, not pixels
        label.setFont(font)
        self.layout.addWidget(label)

        self.layout.addWidget(self.max_slider)
        self.layout.addWidget(self.max_value_label)
        self.setLayout(self.layout)

        self.min_slider.valueChanged.connect(self.update_range)
        self.max_slider.valueChanged.connect(self.update_range)

    def update_maximum(self, max_samples):
        self.max_samples = max_samples
        for slider in [self.min_slider, self.max_slider]:
            slider.setMaximum(max_samples)
        self.min_value_label.setText(str(self.min_slider.value()))
        self.max_value_label.setText(str(self.max_slider.value()))

    def update_range(self):
        min_val = self.min_slider.value()
        max_val = self.max_slider.value()
        self.min_value_label.setText(str(min_val))
        self.max_value_label.setText(str(max_val))
        self.range_changed.emit(min_val, max_val)

class TimeViewFeature:
    def __init__(self, parent, channel=None, model_name="model1", console=None):
        self.parent = parent
        self.channel = channel
        self.model_name = model_name
        self.console = console
        self.widget = None
        self.plot_widgets = []
        self.plots = []
        self.data = []
        self.channel_times = []
        self.sample_rate = 1000
        self.num_channels = 2
        self.scaling_factor = 3.3 / 65535
        self.num_plots = 3
        self.samples_per_channel = 1000
        self.tacho_samples = 1000
        self.vrms_ch1 = None
        self.vrms_ch2 = None
        self.frequency_ch2 = None
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
        self.data_range_end = 1000
        self.range_slider = None
        self.freq_input = None
        self.set_freq_button = None
        self.initUI()
        self.connect_buttons()
        logging.debug("TimeViewFeature initialized with plotting enabled")

    def initUI(self):
        self.widget = QWidget()
        main_layout = QHBoxLayout()

        content_widget = QWidget()
        content_layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        colors = ['r', 'g', 'b']
        channel_labels = ['CH1 Input (V)', 'CH2 Output (V)']

        for i in range(self.num_plots):
            axis_items = {'bottom': TimeAxisItem(orientation='bottom') if i < 2 else AxisItem(orientation='bottom')}
            plot_widget = PlotWidget(axisItems=axis_items, background='w')
            plot_widget.setFixedHeight(250)
            plot_widget.setMinimumWidth(250)
            if i < len(channel_labels):
                plot_widget.setLabel('left', channel_labels[i])
                plot_widget.setYRange(0, 3.0, padding=10)
            elif i == 2:
                plot_widget.setLabel('left', 'Gain (dB)')
                plot_widget.setLabel('bottom', 'Input Frequency (Hz)')
                plot_widget.setYRange(10, -85, padding=10)
                plot_widget.setXRange(0, 1000, padding=10)
            plot_widget.showGrid(x=True, y=True)
            plot_widget.addLegend()
            pen = mkPen(color=colors[i % len(colors)], width=2)
            plot = plot_widget.plot([], [], pen=pen, name=f'Plot {i+1}')
            self.plots.append(plot)
            self.data.append([])
            self.plot_widgets.append(plot_widget)
            scroll_layout.addWidget(plot_widget)

        scroll_area.setWidget(scroll_content)
        content_layout.addWidget(scroll_area)

        bottom_controls = QWidget()
        control_layout = QVBoxLayout()

        self.range_slider = RangeSlider(max_samples=self.samples_per_channel)
        # control_layout.addWidget(QLabel("Data Range Selection:"))
        label=QLabel("Data Range Selection::")
        font = QFont()
        font.setPointSize(10)  # This uses points, not pixels
        label.setFont(font)
        control_layout.addWidget(label)
        control_layout.addWidget(self.range_slider)

        freq_layout = QHBoxLayout()
        # freq_layout.addWidget(QLabel("Frequency Range (Hz):"))
        label = QLabel("Frequency Range (Hz):")
        font = QFont()
        font.setPointSize(10)  # This uses points, not pixels
        label.setFont(font)
        freq_layout.addWidget(label)
        self.freq_input = QLineEdit("1000")
        self.freq_input.setPlaceholderText("Enter frequency (Hz)")
        self.freq_input.setFixedWidth(200)
        self.freq_input.setFixedHeight(50)
        self.set_freq_button = QPushButton("Set Frequency")
        # self.set_freq_button.setStyleSheet("background-color: #FF5722; color: white; padding: 10px; border-radius: 4px;")
        self.set_freq_button.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;  /* Deep orange */
                color: white;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: #BF360C;  /* Much darker deep orange */
            }
        """)

        freq_layout.addWidget(self.freq_input)
        freq_layout.addWidget(self.set_freq_button)
        freq_layout.addStretch()
        control_layout.addLayout(freq_layout)
        info_layout = QHBoxLayout()

        self.vrms_label = QLabel("INPUT Vrms: N/A | OUTPUT 2 Vrms: N/A")
        self.vrms_label.setStyleSheet("font-weight: bold; font-size: 10pt;")

        self.frequency_label = QLabel("OUTPUT FREQUENCY: N/A")
        self.frequency_label.setStyleSheet("font-weight: bold; font-size: 10pt;")

        # Add both labels to the horizontal layout
        info_layout.addWidget(self.vrms_label)
        info_layout.addWidget(self.frequency_label)

        # Add the horizontal layout to your control_layout
        control_layout.addLayout(info_layout)


        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start MQTT Plotting")
        self.stop_button = QPushButton("Stop MQTT Plotting")
        self.clear_gain_button = QPushButton("Clear Gain vs Input Freq Plot")
        self.y_range_3v_button = QPushButton("Set 3V Range")
        self.y_range_auto_button = QPushButton("Auto Y Range")
        self.default_button = QPushButton("Default Settings")

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.clear_gain_button.setEnabled(True)
        self.y_range_auto_button.setEnabled(False)

        # self.start_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 4px;")
        # self.stop_button.setStyleSheet("background-color: #f44336; color: white; padding: 10px; border-radius: 4px;")
        # self.clear_gain_button.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; border-radius: 4px;")
        # self.y_range_3v_button.setStyleSheet("background-color: #FFC107; color: black; padding: 10px; border-radius: 4px;")
        # self.y_range_auto_button.setStyleSheet("background-color: #FFC107; color: black; padding: 10px; border-radius: 4px;")
        # self.default_button.setStyleSheet("background-color: #9C27B0; color: white; padding: 10px; border-radius: 4px;")

        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: #2E7D32;  /* Much darker green */
            }
        """)

        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: #B71C1C;  /* Dark red */
            }
        """)

        self.clear_gain_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: #0D47A1;  /* Dark blue */
            }
        """)

        self.y_range_3v_button.setStyleSheet("""
            QPushButton {
                background-color: #FFC107;
                color: black;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: #FF6F00;  /* Dark amber */
            }
        """)

        self.y_range_auto_button.setStyleSheet("""
            QPushButton {
                background-color: #FFC107;
                color: black;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: #FF6F00;
            }
        """)

        self.default_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: #4A148C;  /* Dark purple */
            }
        """)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.clear_gain_button)
        button_layout.addWidget(self.y_range_3v_button)
        button_layout.addWidget(self.y_range_auto_button)
        button_layout.addWidget(self.default_button)

        control_layout.addLayout(button_layout)
        bottom_controls.setLayout(control_layout)
        content_layout.addWidget(bottom_controls)

        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)
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
        self.set_freq_button.clicked.connect(self.set_input_freq_from_input)
        logging.debug("Button signals connected")

    def start_plotting(self):
        self.is_plotting = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.clear_gain_button.setEnabled(True)
        logging.debug("Started plotting MQTT data")
        if self.console:
            self.console.append_to_console("Started plotting MQTT data")

    def stop_plotting(self):
        self.is_plotting = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.clear_gain_button.setEnabled(True)
        logging.debug("Stopped plotting MQTT data")
        if self.console:
            self.console.append_to_console("Stopped plotting MQTT data")

    def clear_gain_vs_freq_plot(self):
        self.gain_vs_freq_data['gain'] = []
        self.gain_vs_freq_data['input_freq'] = []
        self.plots[2].setData([], [])
        self.plot_widgets[2].setXRange(self.input_freq_range[0], self.input_freq_range[1], padding=0)
        self.plot_widgets[2].setYRange(10, -85, padding=0)
        logging.debug("Cleared Gain vs Input Frequency plot")
        if self.console:
            self.console.append_to_console("Cleared Gain vs Input Frequency plot")

    def set_3v_range(self):
        self.y_range_fixed = True
        self.y_range_3v_button.setEnabled(False)
        self.y_range_auto_button.setEnabled(True)
        for i in range(self.num_channels):
            self.plot_widgets[i].setYRange(0, 3.0, padding=0)
        self.plot_widgets[2].setYRange(10, -85, padding=0)
        logging.debug("Set Y-axis to 0-3V for Channel 1 and 2")
        if self.console:
            self.console.append_to_console("Set Y-axis to 0-3V for Channel 1 and 2")

    def set_auto_y_range(self):
        self.y_range_fixed = False
        self.y_range_3v_button.setEnabled(True)
        self.y_range_auto_button.setEnabled(False)
        for i in range(self.num_channels):
            self.plot_widgets[i].enableAutoRange(axis='y')
        self.plot_widgets[2].enableAutoRange(axis='y')
        logging.debug("Enabled auto Y-axis range for Channel 1 and 2")
        if self.console:
            self.console.append_to_console("Enabled auto Y-axis range for Channel 1 and 2")

    def set_input_freq_from_input(self):
        try:
            freq = float(self.freq_input.text())
            if freq <= 0:
                raise ValueError("Frequency must be positive")
            self.input_freq_range = [0, freq]
            self.plot_widgets[2].setXRange(0, freq, padding=0)
            self.apply_ranges()
            logging.debug(f"Set Input Frequency range to 0-{freq}Hz")
            if self.console:
                self.console.append_to_console(f"Set Input Frequency range to 0-{freq}Hz")
        except ValueError as e:
            logging.error(f"Invalid frequency input: {str(e)}")
            if self.console:
                self.console.append_to_console(f"Invalid frequency input: {str(e)}")

    def update_data_range(self, start, end):
        self.data_range_start = start
        self.data_range_end = end
        self.apply_data_range()
        logging.debug(f"Data range updated to {start}-{end}")
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
        if len(self.data[0]) > 0 and len(self.data[1]) > 0:
            sliced_data_ch1 = self.data[0][self.data_range_start:self.data_range_end]
            sliced_data_ch2 = self.data[1][self.data_range_start:self.data_range_end]
            if len(sliced_data_ch1) > 0 and len(sliced_data_ch2) > 0:
                vrms_ch1 = self.calculate_vrms(sliced_data_ch1)
                vrms_ch2 = self.calculate_vrms(sliced_data_ch2)
                frequency_ch2 = self.calculate_frequency(sliced_data_ch2, self.sample_rate)
                input_frequency = self.calculate_frequency(sliced_data_ch1, self.sample_rate)
                gain_db = self.calculate_gain(vrms_ch1, vrms_ch2)
                # self.vrms_label.setText(f"Channel 1 Vrms: {vrms_ch1:.2f} V | Channel 2 Vrms: {vrms_ch2:.2f} V")
                self.vrms_label.setText(f"INPUT Vrms: {vrms_ch1:.2f} V | OUTPUT 2 Vrms: {vrms_ch2:.2f} V")
                
                
                self.frequency_label.setText(f"Channel 2 Frequency: {frequency_ch2:.2f} Hz")
                if input_frequency > 0 and self.input_freq_range[0] <= input_frequency <= self.input_freq_range[1]:
                    if len(self.gain_vs_freq_data['input_freq']) == 0 or self.gain_vs_freq_data['input_freq'][-1] != input_frequency:
                        self.gain_vs_freq_data['gain'].append(gain_db)
                        self.gain_vs_freq_data['input_freq'].append(input_frequency)
                        logging.debug(f"Appended Gain: {gain_db:.2f} dB at Input Freq: {input_frequency:.2f} Hz")
                    self.apply_ranges()

    def apply_ranges(self):
        filtered_gain = []
        filtered_input_freq = []
        for g, f in zip(self.gain_vs_freq_data['gain'], self.gain_vs_freq_data['input_freq']):
            if self.input_freq_range[0] <= f <= self.input_freq_range[1]:
                filtered_gain.append(g)
                filtered_input_freq.append(f)
        if len(filtered_input_freq) > 0:
            self.plots[2].setData(filtered_input_freq, filtered_gain)
        else:
            self.plots[2].setData([], [])
        self.plot_widgets[2].setXRange(self.input_freq_range[0], self.input_freq_range[1], padding=0)
        self.plot_widgets[2].setYRange(10, -85, padding=0)
        logging.debug(f"Applied ranges: Input Freq {self.input_freq_range}, {len(filtered_input_freq)} points plotted")
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
        self.data_range_end = self.samples_per_channel
        self.range_slider.min_slider.setValue(0)
        self.range_slider.max_slider.setValue(self.samples_per_channel)
        self.gain_vs_freq_data['gain'] = []
        self.gain_vs_freq_data['input_freq'] = []
        self.freq_input.setText("1000")
        self.apply_ranges()
        self.apply_data_range()
        logging.debug("Default settings applied")
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

    def calculate_frequency(self, data, sample_rate):
        if len(data) == 0:
            return 0.0
        data = data - np.mean(data)
        n = len(data)
        fft_result = np.fft.fft(data)
        freqs = np.fft.fftfreq(n, d=1/sample_rate)
        magnitudes = np.abs(fft_result)
        positive_freqs = freqs[:n//2]
        positive_magnitudes = magnitudes[:n//2]
        positive_magnitudes[0] = 0
        dominant_freq_idx = np.argmax(positive_magnitudes)
        dominant_freq = positive_freqs[dominant_freq_idx]
        return abs(dominant_freq)

    def calculate_gain(self, vrms_ch1, vrms_ch2):
        if vrms_ch1 == 0 or vrms_ch2 == 0:
            return 0.0
        gain_db = 20 * np.log10(vrms_ch2 / vrms_ch1)
        return gain_db

    def on_data_received(self, tag_name, model_name, values, sample_rate, samples_per_channel):
        if not self.is_plotting:
            logging.debug("Plotting is disabled, ignoring MQTT data")
            return
        if tag_name != "sarayu/d1/topic1":
            logging.debug(f"Ignoring data for topic {tag_name}, expected sarayu/d1/topic1")
            return
        if self.model_name != model_name:
            logging.debug(f"Ignoring data for model {model_name}, expected {self.model_name}")
            return

        logging.debug(f"Processing MQTT data: {len(values)} channels, sample_rate={sample_rate}, samples_per_channel={samples_per_channel}")
        try:
            if not values or len(values) != 6:
                logging.warning(f"Received incorrect number of sublists: {len(values)}, expected 6")
                if self.console:
                    self.console.append_to_console(f"Received incorrect number of sublists: {len(values)}")
                return

            self.sample_rate = sample_rate
            self.samples_per_channel = samples_per_channel
            self.tacho_samples = samples_per_channel
            self.range_slider.update_maximum(samples_per_channel)
            if self.data_range_end > samples_per_channel:
                self.data_range_end = samples_per_channel
                self.data_range_start = 0
                self.range_slider.min_slider.setValue(self.data_range_start)
                self.range_slider.max_slider.setValue(self.data_range_end)

            for ch in range(4):
                if len(values[ch]) != self.samples_per_channel:
                    logging.warning(f"Channel {ch+1} has {len(values[ch])} samples, expected {self.samples_per_channel}")
                    if self.console:
                        self.console.append_to_console(f"Channel {ch+1} sample mismatch: {len(values[ch])}")
                    return
            if len(values[4]) != self.tacho_samples or len(values[5]) != self.tacho_samples:
                logging.warning(f"Tacho data length mismatch: freq={len(values[4])}, trigger={len(values[5])}, expected={self.tacho_samples}")
                if self.console:
                    self.console.append_to_console(f"Tacho data length mismatch: freq={len(values[4])}, trigger={len(values[5])}")
                return

            current_time = time.time()
            channel_time_step = 1.0 / sample_rate
            self.channel_times = np.array([current_time - (self.samples_per_channel - 1 - i) * channel_time_step for i in range(self.samples_per_channel)])

            self.data[0] = np.array(values[0][:self.samples_per_channel]) * self.scaling_factor
            self.data[1] = np.array(values[1][:self.samples_per_channel]) * self.scaling_factor
            self.data[2] = np.array(values[4][:self.tacho_samples]) / 100

            self.apply_data_range()

            logging.debug(f"Updated {self.num_plots} plots: {self.samples_per_channel} channel samples")
            if self.console:
                self.console.append_to_console(f"Time View ({self.model_name}): Updated {self.num_plots} plots with {self.samples_per_channel} channel samples")

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