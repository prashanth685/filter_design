import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QScrollArea, QLabel, QPushButton, QHBoxLayout, QSlider
from PyQt5.QtCore import QObject, Qt, pyqtSignal, QTimer
from pyqtgraph import PlotWidget, mkPen, AxisItem, SignalProxy
from datetime import datetime
import time
import logging
import paho.mqtt.client as mqtt
import json
import struct
import sys

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeAxisItem(AxisItem):
    """Custom axis to display datetime on x-axis."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        """Convert timestamps to 'YYYY-MM-DD\nHH:MM:SS' format."""
        return [datetime.fromtimestamp(v).strftime('%Y-%m-%d\n%H:%M:%S') for v in values]

class MQTTHandler(QObject):
    data_received = pyqtSignal(str, str, list, int)  # tag_name, model_name, values, sample_rate
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
                # Attempt JSON decode
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
                    values = struct.unpack(f"<{num_samples}H", payload)
                except struct.error as e:
                    logging.error(f"Failed to unpack payload of {num_samples} uint16_t: {str(e)}")
                    return

                samples_per_channel = 4096
                num_channels = 4
                tacho_samples = 4096
                expected_total = samples_per_channel * num_channels + tacho_samples * 2

                if len(values) != expected_total:
                    logging.warning(f"Unexpected data length: got {len(values)}, expected {expected_total}")
                    return

                # Segregate based on index
                channel_data = [[] for _ in range(num_channels)]
                for i in range(0, samples_per_channel * num_channels, num_channels):
                    for ch in range(num_channels):
                        channel_data[ch].append(values[i + ch])
                tacho_freq_data = values[samples_per_channel * num_channels:samples_per_channel * num_channels + tacho_samples]
                tacho_trigger_data = values[samples_per_channel * num_channels + tacho_samples:samples_per_channel * num_channels + tacho_samples * 2]

                values = [[float(v) for v in ch] for ch in channel_data]
                values.append([float(v) for v in tacho_freq_data])
                values.append([float(v) for v in tacho_trigger_data])
                sample_rate = 4096

                logging.debug(f"Parsed binary payload: {num_channels} channels, {len(channel_data[0])} samples/channel")
                logging.debug(f"Tacho freq (first 5): {tacho_freq_data[:5]}")
                logging.debug(f"Tacho trigger (first 20): {tacho_trigger_data[:20]}")

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
        self.sample_rate = 4096
        self.num_channels = 2  # Only Channel 1 and Channel 2
        self.scaling_factor = 3.3 / 65535
        self.num_plots = 5  # Channel 1, Channel 2, Vrms vs Tacho Freq, Log-x Plot, Gain (dB) for Channel 2
        self.channel_samples = 4096
        self.tacho_samples = 4096
        self.proxies = []
        self.peak_to_peak_ch2 = None
        self.peak_to_peak_per_cycle = []
        self.custom_magnitude_ch2 = None
        self.custom_magnitude_per_cycle = []
        self.vrms_label = None
        self.vrms_ch2 = None
        self.frequency_label = None
        self.amplitude_label = None
        self.start_button = None
        self.stop_button = None
        self.clear_button = None
        self.is_plotting = True  # Enable plotting by default
        self.vrms_vs_tacho_data = {'vrms': [], 'tacho_freq': []}
        self.y_range_fixed = True  # Default to 3V fixed range
        self.tacho_freq_range = 1000  # Default to 1000Hz
        self.vrms_slider = None
        self.tacho_slider = None
        self.ok_button = None
        self.vrms_range = [0, 4.0]  # Default Vrms range
        self.tacho_range = [0, 1000]  # Default Tacho range

        self.initUI()
        logging.debug("TimeViewFeature initialized with plotting enabled")

    def initUI(self):
        """Initialize the UI with pyqtgraph subplots, labels, sliders, and buttons."""
        self.widget = QWidget()
        layout = QVBoxLayout()
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        colors = ['r', 'g', 'b', 'k', 'm']
        for i in range(self.num_plots):
            if i < 2:  # Time-based plots for Channel 1 and 2
                axis_items = {'bottom': TimeAxisItem(orientation='bottom')}
            elif i == 2:  # Vrms vs Tacho Frequency plot
                axis_items = None
            elif i == 3:  # Log-x plot for Channel 1 and 2
                axis_items = {'bottom': AxisItem(orientation='bottom')}
                axis_items['bottom'].setLogMode(True)
            else:  # Gain (dB) for Channel 2
                axis_items = None
            plot_widget = PlotWidget(axisItems=axis_items, background='w')
            plot_widget.setFixedHeight(250)
            plot_widget.setMinimumWidth(0)
            if i < self.num_channels:
                plot_widget.setLabel('left', f'CH{i+1} Value (V)')
                plot_widget.setYRange(0, 3.0, padding=0)  # Default 0-3V
            elif i == 2:
                plot_widget.setLabel('left', 'Channel 2 Vrms (V)')
                plot_widget.setLabel('bottom', 'Tacho Frequency (Hz)')
                plot_widget.setYRange(0, 4.0, padding=0)  # Fixed 0-4V
                plot_widget.setXRange(0, 1000, padding=0)  # Fixed 0-1000Hz
            elif i == 3:
                plot_widget.setLabel('left', 'Magnitude')
                plot_widget.setLabel('bottom', 'Frequency (Hz)')
                plot_widget.setXRange(np.log10(0.1), np.log10(1000), padding=0)  # Log scale from 0.1Hz to 1000Hz
                plot_widget.enableAutoRange(axis='y')
            elif i == 4:
                plot_widget.setLabel('left', 'Gain (dB)')
                plot_widget.setLabel('bottom', 'Frequency (Hz)')
                plot_widget.setXRange(0, 1000, padding=0)  # Linear scale from 0 to 1000Hz
                plot_widget.enableAutoRange(axis='y')
            plot_widget.showGrid(x=True, y=True)
            plot_widget.addLegend()
            if i == 3:  # Log-x plot has two lines
                pen1 = mkPen(color='r', width=2)
                pen2 = mkPen(color='g', width=2)
                plot1 = plot_widget.plot([], [], pen=pen1, name='Channel 1')
                plot2 = plot_widget.plot([], [], pen=pen2, name='Channel 2')
                self.plots.append([plot1, plot2])
            else:
                pen = mkPen(color=colors[i % len(colors)], width=2)
                plot = plot_widget.plot([], [], pen=pen, name=f'Plot {i+1}')
                self.plots.append(plot)
            self.plot_widgets.append(plot_widget)
            self.data.append([])

            scroll_layout.addWidget(plot_widget)

            # Add sliders and OK button after Vrms vs Tacho plot
            if i == 2:
                slider_layout = QHBoxLayout()
                self.vrms_slider = QSlider(Qt.Horizontal)
                self.vrms_slider.setMinimum(0)
                self.vrms_slider.setMaximum(400)  # 0 to 4.0V, scaled by 100
                self.vrms_slider.setValue(400)  # Default max
                self.vrms_slider.setTickInterval(100)
                self.vrms_slider.setTickPosition(QSlider.TicksBelow)
                self.vrms_slider.valueChanged.connect(self.update_vrms_range)
                slider_layout.addWidget(QLabel("Vrms Range (V):"))
                slider_layout.addWidget(self.vrms_slider)

                self.tacho_slider = QSlider(Qt.Horizontal)
                self.tacho_slider.setMinimum(0)
                self.tacho_slider.setMaximum(1000)  # 0 to 1000Hz
                self.tacho_slider.setValue(1000)  # Default max
                self.tacho_slider.setTickInterval(100)
                self.tacho_slider.setTickPosition(QSlider.TicksBelow)
                self.tacho_slider.valueChanged.connect(self.update_tacho_range)
                slider_layout.addWidget(QLabel("Tacho Freq Range (Hz):"))
                slider_layout.addWidget(self.tacho_slider)

                self.ok_button = QPushButton("Apply Range")
                self.ok_button.setStyleSheet("background-color: #2196F3; color: white; padding: 8px; border-radius: 4px;")
                self.ok_button.clicked.connect(self.apply_ranges)
                slider_layout.addWidget(self.ok_button)

                scroll_layout.addLayout(slider_layout)

        # Add labels for Vrms, frequency, and amplitude for Channel 2
        self.vrms_label = QLabel("Channel 2 Vrms: N/A")
        self.frequency_label = QLabel("Channel 2 Frequency: N/A")
        self.amplitude_label = QLabel("Channel 2 Amplitude: N/A")
        scroll_layout.addWidget(self.vrms_label)
        scroll_layout.addWidget(self.frequency_label)
        scroll_layout.addWidget(self.amplitude_label)

        # Button layout for controls
        button_layout = QHBoxLayout()
        button_layout.addStretch()  # Push buttons to the right
        self.start_button = QPushButton("Start MQTT Plotting")
        self.stop_button = QPushButton("Stop MQTT Plotting")
        self.clear_button = QPushButton("Clear Vrms vs Tacho Plot")
        self.y_range_3v_button = QPushButton("Set 3V Range")
        self.y_range_auto_button = QPushButton("Auto Y Range")
        self.freq_100hz_button = QPushButton("100 Hz Range")
        self.freq_1000hz_button = QPushButton("1000 Hz Range")

        self.start_button.setEnabled(False)  # Disabled since plotting starts by default
        self.stop_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.y_range_auto_button.setEnabled(False)  # Default is 3V fixed

        # Styling
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; border-radius: 4px;")
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; padding: 8px; border-radius: 4px;")
        self.clear_button.setStyleSheet("background-color: #2196F3; color: white; padding: 8px; border-radius: 4px;")
        self.y_range_3v_button.setStyleSheet("background-color: #FFC107; color: black; padding: 8px; border-radius: 4px;")
        self.y_range_auto_button.setStyleSheet("background-color: #FFC107; color: black; padding: 8px; border-radius: 4px;")
        self.freq_100hz_button.setStyleSheet("background-color: #FF5722; color: white; padding: 8px; border-radius: 4px;")
        self.freq_1000hz_button.setStyleSheet("background-color: #FF5722; color: white; padding: 8px; border-radius: 4px;")

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.y_range_3v_button)
        button_layout.addWidget(self.y_range_auto_button)
        button_layout.addWidget(self.freq_100hz_button)
        button_layout.addWidget(self.freq_1000hz_button)

        scroll_layout.addLayout(button_layout)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        self.widget.setLayout(layout)

        if not self.model_name and self.console:
            self.console.append_to_console("No model selected in TimeViewFeature.")
        if not self.channel and self.console:
            self.console.append_to_console("No channel selected in TimeViewFeature.")

    def start_plotting(self):
        """Enable plotting of MQTT data."""
        self.is_plotting = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        logging.debug("Started plotting MQTT data")
        if self.console:
            self.console.append_to_console("Started plotting MQTT data")

    def stop_plotting(self):
        """Disable plotting of MQTT data."""
        self.is_plotting = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.clear_button.setEnabled(True)
        logging.debug("Stopped plotting MQTT data")
        if self.console:
            self.console.append_to_console("Stopped plotting MQTT data")

    def clear_vrms_vs_tacho_plot(self):
        """Clear the Vrms vs Tacho Frequency plot."""
        self.vrms_vs_tacho_data['vrms'] = []
        self.vrms_vs_tacho_data['tacho_freq'] = []
        self.plots[2].setData([], [])
        self.plot_widgets[2].setXRange(self.tacho_range[0], self.tacho_range[1], padding=0)
        self.plot_widgets[2].setYRange(self.vrms_range[0], self.vrms_range[1], padding=0)
        logging.debug("Cleared Vrms vs Tacho Frequency plot")
        if self.console:
            self.console.append_to_console("Cleared Vrms vs Tacho Frequency plot")

    def set_3v_range(self):
        """Set Y-axis to fixed 0-3V for Channel 1 and 2."""
        self.y_range_fixed = True
        self.y_range_3v_button.setEnabled(False)
        self.y_range_auto_button.setEnabled(True)
        for i in range(self.num_channels):
            self.plot_widgets[i].setYRange(0, 3.0, padding=0)
        logging.debug("Set Y-axis to 0-3V for Channel 1 and 2")
        if self.console:
            self.console.append_to_console("Set Y-axis to 0-3V for Channel 1 and 2")

    def set_auto_y_range(self):
        """Enable auto Y-axis range for Channel 1 and 2."""
        self.y_range_fixed = False
        self.y_range_3v_button.setEnabled(True)
        self.y_range_auto_button.setEnabled(False)
        for i in range(self.num_channels):
            self.plot_widgets[i].enableAutoRange(axis='y')
        logging.debug("Enabled auto Y-axis range for Channel 1 and 2")
        if self.console:
            self.console.append_to_console("Enabled auto Y-axis range for Channel 1 and 2")

    def set_tacho_freq_range(self, freq):
        """Set X-axis range for Vrms vs Tacho Frequency plot."""
        self.tacho_freq_range = freq
        self.tacho_range = [0, freq]
        self.tacho_slider.setMaximum(freq)
        self.tacho_slider.setValue(freq)
        self.plot_widgets[2].setXRange(0, freq, padding=0)
        self.apply_ranges()
        logging.debug(f"Set Tacho Frequency range to 0-{freq}Hz")
        if self.console:
            self.console.append_to_console(f"Set Tacho Frequency range to 0-{freq}Hz")

    def update_vrms_range(self, value):
        """Update Vrms range based on slider value."""
        self.vrms_range[1] = value / 100.0
        logging.debug(f"Vrms slider updated to {self.vrms_range[1]} V")

    def update_tacho_range(self, value):
        """Update Tacho frequency range based on slider value."""
        self.tacho_range[1] = value
        logging.debug(f"Tacho slider updated to {self.tacho_range[1]} Hz")

    def apply_ranges(self):
        """Apply the selected Vrms and Tacho frequency ranges to the Vrms vs Tacho plot."""
        filtered_vrms = []
        filtered_tacho = []
        for v, t in zip(self.vrms_vs_tacho_data['vrms'], self.vrms_vs_tacho_data['tacho_freq']):
            if self.vrms_range[0] <= v <= self.vrms_range[1] and self.tacho_range[0] <= t <= self.tacho_range[1]:
                filtered_vrms.append(v)
                filtered_tacho.append(t)
        self.plots[2].setData(filtered_tacho, filtered_vrms)
        self.plot_widgets[2].setXRange(self.tacho_range[0], self.tacho_range[1], padding=0)
        self.plot_widgets[2].setYRange(self.vrms_range[0], self.vrms_range[1], padding=0)
        logging.debug(f"Applied ranges: Vrms {self.vrms_range}, Tacho {self.tacho_range}")
        if self.console:
            self.console.append_to_console(f"Applied ranges: Vrms {self.vrms_range}, Tacho {self.tacho_range}")

    def get_widget(self):
        """Return the widget containing the plots, labels, sliders, and buttons."""
        return self.widget

    def calculate_custom_magnitude(self, data):
        """Calculate custom magnitude: sqrt(sum(v_i^2)) / n."""
        if len(data) == 0:
            return 0.0
        n = len(data)
        sum_squares = np.sum(data**2)
        return np.sqrt(sum_squares) / n

    def calculate_vrms(self, data):
        """Calculate Vpeak-to-peak: Vmax - Vmin."""
        if len(data) == 0:
            return 0.0
        vmax = np.max(data)
        vmin = np.min(data)
        logging.debug(f"Maximum: {vmax}, Minimum: {vmin}, Vrms: {vmax-vmin}")
        return vmax - vmin

    def calculate_amplitude(self, data):
        """Calculate the amplitude: (max - min) / 2 after removing DC offset."""
        if len(data) == 0:
            return 0.0
        data_centered = data - np.mean(data)
        peak_to_peak = np.max(data_centered) - np.min(data_centered)
        amplitude = peak_to_peak / 2
        logging.debug(f"Amplitude - Mean: {np.mean(data):.4f}, Centered Min: {np.min(data_centered):.4f}, Centered Max: {np.max(data_centered):.4f}, Peak-to-Peak: {peak_to_peak:.4f}, Amplitude: {amplitude:.2f}")
        return amplitude

    def calculate_frequency(self, data, sample_rate):
        """Calculate the dominant frequency of the signal using FFT."""
        if len(data) == 0:
            return 0.0
        data = data - np.mean(data)
        n = len(data)
        fft_result = np.fft.fft(data)
        freqs = np.fft.fftfreq(n, d=1/sample_rate)
        magnitudes = np.abs(fft_result)
        positive_freqs = freqs[:n//2]
        positive_magnitudes = magnitudes[:n//2]
        positive_magnitudes[0] = 0  # Ignore DC component
        dominant_freq_idx = np.argmax(positive_magnitudes)
        dominant_freq = positive_freqs[dominant_freq_idx]
        return abs(dominant_freq)

    def calculate_per_cycle_metrics(self, frequency):
        """Calculate peak-to-peak and custom magnitude for each cycle of channel 2."""
        if len(self.data[1]) == 0 or frequency <= 0:
            return [], []
        samples_per_cycle = self.sample_rate / frequency
        num_cycles = int(np.floor(self.channel_samples / samples_per_cycle))
        peak_to_peak_values = []
        custom_magnitude_values = []
        for i in range(num_cycles):
            start_idx = int(i * samples_per_cycle)
            end_idx = int((i + 1) * samples_per_cycle)
            if end_idx > len(self.data[1]):
                end_idx = len(self.data[1])
            cycle_data = self.data[1][start_idx:end_idx]
            if len(cycle_data) > 0:
                peak_to_peak = np.max(cycle_data) - np.min(cycle_data)
                peak_to_peak_values.append(peak_to_peak)
                custom_magnitude = self.calculate_custom_magnitude(cycle_data)
                custom_magnitude_values.append(custom_magnitude)
        return peak_to_peak_values, custom_magnitude_values

    def on_data_received(self, tag_name, model_name, values, sample_rate):
        """Handle incoming MQTT data, update plots and labels if plotting is enabled."""
        if not self.is_plotting:
            logging.debug("Plotting is disabled, ignoring MQTT data")
            return
        if tag_name != "sarayu/d1/topic1":
            logging.debug(f"Ignoring data for topic {tag_name}, expected sarayu/d1/topic1")
            return
        if self.model_name != model_name:
            logging.debug(f"Ignoring data for model {model_name}, expected {self.model_name}")
            return

        logging.debug(f"Processing MQTT data: {len(values)} channels, sample_rate={sample_rate}")
        try:
            if not values or len(values) != 6:
                logging.warning(f"Received incorrect number of sublists: {len(values)}, expected 6")
                if self.console:
                    self.console.append_to_console(f"Received incorrect number of sublists: {len(values)}")
                return

            self.sample_rate = sample_rate
            self.channel_samples = 4096
            self.tacho_samples = 4096

            for ch in range(4):  # Check first 4 channels (0-3)
                if len(values[ch]) != self.channel_samples:
                    logging.warning(f"Channel {ch+1} has {len(values[ch])} samples, expected {self.channel_samples}")
                    if self.console:
                        self.console.append_to_console(f"Channel {ch+1} sample mismatch: {len(values[ch])}")
                    return
            tacho_freq_samples = len(values[4])
            tacho_trigger_samples = len(values[5])
            if tacho_freq_samples != self.tacho_samples or tacho_trigger_samples != self.tacho_samples:
                logging.warning(f"Tacho data length mismatch: freq={tacho_freq_samples}, trigger={tacho_trigger_samples}, expected={self.tacho_samples}")
                if self.console:
                    self.console.append_to_console(f"Tacho data length mismatch: freq={tacho_freq_samples}, trigger={tacho_trigger_samples}")
                return

            current_time = time.time()
            channel_time_step = 1.0 / sample_rate
            self.channel_times = np.array([current_time - (self.channel_samples - 1 - i) * channel_time_step for i in range(self.channel_samples)])

            # Process Channel 1, Channel 2, and Tacho Freq
            for ch in range(self.num_channels):
                self.data[ch] = np.array(values[ch][:self.channel_samples]) * self.scaling_factor
                logging.debug(f"Channel {ch+1} data: {len(self.data[ch])} samples, scaled with factor {self.scaling_factor}")
            self.data[2] = np.array(values[4][:self.tacho_samples]) / 100  # Tacho frequency divided by 100
            logging.debug(f"Tacho freq data: {len(self.data[2])} samples")

            if len(self.data[1]) > 0:
                raw_min = np.min(self.data[1])
                raw_max = np.max(self.data[1])
                raw_mean = np.mean(self.data[1])
                logging.debug(f"Channel 2 Raw Data - Min: {raw_min:.4f} V, Max: {raw_max:.4f} V, Mean: {raw_mean:.4f} V")
                if self.console:
                    self.console.append_to_console(f"Channel 2 Raw Data - Min: {raw_min:.4f} V, Max: {raw_max:.4f} V, Mean: {raw_mean:.4f} V")

                self.peak_to_peak_ch2 = np.max(self.data[1]) - np.min(self.data[1])
                logging.debug(f"Channel 2 Peak-to-Peak: {self.peak_to_peak_ch2:.4f} V")
                if self.console:
                    self.console.append_to_console(f"Channel 2 Peak-to-Peak: {self.peak_to_peak_ch2:.4f} V")

                self.custom_magnitude_ch2 = self.calculate_custom_magnitude(self.data[1])
                logging.debug(f"Channel 2 Custom Magnitude: {self.custom_magnitude_ch2:.4f} V")
                if self.console:
                    self.console.append_to_console(f"Channel 2 Custom Magnitude: {self.custom_magnitude_ch2:.4f} V")

                self.vrms_ch2 = self.calculate_vrms(self.data[1])
                self.vrms_label.setText(f"Channel 2 Vrms: {self.vrms_ch2:.2f} V")
                logging.debug(f"Channel 2 Vrms: {self.vrms_ch2:.4f} V")
                if self.console:
                    self.console.append_to_console(f"Channel 2 Vrms: {self.vrms_ch2:.4f} V")

                self.amplitude_ch2 = self.calculate_amplitude(self.data[1])
                self.amplitude_label.setText(f"Channel 2 Amplitude: {self.amplitude_ch2:.4f} V")
                logging.debug(f"Channel 2 Amplitude: {self.amplitude_ch2:.4f} V")
                if self.console:
                    self.console.append_to_console(f"Channel 2 Amplitude: {self.amplitude_ch2:.4f} V")

                frequency = self.calculate_frequency(self.data[1], self.sample_rate)
                self.frequency_label.setText(f"Channel 2 Frequency: {frequency:.2f} Hz")
                logging.debug(f"Channel 2 Frequency: {frequency:.2f} Hz")
                if self.console:
                    self.console.append_to_console(f"Channel 2 Frequency: {frequency:.2f} Hz")

                tacho_freq_avg = np.mean(self.data[2]) if len(self.data[2]) > 0 else 0.0
                if tacho_freq_avg > 0:
                    self.vrms_vs_tacho_data['vrms'].append(self.vrms_ch2)
                    self.vrms_vs_tacho_data['tacho_freq'].append(tacho_freq_avg)
                    self.apply_ranges()  # Apply current slider ranges to filter data
                    logging.debug(f"Vrms vs Tacho Freq updated: Vrms={self.vrms_ch2:.4f}, Tacho Freq={tacho_freq_avg:.4f}")
                    if self.console:
                        self.console.append_to_console(f"Vrms vs Tacho Freq updated: Vrms={self.vrms_ch2:.4f}, Tacho Freq={tacho_freq_avg:.4f}")

                if frequency > 0:
                    self.peak_to_peak_per_cycle, self.custom_magnitude_per_cycle = self.calculate_per_cycle_metrics(frequency)
                    if self.peak_to_peak_per_cycle:
                        avg_peak_to_peak = np.mean(self.peak_to_peak_per_cycle)
                        logging.debug(f"Channel 2 Peak-to-Peak Per Cycle: {self.peak_to_peak_per_cycle}")
                        logging.debug(f"Average Peak-to-Peak: {avg_peak_to_peak:.4f} V")
                        if self.console:
                            self.console.append_to_console(f"Channel 2 Peak-to-Peak Per Cycle: {[f'{x:.4f}' for x in self.peak_to_peak_per_cycle]} V")
                            self.console.append_to_console(f"Average Peak-to-Peak: {avg_peak_to_peak:.4f} V")
                    if self.custom_magnitude_per_cycle:
                        avg_custom_magnitude = np.mean(self.custom_magnitude_per_cycle)
                        logging.debug(f"Channel 2 Custom Magnitude Per Cycle: {self.custom_magnitude_per_cycle}")
                        logging.debug(f"Average Custom Magnitude Per Cycle: {avg_custom_magnitude:.4f} V")
                        if self.console:
                            self.console.append_to_console(f"Channel 2 Custom Magnitude Per Cycle: {[f'{x:.4f}' for x in self.custom_magnitude_per_cycle]} V")
                            self.console.append_to_console(f"Average Custom Magnitude Per Cycle: {avg_custom_magnitude:.4f} V")

                # Update Log-x Plot for Channel 1 and 2 and Gain (dB) Plot for Channel 2
                if len(self.data[0]) > 0 and len(self.data[1]) > 0:
                    freqs = np.fft.fftfreq(self.channel_samples, d=1/self.sample_rate)
                    positive_freqs = freqs[:self.channel_samples//2]
                    valid_freqs = positive_freqs[positive_freqs > 0]  # Exclude DC component
                    if len(valid_freqs) > 0:
                        fft_ch1 = np.abs(np.fft.fft(self.data[0]))[:self.channel_samples//2][positive_freqs > 0]
                        fft_ch2 = np.abs(np.fft.fft(self.data[1]))[:self.channel_samples//2][positive_freqs > 0]
                        self.plots[3][0].setData(np.log10(valid_freqs), fft_ch1)
                        self.plots[3][1].setData(np.log10(valid_freqs), fft_ch2)
                        self.plot_widgets[3].setXRange(np.log10(0.1), np.log10(1000), padding=0)
                        # Update Gain (dB) Plot for Channel 2
                        gain_db_ch2 = 20 * np.log10(fft_ch2 + 1e-10)  # Add small offset to avoid log(0)
                        self.plots[4].setData(valid_freqs, gain_db_ch2)
                        self.plot_widgets[4].setXRange(0, 1000, padding=0)
                        logging.debug(f"Updated Log-x plot and Gain (dB) plot with {len(valid_freqs)} frequency points")
                        if self.console:
                            self.console.append_to_console(f"Updated Log-x plot and Gain (dB) plot with {len(valid_freqs)} frequency points")

            for ch in range(self.num_plots - 3):  # Update only Channel 1 and 2 plots
                times = self.channel_times
                data = self.data[ch]
                if len(data) > 0 and len(times) > 0:
                    self.plots[ch].setData(times, data)
                    self.plot_widgets[ch].setXRange(times[0], times[-1], padding=0)
                    if not self.y_range_fixed:
                        self.plot_widgets[ch].enableAutoRange(axis='y')
                else:
                    logging.warning(f"No data for plot {ch}, data_len={len(data)}, times_len={len(times)}")
                    if self.console:
                        self.console.append_to_console(f"No data for plot {ch}")

            logging.debug(f"Updated {self.num_plots} plots: {self.channel_samples} channel samples")
            if self.console:
                self.console.append_to_console(
                    f"Time View ({self.model_name}): Updated {self.num_plots} plots with {self.channel_samples} channel samples"
                )

        except Exception as e:
            logging.error(f"Error updating plots: {str(e)}")
            if self.console:
                self.console.append_to_console(f"Error updating plots: {str(e)}")