import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout
from PyQt5.QtCore import QObject, QEvent, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy

from pyqtgraph import PlotWidget, mkPen, AxisItem, InfiniteLine, SignalProxy
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeAxisItem(AxisItem):
    """Custom axis to display datetime on x-axis."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        """Convert timestamps to 'YYYY-MM-DD\nHH:MM:SS' format."""
        return [datetime.fromtimestamp(v).strftime('%Y-%m-%d\n%H:%M:%S') for v in values]

class MouseTracker(QObject):
    """Event filter to track mouse enter/leave on plot viewport."""
    def __init__(self, parent, idx, feature):
        super().__init__(parent)
        self.idx = idx
        self.feature = feature

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            self.feature.mouse_enter(self.idx)
        elif event.type() == QEvent.Leave:
            self.feature.mouse_leave(self.idx)
        return False

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
        self.tacho_times = []
        self.sample_rate = 4096
        self.num_channels = 4
        self.scaling_factor = 3.3 / 65535
        self.num_plots = 6  # Channels 1-4, tacho freq, tacho trigger, removed Bode plots
        self.channel_samples = 4096
        self.tacho_samples = 4096
        self.vlines = []
        self.proxies = []
        self.trackers = []
        self.trigger_lines = []
        self.active_line_idx = None
        self.peak_to_peak_ch2 = None
        self.peak_to_peak_per_cycle = []
        self.custom_magnitude_ch2 = None
        self.custom_magnitude_per_cycle = []
        self.vrms_values = []  # Store Vrms values
        self.frequencies = []  # Store corresponding frequencies
        self.mqtt_running = False
        self.start_button = None
        self.stop_button = None
        self.last_tacho_freq = None

        self.initUI()

    def initUI(self):
        """Initialize the UI with pyqtgraph subplots and buttons."""
        self.widget = QWidget()
        layout = QVBoxLayout()
        
        # Add MQTT control buttons
        button_layout = QHBoxLayout()
        button_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        self.start_button = QPushButton("Start MQTT")
        self.start_button.setFixedSize(100, 50)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                font-size: 16px;
            }
        """)
        self.start_button.clicked.connect(self.start_mqtt)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop MQTT")
        self.stop_button.setFixedSize(100, 50)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border-radius: 5px;
                font-size: 16px;
            }
        """)
        button_layout.addWidget(self.stop_button)
        self.setLayout(button_layout)


        # self.start_button = QPushButton("Start MQTT")
        # self.start_button.clicked.connect(self.start_mqtt)
        # self.stop_button = QPushButton("Stop MQTT")
        # self.stop_button.clicked.connect(self.stop_mqtt)
        # self.stop_button.setEnabled(False)
        # button_layout.addWidget(self.start_button)
        # button_layout.addWidget(self.stop_button)
        # layout.addLayout(button_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        for i in range(self.num_plots):
            axis_items = {'bottom': TimeAxisItem(orientation='bottom')} if i < self.num_channels + 2 else None
            plot_widget = PlotWidget(axisItems=axis_items, background='w')
            plot_widget.setFixedHeight(250)
            plot_widget.setMinimumWidth(0)
            if i < self.num_channels:
                plot_widget.setLabel('left', f'CH{i+1} Value')
            elif i == self.num_channels:
                plot_widget.setLabel('left', 'Tacho Frequency')
            elif i == self.num_channels + 1:
                plot_widget.setLabel('left', 'Tacho Trigger')
                plot_widget.setYRange(-0.5, 1.5, padding=0)
            plot_widget.showGrid(x=True, y=True)
            plot_widget.addLegend()
            pen = mkPen(color=colors[i % len(colors)], width=2)
            plot = plot_widget.plot([], [], pen=pen, name=f'Plot {i+1}')
            self.plots.append(plot)
            self.plot_widgets.append(plot_widget)
            self.data.append([])

            vline = InfiniteLine(angle=90, movable=False, pen=mkPen('r', width=2))
            vline.setVisible(False)
            plot_widget.addItem(vline)
            self.vlines.append(vline)

            if i == self.num_plots - 1 or i == self.num_plots - 2:
                self.trigger_lines = []
            else:
                self.trigger_lines.append(None)

            proxy = SignalProxy(plot_widget.scene().sigMouseMoved, rateLimit=60, slot=lambda evt, idx=i: self.mouse_moved(evt, idx))
            self.proxies.append(proxy)

            tracker = MouseTracker(plot_widget.viewport(), i, self)
            plot_widget.viewport().installEventFilter(tracker)
            self.trackers.append(tracker)

            scroll_layout.addWidget(plot_widget)

        # Add new plot for Vrms vs Frequency
        vrms_plot = PlotWidget(background='w')
        vrms_plot.setFixedHeight(250)
        vrms_plot.setLabel('left', 'Vrms (V)')
        vrms_plot.setLabel('bottom', 'Frequency (Hz)')
        vrms_plot.showGrid(x=True, y=True)
        vrms_plot.addLegend()
        pen = mkPen(color='b', width=2)
        vrms_line = vrms_plot.plot([], [], pen=pen, name='Vrms vs Freq')
        self.plot_widgets.append(vrms_plot)
        self.plots.append(vrms_line)
        self.data.append([])  # For Vrms data
        self.num_plots += 1

        vline = InfiniteLine(angle=90, movable=False, pen=mkPen('r', width=2))
        vline.setVisible(False)
        vrms_plot.addItem(vline)
        self.vlines.append(vline)

        proxy = SignalProxy(vrms_plot.scene().sigMouseMoved, rateLimit=60, slot=lambda evt, idx=self.num_plots-1: self.mouse_moved(evt, idx))
        self.proxies.append(proxy)

        tracker = MouseTracker(vrms_plot.viewport(), self.num_plots-1, self)
        vrms_plot.viewport().installEventFilter(tracker)
        self.trackers.append(tracker)

        scroll_layout.addWidget(vrms_plot)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        self.widget.setLayout(layout)

        if not self.model_name and self.console:
            self.console.append_to_console("No model selected in TimeViewFeature.")
        if not self.channel and self.console:
            self.console.append_to_console("No channel selected in TimeViewFeature.")

    def get_widget(self):
        """Return the widget containing the plots."""
        return self.widget

    def calculate_vrms(self, data):
        """Calculate Vrms as Vmax - Vmin."""
        if len(data) == 0:
            return 0.0
        vmax = np.max(data)
        vmin = np.min(data)
        return vmax - vmin

    def calculate_per_cycle_metrics(self, frequency):
        """Calculate peak-to-peak and custom magnitude for each cycle of channel 2."""
        if len(self.data[1]) == 0:
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

    def calculate_custom_magnitude(self, data):
        """Calculate custom magnitude: sqrt(sum(v_i^2)) / n."""
        if len(data) == 0:
            return 0.0
        n = len(data)
        sum_squares = np.sum(data**2)
        return np.sqrt(sum_squares) / n

    def start_mqtt(self):
        """Start MQTT subscription."""
        self.mqtt_running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        if self.console:
            self.console.append_to_console("MQTT subscription started.")
        logging.debug("MQTT subscription started.")

    def stop_mqtt(self):
        """Stop MQTT subscription."""
        self.mqtt_running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.console:
            self.console.append_to_console("MQTT subscription stopped.")
        logging.debug("MQTT subscription stopped.")

    def on_data_received(self, tag_name, model_name, values, sample_rate):
        """Handle incoming MQTT data, update plots."""
        if not self.mqtt_running:
            return

        if tag_name != "sarayu/d1/topic1":
            logging.debug(f"Ignoring data for topic {tag_name}, expected sarayu/d1/topic1")
            return
        if self.model_name != model_name:
            logging.debug(f"Ignoring data for model {model_name}, expected {self.model_name}")
            return

        try:
            if not values or len(values) != 6:
                logging.warning(f"Received incorrect number of sublists: {len(values)}, expected 6")
                if self.console:
                    self.console.append_to_console(f"Received incorrect number of sublists: {len(values)}")
                return

            self.sample_rate = sample_rate
            self.channel_samples = 4096
            self.tacho_samples = 4096

            for ch in range(self.num_channels):
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
            tacho_time_step = 1.0 / sample_rate
            self.channel_times = np.array([current_time - (self.channel_samples - 1) * channel_time_step + i * channel_time_step for i in range(self.channel_samples)])
            self.tacho_times = np.array([current_time - (self.tacho_samples - 1) * tacho_time_step + i * tacho_time_step for i in range(self.tacho_samples)])

            # Apply scaling factor to channel data
            for ch in range(self.num_channels):
                self.data[ch] = np.array(values[ch][:self.channel_samples]) * self.scaling_factor
                logging.debug(f"Channel {ch+1} data: {len(self.data[ch])} samples, scaled with factor {self.scaling_factor}")

            # Assign tacho data without scaling
            self.data[self.num_channels] = np.array(values[4][:self.tacho_samples]) / 100
            self.data[self.num_channels + 1] = np.array(values[5][:self.tacho_samples])
            logging.debug(f"Tacho freq data: {len(self.data[self.num_channels])} samples")
            logging.debug(f"Tacho trigger data: {len(self.data[self.num_channels + 1])} samples, first 10: {self.data[self.num_channels + 1][:10]}")

            # Calculate metrics for channel 2
            if len(self.data[1]) > 0:
                self.peak_to_peak_ch2 = np.max(self.data[1]) - np.min(self.data[1])
                logging.debug(f"Channel 2 Peak-to-Peak (Entire Array): {self.peak_to_peak_ch2:.4f} V")
                if self.console:
                    self.console.append_to_console(f"Channel 2 Peak-to-Peak (Entire Array): {self.peak_to_peak_ch2:.4f} V")

                self.custom_magnitude_ch2 = self.calculate_custom_magnitude(self.data[1])
                logging.debug(f"Channel 2 Custom Magnitude (Entire Array): {self.custom_magnitude_ch2:.4f} V")
                if self.console:
                    self.console.append_to_console(f"Channel 2 Custom Magnitude (Entire Array): {self.custom_magnitude_ch2:.4f} V")

                # Calculate Vrms for Channel 2
                vrms = self.calculate_vrms(self.data[1])
                logging.debug(f"Channel 2 Vrms (Vmax - Vmin): {vrms:.4f} V")
                if self.console:
                    self.console.append_to_console(f"Channel 2 Vrms (Vmax - Vmin): {vrms:.4f} V")

            # Calculate per-cycle metrics and get tacho frequency
            if len(self.data[self.num_channels]) > 0:
                frequency = self.data[self.num_channels][0]  # Tacho frequency (e.g., 10 Hz)
                self.peak_to_peak_per_cycle, self.custom_magnitude_per_cycle = self.calculate_per_cycle_metrics(frequency)
                if self.peak_to_peak_per_cycle:
                    avg_peak_to_peak = np.mean(self.peak_to_peak_per_cycle)
                    logging.debug(f"Channel 2 Peak-to-Peak Per Cycle: {self.peak_to_peak_per_cycle}")
                    logging.debug(f"Average Peak-to-Peak Per Cycle: {avg_peak_to_peak:.4f} V")
                    if self.console:
                        self.console.append_to_console(f"Channel 2 Peak-to-Peak Per Cycle: {[f'{x:.4f}' for x in self.peak_to_peak_per_cycle]} V")
                        self.console.append_to_console(f"Average Peak-to-Peak Per Cycle: {avg_peak_to_peak:.4f} V")
                if self.custom_magnitude_per_cycle:
                    avg_custom_magnitude = np.mean(self.custom_magnitude_per_cycle)
                    logging.debug(f"Channel 2 Custom Magnitude Per Cycle: {self.custom_magnitude_per_cycle}")
                    logging.debug(f"Average Custom Magnitude Per Cycle: {avg_custom_magnitude:.4f} V")
                    if self.console:
                        self.console.append_to_console(f"Channel 2 Custom Magnitude Per Cycle: {[f'{x:.4f}' for x in self.custom_magnitude_per_cycle]} V")
                        self.console.append_to_console(f"Average Custom Magnitude Per Cycle: {avg_custom_magnitude:.4f} V")

                # Update Vrms vs Frequency plot
                if len(self.data[1]) > 0 and frequency > 0:
                    vrms = self.calculate_vrms(self.data[1])
                    if self.last_tacho_freq is None or not np.isclose(self.last_tacho_freq, frequency, rtol=1e-5):
                        self.frequencies.append(frequency)
                        self.vrms_values.append(vrms)
                        self.last_tacho_freq = frequency
                    self.data[self.num_plots-1] = self.vrms_values
                    self.plots[self.num_plots-1].setData(self.frequencies, self.vrms_values)
                    if self.frequencies:
                        self.plot_widgets[self.num_plots-1].setXRange(min(self.frequencies), max(self.frequencies), padding=0.05)
                    if self.vrms_values:
                        self.plot_widgets[self.num_plots-1].setYRange(min(self.vrms_values), max(self.vrms_values), padding=0.05)

            # Update other plots
            for ch in range(self.num_channels + 2):
                times = self.tacho_times if ch >= self.num_channels else self.channel_times
                data = self.data[ch]
                if len(data) > 0 and len(times) > 0:
                    self.plots[ch].setData(times, data)
                    self.plot_widgets[ch].setXRange(times[0], times[-1], padding=0)
                    if ch < self.num_channels:
                        self.plot_widgets[ch].enableAutoRange(axis='y')
                    elif ch == self.num_channels:
                        self.plot_widgets[ch].enableAutoRange(axis='y')
                    elif ch == self.num_channels + 1:
                        self.plot_widgets[ch].setYRange(0, 1.0, padding=0)
                else:
                    logging.warning(f"No data for plot {ch}, data_len={len(data)}, times_len={len(times)}")
                    if self.console:
                        self.console.append_to_console(f"No data for plot {ch}")

            # Add trigger lines for tacho trigger plot
            if len(self.data[self.num_channels + 1]) > 0:
                if self.trigger_lines:
                    for line in self.trigger_lines:
                        if line:
                            self.plot_widgets[self.num_channels + 1].removeItem(line)
                self.trigger_lines = []

                trigger_indices = np.where(self.data[self.num_channels + 1] == 1)[0]
                logging.debug(f"Tacho trigger indices (value=1): {len(trigger_indices)} points")
                for idx in trigger_indices:
                    if idx < len(self.tacho_times):
                        line = InfiniteLine(
                            pos=self.tacho_times[idx],
                            angle=90,
                            movable=False,
                            pen=mkPen('k', width=2, style=Qt.SolidLine)
                        )
                        self.plot_widgets[self.num_channels + 1].addItem(line)
                        self.trigger_lines.append(line)

            logging.debug(f"Updated {self.num_plots} plots: {self.channel_samples} channel samples, {self.tacho_samples} tacho samples")
            if self.console:
                self.console.append_to_console(
                    f"Time View ({self.model_name}): Updated {self.num_plots} plots with {self.channel_samples} channel samples, {self.tacho_samples} tacho samples"
                )

        except Exception as e:
            logging.error(f"Error updating plots: {str(e)}")
            if self.console:
                self.console.append_to_console(f"Error updating plots: {str(e)}")

    def mouse_enter(self, idx):
        """Called when mouse enters plot idx viewport."""
        self.active_line_idx = idx
        self.vlines[idx].setVisible(True)
        logging.debug(f"Mouse entered plot {idx}")

    def mouse_leave(self, idx):
        """Called when mouse leaves plot idx viewport."""
        self.active_line_idx = None
        for vline in self.vlines:
            vline.setVisible(False)
        logging.debug(f"Mouse left plot {idx}")

    def mouse_moved(self, evt, idx):
        """Update vertical lines on mouse move."""
        if self.active_line_idx is None:
            return

        pos = evt[0]
        if not self.plot_widgets[idx].sceneBoundingRect().contains(pos):
            return

        mouse_point = self.plot_widgets[idx].plotItem.vb.mapSceneToView(pos)
        x = mouse_point.x()

        if idx == self.num_plots - 1:  # Vrms plot
            times = self.frequencies
        else:
            times = self.tacho_times if idx >= self.num_channels else self.channel_times

        if len(times) > 0:
            if x < times[0]:
                x = times[0]
            elif x > times[-1]:
                x = times[-1]

        for vline in self.vlines:
            vline.setPos(x)
            vline.setVisible(True)