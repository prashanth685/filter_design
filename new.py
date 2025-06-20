import math
import struct
import paho.mqtt.publish as publish
from PyQt5.QtCore import QTimer, QObject
from PyQt5.QtWidgets import QApplication
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MQTTPublisher(QObject):
    def __init__(self, broker, topics):
        super().__init__()
        self.broker = broker
        self.topics = topics if isinstance(topics, list) else [topics]
        self.count = 1

        self.frequency = 100  # Hz
        self.amplitude = 2.5  # Reduced amplitude for clarity (0-3.3V range)
        self.amplitude_scaled = (self.amplitude * 0.5) / (3.3 / 65535)  # Scale for 16-bit ADC
        self.offset = 32768  # Midpoint for 16-bit unsigned (0-65535)
        self.sample_rate = 4096  # Samples per second
        self.time_per_message = 1.0  # 1 second for 4096 samples
        self.current_time = 0.0
        self.num_channels = 2  # Only Channel 1 and Channel 2
        self.samples_per_channel = 4096  # Samples per channel
        self.num_tacho_channels = 1  # Only Tacho Frequency
        self.frame_index = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.publish_message)
        self.timer.start(1000)  # Publish every 1 second
        logging.debug(f"Initialized MQTTPublisher with broker: {self.broker}, topics: {self.topics}")

    def set_frequency(self, freq):
        """Set the frequency for the sine wave."""
        self.frequency = freq
        logging.debug(f"Updated frequency to {self.frequency} Hz")

    def publish_message(self):
        try:
            # Generate sine wave samples for all main channels
            all_channel_data = []
            for i in range(self.samples_per_channel):
                t = self.current_time + (i / self.sample_rate)
                base_value = self.offset + self.amplitude_scaled * math.sin(2 * math.pi * self.frequency * t)
                rounded_value = int(round(base_value))
                all_channel_data.append(rounded_value)

            self.current_time += self.time_per_message

            # Interleave channel data (8192 = 4096 samples * 2 channels)
            interleaved = []
            for i in range(self.samples_per_channel):
                for ch in range(self.num_channels):
                    interleaved.append(all_channel_data[i])  # Same data for both channels

            if len(interleaved) != self.samples_per_channel * self.num_channels:
                logging.error(f"Interleaved data length incorrect: expected {self.samples_per_channel * self.num_channels}, got {len(interleaved)}")
                return

            # Generate tacho frequency data (4096 samples, constant frequency)
            tacho_freq_data = [self.frequency * 100] * self.samples_per_channel  # Scaled up for compatibility

            # Build header
            header = [
                self.frame_index % 65535,  # Frame index low
                self.frame_index // 65535,  # Frame index high
                self.num_channels,         # Number of channels (2)
                self.sample_rate,          # Sample rate (4096)
                16,                        # Bit depth
                self.samples_per_channel,  # Samples per channel (4096)
                self.num_tacho_channels,   # Number of tacho channels (1)
                0, 0, 0                   # Reserved
            ]
            while len(header) < 100:
                header.append(0)

            # Combine all data
            message_values = header + interleaved + tacho_freq_data
            total_expected = 100 + (self.samples_per_channel * self.num_channels) + (self.samples_per_channel * self.num_tacho_channels)
            if len(message_values) != total_expected:
                logging.error(f"Message length incorrect: expected {total_expected}, got {len(message_values)}")
                return

            # Log sample data for debugging
            logging.debug(f"Header: {header}")
            logging.debug(f"Main channel data (first 5): {interleaved[:5]}")
            logging.debug(f"Tacho freq data (first 5): {tacho_freq_data[:5]}")

            # Convert to binary
            binary_message = struct.pack(f"<{len(message_values)}H", *message_values)

            # Publish to all topics
            for topic in self.topics:
                try:
                    publish.single(topic, binary_message, hostname=self.broker, qos=1)
                    logging.info(f"[{self.count}] Published to {topic}: frame {self.frame_index}, {len(message_values)} values")
                except Exception as e:
                    logging.error(f"Failed to publish to {topic}: {str(e)}")

            self.frame_index += 1
            self.count += 1
        except Exception as e:
            logging.error(f"Error in publish_message: {str(e)}")

if __name__ == "__main__":
    app = QApplication([])
    broker = "192.168.1.233"
    topics = ["sarayu/d1/topic1"]
    mqtt_publisher = MQTTPublisher(broker, topics)
    app.exec_()