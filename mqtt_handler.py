import paho.mqtt.client as mqtt
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import struct
import json
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
                if payload_length < 20 or payload_length % 2 != 0:
                    logging.warning(f"Invalid payload length: {payload_length} bytes")
                    return

                num_samples = payload_length // 2
                try:
                    values = struct.unpack(f"<{num_samples}H", payload)
                except struct.error as e:
                    logging.error(f"Failed to unpack payload of {num_samples} uint16_t: {str(e)}")
                    return

                if len(values) < 100:
                    logging.warning(f"Payload too short: {len(values)} samples")
                    return

                header = values[:100]
                total_values = values[100:]

                num_channels = header[2] if len(header) > 2 and header[2] > 0 else 4
                sample_rate = header[3] if len(header) > 3 and header[3] > 0 else 4096
                samples_per_channel = 4096
                num_tacho_channels = header[6] if len(header) > 6 and header[6] > 0 else 2

                expected_main = samples_per_channel * num_channels
                expected_tacho = 4096 * num_tacho_channels
                expected_total = expected_main + expected_tacho

                if len(total_values) != expected_total:
                    logging.warning(f"Unexpected data length: got {len(total_values)}, expected {expected_total}")
                    return

                main_data = total_values[:expected_main]
                tacho_freq_data = total_values[expected_main:expected_main + 4096]
                tacho_trigger_data = total_values[expected_main + 4096:expected_main + 8192]

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