import os, json
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

def publish_to_mqtt(result: dict):
    broker = os.getenv("MQTT_BROKER_URL")
    username = os.getenv("MQTT_USERNAME")
    password = os.getenv("MQTT_PASSWORD")

    client = mqtt.Client()
    client.username_pw_set(username.split(":")[1], password)
    client.tls_set()
    client.connect(broker, 8883)
    client.loop_start()
    client.publish("coldconnect/shelf/scan", json.dumps(result))
    client.loop_stop()
