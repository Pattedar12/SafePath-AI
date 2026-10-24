pip install -r requirements.txt
import base64
import os
import json
import time
import threading
from datetime import datetime

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

# Native computer audio avoids browser iframe autoplay restrictions.
try:
    import winsound  # Windows beep output
except ImportError:
    winsound = None
try:
    import pyttsx3  # Offline text-to-speech
except ImportError:
    pyttsx3 = None
from urllib.parse import quote_plus

import folium
from streamlit_folium import st_folium

import cv2
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from ultralytics import YOLO

# Load API keys from a local .env file automatically.
load_dotenv()

# ============================================================
# SAFEPATH AI — Computer vision + weather + optional Google APIs
# ============================================================
st.set_page_config(page_title="SafePath AI", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at 12% 0%, #163c50 0%, transparent 38%), radial-gradient(ellipse at 92% 12%, #183b37 0%, transparent 34%), linear-gradient(145deg,#07111d 0%,#0a1724 52%,#0b2028 100%); color:#e8f4fb; font-family:Inter,system-ui,sans-serif; }
[data-testid="stHeader"] { background:rgba(7,17,29,.72); }
[data-testid="stSidebar"] { background:linear-gradient(165deg,rgba(13,34,51,.98),rgba(7,20,31,.99)); border-right:1px solid rgba(113,218,209,.18); box-shadow:10px 0 45px rgba(0,0,0,.22); }
[data-testid="stSidebar"] * { color:#e6f7fb !important; }
[data-testid="stMain"] { color:#e8f4fb; }
h1,h2,h3 { color:#e8fbff !important; letter-spacing:-.04em; font-weight:850 !important; }
h1 { text-shadow:0 0 28px rgba(59,210,198,.17); }
p, label, [data-testid="stCaptionContainer"] { color:#a8c2d0; }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(28,58,78,.86),rgba(13,35,48,.9)); border:1px solid rgba(106,207,211,.24); padding:18px; border-radius:18px; box-shadow:0 12px 30px rgba(0,0,0,.24),inset 0 1px 0 rgba(255,255,255,.07); }
[data-testid="stMetricLabel"] { color:#a7c9d5 !important; }
[data-testid="stMetricValue"] { color:#73f0d1 !important; text-shadow:0 0 20px rgba(65,224,194,.12); }
.stButton button, [data-testid="stFormSubmitButton"] button { border-radius:12px; font-weight:750; border:1px solid rgba(108,217,210,.28); background:linear-gradient(145deg,rgba(31,68,88,.95),rgba(15,42,57,.95)); color:#e8fbff; box-shadow:0 7px 18px rgba(0,0,0,.18); transition:all .2s ease; }
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { background:linear-gradient(120deg,#12a98f,#47d9b4); color:#061b20; border:0; box-shadow:0 8px 24px rgba(27,211,173,.24); }
.stButton button:hover { border-color:#5ae6d1; transform:translateY(-2px); box-shadow:0 12px 25px rgba(0,0,0,.26); }
[data-testid="stAlert"] { border-radius:14px; background:rgba(17,43,57,.9); }
[data-testid="stExpander"] { border:1px solid rgba(108,207,211,.22); border-radius:15px; background:rgba(15,38,52,.72); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; border-color:rgba(108,207,211,.22) !important; background:rgba(14,36,49,.4); }
hr { border-color:rgba(113,218,209,.18); }
[data-testid="stDataFrame"], iframe, [data-testid="stImage"] img { border-radius:16px; }
[data-testid="stImage"] img { border:1px solid rgba(109,218,211,.2); box-shadow:0 18px 44px rgba(0,0,0,.28); }
.small-muted { color:#9bb7c3; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
# YOLOv8n pretrained on COCO can detect 80 common classes (people, vehicles,
# traffic lights, signs, animals, furniture, etc.). Doors/windows are NOT COCO classes.
COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
}
ROAD_CLASSES = {"person", "car", "motorcycle", "bus", "truck", "bicycle", "traffic light", "stop sign", "fire hydrant", "bench"}

@st.cache_resource
def load_model(model_name=None):
    """Load configured custom weights when supplied; otherwise use COCO YOLO."""
    model_path = model_name or os.getenv("SAFEPATH_YOLO_MODEL", "yolov8n.pt")
    return YOLO(model_path)




def play_local_voice_alert(message="Warning! Obstacle detected. Please check your surroundings and move carefully."):
    """Play a local Windows beep and speak a concise obstacle warning."""
    errors = []
    if winsound is not None:
        try:
            for frequency, duration in ((1400, 250), (1800, 250), (1400, 350)):
                winsound.Beep(frequency, duration)
                time.sleep(0.08)
        except Exception as exc:
            errors.append(f"Beep failed: {exc}")
    else:
        errors.append("Windows beep output (winsound) is unavailable on this computer.")

    if pyttsx3 is not None:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            engine.say(message)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            errors.append(f"Voice speech failed: {exc}")
    else:
        errors.append("pyttsx3 is not installed. Run: python -m pip install pyttsx3")
    return errors

def start_audio_alert_async(message):
    """Run local audio in a background thread so live camera frames keep updating."""
    lock = st.session_state.get("audio_alert_lock")
    if lock is None:
        lock = threading.Lock()
        st.session_state["audio_alert_lock"] = lock

    def worker():
        if not lock.acquire(blocking=False):
            return
        try:
            play_local_voice_alert(message)
        finally:
            lock.release()

    threading.Thread(target=worker, daemon=True).start()


# Thread-safe local audio controller. Tiers are intentionally quiet at distance and
# reserve spoken warnings for the close danger zone.
_AUDIO_STATE = {"active": False, "distance": None, "level": 0}
_AUDIO_LOCK = threading.Lock()
_AUDIO_THREAD_STARTED = False


def proximity_level(distance):
    """Return 0=off, 1=slow, 2=medium, 3=fast, 4=critical."""
    if distance is None or distance > 50:
        return 0
    if distance > 35:
        return 1
    if distance > 20:
        return 2
    if distance > 10:
        return 3
    return 4


def update_proximity_audio(active, distance=None):
    """Update the audio controller without blocking camera processing."""
    level = proximity_level(distance) if active else 0
    with _AUDIO_LOCK:
        _AUDIO_STATE["active"] = level > 0
        _AUDIO_STATE["distance"] = distance
        _AUDIO_STATE["level"] = level
    global _AUDIO_THREAD_STARTED
    if not _AUDIO_THREAD_STARTED:
        _AUDIO_THREAD_STARTED = True
        threading.Thread(target=_proximity_audio_worker, daemon=True, name="SafePathAudio").start()


def _proximity_audio_worker():
    """Non-blocking-to-camera alert loop; speak only when the object is close."""
    last_voice = 0.0
    engine = None
    if pyttsx3 is not None:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
        except Exception:
            engine = None
    last_beep_at = 0.0
    last_level = 0
    while True:
        with _AUDIO_LOCK:
            level = _AUDIO_STATE["level"]
            distance = _AUDIO_STATE["distance"]
        if level == 0 or distance is None:
            last_level = 0
            time.sleep(0.08)
            continue
        # Far: soft/short beep every 2 seconds. Closer: more frequent and higher pitch.
        settings = {
            1: (850, 65, 1.9),
            2: (1150, 75, 0.9),
            3: (1550, 85, 0.38),
            4: (1950, 100, 0.16),
        }
        frequency, duration, interval = settings[level]
        now = time.monotonic()
        if winsound is not None and now - last_beep_at >= interval:
            try:
                winsound.Beep(frequency, duration)
            except Exception:
                pass
            last_beep_at = time.monotonic()
        # No spoken warning for distant objects. Voice only at <=20 cm, repeated slowly.
        if distance <= 20 and engine is not None and time.time() - last_voice >= 4.5:
            try:
                engine.say("Warning. Object very close. Stop and check your path.")
                engine.runAndWait()
            except Exception:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 175)
                    engine.setProperty("volume", 1.0)
                except Exception:
                    engine = None
            last_voice = time.time()
        last_level = level
        time.sleep(0.04)


def send_esp8266_camera_alert(active):
    """Send a backward-compatible ON/OFF command to the connected board."""
    connection = st.session_state.get("esp32_monitor_serial")
    if connection is None or not getattr(connection, "is_open", False):
        return False
    try:
        command = "CAMERA_ALERT:ON\n" if active else "CAMERA_ALERT:OFF\n"
        connection.write(command.encode("utf-8"))
        connection.flush()
        return True
    except Exception:
        return False


def send_hardware_alert_level(level):
    """Tell compatible ESP32 firmware the graded beep/vibration level (0-4)."""
    connection = st.session_state.get("esp32_monitor_serial")
    if connection is None or not getattr(connection, "is_open", False):
        return False
    try:
        connection.write(f"ALERT_LEVEL:{int(level)}\n".encode("utf-8"))
        connection.flush()
        return True
    except Exception:
        return False


def available_serial_ports():
    """Return detected serial port names, or an empty list if pyserial is missing."""
    if list_ports is None:
        return []
    try:
        return [port.device for port in list_ports.comports()]
    except Exception:
        return []


def open_camera(camera_index=0):
    """Open a local webcam using common Windows backends, releasing failed attempts."""
    if os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    errors = []
    # Try the selected camera first, then other common indexes.
    indexes = list(dict.fromkeys([int(camera_index), 0, 1, 2]))
    for index in indexes:
        for backend in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        return cap, index, None
                errors.append(f"Camera {index}, backend {backend}: no frame")
            except Exception as exc:
                errors.append(f"Camera {index}, backend {backend}: {exc}")
            if cap is not None:
                cap.release()
    return None, None, "Could not open a working webcam. Close Zoom/Teams/Camera apps, allow camera access in Windows Settings, and try another camera device. " + "; ".join(errors[-3:])


def read_ultrasonic_distance(serial_connection):
    """Read distance values from common ESP32 serial formats and retain partial packets."""
    import re
    if serial_connection is None or not getattr(serial_connection, "is_open", False):
        return None, "ESP32 is not connected."
    try:
        if "esp32_serial_buffer" not in st.session_state:
            st.session_state["esp32_serial_buffer"] = b""
        waiting = int(getattr(serial_connection, "in_waiting", 0))
        if waiting:
            st.session_state["esp32_serial_buffer"] += serial_connection.read(waiting)
        buffer = st.session_state["esp32_serial_buffer"]
        lines = buffer.split(b"\n")
        st.session_state["esp32_serial_buffer"] = lines.pop()[-2048:]
        latest_distance = None
        latest_line = ""
        for raw_line in lines:
            line = raw_line.decode("utf-8", errors="ignore").replace("\r", "").strip()
            if not line:
                continue
            latest_line = line
            # Accept DISTANCE_CM:23.4, DISTANCE_CM=23.4, Distance: 23.4 cm,
            # JSON such as {"distance_cm":23.4}, or a line containing just 23.4.
            candidates = []
            match = re.search(r"(?:DISTANCE(?:_CM)?|distance_cm|Ultrasonic\s+distance)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", line, re.I)
            if match:
                candidates.append(match.group(1))
            else:
                try:
                    payload = json.loads(line)
                    for key in ("distance_cm", "distance", "Distance", "DISTANCE_CM"):
                        if key in payload:
                            candidates.append(str(payload[key]).replace("cm", "").strip())
                            break
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
                if not candidates:
                    numeric = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(?:cm)?\s*", line, re.I)
                    if numeric:
                        candidates.append(numeric.group(1))
            for raw_value in candidates:
                try:
                    value = float(raw_value)
                    if 2.0 <= value <= 400.0:
                        latest_distance = value
                except (ValueError, TypeError):
                    continue
        if latest_line:
            st.session_state["esp32_last_serial_line"] = latest_line
        if latest_distance is not None:
            st.session_state["esp32_last_distance"] = latest_distance
            st.session_state["esp32_last_distance_time"] = time.time()
            return latest_distance, None
        last_time = st.session_state.get("esp32_last_distance_time", 0)
        if last_time and time.time() - last_time <= 2.5:
            return st.session_state.get("esp32_last_distance"), None
        last_line = st.session_state.get("esp32_last_serial_line", "")
        if last_line:
            return None, f"ESP8266 data received but not a distance reading. Received: {last_line[:120]}"
        return None, "No serial data received. Upload the SafePath ESP8266 firmware, select the correct COM port, and use 115200 baud."
    except Exception as exc:
        return None, f"Serial read failed: {exc}"


def get_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def analyze_frame(model, image, confidence=0.30, image_gap_threshold=0.08, imgsz=416):
    """Run YOLO + OpenCV-compatible image analysis. Proximity is image-space only."""
    result = model.predict(source=image, conf=confidence, verbose=False, imgsz=imgsz, max_det=25)[0]
    annotated = result.plot()
    height, width = image.shape[:2]
    diagonal = max(float(np.hypot(width, height)), 1.0)
    detections = []
    people = []
    vehicles = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label = result.names[class_id]
        score = float(box.conf[0].item())
        coords = [float(v) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = coords
        record = {"Object": label, "Confidence": round(score * 100, 1), "Box": (x1, y1, x2, y2)}
        detections.append(record)
        if label == "person":
            people.append(record)
        elif label in VEHICLE_CLASSES:
            vehicles.append(record)

    closest_gap = None
    close_pair = False
    for person in people:
        px1, py1, px2, py2 = person["Box"]
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle["Box"]
            # Minimum gap between the two rectangles; 0 means overlapping boxes.
            dx = max(vx1 - px2, px1 - vx2, 0.0)
            dy = max(vy1 - py2, py1 - vy2, 0.0)
            gap_ratio = float(np.hypot(dx, dy) / diagonal)
            closest_gap = gap_ratio if closest_gap is None else min(closest_gap, gap_ratio)
            if gap_ratio <= image_gap_threshold:
                close_pair = True

    summary = {
        "objects": len(detections),
        "pedestrians": len(people),
        "vehicles": len(vehicles),
        "both_present": bool(people and vehicles),
        "close_pair": close_pair,
        "closest_gap": closest_gap,
    }
    clean_detections = [{"Object": d["Object"], "Confidence (%)": d["Confidence"]} for d in detections]
    return annotated, clean_detections, summary, result


def show_detection_summary(detections, summary, show_details=True, show_safety_messages=True):
    a, b, c, d = st.columns(4)
    a.metric("Objects detected", summary["objects"])
    b.metric("Pedestrians", summary["pedestrians"])
    c.metric("Vehicles / bicycles", summary["vehicles"])
    d.metric("Visual proximity", "CHECK" if summary["close_pair"] else "Not flagged")

    if show_safety_messages:
        if summary["both_present"] and summary["close_pair"]:
            st.error("🚨 Possible visual overlap detected. This image-space estimate is not a real-world distance measurement.")
        elif summary["both_present"]:
            st.info("Pedestrian and vehicle are visible. Real-world proximity requires a distance sensor.")
        elif summary["pedestrians"]:
            st.info("Pedestrian detected. Real-world distance is not measured by the camera.")
        elif summary["vehicles"]:
            st.info("Vehicle detected. Real-world distance is not measured by the camera.")
        else:
            st.success("No pedestrian or listed vehicle detected in this frame.")

        if summary["closest_gap"] is not None:
            st.caption(f"Visual bounding-box gap: {summary['closest_gap']:.1%} of the image diagonal. This is not a real-world distance or collision probability.")
    if detections:
        counts = {}
        for item in detections:
            counts[item["Object"]] = counts.get(item["Object"], 0) + 1
        st.caption("Detected objects: " + ", ".join(f"{name} ×{count}" for name, count in sorted(counts.items())))
    if show_details and detections:
        with st.expander("View detection details", expanded=False):
            st.dataframe(detections, use_container_width=True, hide_index=True)


def gemini_scene_summary(api_key, image_bgr, summary, detections):
    """Optional Google Gemini API call using the current image and YOLO results."""
    if not api_key.strip():
        raise ValueError("Add your Gemini API key in the sidebar first.")
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise ValueError("Could not encode the image for Gemini.")
    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    prompt = (
        "You are assisting a student road-safety research prototype. Describe only visible, cautious observations. "
        "Do not claim to know actual distance, speed, intent, collision probability, or that the road is safe. "
        "Give: (1) brief scene summary, (2) observed road users, (3) potential concerns to inspect, (4) limitations. "
        f"YOLO summary: {summary}. Detected labels/confidences: {detections[:30]}"
    )
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}]}],
               "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500}}
    response = requests.post(endpoint, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"}, json=payload, timeout=35)
    if not response.ok:
        try:
            details = response.json().get("error", {}).get("message", response.text)
        except Exception:
            details = response.text
        raise RuntimeError(f"Gemini API error ({response.status_code}): {details}")
    data = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    if not text:
        raise RuntimeError("Gemini returned no text. Check the key, model access, and API quota.")
    return text



def groq_scene_explanation(api_key, summary, detections):
    """Use Groq to explain YOLO's textual detections; the model is not given the image itself."""
    if not api_key.strip():
        raise ValueError("Add your Groq API key to .env as GROQ_API_KEY first.")
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You assist with a student road-safety research prototype. "
                    "Only discuss the supplied object-detection labels and counts. "
                    "Do not claim actual distances, speeds, intent, collision probability, "
                    "or that a scene is safe. Clearly say that detections may be wrong. "
                    "Give a concise scene summary, potential things to inspect, and limitations."
                ),
            },
            {
                "role": "user",
                "content": f"YOLO detection summary: {summary}. Detected labels and confidences: {detections[:40]}. Explain these results cautiously.",
            },
        ],
        "temperature": 0.2,
        "max_tokens": 450,
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"},
        json=payload,
        timeout=35,
    )
    if not response.ok:
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"Groq API error ({response.status_code}): {detail}")
    data = response.json()
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Groq returned an empty response. Check model access and API limits.")
    return answer



@st.cache_data(ttl=300, show_spinner=False)
def geocode_place(place):
    """Find a place coordinate using OpenStreetMap Nominatim."""
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": "SafePathAI-StudentPrototype/1.0 (route planner)"},
        timeout=20,
    )
    response.raise_for_status()
    items = response.json()
    if not items:
        raise ValueError(f"Could not find the place: {place}. Try adding city/area details.")
    return {"lat": float(items[0]["lat"]), "lon": float(items[0]["lon"]), "name": items[0].get("display_name", place)}


@st.cache_data(ttl=300, show_spinner=False)
def get_walking_route(origin_lat, origin_lon, dest_lat, dest_lon):
    """Fetch a walking route from an OpenStreetMap-based foot-routing service."""
    url = (
        "https://routing.openstreetmap.de/routed-foot/route/v1/driving/"
        f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    )
    response = requests.get(
        url,
        params={"overview": "full", "geometries": "geojson", "steps": "false"},
        headers={"User-Agent": "SafePathAI-StudentPrototype/1.0"},
        timeout=35,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError("The walking route service could not calculate this route. Try different place names.")
    route = data["routes"][0]
    return {
        "coordinates": route["geometry"]["coordinates"],
        "distance_m": float(route.get("distance", 0)),
        "duration_s": float(route.get("duration", 0)),
    }


def get_traffic_flow(latitude, longitude, api_key):
    """Retrieve live road-flow data from TomTom when the user supplies a key."""
    if not api_key.strip():
        return None
    response = requests.get(
        "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
        params={"point": f"{latitude},{longitude}", "unit": "KMPH", "key": api_key.strip()},
        timeout=20,
    )
    if not response.ok:
        try:
            message = response.json().get("error", {}).get("description", response.text)
        except Exception:
            message = response.text
        raise RuntimeError(f"Traffic provider returned HTTP {response.status_code}: {message}")
    flow = response.json().get("flowSegmentData")
    if not flow:
        raise RuntimeError("No live traffic-flow data was returned for this point.")
    current_speed = float(flow.get("currentSpeed", 0))
    free_speed = float(flow.get("freeFlowSpeed", 0))
    ratio = current_speed / free_speed if free_speed > 0 else None
    if flow.get("roadClosure"):
        label = "Road closure reported"
    elif ratio is None:
        label = "Traffic data available; speed comparison unavailable"
    elif ratio < 0.35:
        label = "Heavy / very slow traffic"
    elif ratio < 0.65:
        label = "Slow traffic"
    elif ratio < 0.85:
        label = "Moderate traffic"
    else:
        label = "Near free-flow speed"
    return {
        "current_speed": current_speed,
        "free_speed": free_speed,
        "travel_time_s": flow.get("currentTravelTime"),
        "free_time_s": flow.get("freeFlowTravelTime"),
        "road_closure": bool(flow.get("roadClosure", False)),
        "label": label,
        "confidence": flow.get("confidence"),
    }


# Sidebar / API setup
st.sidebar.title("🚦 SafePath AI")
page = st.sidebar.radio("Navigation", ["Dashboard", "AI Object Detection", "Live Traffic Monitor", "Weather & Safety", "Safe Route Planner", "ESP32 Hardware"])
st.sidebar.divider()
voice_alerts_enabled = st.sidebar.checkbox(
    "🔊 Enable close-hazard alerts", value=True,
    help="Audio is used only when a person is detected and the connected ultrasonic sensor reads within the alert distance."
)
st.sidebar.caption(datetime.now().strftime("%A, %d %B %Y"))
def configured_secret(name, default=""):
    """Read a key from Streamlit secrets first, then environment/.env."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()


with st.sidebar.expander("🔑 API connections & setup", expanded=False):
    st.caption("Keys auto-load from your local .env file or Streamlit secrets. You can also enter them here for this session.")
    gemini_key = st.text_input(
        "Google Gemini API key",
        value=configured_secret("GEMINI_API_KEY"),
        type="password",
        help="Loaded from .env automatically when configured.",
        key="gemini_api_key_input",
    ).strip()
    google_maps_key = st.text_input(
        "Google Maps JavaScript API key",
        value=configured_secret("GOOGLE_MAPS_API_KEY"),
        type="password",
        help="Used to render Google Maps and its live traffic layer inside this app. Restrict this key to Maps JavaScript API and your local/deployed website referrers.",
        key="google_maps_api_key_input",
    ).strip()
    groq_key = st.text_input(
        "Groq API key",
        value=configured_secret("GROQ_API_KEY"),
        type="password",
        help="Used for AI explanations of YOLO detection results. Keep the key in .env, not in source code.",
        key="groq_api_key_input",
    ).strip()
    tomtom_key = st.text_input(
        "TomTom Traffic API key",
        value=configured_secret("TOMTOM_API_KEY"),
        type="password",
        help="Required to display real-time traffic speeds.",
        key="tomtom_api_key_input",
    ).strip()
    st.caption("Gemini: " + ("🟢 Key configured" if gemini_key else "🟠 Key missing"))
    st.caption("Google Maps: " + ("🟢 Key configured" if google_maps_key else "🟠 Key missing"))
    st.caption("Groq: " + ("🟢 Key configured" if groq_key else "🟠 Key missing"))
    st.caption("TomTom Traffic: " + ("🟢 Key configured" if tomtom_key else "🟠 Key missing"))
    if st.button("💾 Save API keys on this computer", use_container_width=True):
        # Persist keys locally in .env; never display their values.
        key_values = {
            "GEMINI_API_KEY": gemini_key,
            "GOOGLE_MAPS_API_KEY": google_maps_key,
            "GROQ_API_KEY": groq_key,
            "TOMTOM_API_KEY": tomtom_key,
        }
        try:
            env_path = os.path.join(os.getcwd(), ".env")
            existing = {}
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as env_file:
                    for line in env_file:
                        if "=" in line and not line.lstrip().startswith("#"):
                            k, v = line.rstrip("\n").split("=", 1)
                            existing[k.strip()] = v.strip().strip('"').strip("'")
            existing.update({k: v for k, v in key_values.items() if v})
            with open(env_path, "w", encoding="utf-8") as env_file:
                for k, v in existing.items():
                    env_file.write(f'{k}="{v.replace(chr(34), chr(92)+chr(34))}"\n')
            st.success("Saved locally to .env. Restart SafePath AI to reload saved keys. Do not share this file or commit it to Git.")
        except Exception as exc:
            st.error(f"Could not save .env: {exc}")

    if st.button("🧪 Test Groq API", use_container_width=True):
        if not groq_key:
            st.error("Groq key missing. Add GROQ_API_KEY to your .env file or enter it above.")
        else:
            try:
                test_response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Reply with exactly: SafePath Groq API connected"}], "temperature": 0, "max_tokens": 30},
                    timeout=25,
                )
                if test_response.ok:
                    st.success("Groq API connected successfully.")
                else:
                    try:
                        detail = test_response.json().get("error", {}).get("message", test_response.text)
                    except Exception:
                        detail = test_response.text
                    st.error(f"Groq API test failed ({test_response.status_code}): {detail}")
            except Exception as exc:
                st.error(f"Groq connection failed: {exc}")
    if st.button("🧪 Test Google Gemini API", use_container_width=True):
        if not gemini_key:
            st.error("Gemini key missing. Add GEMINI_API_KEY to your .env file or enter it above.")
        else:
            try:
                test_response = requests.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent",
                    headers={"x-goog-api-key": gemini_key, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": "Reply with exactly: SafePath API connected"}]}],
                          "generationConfig": {"temperature": 0, "maxOutputTokens": 30}},
                    timeout=25,
                )
                if test_response.ok:
                    st.success("Gemini API connected successfully.")
                else:
                    try:
                        detail = test_response.json().get("error", {}).get("message", test_response.text)
                    except Exception:
                        detail = test_response.text
                    st.error(f"Gemini API test failed ({test_response.status_code}): {detail}")
            except Exception as exc:
                st.error(f"Gemini connection failed: {exc}")
    if st.button("🧪 Check Google Maps key setup", use_container_width=True):
        if not google_maps_key:
            st.error("Google Maps key missing. Add GOOGLE_MAPS_API_KEY to .env or enter it above.")
        else:
            st.info("Google Maps key is configured. The embedded map will confirm whether Maps JavaScript API is enabled and the key's restrictions allow this app.")
    if st.button("🧪 Test TomTom Traffic API", use_container_width=True):
        if not tomtom_key:
            st.error("TomTom key missing. Add TOMTOM_API_KEY to your .env file or enter it above.")
        else:
            try:
                test_flow = get_traffic_flow(12.9716, 77.5946, tomtom_key)
                st.success(f"TomTom API connected. Current speed near central Bengaluru: {test_flow['current_speed']:.0f} km/h; {test_flow['label']}.")
            except Exception as exc:
                st.error(f"TomTom API test failed: {exc}")
    st.caption("Google Maps and its live traffic layer are rendered inside this app when GOOGLE_MAPS_API_KEY is configured. OpenStreetMap routing is used to calculate the walking path.")
st.sidebar.caption("Research prototype — not a certified road-safety system.")

st.title("🚦 SafePath AI")
st.subheader("INTELLIGENT MOBILITY  ·  REAL-TIME SAFETY")
st.caption("AI vision  ·  Live traffic intelligence  ·  Weather-aware routes  ·  Assistive mobility")
st.divider()

if page == "Dashboard":
    st.header("📊 System Dashboard")
    a, b, c, d = st.columns(4)
    a.metric("Application", "Running")
    b.metric("Vision model", "YOLOv8n")
    c.metric("Image processing", "OpenCV")
    d.metric("Hardware", "Not connected")
    st.markdown("### Modules")
    x, y = st.columns(2)
    with x:
        st.info("**🤖 Live AI Vision**\n\nUse the live camera to detect supported objects. Beep cadence increases as a fresh ultrasonic reading gets closer; voice warnings begin in the critical zone.")
        st.info("**🌦️ Weather awareness**\n\nRetrieve temperature, humidity, precipitation, and wind speed for entered coordinates.")
    with y:
        st.info("**⚡ Groq AI + Google Gemini**\n\nGroq can explain YOLO detection labels and counts. Gemini can optionally analyze the actual image. Each requires its own API key.")
        st.info("**🚶 In-app Walking Route**\n\nView walking directions and the Google live traffic overlay inside SafePath AI. Configure GOOGLE_MAPS_API_KEY for the embedded map; OpenStreetMap routing supplies the walking line.")
    hardware = st.session_state.get("esp32_monitor_serial")
    hardware_connected = hardware is not None and getattr(hardware, "is_open", False)
    st.info("Hardware status: " + (f"Connected on {st.session_state.get('esp32_monitor_port', 'serial port')}" if hardware_connected else "Not connected — open ESP32 Hardware to connect the board."))
    st.warning("The camera's visual proximity estimate is not a real-world distance or collision prediction. The physical buzzer is controlled from the ultrasonic sensor threshold, not merely because a person or vehicle appears in the camera.")

elif page == "AI Object Detection":
    st.header("🤖 Live AI Road-Safety Detection")
    st.markdown("Monitor the live camera feed. The detection model uses sensible defaults automatically, so there are no confusing tuning controls.")
    st.caption("For proximity alerts, connect the ultrasonic sensor in ESP32 Hardware. Camera proximity alerts require a detected object and a fresh ultrasonic reading within 50 cm. Beeps accelerate at closer distances; voice warning starts at 15 cm.")
    camera_index = 0  # Use the default webcam; advanced device selector removed.
    if st.button("▶ Start Live Detection", type="primary", use_container_width=True):
        cap = None
        try:
            with st.spinner("Loading AI vision model..."):
                model = load_model()
            cap, actual_camera_index, camera_error = open_camera(camera_index)
            if cap is None:
                st.error(camera_error or "Could not open camera.")
            else:
                st.success(f"Camera connected (device {actual_camera_index}). Live analysis started.")
                # Low-latency capture settings reduce stale-frame buildup.
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                except Exception:
                    pass
                frame_area = st.empty()
                status_area = st.empty()
                metric_area = st.empty()
                previous_obstacle_state = False
                previous_camera_buzzer_state = False
                last_audio_alert_time = 0.0
                last_camera_detection_alert = 0.0
                frame_count = 0
                session_started = time.time()
                # Fixed internal defaults keep the user interface simple.
                confidence = 0.30
                gap_threshold = 0.08
                cached_annotated = None
                detections, summary, result = [], {"objects": 0, "pedestrians": 0, "vehicles": 0, "both_present": False, "close_pair": False, "closest_gap": None}, None
                while time.time() - session_started < 300:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        status_area.warning("Waiting for a camera frame… Check that another app is not using the webcam.")
                        time.sleep(0.08)
                        continue
                    frame_count += 1
                    # Run AI on every third frame at lower resolution; display frames between
                    # inferences so the preview keeps moving on ordinary laptops.
                    run_inference = (frame_count == 1 or frame_count % 3 == 0 or cached_annotated is None)
                    if run_inference:
                        h, w = frame.shape[:2]
                        if w > 480:
                            frame = cv2.resize(frame, (480, int(h * 480 / w)), interpolation=cv2.INTER_AREA)
                        annotated, detections, summary, result = analyze_frame(
                            model, frame, confidence, gap_threshold, imgsz=320
                        )
                        cached_annotated = annotated
                    frame_area.image(cv2.cvtColor(cached_annotated, cv2.COLOR_BGR2RGB), caption="Live AI camera · lightweight processing", use_container_width=True)
                    with metric_area.container():
                        show_detection_summary(detections, summary, show_details=False, show_safety_messages=False)
                    visible_labels = sorted({item["Object"] for item in detections})
                    any_camera_object = bool(detections)
                    active_serial = st.session_state.get("esp32_monitor_serial")
                    if active_serial is not None and getattr(active_serial, "is_open", False):
                        read_ultrasonic_distance(active_serial)
                    distance = st.session_state.get("esp32_last_distance")
                    distance_time = st.session_state.get("esp32_last_distance_time", 0)
                    sensor_is_fresh = distance is not None and (time.time() - distance_time) <= 2.5
                    alert_threshold = float(st.session_state.get("hardware_alert_threshold", 50))
                    close_hazard = bool(sensor_is_fresh and distance <= alert_threshold)
                    # Distance sensor drives graded alerts even if YOLO misses an object label.
                    current_level = (proximity_level(distance) if sensor_is_fresh and distance <= alert_threshold else 0)
                    update_proximity_audio(close_hazard, distance if sensor_is_fresh else None)
                    if current_level != st.session_state.get("last_hardware_alert_level"):
                        send_hardware_alert_level(current_level)
                        st.session_state["last_hardware_alert_level"] = current_level

                    # Camera detections also trigger a local PC beep + spoken warning,
                    # independently of the ultrasonic sensor. Cooldown prevents constant speech.
                    now_alert = time.time()
                    if any_camera_object and now_alert - last_camera_detection_alert >= 5.0:
                        start_audio_alert_async(
                            "Warning. Object detected. Please check your surroundings and move carefully."
                        )
                        last_camera_detection_alert = now_alert

                    if close_hazard and distance <= 15:
                        status_area.error(f"🛑 CRITICAL — STOP: {', '.join(visible_labels[:4])}. {distance:.1f} cm. Rapid beeps + voice warning active.")
                    elif close_hazard and distance <= 20:
                        status_area.error(f"🚨 VERY CLOSE: {', '.join(visible_labels[:4])}. {distance:.1f} cm. Rapid high-pitched beeps.")
                    elif close_hazard and distance <= 35:
                        status_area.info(f"Proximity signal: {distance:.1f} cm. Beep and vibration are increasing.")
                    elif close_hazard:
                        status_area.info(f"Object nearby at {distance:.1f} cm. Soft beep/vibration only; no spoken warning.")
                    elif not any_camera_object:
                        status_area.success("Clear view — no supported object detected. Camera alert is silent.")
                    elif sensor_is_fresh:
                        status_area.success(f"Object detected ({', '.join(visible_labels[:4])}); sensor reads {distance:.1f} cm. Outside alert zone — no camera-triggered warning or beep.")
                    else:
                        status_area.warning(
                            f"Camera detected: {', '.join(visible_labels[:4])}. PC beep/voice warning enabled; "
                            "distance-based hardware alert needs a fresh ultrasonic reading."
                        )

                    if close_hazard != previous_camera_buzzer_state:
                        send_esp8266_camera_alert(close_hazard)
                        previous_camera_buzzer_state = close_hazard
                    previous_obstacle_state = close_hazard
                    time.sleep(0.025)
                status_area.info("Live session ended after five minutes. Start Live Detection again to run another session.")
        except Exception as exc:
            st.error(f"Live camera error: {exc}")
        finally:
            send_esp8266_camera_alert(False)
            send_hardware_alert_level(0)
            st.session_state["last_hardware_alert_level"] = 0
            update_proximity_audio(False, None)
            if cap is not None:
                cap.release()

elif page == "Live Traffic Monitor":
    st.header("🚦 Live Traffic Intelligence")
    st.write("Check current road-flow conditions at a selected location. Traffic speed figures require a TomTom Traffic API key; the interactive Google traffic map requires a Google Maps JavaScript API key.")
    with st.container(border=True):
        st.markdown("#### 📍 Select monitoring location")
        loc_a, loc_b = st.columns(2)
        with loc_a:
            traffic_lat = st.number_input("Latitude", value=12.9716, min_value=-90.0, max_value=90.0, format="%.5f", key="traffic_monitor_lat")
        with loc_b:
            traffic_lon = st.number_input("Longitude", value=77.5946, min_value=-180.0, max_value=180.0, format="%.5f", key="traffic_monitor_lon")
        st.caption("Default location: central Bengaluru. Change the coordinates to monitor another road segment.")
        refresh_traffic = st.button("🔄 Fetch Latest Traffic Data", type="primary", use_container_width=True)
    if refresh_traffic:
        if not tomtom_key:
            st.session_state["traffic_monitor_error"] = "Add a TomTom Traffic API key in the API connections panel to retrieve numerical live speed data."
            st.session_state.pop("traffic_monitor_result", None)
        else:
            try:
                st.session_state["traffic_monitor_result"] = get_traffic_flow(traffic_lat, traffic_lon, tomtom_key)
                st.session_state["traffic_monitor_checked_at"] = datetime.now().strftime("%d %b %Y, %H:%M:%S")
                st.session_state.pop("traffic_monitor_error", None)
            except Exception as exc:
                st.session_state["traffic_monitor_error"] = str(exc)
                st.session_state.pop("traffic_monitor_result", None)
    if st.session_state.get("traffic_monitor_error"):
        st.warning(st.session_state["traffic_monitor_error"])
    flow = st.session_state.get("traffic_monitor_result")
    if flow:
        st.caption(f"Latest provider response: {st.session_state.get('traffic_monitor_checked_at', 'time unavailable')} · refresh manually for an updated reading.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current road speed", f"{flow['current_speed']:.0f} km/h")
        m2.metric("Free-flow speed", f"{flow['free_speed']:.0f} km/h")
        m3.metric("Road status", "Closure reported" if flow["road_closure"] else flow["label"])
        m4.metric("Traffic data confidence", str(flow["confidence"] if flow["confidence"] is not None else "N/A"))
        if flow["free_speed"] > 0:
            slowdown = max(0.0, (1 - flow["current_speed"] / flow["free_speed"]) * 100)
            st.progress(min(1.0, slowdown / 100.0), text=f"Estimated speed reduction versus free flow: {slowdown:.0f}%")
        if flow["road_closure"]:
            st.error("⚠️ The traffic provider reports a road closure for this road segment. Verify local signage before travelling.")
        elif flow["label"] == "Heavy / very slow traffic":
            st.warning("Heavy traffic reported. Use designated pedestrian crossings and allow extra time.")
        elif flow["label"] == "Slow traffic":
            st.warning("Slow traffic reported. Stay alert near intersections and turning vehicles.")
        else:
            st.success("Traffic speed is not currently classified as slow by the provider. Always follow road signals and local conditions.")
    st.subheader("🗺️ Live Traffic Map")
    if google_maps_key:
        traffic_map_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>html,body,#map{{height:100%;width:100%;margin:0;background:#0b1120;font-family:Arial}}</style></head><body><div id="map" style="height:500px"></div><script>function initMap(){{const center={{lat:{traffic_lat},lng:{traffic_lon}}};const map=new google.maps.Map(document.getElementById('map'),{{center:center,zoom:13,mapTypeControl:true,streetViewControl:false,fullscreenControl:true}});new google.maps.Marker({{map:map,position:center,title:'Selected traffic monitoring point'}});new google.maps.TrafficLayer().setMap(map);}}window.gm_authFailure=function(){{document.body.innerHTML='<p style="color:white;padding:20px">Google Maps authentication failed. Check API enablement and key restrictions.</p>'}};</script><script async defer src="https://maps.googleapis.com/maps/api/js?key={google_maps_key}&callback=initMap&v=weekly"></script></body></html>"""
        components.html(traffic_map_html, height=515, scrolling=False)
        st.caption("Google's traffic layer displays road congestion where coverage is available and refreshes periodically. Numerical speeds come separately from TomTom.")
    else:
        st.info("Configure GOOGLE_MAPS_API_KEY in the sidebar to show the interactive map with the live traffic layer. You can still use TomTom numerical readings above.")
    st.caption("Research prototype: traffic provider coverage, update frequency, and confidence vary by road. Traffic readings are not a guarantee of pedestrian safety.")

elif page == "Weather & Safety":
    st.header("🌦️ Weather & Environmental Awareness")
    c1, c2 = st.columns(2)
    with c1:
        latitude = st.number_input("Latitude", value=12.9716, min_value=-90.0, max_value=90.0, format="%.4f")
    with c2:
        longitude = st.number_input("Longitude", value=77.5946, min_value=-180.0, max_value=180.0, format="%.4f")
    if st.button("🌍 Get Current Weather", type="primary"):
        try:
            with st.spinner("Retrieving weather..."):
                current = get_weather(latitude, longitude)["current"]
            a, b = st.columns(2)
            a.metric("Temperature", f"{current['temperature_2m']} °C")
            b.metric("Humidity", f"{current['relative_humidity_2m']}%")
            a.metric("Precipitation", f"{current['precipitation']} mm")
            b.metric("Wind speed", f"{current['wind_speed_10m']} km/h")
            st.caption("Weather is contextual information, not a validated safety score.")
        except Exception as exc:
            st.error(f"Weather request failed: {exc}")

elif page == "Safe Route Planner":
    st.header("🗺️ In-App Walking Route & Live Traffic")
    st.write("Google Maps and its traffic overlay are displayed inside SafePath AI. No separate Google Maps webpage is opened. The walking path is calculated using an OpenStreetMap-based routing service.")

    with st.form("route_form"):
        origin = st.text_input("Starting point", value="HKBK College of Engineering, Bengaluru")
        destination = st.text_input("Destination", value="Yeshwantpur Railway Station, Bengaluru")
        submitted = st.form_submit_button("🧭 Show Walking Route Inside App", type="primary", use_container_width=True)

    if submitted:
        if not origin.strip() or not destination.strip():
            st.error("Enter both a starting point and a destination.")
        else:
            try:
                with st.spinner("Finding locations and calculating the walking route..."):
                    start_place = geocode_place(origin.strip())
                    end_place = geocode_place(destination.strip())
                    route_data = get_walking_route(
                        start_place["lat"], start_place["lon"],
                        end_place["lat"], end_place["lon"],
                    )
                st.session_state["route_result"] = {
                    "origin": start_place, "destination": end_place, "route": route_data,
                    "origin_text": origin.strip(), "destination_text": destination.strip(),
                }
            except Exception as exc:
                st.error(f"Could not create the route: {exc}")

    saved = st.session_state.get("route_result")
    if saved:
        route_data = saved["route"]
        origin_place = saved["origin"]
        destination_place = saved["destination"]
        distance_km = route_data["distance_m"] / 1000.0
        duration_min = route_data["duration_s"] / 60.0
        a, b, c = st.columns(3)
        a.metric("Walking distance (estimate)", f"{distance_km:.2f} km")
        b.metric("Walking time (estimate)", f"{duration_min:.0f} min")
        c.metric("Live traffic layer", "Enabled" if google_maps_key else "Needs API key")

        st.subheader("📍 Map, walking route & traffic")
        if not google_maps_key:
            st.warning("Add GOOGLE_MAPS_API_KEY to your .env file or enter it in the API Connections sidebar to display the embedded Google Map and live traffic layer. The route locations were calculated, but the Google map cannot load without the key.")
        else:
            map_data = {
                "origin": {"lat": origin_place["lat"], "lng": origin_place["lon"], "label": saved["origin_text"]},
                "destination": {"lat": destination_place["lat"], "lng": destination_place["lon"], "label": saved["destination_text"]},
                "path": [{"lat": lat, "lng": lon} for lon, lat in route_data["coordinates"]],
            }
            map_data_json = json.dumps(map_data).replace("</", "<\\/")
            html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
html,body,#map{{height:100%;width:100%;margin:0;padding:0;background:#0b1120;font-family:Arial,sans-serif;}}
#map{{min-height:500px}} #error{{padding:16px;color:#b91c1c;background:#fff;font-size:15px}}
</style></head><body><div id="map"></div><div id="error" style="display:none"></div>
<script>
const ROUTE_DATA = {map_data_json};
function showError(message) {{ const e=document.getElementById('error'); e.style.display='block'; e.textContent=message; }}
window.gm_authFailure = function() {{ showError('Google Maps could not authenticate this key. Check that Maps JavaScript API is enabled and the key restrictions allow this app.'); }};
async function initMap() {{
  try {{
    const {{Map, TrafficLayer}} = await google.maps.importLibrary('maps');
    const {{AdvancedMarkerElement}} = await google.maps.importLibrary('marker');
    const map = new Map(document.getElementById('map'), {{
      center: ROUTE_DATA.origin, zoom: 13, mapTypeControl: true, streetViewControl: true,
      fullscreenControl: true, mapTypeId: 'roadmap',
    }});
    new google.maps.Marker({{map, position: ROUTE_DATA.origin, title: 'Start: ' + ROUTE_DATA.origin.label, label: 'A'}});
    new google.maps.Marker({{map, position: ROUTE_DATA.destination, title: 'Destination: ' + ROUTE_DATA.destination.label, label: 'B'}});
    const path = ROUTE_DATA.path;
    new google.maps.Polyline({{map, path, geodesic: true, strokeColor: '#00A6FF', strokeOpacity: 0.95, strokeWeight: 6}});
    const bounds = new google.maps.LatLngBounds();
    bounds.extend(ROUTE_DATA.origin); bounds.extend(ROUTE_DATA.destination);
    path.forEach(p => bounds.extend(p)); map.fitBounds(bounds, 45);
    const traffic = new TrafficLayer({{autoRefresh: true}}); traffic.setMap(map);
    const info = new google.maps.InfoWindow({{content: '<b>Walking route</b><br>Blue line: estimated walking path.<br>Colored roads: current traffic where available.'}});
    info.setPosition(ROUTE_DATA.origin); info.open({{map}});
  }} catch (err) {{ showError('Map initialization failed: ' + (err && err.message ? err.message : err)); }}
}}
</script>
<script async defer src="https://maps.googleapis.com/maps/api/js?key={google_maps_key}&callback=initMap&v=weekly"></script>
</body></html>'''
            components.html(html, height=560, scrolling=False)
            st.caption("Google's TrafficLayer shows current traffic where available and refreshes periodically. It is not a historical daily traffic chart and may not cover every road. Walking path is an estimate and has not been validated for safety.")

        st.subheader("🧑‍🦯 Walker Assistance")
        st.write("Choose how you want to begin. The route and AI camera are separate modules in this prototype; starting camera assistance requires switching to AI Object Detection and starting Live Camera.")
        nav_a, nav_b = st.columns(2)
        with nav_a:
            if st.button("🧭 Start / Go to destination", use_container_width=True):
                st.session_state["walker_mode"] = "route"
                st.info("Route is displayed above. Follow the mapped path, obey pedestrian signals, and verify directions in the real world.")
        with nav_b:
            if st.button("👁️ Start with AI Vision", use_container_width=True):
                st.session_state["walker_mode"] = "ai_vision"
                st.warning("AI Vision selected. Open AI Object Detection → Live Camera, connect the ESP8266 sensor, then press Start Live Detection. Camera assistance is not yet an autonomous turn-by-turn navigation service.")
        st.write("Start a spoken pre-walk briefing with the latest weather and available traffic readings. Keep your phone accessible and remain aware of your surroundings.")
        if st.button("▶ Start Walker Assistance", type="primary", use_container_width=True):
            briefing_parts = [f"Walking guidance started. Your destination is {saved['destination_text']}.", f"Estimated walking distance is {distance_km:.1f} kilometres and estimated duration is {duration_min:.0f} minutes."]
            try:
                wx = get_weather(origin_place["lat"], origin_place["lon"])["current"]
                temp = wx.get("temperature_2m")
                rain = wx.get("precipitation")
                wind = wx.get("wind_speed_10m")
                briefing_parts.append(f"Current weather near your start is {temp} degrees Celsius, precipitation {rain} millimetres, wind {wind} kilometres per hour.")
                st.info(f"Weather at start: {temp} °C · precipitation {rain} mm · wind {wind} km/h")
                if rain and float(rain) > 0:
                    briefing_parts.append("Precipitation is reported. Watch for slippery surfaces and reduced visibility.")
            except Exception as exc:
                st.warning(f"Weather briefing unavailable: {exc}")
                briefing_parts.append("Weather data is currently unavailable. Check local conditions before walking.")
            if tomtom_key:
                try:
                    flow = get_traffic_flow(origin_place["lat"], origin_place["lon"], tomtom_key)
                    st.info(f"Traffic near start: {flow['label']}; current speed {flow['current_speed']:.0f} km/h versus free-flow {flow['free_speed']:.0f} km/h.")
                    briefing_parts.append(f"Traffic near the starting point is {flow['label']}. Current reported road speed is {flow['current_speed']:.0f} kilometres per hour. This is a nearby road reading, not a full pedestrian-route traffic assessment.")
                except Exception as exc:
                    st.warning(f"Live traffic reading unavailable: {exc}")
                    briefing_parts.append("Live traffic data is unavailable. Use crossings carefully and follow pedestrian signals.")
            else:
                briefing_parts.append("Live traffic speed data is not configured. Check traffic visually and follow pedestrian signals.")
                st.info("For spoken live-traffic readings, configure the TomTom Traffic API key in the sidebar.")
            briefing_parts.append("Follow pedestrian crossings and traffic signals. This prototype does not guarantee a safe route.")
            st.success("Walker briefing ready.")
            # Keep this informational briefing silent; spoken audio is reserved for live close-hazard alerts.
            st.write(" ".join(briefing_parts))

        st.subheader("🚦 Optional traffic speed readings (TomTom)")
        if not tomtom_key:
            st.info("TomTom speed readings are optional. Add TOMTOM_API_KEY if you want numerical current/free-flow speed readings at the start and destination. Google's embedded traffic layer can still display without TomTom.")
        else:
            if st.button("🔄 Refresh TomTom traffic readings", use_container_width=True):
                refreshed = {}
                errors = []
                for label, place in (("Starting point", origin_place), ("Destination", destination_place)):
                    try:
                        refreshed[label] = get_traffic_flow(place["lat"], place["lon"], tomtom_key)
                    except Exception as exc:
                        errors.append(f"{label}: {exc}")
                st.session_state["route_traffic"] = refreshed
                st.session_state["route_traffic_error"] = "\n".join(errors) if errors else None
            traffic = st.session_state.get("route_traffic")
            if traffic:
                st.caption(f"TomTom response checked at {datetime.now().strftime('%d %b %Y, %H:%M:%S')}.")
                cols = st.columns(len(traffic))
                for idx, (label, flow) in enumerate(traffic.items()):
                    with cols[idx]:
                        st.markdown(f"**{label}**")
                        st.metric("Current speed", f"{flow['current_speed']:.0f} km/h")
                        st.metric("Free-flow speed", f"{flow['free_speed']:.0f} km/h")
                        st.info(flow["label"])
            elif st.session_state.get("route_traffic_error"):
                st.warning(st.session_state["route_traffic_error"])

elif page == "ESP32 Hardware":
    st.header("🔌 Hardware Control Center")
    st.markdown("Monitor the ultrasonic sensor and control the buzzer from a distance threshold. Keep this page open while testing the board.")
    st.caption("Expected board: ESP32 DevKit V1 · Serial speed: 115200 baud · Default alert distance: 50 cm")
    ports = available_serial_ports()

    st.markdown("### 📏 Live distance monitor")
    st.caption("Use a USB data cable, select the board's COM port, and press Start live readings. Close Arduino Serial Monitor first because only one app can normally own the port at a time.")
    if "esp32_monitor_serial" not in st.session_state:
        st.session_state.esp32_monitor_serial = None
    if "esp32_monitor_port" not in st.session_state:
        st.session_state.esp32_monitor_port = None
    port_choices = ports if ports else ["No COM port found"]
    monitor_port = st.selectbox("ESP32 COM port", port_choices, key="esp32_monitor_port_select")
    start_col, stop_col = st.columns(2)
    with start_col:
        if st.button("▶ Start live readings", use_container_width=True, disabled=not ports):
            try:
                current = st.session_state.esp32_monitor_serial
                if current is not None and current.is_open:
                    current.close()
                st.session_state["esp32_serial_buffer"] = b""
                st.session_state["hardware_buzzer_requested"] = None
                st.session_state.pop("esp32_last_distance", None)
                st.session_state.pop("esp32_last_distance_time", None)
                st.session_state.pop("esp32_last_serial_line", None)
                st.session_state.esp32_monitor_serial = serial.Serial(monitor_port, 115200, timeout=0.2)
                st.session_state.esp32_monitor_port = monitor_port
                time.sleep(2.0)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not open {monitor_port}: {exc}")
    with stop_col:
        if st.button("■ Stop live readings", use_container_width=True):
            current = st.session_state.esp32_monitor_serial
            if current is not None:
                try:
                    current.close()
                except Exception:
                    pass
            st.session_state.esp32_monitor_serial = None
            st.session_state.esp32_monitor_port = None
            st.rerun()

    @st.fragment(run_every="1s")
    def show_live_distance():
        connection = st.session_state.get("esp32_monitor_serial")
        if connection is not None and getattr(connection, "is_open", False):
            distance, message = read_ultrasonic_distance(connection)
            if distance is not None:
                a, b = st.columns(2)
                a.metric("Live distance", f"{distance:.1f} cm")
                threshold = float(st.session_state.get("hardware_alert_threshold", 50))
                level = proximity_level(distance) if distance <= threshold else 0
                labels = {0: "NORMAL", 1: "LOW · slow beep", 2: "MEDIUM · quicker beep", 3: "HIGH · fast beep + vibration", 4: "CRITICAL · rapid beep + vibration + voice"}
                b.metric("Proximity", labels[level])
                if level == 0:
                    st.success("No proximity signal. Beep and vibration are off; no spoken warning.")
                elif level == 1:
                    st.info(f"{distance:.1f} cm — soft beep/vibration about every 2 seconds. No voice warning.")
                elif level == 2:
                    st.info(f"{distance:.1f} cm — beep/vibration about once per second. No voice warning.")
                elif level == 3:
                    st.warning(f"{distance:.1f} cm — fast beep and vibration. Spoken warning enabled in close zone.")
                else:
                    st.error(f"{distance:.1f} cm — critical rapid beep/vibration and spoken danger warning.")
                previous_level = st.session_state.get("hardware_buzzer_requested")
                if previous_level != level:
                    sent = send_hardware_alert_level(level)
                    st.session_state["hardware_buzzer_requested"] = level if sent else None
                    if not sent:
                        st.warning("Distance is being read, but graded alert commands were not sent. Upload the matching ESP32 firmware and check the serial connection.")
                st.caption(f"Connected to {st.session_state.get('esp32_monitor_port', 'ESP8266')} · refreshed automatically")
            else:
                # Fail safe: request buzzer OFF if valid distance readings disappear.
                previous_state = st.session_state.get("hardware_buzzer_requested")
                if previous_state:
                    send_hardware_alert_level(0)
                st.session_state["hardware_buzzer_requested"] = 0
                st.metric("Live distance", "Waiting for sensor…")
                st.warning(message or "No valid reading received. Check sensor wiring.")
        else:
            st.info("Monitoring is stopped. Choose the ESP32 COM port and press Start live readings.")
    show_live_distance()

    c1, c2 = st.columns(2)
    c1.metric("Detected serial ports", str(len(ports)))
    threshold = st.slider("Buzzer alert threshold (cm)", min_value=10, max_value=150, value=50, step=5, key="hardware_alert_threshold")
    c2.metric("Ultrasonic alert threshold", f"{threshold} cm")
    with st.expander("Hardware troubleshooting checklist"):
        st.markdown("""
- Close Arduino IDE Serial Monitor/Plotter before connecting here.
- Select the COM port that appears/disappears when you unplug/reconnect the board.
- Firmware must print distance readings such as `DISTANCE_CM:23.4` or `Distance: 23.4 cm` at **115200 baud**.
- For graded sound/vibration, upload the matching ESP32 firmware and use the `ALERT_LEVEL:0..4` serial protocol. Confirm the buzzer and vibration-motor pin wiring before powering the hardware.
- If using an HC-SR04 with an ESP8266, do **not** connect the sensor's 5 V ECHO signal directly to an ESP8266 GPIO; use a suitable voltage divider/level shifter.
- Test the sensor with the board powered from USB and keep the sensor's ground connected to the board's ground.
""")
    if ports:
        st.success("Detected ports: " + ", ".join(ports))
    else:
        st.warning("No serial port detected. Connect the NodeMCU ESP8266 with a USB data cable and install its USB-to-serial driver if needed.")

