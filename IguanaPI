import RPi.GPIO as GPIO
import time, os, cv2
from datetime import datetime, date, timedelta
from picamera2 import Picamera2
from ultralytics import YOLO
from astral import LocationInfo
from astral.sun import sun
import pytz
from flask import Flask, Response, jsonify
from threading import Thread, Lock

# ==== CONFIG ====
DEBUG_MODE = True
FAST_MODE = True
CITY = LocationInfo("Florida", "USA", "US/Eastern", 27.994402, -81.760254)
tz = pytz.timezone("US/Eastern")

folder_path = "/home/pi/Images"
os.makedirs(folder_path, exist_ok=True)

# ==== ULTRASONIC SENSOR ====
TRIG, ECHO = 23, 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, False)
_distance_ema = None
EMA_ALPHA = 0.45  # smoothing

# ==== STEPPER CONFIG ====
IN1, IN2, IN3, IN4 = 17, 18, 27, 22
STEPPER_PINS = [IN1, IN2, IN3, IN4]
for pin in STEPPER_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, 0)

step_sequence = [
    [1,0,0,0],
    [1,1,0,0],
    [0,1,0,0],
    [0,1,1,0],
    [0,0,1,0],
    [0,0,1,1],
    [0,0,0,1],
    [1,0,0,1]
]
trap_closed = False

# ==== CAMERA + YOLO ====
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"format": "RGB888", "size": (640,480)}))

# Load model once
model = YOLO("/home/pi/Downloads/best2.pt")
print("YOLO model loaded.")
YOLO_CONF = 0.6
YOLO_IOU = 0.45

# Detection debounce state
_detection_hits = 0
_detection_last_time = 0.0
REQUIRED_HITS = 2
HIT_WINDOW_SEC = 1.2

# ==== FLASK STREAM ====
app = Flask(__name__)
frame_lock = Lock()
latest_frame = None
YOLO_SKIP = 2
frame_count = 0
last_boxes = []

def yolo_capture_loop():
    """Capture camera frames, run YOLO every YOLO_SKIP frames, and encode MJPEG."""
    global latest_frame, frame_count, last_boxes
    while True:
        frame = camera.capture_array()
        frame_count += 1

        # YOLO inference on smaller frame every YOLO_SKIP frames
        if frame_count % YOLO_SKIP == 0:
            small = cv2.resize(frame, (320, 240))
            results = model(small, conf=YOLO_CONF, iou=YOLO_IOU)

            new_boxes = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{model.names[cls]} ({conf:.2f})"
                    # scale coords back to 640x480
                    x1, y1, x2, y2 = map(lambda v: int(v*2), box.xyxy[0])
                    new_boxes.append((x1, y1, x2, y2, label))
            last_boxes = new_boxes

        # Draw YOLO boxes and labels
        for (x1, y1, x2, y2, label) in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0,255,0), -1)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # Encode MJPEG with no-cache
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        with frame_lock:
            latest_frame = buffer.tobytes()
        time.sleep(0.005)

def generate_frames():
    global latest_frame
    while True:
        if latest_frame:
            with frame_lock:
                frame = latest_frame
            yield (b"--frame\r\n"
                   b"Cache-Control: no-cache\r\n"
                   b"Pragma: no-cache\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   frame + b"\r\n")
        else:
            time.sleep(0.01)

@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==== Manual Trap Endpoints ====
@app.route("/trigger_trap", methods=["POST"])
def trigger_trap():
    try:
        print("Manual trap trigger request received.")
        Thread(target=activate_trap, daemon=True).start()
        return jsonify({"message": "Trap activated successfully"}), 200
    except Exception as e:
        return jsonify({"message": f"Failed to trigger trap: {str(e)}"}), 500

@app.route("/reset_trap", methods=["POST"])
def reset_trap():
    try:
        global trap_closed
        print("Manual trap reset request received.")
        Thread(target=lambda: step_motor(600, -1), daemon=True).start()
        trap_closed = False
        return jsonify({"message": "Trap reset successfully"}), 200
    except Exception as e:
        return jsonify({"message": f"Failed to reset trap: {str(e)}"}), 500

def run_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)

# ==== SUN CONTROL ====
def get_sun_times():
    s = sun(CITY.observer, date=date.today(), tzinfo=tz)
    return s["sunrise"], s["sunset"]

def is_daytime():
    sunrise, sunset = get_sun_times()
    now = datetime.now(tz)
    return sunrise <= now <= sunset

def wait_until_sunrise():
    sunrise, _ = get_sun_times()
    now = datetime.now(tz)
    if now < sunrise:
        seconds = (sunrise - now).total_seconds()
    else:
        tomorrow = date.today() + timedelta(days=1)
        s = sun(CITY.observer, date=tomorrow, tzinfo=tz)
        seconds = (s["sunrise"] - now).total_seconds()
    print(f"Sleeping {seconds/3600:.2f}h until sunrise.")
    time.sleep(seconds)

# ==== MOTOR CONTROL ====
def step_motor(steps, direction=1):
    if steps <= 0:
        return
    if FAST_MODE:
        start_delay, end_delay = 0.0012, 0.0006
    else:
        start_delay, end_delay = 0.0025, 0.001
    seq = step_sequence if direction == 1 else list(reversed(step_sequence))
    try:
        for i in range(steps):
            delay = max(end_delay, start_delay - (i / steps) * (start_delay - end_delay))
            for step in seq:
                for pin, val in zip(STEPPER_PINS, step):
                    GPIO.output(pin, val)
                time.sleep(delay)
    finally:
        for p in STEPPER_PINS:
            GPIO.output(p, 0)

# ==== ULTRASONIC + YOLO DETECTION ====
def get_distance():
    global _distance_ema
    try:
        GPIO.output(TRIG, False)
        time.sleep(0.00005)
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        start = time.time()
        timeout_start = start + 0.02
        while GPIO.input(ECHO) == 0:
            start = time.time()
            if start > timeout_start:
                return None
        stop = time.time()
        timeout_stop = stop + 0.02
        while GPIO.input(ECHO) == 1:
            stop = time.time()
            if stop > timeout_stop:
                return None
        elapsed = stop - start
        distance = (elapsed * 34300) / 2
        if not (0 < distance < 500):
            return None
        if _distance_ema is None:
            _distance_ema = distance
        else:
            _distance_ema = EMA_ALPHA * distance + (1 - EMA_ALPHA) * _distance_ema
        return round(_distance_ema, 1)
    except Exception as e:
        if DEBUG_MODE:
            print(f"Ultrasonic read error: {e}")
        return None

def detect_cat(frame):
    global _detection_hits, _detection_last_time
    try:
        small = cv2.resize(frame, (320, 240))
        results = model(small, conf=YOLO_CONF, iou=YOLO_IOU)

        positive = False
        best_conf = 0.0
        names = model.names

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = names[cls].lower() if isinstance(names, list) else names.get(cls, str(cls)).lower()
                if DEBUG_MODE:
                    print(f"Detected {label} conf={conf:.2f}")
                if label in ["cat", "iguana", "iguana_detect"]:
                    if conf > best_conf:
                        best_conf = conf
                    if conf >= YOLO_CONF:
                        positive = True

        now = time.time()
        if positive:
            if now - _detection_last_time <= HIT_WINDOW_SEC:
                _detection_hits += 1
            else:
                _detection_hits = 1
            _detection_last_time = now
        else:
            if now - _detection_last_time > HIT_WINDOW_SEC:
                _detection_hits = 0

        if DEBUG_MODE:
            print(f"detection_hits={_detection_hits}, best_conf={best_conf:.2f}")

        if _detection_hits >= REQUIRED_HITS:
            _detection_hits = 0
            return True
        return False
    except Exception as e:
        print(f"Detection error: {e}")
        return False

# ==== TRAP CONTROL ====
def activate_trap():
    global trap_closed
    if trap_closed:
        print("Trap already closed.")
        return
    try:
        print("Closing trap...")
        step_motor(720, 1)
        trap_closed = True
        print("Trap closed.")
        time.sleep(5)
    except Exception as e:
        print(f"Activate trap error: {e}")
    finally:
        try:
            print("Reopening trap...")
            step_motor(300, -1)
            trap_closed = False
            print("Ready again.")
        except Exception as e:
            print(f"Error reopening trap: {e}")

# ==== MAIN LOOP ====
def idle_loop():
    sleep_interval = 1.0
    while True:
        if not is_daytime():
            print(" Nighttime - sleeping until sunrise.")
            wait_until_sunrise()
            continue

        distance = get_distance()
        if distance is not None and distance < 30:
            print(f" Object detected at {distance:.1f} cm")
            frame = camera.capture_array()
            if detect_cat(frame):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(folder_path, f"cat_{timestamp}.jpg")
                cv2.imwrite(path, cv2.resize(frame, (1280, 720)))
                print(f"Saved capture: {path}")
                activate_trap()
            sleep_interval = 1
        else:
            if distance is None:
                print("No reading from sensor.")
            else:
                print("Nothing nearby.")
            sleep_interval = min(sleep_interval + 0.5, 5.0)
        time.sleep(sleep_interval)

# ==== STARTUP ====
print("System armed (ultrasonic + YOLO mode).")
camera.start()
Thread(target=yolo_capture_loop, daemon=True).start()
Thread(target=run_flask, daemon=True).start()
print("Stream running at https://feed.iguanafeed.com/stream")

try:
    idle_loop()
except KeyboardInterrupt:
    print(" Stopped.")
finally:
    for p in STEPPER_PINS:
        GPIO.output(p, 0)
    GPIO.cleanup()
    camera.stop()
    print("Shutdown complete.")
