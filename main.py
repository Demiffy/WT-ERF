import os, sys, time, math, threading, signal, pyautogui, cv2, numpy as np, pytesseract, re
from flask import Flask, render_template, jsonify
import mss, mss.tools
from PIL import Image
import tkinter as tk
import state
import utils
import logging
from utils import log as custom_log
exit_event = threading.Event()

def signal_handler(sig, frame):
    custom_log("CTRL+C pressed. Exiting...", level="INFO", tag="EVENT")
    exit_event.set()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# -----------------------------------------------------------
# Global Flags and Configurations
# -----------------------------------------------------------
CAPTURE_REGION = (1473, 635, 432, 432)
OCR_REGION = (1800, 1044, 106, 20)
DIR_OUTPUT = os.path.join("static", "screenshots", "output")
os.makedirs(DIR_OUTPUT, exist_ok=True)
DIR_OCR = os.path.join("static", "screenshots", "ocr")
os.makedirs(DIR_OCR, exist_ok=True)
OUTPUT_IMAGE_PATH = os.path.join(DIR_OUTPUT, "tracked_target.png")
LATEST_OCR_PATH = os.path.join(DIR_OCR, "latest_ocr.png")
LATEST_EDITED_OCR_PATH = os.path.join(DIR_OCR, "latest_edited_ocr.png")
TESS_CONFIG = '--psm 7 -c tessedit_char_whitelist=0123456789.'

latest_range_m = 0.0
scale_number = ""
latest_ocr_snapshot = "/static/screenshots/ocr/latest_ocr.png"
latest_edited_ocr_snapshot = os.path.join("static", "screenshots", "ocr", "latest_edited_ocr.png")
scale_pixels = 0
conversion_factor = 0
debug_info = ""
tracking_active = False

scale_locked = False
last_scale_value = None
last_scale_time = 0

# -----------------------------------------------------------
# Colors & Utility Functions
# -----------------------------------------------------------
player_hex_colors = [
    "f2c52f", "caa21f", "f4c832", "f8d970", "c6a228", "8f7520", "f1ca46", "f3d15d",
    "f4c730", "f0c42f", "f2c530", "c39f26", "93781d", "6d5a15", "9b7c18", "f7d35a",
    "f6d050", "f7d563", "e0b628", "f5c830", "f5c937", "b59019",
    "f6db7d", "eabf2d", "fae5a0", "fbeab1", "f9df87", "a2811e", "987815", "987711", "b28c16",
    "c2ad6c", "b49a4a", "f6cc3f", "f8da77", "f9e08c", "f8da75", "fae398",
    "fbe8ac", "f7d76b", "f9e193", "ddb940", "d0aa29", "f1c739", "f6cd44",
    "cdaf4a", "efce60", "a88a23", "dab32d", "e5bb2d", "f7d45f"
]
ping_hex_colors = ["d8d807", "d6d607", "d0d007", "d1d107", "c8c807", "adad06", "b9b906", "bebe06"]
hex_to_bgr = lambda s: np.array([int(s[i:i+2], 16) for i in (4, 2, 0)], dtype=np.uint8)
target_colors = [hex_to_bgr(h) for h in player_hex_colors]
ping_target_colors = [hex_to_bgr(h) for h in ping_hex_colors]
tolerance = 4
max_radius = 10

def process_image(img):
    output_img = img.copy()
    reshaped = img.reshape(-1, 3)
    mask = np.zeros((reshaped.shape[0],), dtype=bool)
    for target in target_colors:
        mask |= (np.linalg.norm(reshaped.astype(np.int16) - target.astype(np.int16), axis=1) < tolerance)
    output_img.reshape(-1, 3)[mask] = [0, 0, 255]
    return output_img, mask

def process_ping(img):
    output_img = img.copy()
    reshaped = img.reshape(-1, 3)
    ping_mask = np.zeros((reshaped.shape[0],), dtype=bool)
    for target in ping_target_colors:
        ping_mask |= (np.linalg.norm(reshaped.astype(np.int16) - target.astype(np.int16), axis=1) < tolerance)
    return output_img, ping_mask

def get_enclosing_circle(mask, shape):
    h, w = shape[:2]
    mask_2d = mask.reshape(h, w).astype(np.uint8)
    pts = cv2.findNonZero(mask_2d)
    if pts is not None:
        center, radius = cv2.minEnclosingCircle(pts)
        return (int(center[0]), int(center[1])), min(int(radius), max_radius), pts.shape[0]
    return None, None, 0

draw_filled_circle = lambda img, c, r, col=(0, 0, 255): cv2.circle(img.copy(), c, r, col, -1) if c and r else img.copy()
overlay_text = lambda img, text, col=(255, 255, 255), pos=(10, 30): cv2.putText(img.copy(), text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

prev_center = None
prev_count = 0
stable_count = 0
stable_threshold = 3
distance_threshold = 20
min_count_threshold = 2

def write_placeholder():
    placeholder = np.zeros((CAPTURE_REGION[3], CAPTURE_REGION[2], 3), dtype=np.uint8)
    cv2.imwrite(OUTPUT_IMAGE_PATH, overlay_text(placeholder, "Tracking paused", (0, 0, 255), (10, 30)))

# -----------------------------------------------------------
# Combined Capture Loop (Tracking)
# -----------------------------------------------------------
def combined_loop():
    global prev_center, prev_count, stable_count, latest_range_m, tracking_active, conversion_factor, debug_info, scale_pixels, scale_number
    with mss.mss() as sct:
        monitor = {"left": CAPTURE_REGION[0], "top": CAPTURE_REGION[1], "width": CAPTURE_REGION[2], "height": CAPTURE_REGION[3]}
        while not exit_event.is_set():
            if not utils.is_aces_in_focus():
                write_placeholder()
                time.sleep(0.1)
                continue

            if state.ui_state != "ingame":
                write_placeholder()
                time.sleep(0.1)
                continue

            tracking_active = True
            sct_img = sct.grab(monitor)
            img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            processed_img, mask = process_image(img)
            center, radius, count = get_enclosing_circle(mask, img.shape)
            msg, text_color = (f"Player: {count} pixels", (255, 255, 255)) if count > 0 else ("No Player", (0, 0, 255))
            if count < min_count_threshold:
                prev_center, prev_count, stable_count = None, 0, 0
            if center:
                if prev_center is None:
                    prev_center, prev_count, stable_count = center, count, 1
                else:
                    dist = np.linalg.norm(np.array(center) - np.array(prev_center))
                    if dist > distance_threshold:
                        if count > 1.5 * prev_count:
                            stable_count += 1
                            if stable_count >= stable_threshold:
                                prev_center, prev_count, stable_count = center, count, 0
                        else:
                            center, radius, stable_count = prev_center, int(prev_count / 10), 0
                    else:
                        prev_center, prev_count, stable_count = center, count, 0
            else:
                prev_center, prev_count, stable_count = None, 0, 0
            circled = draw_filled_circle(processed_img, center, radius)
            output_img = overlay_text(circled, msg, text_color, (10, 30))
            
            def process_ping_local(img):
                reshaped = img.reshape(-1, 3)
                local_ping_mask = np.zeros((reshaped.shape[0],), dtype=bool)
                for target in ping_target_colors:
                    local_ping_mask |= (np.linalg.norm(reshaped.astype(np.int16) - target.astype(np.int16), axis=1) < tolerance)
                return img.copy(), local_ping_mask
            
            _, ping_mask = process_ping_local(img)
            ping_center, ping_radius, ping_count = get_enclosing_circle(ping_mask, img.shape)
            if ping_count > 0:
                output_img = draw_filled_circle(output_img, ping_center, ping_radius, (0, 255, 255))
                if center and ping_center:
                    cv2.line(output_img, ping_center, center, (255, 255, 255), 2)
                    dx, dy = ping_center[0] - center[0], ping_center[1] - center[1]
                    pixel_distance = math.sqrt(dx * dx + dy * dy)
                    try:
                        scale_val = int(round(float(scale_number)))
                    except:
                        scale_val = 0
                    if scale_val == 0 or scale_pixels == 0:
                        conversion_factor = 0
                    else:
                        conversion_factor = float(scale_val) / float(scale_pixels)
                    debug_info = f"Scale Number: {scale_val}, Scale Pixels: {scale_pixels}, CF: {conversion_factor:.2f}"
                    cv2.putText(output_img, debug_info, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    range_m = pixel_distance * conversion_factor
                    latest_range_m = range_m
                    cv2.putText(output_img, f"Range: {range_m:.2f} m", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(OUTPUT_IMAGE_PATH, output_img)
            time.sleep(0.1)

# -----------------------------------------------------------
# OCR Loop for Scale Number and Snapshot Saving with Postprocessing
# -----------------------------------------------------------
def ocr_scale_loop():
    global scale_number, scale_pixels, scale_locked, last_scale_value, last_scale_time
    OCR_BLACK_THRESHOLD = 12
    LINE_BLACK_THRESHOLD = 1
    scale_locked = False
    last_scale_value = None
    last_scale_time = time.time()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    while not exit_event.is_set():
        if not utils.is_aces_in_focus() or state.ui_state != "ingame":
            if scale_locked:
                utils.log("Focus lost or state changed; unlocking scale.", level="DEBUG", tag="OCR")
            scale_locked = False
            last_scale_value = None
            last_scale_time = time.time()
            time.sleep(2)
            continue

        try:
            ocr_img = pyautogui.screenshot(region=OCR_REGION)
            ocr_img.save(LATEST_OCR_PATH)
            ocr_gray = cv2.cvtColor(np.array(ocr_img), cv2.COLOR_RGB2GRAY)
            
            ocr_binary_img = np.where(ocr_gray < OCR_BLACK_THRESHOLD, 0, 255).astype(np.uint8)
            inverted = cv2.bitwise_not(ocr_binary_img)
            dilated = cv2.dilate(inverted, kernel, iterations=1)
            ocr_binary_img = cv2.bitwise_not(dilated)
            
            cv2.imwrite(LATEST_EDITED_OCR_PATH, ocr_binary_img)
            
            line_binary_img = np.where(ocr_gray < LINE_BLACK_THRESHOLD, 0, 255).astype(np.uint8)
            last_row = line_binary_img[-1, :]
            max_run = 0
            current_run = 0
            for pixel in last_row:
                if pixel == 0:
                    current_run += 1
                else:
                    max_run = max(max_run, current_run)
                    current_run = 0
            scale_pixels = max(max_run, current_run)
            
            raw_text = pytesseract.image_to_string(ocr_binary_img, config=TESS_CONFIG).strip()
            clean_text = re.sub(r'\D', '', raw_text)
            
            if len(clean_text) == 3 and clean_text.endswith('0'):
                if not scale_locked:
                    if last_scale_value == clean_text:
                        if time.time() - last_scale_time >= 10:
                            scale_locked = True
                            utils.log(f"Scale locked to {clean_text}", level="DEBUG", tag="OCR")
                    else:
                        scale_locked = False
                        last_scale_value = clean_text
                        last_scale_time = time.time()
                        scale_number = clean_text
                        utils.log(f"Scale Number: {scale_number}", level="INFO", tag="OCR")
        except Exception as e:
            utils.log(f"OCR Error: {e}", level="ERROR", tag="OCR")
        time.sleep(2)

# -----------------------------------------------------------
# Flask Web Server for Rangefinder Interface
# -----------------------------------------------------------
app = Flask(__name__, static_url_path='/static', static_folder='static')

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.disabled = True
app.logger.disabled = True

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/latest")
def latest():
    return jsonify({
        "image": OUTPUT_IMAGE_PATH,
        "edited_ocr_snapshot": latest_edited_ocr_snapshot,
        "range": latest_range_m,
        "number": scale_number,
        "scale_pixels": scale_pixels,
        "conversion_factor": conversion_factor,
        "debug_info": debug_info,
        "state_snapshot": state.latest_state_snapshot,
        "state_extra_snapshot": state.latest_state_extra_snapshot,
        "state_ingame_snapshot": state.latest_state_ingame_snapshot,
        "ui_state": state.ui_state
    })

def start_overlay():
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-transparentcolor", "black")
    root.geometry("500x50+50+50")
    frame = tk.Frame(root, width=500, height=50, bg="black")
    frame.pack_propagate(False)
    frame.pack()
    label = tk.Label(frame, text="Range: 0.00 m", font=("Arial", 24, "bold"), fg="lime", bg="black")
    label.pack(expand=True)
    
    def on_escape(event):
        exit_event.set()
        root.destroy()
    root.bind("<Escape>", on_escape)
    
    def refresh_overlay():
        if state.ui_state == "ingame" and utils.is_aces_in_focus():
            root.deiconify()
            label.config(text=f"Range: {latest_range_m:.2f} m")
        else:
            root.withdraw()
        if not exit_event.is_set():
            root.after(100, refresh_overlay)
        else:
            root.destroy()
    refresh_overlay()
    try:
        root.mainloop()
    except Exception as e:
        print(f"Overlay error: {e}")

def start_rangefinder():
    threading.Thread(target=ocr_scale_loop, daemon=True).start()
    threading.Thread(target=combined_loop, daemon=True).start()
    threading.Thread(target=state.ocr_state_loop, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5001, debug=False), daemon=True).start()
    threading.Thread(
        target=lambda: utils.handle_focus_loss(
            lambda: utils.log("Detection paused due to focus loss.", level="WARN", tag="PROCESS"),
            lambda: utils.log("Detection resumed.", level="INFO", tag="PROCESS")
        ),
        daemon=True
    ).start()
    start_overlay()

if __name__ == "__main__":
    utils.check_resolution()
    utils.wait_for_aces()
    try:
        start_rangefinder()
    except KeyboardInterrupt:
        exit_event.set()
        sys.exit(0)