import cv2, time, math, collections
import numpy as np
import mediapipe as mp

cv2.setUseOptimized(True)

# =========================
# Config (tweak if needed)
# =========================
WIN_NAME        = "Gesture Keyboard (Pinch+Calibrated)"

# ---- Theme (BGR) ----
UI_BG           = (30, 24, 40)
KB_BG           = (48, 36, 66)
KEY_PURPLE      = (180, 80, 255)
KEY_HOVER       = (200, 120, 255)
KEY_PRESSED     = (150, 40, 230)
KEY_BORDER      = (220, 200, 255)
KEY_SHADOW      = (40, 28, 56)
TEXT_LIGHT      = (255, 255, 255)
TEXT_DIM        = (220, 220, 230)
SHADOW_OFF      = 4
KEY_RADIUS      = 12

# ---- Behavior ----
USE_WEBCAM      = True
VIDEO_PATH      = "hand_demo.mp4"
SHOW_CAMERA     = False            # UI only by default; press V to toggle
START_MODE      = "pinch"          # "pinch" or "dwell"

# ---- Performance knobs ----
RESIZE_WIDTH    = 960              # lower to 800 for more FPS
PROCESS_EVERY_N = 1                # process landmarks every frame for reliable pinch
MAX_HANDS       = 1
MODEL_COMPLEXITY= 0
SMOOTH_ALPHA    = 0.45             # EMA for cursor (fingertip) smoothing

# ---- Press logic ----
DWELL_MS        = 450
COOLDOWN_MS     = 160              # shorter gap feels snappier
CORE_INSET      = 0.18

# ---- Pinch detection (adaptive capable) ----
# Base thresholds used before/without calibration (normalized units).
PINCH_ON_BASE   = 0.22             # trigger when pinch_norm_smooth < ON
PINCH_OFF_BASE  = 0.27             # re-arm when > OFF
PINCH_HOLD_MS   = 55               # must stay under ON for this many ms to commit
PINCH_SMOOTH_A  = 0.60             # EMA just for pinch distance (independent of cursor)
PINCH_DEBUG     = False            # toggle with 'd'

# Calibration (press K): we sample "open" (no pinch) then "pinched"
CALIB_ON_FACT   = 0.58             # ON = open_mean * this (lower means stricter)
CALIB_OFF_FACT  = 0.72             # OFF = open_mean * this
CALIB_OPEN_MS   = 700              # collect open samples for this long
CALIB_PINCH_MS  = 700              # collect pinched samples for this long

# Optional: send OS keystrokes (off)
SEND_OS_KEYS    = False
# if SEND_OS_KEYS: from pynput.keyboard import Controller, Key; kb = Controller()

# =========================
# Helpers
# =========================
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def inset_rect(r, frac):
    x1,y1,x2,y2 = r
    dx = int((x2-x1)*frac/2); dy = int((y2-y1)*frac/2)
    return (x1+dx, y1+dy, x2-dx, y2-dy)

def pt_in_rect(pt, r):
    x,y = pt; x1,y1,x2,y2 = r
    return x1 <= x <= x2 and y1 <= y <= y2

def fill_rounded_rect(img, rect, color, radius=12):
    x1,y1,x2,y2 = rect
    rad = min(radius, (x2-x1)//2, (y2-y1)//2)
    cv2.rectangle(img, (x1+rad, y1), (x2-rad, y2), color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1+rad), (x2, y2-rad), color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x1+rad, y1+rad), rad, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x2-rad, y1+rad), rad, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x1+rad, y2-rad), rad, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x2-rad, y2-rad), rad, color, -1, lineType=cv2.LINE_AA)

def draw_key(img, r, label, hovered=False, pressed=False):
    x1, y1, x2, y2 = r
    dy = 2 if pressed else 0
    y1p, y2p = y1 + dy, y2 + dy
    fill = KEY_PURPLE
    if hovered: fill = KEY_HOVER
    if pressed: fill = KEY_PRESSED
    cv2.rectangle(img, (x1+SHADOW_OFF, y1p+SHADOW_OFF), (x2+SHADOW_OFF, y2p+SHADOW_OFF),
                  KEY_SHADOW, -1, lineType=cv2.LINE_AA)
    fill_rounded_rect(img, (x1, y1p, x2, y2p), fill, KEY_RADIUS)
    cv2.rectangle(img, (x1, y1p), (x2, y2p), KEY_BORDER, 2, lineType=cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cx = (x1+x2)//2 - tw//2; cy = (y1p+y2p)//2 + th//3
    cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_LIGHT, 2, cv2.LINE_AA)

def build_keyboard_layout(W, H):
    margin = 20
    gutter = 10
    kb_height = int(H * 0.42)
    y_top = H - kb_height - margin
    key_h = int((kb_height - 4*gutter) / 4)
    rows = [list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM") + ["BKSP"]]
    key_list = []
    y = y_top
    for row in rows:
        cols = len(row)
        key_w = int((W - 2*margin - (cols-1)*gutter) / cols)
        row_width = cols*key_w + (cols-1)*gutter
        x = (W - row_width)//2
        for k in row:
            r = (x, y, x+key_w, y+key_h)
            key_list.append({
                "label": k,
                "rect": r,
                "core_rect": inset_rect(r, CORE_INSET),
                "center": ((x+x+key_w)//2, (y+y+key_h)//2),
            })
            x += key_w + gutter
        y += key_h + gutter
    # space bar
    space_w = int(W * 0.60)
    x1 = (W - space_w)//2
    r = (x1, y, x1 + space_w, y + key_h)
    key_list.append({
        "label": "SPACE",
        "rect": r,
        "core_rect": inset_rect(r, CORE_INSET),
        "center": ((r[0]+r[2])//2, (r[1]+r[3])//2),
    })
    rows_area = (margin, y_top, W-margin, H-margin)
    return key_list, rows_area

def nearest_key(keys, pt):
    for k in keys:
        if pt_in_rect(pt, k["rect"]):
            return k
    best = None; best_d = 1e9
    for k in keys:
        d = dist(pt, k["center"])
        if d < best_d:
            best_d = d; best = k
    return best

def commit(char, typed_so_far):
    if char == "BKSP":
        return typed_so_far[:-1]
    elif char == "SPACE":
        return typed_so_far + " "
    else:
        return typed_so_far + char

# Click pulse fx
click_fx = []
FX_DURATION = 160
def trigger_click_fx(key_obj):
    x1,y1,x2,y2 = key_obj["rect"]
    max_r = int(0.55 * min(x2-x1, y2-y1))
    click_fx.append({"t0": int(time.time()*1000), "center": key_obj["center"], "rmax": max_r})

# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# =========================
# Video / Window
# =========================
cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH, cv2.CAP_DSHOW if USE_WEBCAM else 0)
if not cap.isOpened():
    raise SystemExit("ERROR: Cannot open camera/video.")
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

typed = ""
mode = START_MODE
last_press_ms = 0
hover_key = None
hover_start_ms = 0
pressed_flash_until = 0
smoothed = None
frame_idx = 0

# Pinch state
pinch_gate = False           # hysteresis gate (True while pinched)
pinch_under_since = None     # ms when we first went under ON
pinch_norm_smooth = None     # EMA for pinch itself

# Calibration state
calibrated = False
pinch_on = PINCH_ON_BASE
pinch_off = PINCH_OFF_BASE
calib_phase = None           # None | "open" | "pinch"
calib_samples = []

# First frame -> set size & layout
ret, frame0 = cap.read()
if not ret:
    raise SystemExit("ERROR: Could not read first frame.")
frame0 = cv2.flip(frame0, 1)
h0, w0 = frame0.shape[:2]
scale = RESIZE_WIDTH / float(w0)
W = RESIZE_WIDTH
H = int(round(h0 * scale))
frame0 = cv2.resize(frame0, (W, H), interpolation=cv2.INTER_LINEAR)
keys, kb_area = build_keyboard_layout(W, H)

def compute_scale(lm):
    """Robust hand scale from two baselines: width across knuckles and wrist→middle span."""
    # indices: 0 wrist, 9 middle_mcp, 5 index_mcp, 17 pinky_mcp
    def to_px(i): return int(lm[i].x * W), int(lm[i].y * H)
    wrist = to_px(0); mid_mcp = to_px(9); idx_mcp = to_px(5); pky_mcp = to_px(17)
    s1 = dist(idx_mcp, pky_mcp)
    s2 = dist(wrist,   mid_mcp)
    return max(1.0, 0.5 * (s1 + s2))

def draw_ui(img, fingertip, hovered_key, pressed_label=None, pinch_info=None):
    img[:] = UI_BG
    x1,y1,x2,y2 = kb_area
    fill_rounded_rect(img, (x1-8, y1-8, x2+8, y2+8), KB_BG, 16)

    now = int(time.time()*1000)
    for k in keys:
        pressed = (pressed_label == k["label"]) and (now < pressed_flash_until)
        draw_key(img, k["rect"], k["label"], hovered_key is k, pressed)

    # text bar
    fill_rounded_rect(img, (15, 15, W-15, 75), KB_BG, 16)
    cv2.putText(img, "Typed: " + typed, (28, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, TEXT_LIGHT, 2, cv2.LINE_AA)

    # footer
    footer = f"Mode: {mode.upper()}  [M]  K=calibrate  D=debug {'ON' if PINCH_DEBUG else 'OFF'}  C=clear  Q=quit  V=video {'ON' if SHOW_CAMERA else 'OFF'}"
    cv2.putText(img, footer, (20, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_DIM, 2, cv2.LINE_AA)

    # click pulse
    if click_fx:
        overlay = img.copy()
        alive = []
        for fx in click_fx:
            age = now - fx["t0"]
            if age < FX_DURATION:
                p = age / FX_DURATION
                r = int(5 + p * fx["rmax"])
                thickness = max(1, int(5 * (1 - p)))
                cv2.circle(overlay, fx["center"], r, KEY_BORDER, thickness, lineType=cv2.LINE_AA)
                alive.append(fx)
        click_fx[:] = alive
        img[:] = cv2.addWeighted(overlay, 0.32, img, 0.68, 0)

    # dwell progress ring
    if fingertip and mode == "dwell" and hovered_key is not None:
        elapsed = max(0, int(time.time()*1000) - hover_start_ms)
        p = np.clip(elapsed / max(DWELL_MS, 1), 0.0, 1.0)
        end_angle = int(-90 + 270 * p)
        cv2.ellipse(img, fingertip, (15,15), 0, -90, end_angle, KEY_BORDER, 3, lineType=cv2.LINE_AA)

    # pointer
    if fingertip:
        cv2.circle(img, fingertip, 7, (255, 180, 90), -1, lineType=cv2.LINE_AA)

    # debug HUD for pinch
    if PINCH_DEBUG and pinch_info is not None:
        s = f"pinch={pinch_info['pn']:.3f}  smooth={pinch_info['pns']:.3f}  ON={pinch_on:.3f} OFF={pinch_off:.3f}  gate={pinch_info['gate']}  hold={pinch_info['hold']}"
        cv2.putText(img, s, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_DIM, 2, cv2.LINE_AA)

    # calibration HUD
    if calib_phase is not None:
        msg = "Calibrating: SHOW OPEN HAND" if calib_phase == "open" else "Calibrating: MAKE A PINCH"
        cv2.putText(img, msg, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

# =========================
# Main loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

    do_process = (frame_idx % PROCESS_EVERY_N == 0)
    frame_idx += 1

    fingertip_px = None
    pinch_norm = None
    scale_val = None
    pinch_info = None

    if do_process:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hls = res.multi_hand_landmarks[0]
            lm = hls.landmark

            def to_px(i): return int(lm[i].x * W), int(lm[i].y * H)
            idx_tip = to_px(8)
            thm_tip = to_px(4)

            # cursor smoothing (EMA)
            if smoothed is None:
                smoothed = idx_tip
            else:
                smoothed = (
                    int(SMOOTH_ALPHA*idx_tip[0] + (1-SMOOTH_ALPHA)*smoothed[0]),
                    int(SMOOTH_ALPHA*idx_tip[1] + (1-SMOOTH_ALPHA)*smoothed[1]),
                )
            fingertip_px = smoothed

            # robust scale & pinch norm
            scale_val = compute_scale(lm)
            pinch_dist = dist(thm_tip, idx_tip)
            pinch_norm = pinch_dist / scale_val

            # pinch EMA (smoother than cursor)
            if pinch_norm_smooth is None:
                pinch_norm_smooth = pinch_norm
            else:
                pinch_norm_smooth = PINCH_SMOOTH_A*pinch_norm + (1-PINCH_SMOOTH_A)*pinch_norm_smooth

            # calibration sampling
            if calib_phase is not None:
                calib_samples.append(pinch_norm)
        else:
            # keep last fingertip/pinch if hand lost
            pass
    else:
        fingertip_px = smoothed

    # Determine hovered key
    hovered = None
    if fingertip_px is not None:
        k = nearest_key(keys, fingertip_px)
        if pt_in_rect(fingertip_px, k["rect"]):
            hovered = k

    now_ms = int(time.time()*1000)
    pressed_label = None

    # Track hover (for dwell)
    if hovered is not None:
        core_ok = pt_in_rect(fingertip_px, hovered["core_rect"])
        if hover_key is not hovered:
            hover_key = hovered
            hover_start_ms = now_ms
    else:
        hover_key = None
        hover_start_ms = now_ms

    # ===== Calibration routine =====
    if calib_phase == "open":
        # collect for CALIB_OPEN_MS then switch to pinch
        # timers handled by sample count & time; here we just wait and user toggles again to proceed auto
        pass
    elif calib_phase == "pinch":
        pass

    # automatically end phases after durations once we started
    # we need to store a start time per phase:
    # We'll cleverly piggyback on 'last_press_ms' as phase timer store when phase starts
    # but better: use separate vars
    try:
        calib_phase_start
    except NameError:
        calib_phase_start = None

    # Process phase timeout
    if calib_phase is not None and pinch_norm is not None:
        if calib_phase_start is None:
            calib_phase_start = now_ms
        elapsed = now_ms - calib_phase_start
        need = CALIB_OPEN_MS if calib_phase == "open" else CALIB_PINCH_MS
        if elapsed >= need:
            if calib_phase == "open" and len(calib_samples) >= 5:
                open_mean = float(np.median(calib_samples))
                # switch to pinch phase
                calib_phase = "pinch"
                calib_phase_start = now_ms
                calib_samples = []
                # store interim
                calib_open_mean = open_mean
            elif calib_phase == "pinch" and len(calib_samples) >= 5:
                pinch_mean = float(np.median(calib_samples))
                # finalize thresholds based on open baseline
                base = calib_open_mean if 'calib_open_mean' in globals() else pinch_mean*1.5
                pinch_on = base * CALIB_ON_FACT
                pinch_off = base * CALIB_OFF_FACT
                calibrated = True
                calib_phase = None
                calib_phase_start = None
                calib_samples = []

    # ---------- DWELL ----------
    if hovered is not None and mode == "dwell":
        if core_ok and (now_ms - hover_start_ms) >= DWELL_MS and (now_ms - last_press_ms) >= COOLDOWN_MS:
            typed = commit(hovered["label"], typed)
            last_press_ms = now_ms
            pressed_label = hovered["label"]
            pressed_flash_until = now_ms + 120
            trigger_click_fx(hovered)
            hover_start_ms = now_ms   # reset dwell timer for repeated letters

    # ---------- PINCH (with smoothing, hold confirm, adaptive thresholds) ----------
    # decide which thresholds to use
    on_th  = pinch_on
    off_th = pinch_off
    # if not calibrated yet, use base values
    if not calibrated:
        on_th, off_th = PINCH_ON_BASE, PINCH_OFF_BASE

    if hovered is not None and pinch_norm_smooth is not None and mode == "pinch":
        # crossing under ON -> start hold timer
        if not pinch_gate and pinch_norm_smooth < on_th and core_ok:
            if pinch_under_since is None:
                pinch_under_since = now_ms
            # commit only if we've stayed under ON for PINCH_HOLD_MS
            if (now_ms - pinch_under_since) >= PINCH_HOLD_MS and (now_ms - last_press_ms) >= COOLDOWN_MS:
                typed = commit(hovered["label"], typed)
                last_press_ms = now_ms
                pressed_label = hovered["label"]
                pressed_flash_until = now_ms + 120
                trigger_click_fx(hovered)
                hover_start_ms = now_ms
                pinch_gate = True  # latched until OFF
        # went above OFF -> re-arm
        elif pinch_gate and pinch_norm_smooth > off_th:
            pinch_gate = False
            pinch_under_since = None
        # if under ON but moved off key core, reset hold timer
        elif pinch_under_since is not None and not core_ok:
            pinch_under_since = None
        # if above ON while not gated, clear hold
        elif not pinch_gate and pinch_norm_smooth >= on_th:
            pinch_under_since = None

    # Debug info pack
    if PINCH_DEBUG and pinch_norm is not None:
        pinch_info = {
            "pn": pinch_norm,
            "pns": pinch_norm_smooth if pinch_norm_smooth is not None else -1,
            "gate": pinch_gate,
            "hold": 0 if pinch_under_since is None else (now_ms - pinch_under_since)
        }

    # Choose canvas (camera hidden by default)
    if SHOW_CAMERA:
        canvas = frame.copy()
        canvas = cv2.addWeighted(canvas, 0.85, np.full_like(canvas, UI_BG, dtype=canvas.dtype), 0.15, 0)
    else:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

    draw_ui(canvas, fingertip_px, hovered if hovered is not None else None, pressed_label, pinch_info)

    # show
    try:
        cv2.imshow(WIN_NAME, canvas)
    except cv2.error:
        break

    # window close via X
    try:
        if (cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1 or
            cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0):
            break
    except cv2.error:
        break

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('c'):
        typed = ""
    elif key == ord('m'):
        mode = "dwell" if mode == "pinch" else "pinch"
    elif key == ord('v'):
        SHOW_CAMERA = not SHOW_CAMERA
    elif key == ord('d'):
        PINCH_DEBUG = not PINCH_DEBUG
    elif key == ord('k'):
        # start calibration sequence
        calibrated = False
        calib_phase = "open"
        calib_phase_start = None
        calib_samples = []
        pinch_under_since = None
        pinch_gate = False

cap.release()
cv2.destroyAllWindows()
hands.close()
