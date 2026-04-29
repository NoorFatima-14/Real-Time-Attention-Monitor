

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from collections import deque

# ── Config ─────────────────────────────────────────────────────────────────────
SCREENSHOTS_DIR  = "./screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

EAR_THRESH       = 0.22    # below this = eye closing
EAR_CONSEC       = 15      # frames below threshold to confirm drowsy
DISPLAY_W        = 1000
DISPLAY_H        = 620
PANEL_W          = 280
CAM_W            = DISPLAY_W - PANEL_W
CAM_H            = DISPLAY_H - 50
# ───────────────────────────────────────────────────────────────────────────────

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=6,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (MediaPipe Face Mesh)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── Colors (BGR) ───────────────────────────────────────────────────────────────
COLORS = {
    "Attentive": (0,   220, 80),
    "Drowsy":    (0,   80,  255),
    "No Face":   (80,  80,  80),
}
BG_COLORS = {
    "Attentive": (0,   50,  20),
    "Drowsy":    (0,   20,  80),
    "No Face":   (30,  30,  30),
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EAR CALCULATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_ear(landmarks, eye_indices, w, h):
    """
    Eye Aspect Ratio:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    When eye is open  → EAR ~ 0.3
    When eye is closed → EAR ~ 0.0
    """
    pts = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in eye_indices
    ])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DRAWING HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_ear_bar(canvas, ear_val, x, y, bar_w=200, bar_h=18):
    """Draw EAR level bar — green when open, red when closing."""
    ratio = min(ear_val / 0.35, 1.0)
    color = (0, int(220 * ratio), int(255 * (1 - ratio)))
    cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), (30, 30, 45), -1)
    cv2.rectangle(canvas, (x, y), (x + int(bar_w * ratio), y + bar_h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), (60, 60, 80), 1)
    # Threshold marker
    thresh_x = x + int(bar_w * (EAR_THRESH / 0.35))
    cv2.line(canvas, (thresh_x, y - 3), (thresh_x, y + bar_h + 3), (0, 200, 255), 2)
    cv2.putText(canvas, f"EAR: {ear_val:.2f}",
                (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 180), 1)


def draw_gauge(canvas, value, cx, cy, radius=50, color=(0, 220, 80), label=""):
    """Circular percentage gauge."""
    cv2.circle(canvas, (cx, cy), radius, (30, 30, 45), -1)
    cv2.circle(canvas, (cx, cy), radius, (55, 55, 75), 2)
    angle = int(360 * value / 100)
    for a in range(-90, -90 + angle, 3):
        rad = np.radians(a)
        px  = int(cx + (radius - 7) * np.cos(rad))
        py  = int(cy + (radius - 7) * np.sin(rad))
        cv2.circle(canvas, (px, py), 3, color, -1)
    cv2.putText(canvas, f"{int(value)}%",
                (cx - 22, cy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    if label:
        cv2.putText(canvas, label,
                    (cx - len(label) * 4, cy + radius + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 160), 1)


def draw_mini_bar(canvas, label, value, x, y, bar_w=180, bar_h=14, color=(0,220,80)):
    cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), (30, 30, 45), -1)
    fill = int(bar_w * value / 100)
    if fill > 0:
        cv2.rectangle(canvas, (x, y), (x + fill, y + bar_h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), (55, 55, 70), 1)
    cv2.putText(canvas, f"{label}: {value:.0f}%",
                (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 180), 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Session:
    def __init__(self):
        self.start_time   = time.time()
        self.counts       = {"Attentive": 0, "Drowsy": 0, "No Face": 0}
        self.ear_counters = {}       # drowsiness frame counter per face
        self.blink_counts = {}       # blink counter per face
        self.in_blink     = {}       # blink state per face
        self.history      = deque(maxlen=400)
        self.screenshot_n = 0

    def reset(self):
        self.__init__()

    def pct(self):
        total = sum(self.counts.values()) or 1
        return {k: v / total * 100 for k, v in self.counts.items()}

    def elapsed(self):
        s = int(time.time() - self.start_time)
        return f"{s // 60:02d}:{s % 60:02d}"

    def update_blink(self, face_idx, ear_val):
        """Detect blinks: a quick dip below threshold that comes back up."""
        if ear_val < EAR_THRESH:
            self.in_blink[face_idx] = True
        else:
            if self.in_blink.get(face_idx, False):
                self.blink_counts[face_idx] = self.blink_counts.get(face_idx, 0) + 1
            self.in_blink[face_idx] = False
        return self.blink_counts.get(face_idx, 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    session   = Session()
    paused    = False
    fps_times = deque(maxlen=30)

    cv2.namedWindow("Classroom Attention Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Classroom Attention Monitor", DISPLAY_W, DISPLAY_H)

    print("=" * 50)
    print("  📚 Classroom Attention Monitoring System")
    print("=" * 50)
    print("  SPACE=Pause  S=Screenshot  R=Reset  Q=Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        fps_times.append(time.time())
        fps = len([t for t in fps_times if time.time() - t < 1])

        # ── MediaPipe inference ───────────────────────────────────────────────
        face_states  = []
        face_ears    = []
        face_blinks  = []

        if not paused:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results and results.multi_face_landmarks:
                for idx, face_lm in enumerate(results.multi_face_landmarks):
                    lm = face_lm.landmark

                    # EAR
                    ear_l = calc_ear(lm, LEFT_EYE,  w, h)
                    ear_r = calc_ear(lm, RIGHT_EYE, w, h)
                    ear_v = (ear_l + ear_r) / 2.0

                    # Drowsiness counter
                    if ear_v < EAR_THRESH:
                        session.ear_counters[idx] = session.ear_counters.get(idx, 0) + 1
                    else:
                        session.ear_counters[idx] = 0

                    # Blink
                    blinks = session.update_blink(idx, ear_v)

                    # Classify
                    state = "Drowsy" if session.ear_counters.get(idx, 0) >= EAR_CONSEC \
                            else "Attentive"

                    face_states.append(state)
                    face_ears.append(ear_v)
                    face_blinks.append(blinks)
                    session.counts[state] += 1

                    # ── Face bounding box ──
                    xs = [int(p.x * w) for p in lm]
                    ys = [int(p.y * h) for p in lm]
                    x1 = max(0,  min(xs) - 12)
                    y1 = max(0,  min(ys) - 12)
                    x2 = min(w,  max(xs) + 12)
                    y2 = min(h,  max(ys) + 12)

                    color = COLORS[state]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # State badge above box
                    badge = f"#{idx+1}  {state}"
                    bw    = len(badge) * 11
                    cv2.rectangle(frame, (x1, y1-30), (x1+bw, y1), BG_COLORS[state], -1)
                    cv2.putText(frame, badge, (x1+4, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                    # EAR value below box
                    cv2.putText(frame, f"EAR: {ear_v:.2f}  Blinks: {blinks}",
                                (x1, y2+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)

                    # Eye dots
                    for ei in LEFT_EYE + RIGHT_EYE:
                        ex = int(lm[ei].x * w)
                        ey = int(lm[ei].y * h)
                        cv2.circle(frame, (ex, ey), 2, color, -1)

                    # Drowsy alert overlay
                    if state == "Drowsy":
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                                      (0, 0, 200), -1)
                        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                        cv2.putText(frame, "⚠ DROWSY",
                                    (x1+4, (y1+y2)//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 80, 255), 2)
            else:
                session.counts["No Face"] += 1

        # Track history
        session.history.append(face_states[0] if face_states else "No Face")

        # ── Resize camera feed ────────────────────────────────────────────────
        cam = cv2.resize(frame, (CAM_W, CAM_H))

        cv2.putText(cam, f"FPS {fps}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        if paused:
            cv2.putText(cam, "PAUSED", (CAM_W//2-70, CAM_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,200,255), 3)

        # ── Right panel ───────────────────────────────────────────────────────
        panel = np.zeros((CAM_H, PANEL_W, 3), dtype=np.uint8)
        panel[:] = (16, 16, 28)

        cv2.putText(panel, "ATTENTION", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200,210,255), 2)
        cv2.putText(panel, "MONITOR",   (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200,210,255), 2)
        cv2.line(panel, (10,68), (PANEL_W-10,68), (45,45,65), 1)

        cv2.putText(panel, f"Session:  {session.elapsed()}",
                    (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120,120,150), 1)
        cv2.putText(panel, f"Faces:    {len(face_states)}",
                    (12,110), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120,120,150), 1)
        cv2.line(panel, (10,122), (PANEL_W-10,122), (35,35,55), 1)

        # Attention gauge
        pcts = session.pct()
        att  = pcts["Attentive"]
        gauge_color = (0,220,80) if att > 60 else (0,180,255) if att > 30 else (0,60,255)
        draw_gauge(panel, att, PANEL_W//2, 188, radius=52,
                   color=gauge_color, label="Attentive")

        cv2.line(panel, (10,255), (PANEL_W-10,255), (35,35,55), 1)
        draw_mini_bar(panel, "Attentive", pcts["Attentive"],
                      12, 275, color=(0,220,80))
        draw_mini_bar(panel, "Drowsy",    pcts["Drowsy"],
                      12, 310, color=(0,80,255))
        draw_mini_bar(panel, "No Face",   pcts["No Face"],
                      12, 345, color=(80,80,80))

        # Per-face EAR bars
        cv2.line(panel, (10,375), (PANEL_W-10,375), (35,35,55), 1)
        cv2.putText(panel, "Eye Openness:", (12,393),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140,140,160), 1)
        for i, (ear_v, blinks) in enumerate(zip(face_ears[:3], face_blinks[:3])):
            yy = 410 + i * 55
            cv2.putText(panel, f"Person {i+1}  Blinks: {blinks}",
                        (12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,140), 1)
            draw_ear_bar(panel, ear_v, 12, yy+8, bar_w=PANEL_W-30)

        # Current states list
        cv2.line(panel, (10, CAM_H-100), (PANEL_W-10, CAM_H-100), (35,35,55), 1)
        for i, s in enumerate(face_states[:4]):
            color = COLORS[s]
            cv2.circle(panel, (20, CAM_H-83+i*20), 5, color, -1)
            cv2.putText(panel, f"Person {i+1}: {s}",
                        (32, CAM_H-78+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # Controls
        cv2.line(panel, (10, CAM_H-25), (PANEL_W-10, CAM_H-25), (35,35,55), 1)
        cv2.putText(panel, "[SPACE] Pause  [S] Shot  [R] Reset  [Q] Quit",
                    (8, CAM_H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (70,70,100), 1)

        # ── Combine ───────────────────────────────────────────────────────────
        top    = np.hstack([cam, panel])
        bottom = np.zeros((50, DISPLAY_W, 3), dtype=np.uint8)
        bottom[:] = (10, 10, 20)
        cv2.putText(bottom, "Classroom Attention Monitoring System",
                    (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55,55,80), 1)

        # History color strip
        hist = list(session.history)
        if hist:
            seg = DISPLAY_W / max(len(hist), 1)
            for i, s in enumerate(hist):
                c  = COLORS.get(s, (60,60,60))
                x1 = int(i * seg)
                x2 = int((i+1) * seg)
                cv2.rectangle(bottom, (x1,32), (x2,48), c, -1)

        canvas = np.vstack([top, bottom])
        cv2.imshow("Classroom Attention Monitor", canvas)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
            print("⏸ Paused" if paused else "▶ Resumed")
        elif key in (ord('s'), ord('S')):
            session.screenshot_n += 1
            path = os.path.join(SCREENSHOTS_DIR,
                                f"attention_{session.screenshot_n:03d}.png")
            cv2.imwrite(path, canvas)
            print(f"📸 Saved: {path}")
        elif key in (ord('r'), ord('R')):
            session.reset()
            print("🔄 Session reset")

    cap.release()
    cv2.destroyAllWindows()

    pcts = session.pct()
    print("\n" + "="*45)
    print("  📊 SESSION SUMMARY")
    print("="*45)
    print(f"  Duration   : {session.elapsed()}")
    print(f"  Attentive  : {pcts['Attentive']:.1f}%")
    print(f"  Drowsy     : {pcts['Drowsy']:.1f}%")
    print("="*45)


if __name__ == "__main__":
    main()