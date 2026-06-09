import cv2
import math
import numpy as np
from ultralytics import YOLO

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = "/Users/takneekmacmini/Documents/Ball track detection/V4/best.pt"
VIDEO_PATH = "/Users/takneekmacmini/Documents/Ball track detection/input.mp4"
OUTPUT_PATH = "/Users/takneekmacmini/Documents/Ball track detection/input_track.mp4"

BALL_CLASS_ID = 0

# ------------------------------------------
# CALIBRATION
# ------------------------------------------

VISIBLE_PITCH_METERS = 3.2
VISIBLE_PITCH_PIXELS = 700

METERS_PER_PIXEL = (
    VISIBLE_PITCH_METERS /
    VISIBLE_PITCH_PIXELS
)

# use first N detections
MAX_PIXEL_JUMP = 550
MIN_DETECTIONS_FOR_SPEED = 6

# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO(MODEL_PATH)

# ==========================================
# VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# ==========================================
# STORAGE
# ==========================================

detections = []

frame_id = 0

speed_kmh = None

# ==========================================
# PROCESS
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        conf=0.80,
        verbose=False
    )

    ball_found = False

    for box in results[0].boxes:

        cls = int(box.cls[0])

        if cls != BALL_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if len(detections) > 0:

            _, prev_x, prev_y = detections[-1]

            jump = math.sqrt(
                (cx - prev_x) ** 2 +
                (cy - prev_y) ** 2
            )

            if jump > MAX_PIXEL_JUMP:

                print(
                    f"Rejected jump: {jump:.1f}px "
                    f"at frame {frame_id}"
                )

                continue

        detections.append(
            (
                frame_id,
                cx,
                cy
            )
        )

        ball_found = True

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            4,
            (0,0,255),
            -1
        )

        break
    
    # ======================================
    # SPEED CALCULATION
    # ======================================

    if (
    speed_kmh is None and
    len(detections) >= MIN_DETECTIONS_FOR_SPEED
):

        first = detections[0]
        last = detections[-1]

        f1, x1, y1 = first
        f2, x2, y2 = last

        pixel_distance = math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

        real_distance = (
            pixel_distance *
            METERS_PER_PIXEL
        )

        elapsed_time = (
            (f2 - f1) / fps
        )

        if elapsed_time > 0:

            speed_mps = (
                real_distance /
                elapsed_time
            )

            speed_kmh = (
                speed_mps * 3.6
            )

            print("\n========== SPEED ==========")
            print("First :", first)
            print("Last  :", last)
            print("Pixel Distance :", pixel_distance)
            print("Time :", elapsed_time)
            print("Speed :", speed_kmh)
            print("===========================\n")
    # ======================================
    # DRAW TRAJECTORY
    # ======================================

    if len(detections) == 3:

        for i in range(
            1,
            len(detections)
        ):

            _, px1, py1 = detections[i-1]
            _, px2, py2 = detections[i]

            cv2.line(
                frame,
                (px1, py1),
                (px2, py2),
                (255,255,0),
                2
            )

    # ======================================
    # DISPLAY SPEED
    # ======================================

    if speed_kmh is not None:

        cv2.putText(
            frame,
            f"Ball Speed: {speed_kmh:.1f} km/h",
            (30,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

    cv2.putText(
        frame,
        f"Detections: {len(detections)}",
        (30,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    writer.write(frame)

    cv2.imshow(
        "Ball Speed",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

    frame_id += 1

# ==========================================
# CLEANUP
# ==========================================

cap.release()
writer.release()

cv2.destroyAllWindows()

for d in detections:
    print(d)

print(
    f"Saved: {OUTPUT_PATH}"
)

if speed_kmh is not None:

    print(
        f"Final Speed = "
        f"{speed_kmh:.2f} km/h"
    )

else:

    print(
        "Not enough detections."
    )