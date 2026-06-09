import cv2
from ultralytics import YOLO
from collections import deque

# ==========================
# CONFIG
# ==========================

MODEL_PATH = "/Users/takneekmacmini/Documents/Ball track detection/v4/best.pt"
VIDEO_PATH = "/Users/takneekmacmini/Documents/Ball track detection/input.mp4"
OUTPUT_PATH = "/Users/takneekmacmini/Documents/Ball track detection/tracked_input.mp4"

BALL_CLASS_ID = 0

# ==========================
# LOAD MODEL
# ==========================

model = YOLO(MODEL_PATH)

# ==========================
# VIDEO
# ==========================

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# ==========================
# TRAJECTORY
# ==========================

trajectory = deque(maxlen=300)

# ==========================
# LOOP
# ==========================

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

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        trajectory.append((cx, cy))

        ball_found = True

        # Ball box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Ball center
        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )

        break

    # ==========================
    # DRAW TRAJECTORY
    # ==========================

    # ==========================
    # DRAW CRICKET STYLE TRAJECTORY
    # ==========================

    overlay = frame.copy()

    for i in range(1, len(trajectory)):

        # outer glow
        cv2.line(
            overlay,
            trajectory[i - 1],
            trajectory[i],
            (255, 255, 255),
            20
        )

        # bright center
        cv2.line(
            overlay,
            trajectory[i - 1],
            trajectory[i],
            (255, 255, 255),
            8
        )

    frame = cv2.addWeighted(
        overlay,
        0.5,
        frame,
        0.5,
        0
    )

    writer.write(frame)

    cv2.imshow("Ball Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Done")