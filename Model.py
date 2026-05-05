from ultralytics import YOLO
import cv2

model = YOLO("/home/code/Desktop/Model/yolov5nu_ncnn_model")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0

SKIP_FRAMES = 4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        continue

    results = model.predict(
        frame,
        device="cpu",
        imgsz=320,
        conf=0.6,
        verbose=False
    )

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
