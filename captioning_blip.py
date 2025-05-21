import cv2
from ultralytics import YOLO
from models.blip import Blip

print("Loading models")
detector = YOLO("yolov8n.pt")

blip = Blip()
blip.build_model()
print("Models loaded")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera is not opened")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector(frame, conf=0.6)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        caption = blip.generate(roi)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,caption,(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,
            0.5,(0, 255, 0),1,cv2.LINE_AA)

    cv2.imshow("Dense Captioning", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
