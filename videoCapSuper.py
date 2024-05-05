import time
import cv2 
from ultralytics import YOLO
from OrangeDetector import OrangeDetector
from KalmanFilter import KalmanFilter

# importing necessary for kalman filter

od = OrangeDetector()
kf = KalmanFilter()

# end of importing necessary for kalman filter


# model_path = '../ourDataSet2.pt'
model_path = 'yolov8n-seg.pt'
model = YOLO(model_path)

my_vid = 'vid.mp4'

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if success:
        start = time.perf_counter()
        results = model(frame)
        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time
        annotated_frame = results[0].plot()


# this zone is drawing a circle to the center of the detected object

        orange_bbox = od.detect(frame)
        x, y, x2, y2 = orange_bbox
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

# production of kalman filter

        predicted = kf.predict(cx, cy)

# end of production of kalman filter
        # 20 is the radius of the circle and -2 is the thickness of the circle

        cv2.circle(annotated_frame, (cx, cy), 20, (0, 0, 255), 4)
        cv2.circle(annotated_frame, (predicted[0], predicted[1]),20,(255,0,0), -2)
        cv2.line(annotated_frame, (cx, cy), (predicted[0], predicted[1]), (0, 255, 0), 5)

# kalman filter zone end the ditection object work


        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Annotated Frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release() 
cv2.destroyAllWindows()


