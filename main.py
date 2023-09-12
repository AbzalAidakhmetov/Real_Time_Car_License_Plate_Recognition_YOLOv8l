from ultralytics import YOLO
import cv2
import cvzone
import util
from sort import *
from util import get_car, read_license_plate, write_csv

license_plate_detector = YOLO('yolov8l.pt')  # load an official model
license_plate_detector = YOLO('/home/theballer/Downloads/best.pt')  # load a custom model

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8l.pt')
#license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('/home/theballer/Desktop/DS_Learning/PetProjects/PlateRecognitionYOLOv8/sample_final.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            xcar1, ycar1, xcar2, ycar2 = int(xcar1), int(ycar1), int(xcar2), int(ycar2)
            cv2.rectangle(frame, (xcar1,ycar1),(xcar2, ycar2), (255,0,255), 3)
            
            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                
                
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                
                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                if license_plate_text_score is not None and license_plate_text_score > 0.5:
                    cv2.rectangle(frame, (x1,y1),(x2, y2), (255,0,255), 3) 
                    cvzone.putTextRect(frame, f'{license_plate_text}', (max(0, x1), max(35, y1)), scale = 1.5, thickness=2, offset=3)
    
    cv2.imshow('Image', frame)
    #cv2.imshow('ImageRegion', imgRegion)
    cv2.waitKey(1)

