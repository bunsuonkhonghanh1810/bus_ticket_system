import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def process_faces(img):
    process_faces = []

    for each_img in img:
        rgb_img = cv2.cvtColor(each_img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_img)
        if results.detections:
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = each_img.shape
                x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                process_faces.append(each_img[y:y+h, x:x+w])
    
    return process_faces