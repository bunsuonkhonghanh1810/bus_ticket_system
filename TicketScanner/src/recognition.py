import cv2
import mediapipe as mp
import numpy as np
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from db import connect_db, close_db

embedder = FaceNet()
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8)

def detect_face(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt
    results = face_detection.process(image_rgb)

    # Gắn cờ phát hiện
    if results.detections:
        return True
    
    return False

def check_existed_faces(captured_image, status_label, root):
    print("Bắt đầu check_existed_faces")
    conn, cursor = connect_db()
    if not conn:
        print("Lỗi: Không kết nối được cơ sở dữ liệu")
        return "unknown"

    cursor.execute('SELECT * FROM PASSENGERS')
    faceData = cursor.fetchall()
    close_db(conn, cursor)

    existed = {}
    print('hehe')

    for img in captured_image:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x, y, w_bbox, h_bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                if x < 0 or y < 0 or x + w_bbox > w or y + h_bbox > h:
                    continue

                print('huhu')

                face_crop = img[y:y+h_bbox, x:x+w_bbox]

                if face_crop.size == 0:
                    continue

                cv2.imwrite('1.jpg', face_crop)

                passengerId = recognize_face(face_crop, faceData)
                existed[passengerId] = existed.get(passengerId, 0) + 1
                print(passengerId)

        cv2.waitKey(10)

    s = sorted(existed, key=existed.get, reverse=True)
    if not s:
        return "unknown"
    if s[0] == "unknown" and len(s) > 1:
        return s[1]
    return s[0]


def recognize_face(face_image, faceData):
    try:
        face_image = cv2.resize(face_image, (160, 160))
        face_image = np.expand_dims(face_image, axis=0)
        emb = embedder.embeddings(face_image)[0]
        emb = emb / np.linalg.norm(emb)

        best_match = None
        best_score = float("inf")

        for face in faceData:
            stored_emb = np.frombuffer(face[1], dtype=np.float32).reshape((512,))
            # print(stored_emb)
            stored_emb = stored_emb / np.linalg.norm(stored_emb)
            score = cosine(emb, stored_emb)

            if score < best_score:
                best_score = score
                best_match = face[0]

        return best_match if best_score < 0.3 else "unknown"
    except Exception as e:
        print(f"Lỗi nhận diện khuôn mặt: {e}")
        return "unknown"