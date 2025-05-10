import cv2
import mediapipe as mp
import numpy as np
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from db import connect_db, close_db

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
embedder = FaceNet()
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def capture_face(captured_images, cap, status_label=None, root=None):
    # Số lượng ảnh cần chụp cho mỗi hướng
    max_images_per_direction = 10
    forward_count = 0
    left_count = 0
    right_count = 0

    # Giai đoạn hiện tại (0: trước, 1: trái, 2: phải)
    stage = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if status_label and root:
                root.after(0, lambda: status_label.configure(text="Trạng thái: Lỗi webcam"))
            print("Không thể đọc khung hình từ webcam")
            return captured_images

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_direction = detect_face_pose(face_landmarks, w, h)

                if stage == 0: 
                    root.after(0, lambda: status_label.configure(
                        text="Mời nhìn về phía trước"
                    ))
                elif stage == 1:
                    root.after(0, lambda: status_label.configure(
                        text="Mời nhìn qua bên trái"
                    ))
                else:
                    root.after(0, lambda: status_label.configure(
                        text="Mời nhìn qua bên phải"
                    ))

                if stage == 0 and face_direction == "Looking Forward" and forward_count < max_images_per_direction:
                    captured_images.append(frame.copy())
                    forward_count += 1
                    if forward_count >= max_images_per_direction:
                        stage = 1  # Chuyển sang chụp trái

                elif stage == 1 and face_direction == "Looking Left" and left_count < max_images_per_direction:
                    captured_images.append(frame.copy())
                    left_count += 1
                    if left_count >= max_images_per_direction:
                        stage = 2  # Chuyển sang chụp phải

                elif stage == 2 and face_direction == "Looking Right" and right_count < max_images_per_direction:
                    captured_images.append(frame.copy())
                    right_count += 1
                    if right_count >= max_images_per_direction:
                        # Hoàn tất, thoát vòng lặp
                        root.after(0, lambda: status_label.configure(text="Trạng thái: Hoàn tất chụp ảnh"))
                        face_mesh.close()
                        return captured_images

        # Đợi nhẹ để tránh CPU overload
        cv2.waitKey(30)

    face_mesh.close()
    return captured_images

def detect_face_pose(face_landmarks, img_w, img_h):
    left_eye = face_landmarks.landmark[33]  # Right eye corner
    right_eye = face_landmarks.landmark[263]  # Left eye corner
    nose_tip = face_landmarks.landmark[1]  # Nose tip
    
    x_left, x_right, x_nose = int(left_eye.x * img_w), int(right_eye.x * img_w), int(nose_tip.x * img_w)

    if x_nose < x_left:
        return "Looking Right"
    elif x_nose > x_right:
        return "Looking Left"
    else:
        return "Looking Forward"
        
# Hàm check_existed_faces (sửa để xử lý lỗi và in hehe)
def check_existed_faces(cap, status_label, root):
    print("Bắt đầu check_existed_faces")
    conn, cursor = connect_db()
    if not conn:
        print("Lỗi: Không kết nối được cơ sở dữ liệu")
        return "unknown"

    cursor.execute('SELECT * FROM PASSENGERS')
    faceData = cursor.fetchall()

    existed = {}
    max_attempts = 100

    for _ in range(max_attempts):
        ret, frame = cap.read()
        
        if not ret:
            print("Lỗi: Không đọc được khung hình")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_bbox, h_bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                if x < 0 or y < 0 or x + w_bbox > w or y + h_bbox > h:
                    continue
                face_crop = frame[y:y+h_bbox, x:x+w_bbox]
                if face_crop.size == 0:
                    continue
                
                passengerId = recognize_face(face_crop, faceData)
                existed[passengerId] = existed.get(passengerId, 0) + 1

        if sum(existed.values()) > 10:
            break

        cv2.waitKey(10)

    close_db(conn, cursor)

    s = sorted(existed, key=existed.get, reverse=True)
    if not s:
        return "unknown"
    if s[0] == "unknown" and len(s) > 1:
        return s[1]
    return s[0]

# Hàm recognize_face
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