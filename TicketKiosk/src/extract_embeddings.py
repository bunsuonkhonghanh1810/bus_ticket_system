import numpy as np
from keras_facenet import FaceNet
import cv2
import time

embedder = FaceNet()

def extract_embeddings(image):
    if not image:
        print("Lỗi: Danh sách ảnh rỗng trong extract_embeddings")
        return None

    person_embeddings = []
    max_images = len(image)  # Giới hạn số ảnh xử lý
    print(f"Extracting {min(len(image), max_images)}/{len(image)} images")

    for i, img in enumerate(image[:max_images]):
        if not isinstance(img, np.ndarray):
            print(f"Lỗi: Ảnh {i} không hợp lệ trong extract_embeddings")
            continue

        try:
            start_time = time.time()
            img = cv2.resize(img, (160, 160))
            img = np.expand_dims(img, axis=0)
            emb = embedder.embeddings(img)[0]
            person_embeddings.append(emb)
            print(f"Extracting image {i+1}/{min(len(image), max_images)} in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"Lỗi khi tạo embedding cho ảnh {i}: {e}")
            continue

    if not person_embeddings:
        print("Lỗi: Không tạo được embedding nào")
        return None

    embedding = np.mean(person_embeddings, axis=0)
    embedding = embedding / np.linalg.norm(embedding)

    return embedding