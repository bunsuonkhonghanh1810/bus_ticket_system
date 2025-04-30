import tkinter as tk
import cv2
import threading
import queue
import uuid
import random
import time
import numpy as np

from PIL import Image, ImageTk

from db import connect_db, close_db
from recognition import check_existed_faces, detect_face, embedder

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")

        # Biến lưu trạng thái
        self.frame_counter = 0
        self.face_detected = False
        self.face_queue = queue.Queue(maxsize=2)
        self.detecting = True  # Cờ cho phép detect face
        self.processing_passenger = False  # Đang xử lý nhận diện không?
        embedder.embeddings(np.zeros((1, 160, 160, 3), dtype=np.uint8))
        threading.Thread(target=self.detect_face_thread, daemon=True).start()


        # Tạo webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Không thể mở webcam!")
            return

        # Đặt độ phân giải webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Khu vực hiển thị video
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Nhãn hiển thị tên
        self.status_label = tk.Label(root, text="Đang xử lý", font=("Arial", 14))
        self.status_label.pack(pady=50)

        # Cập nhật video
        self.update_video()

    def detect_face_thread(self):
        while True:
            if not self.detecting:
                continue
            try:
                frame = self.face_queue.get(timeout=1)
                self.face_detected = detect_face(frame)
            except queue.Empty:
                continue

    def check_faces_and_find_ticket(self, cap):
        passenger_id = check_existed_faces(cap, self.status_label, self.root)
        if passenger_id != "unknown":
            conn, cursor = connect_db()
            if not conn:
                print("Không thể kết nối để thực hiện INSERT")
                return
            cursor.execute(f"SELECT * FROM TICKETS JOIN TICKETCLASS ON TICKETS.TicketClassId = TICKETCLASS.TicketClassId WHERE PassengerId = '{passenger_id}' AND TicketType = 'Single' AND TicketState = 'Unused' ORDER BY PurchaseTime ASC")
            tickets = cursor.fetchall()
            if len(tickets) > 0:
                ticket = tickets[0]
                cursor.execute("INSERT INTO BUSENTRY (EntryId, Fare, PassengerId, TicketId, StopId, BusId) VALUES (?, ?, ?, ?, ?, ?)", 
                               (str(uuid.uuid4()), ticket[12], ticket[5], ticket[0], random.randint(1, 23), random.randint(1, 40)))
                self.root.after(0, lambda: self.status_label.configure(text="Chào mừng hành khách"))
            else:
                cursor.execute(f"SELECT * FROM TICKETS JOIN TICKETCLASS ON TICKETS.TicketClassId = TICKETCLASS.TicketClassId WHERE PassengerId = '{passenger_id}' AND TicketType = 'Monthly' AND TicketState = 'Active'")
                tickets = cursor.fetchall()
                if len(tickets) == 1: 
                    ticket = tickets[0]
                    cursor.execute("INSERT INTO BUSENTRY (EntryId, Fare, PassengerId, TicketId, StopId, BusId) VALUES (?, ?, ?, ?, ?, ?)", 
                               (str(uuid.uuid4()), ticket[12], ticket[5], ticket[0], random.randint(1, 23), random.randint(1, 40)))
                    self.root.after(0, lambda: self.status_label.configure(text="Chào mừng hành khách"))
                else:
                    self.root.after(0, lambda: self.status_label.configure(text="Hành khách không có vé"))
            close_db(conn, cursor)
        else:
            print('Không tìm thấy hành khách')
            self.root.after(0, lambda: self.status_label.configure(text="Không tìm thấy hành khách"))

        # Khởi động lại detect face sau khi xử lý xong
        self.root.after(4000, self.continue_processing)

    def continue_processing(self):
        self.root.after(0, lambda: self.status_label.configure(text="Đang xử lý"))
        self.detecting = True
        self.processing_passenger = False

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Chuyển khung hình từ BGR (OpenCV) sang RGB
            temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Chuyển thành định dạng ImageTk
            img = Image.fromarray(temp_frame)
            # Resize khung hình nếu muốn lớn hơn (ví dụ: 800x600)
            img = img.resize((960, 720), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk  # Giữ tham chiếu để tránh garbage collection

            self.frame_counter += 1

            if self.frame_counter % 5 == 0:
                if not self.processing_passenger:
                    self.face_queue.put(frame.copy())

                    if self.face_detected:
                        print("✅ Có khuôn mặt")
                        self.processing_passenger = True
                        self.detecting = False  # Dừng phát hiện thêm mặt
                        threading.Thread(target=self.check_faces_and_find_ticket, args=(self.cap,), daemon=True).start()
                    else:
                        print("❌ Không có khuôn mặt")

                self.frame_counter = 0

        # Cập nhật liên tục
        self.root.after(30, self.update_video)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()