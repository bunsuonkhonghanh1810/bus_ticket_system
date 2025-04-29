import tkinter as tk
import cv2
import threading
import queue
import uuid
import numpy as np
from PIL import Image, ImageTk

from db import connect_db, close_db
from capture_face import capture_face, check_existed_faces
from process_faces import process_faces
from extract_embeddings import extract_embeddings

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1800x800")  # Tăng kích thước cửa sổ chính

        # Biến lưu trạng thái
        self.img = []
        self.processed_img = []
        self.passengerId = ""
        self.cap_lock = threading.Lock()

        # Tạo webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Không thể mở webcam!")
            return

        # Đặt độ phân giải webcam cao hơn
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

        # Chia giao diện thành hai khung: trái và phải
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)

        # Khu vực hiển thị video (bên trái) với kích thước lớn hơn
        self.video_label = tk.Label(self.left_frame, width=1280, height=700)  # Tăng kích thước Label
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Các thành phần bên phải
        self.btn_buy_tickets = tk.Button(self.right_frame, text="Mua vé", command=self.buy_ticket, font=("Arial", 15), width=20)
        self.btn_buy_tickets.pack(pady=20)

        # 
        self.status_label = tk.Label(self.right_frame, text="Trạng thái: Đang chờ...", font=("Arial", 14))
        self.status_label.pack_forget() #pady=10, expand=True, fill=tk.X

        # Cập nhật video
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Chuyển khung hình từ BGR (OpenCV) sang RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize khung hình về (1024, 768) để khớp với Label
            frame = cv2.resize(frame, (1280, 700))
            # Chuyển thành định dạng ImageTk
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk  # Giữ tham chiếu

        # Cập nhật liên tục
        self.root.after(10, self.update_video)

    def check_and_capture(self, result_queue):
        try:
            # Chạy check_existed_faces
            with self.cap_lock:
                person_name = check_existed_faces(
                    self.cap, self.status_label, self.root
                )
            result_queue.put(person_name)

            # Chạy capture_face nếu cần
            if person_name == "unknown":
                with self.cap_lock:
                    capture_face(self.img, self.cap, self.status_label, self.root)
                
                self.root.after(0, lambda: self.status_label.configure(text="Đang xử lý"))

                self.processed_img = []
                self.processed_img = process_faces(self.img)
                
        except Exception as e:
            print(f"Lỗi trong check_and_capture: {e}")
            result_queue.put("error")
    
    def extract(self, embedding_queue):
        with self.cap_lock:
            embedding = extract_embeddings(self.processed_img)
        embedding_queue.put(embedding)

    def handle_tickets(self, ticketType):
        def db_insert():
            try:
                conn, cursor = connect_db()
                if not conn:
                    print("Không thể kết nối để thực hiện INSERT")
                    return
                
                final_price = float(ticketType[3]) - float(ticketType[3]) * ticketType[4]
                
                cursor.execute(
                    "INSERT INTO TICKETS (TicketId, PurchaseMethod, TicketState, PassengerId, TicketClassId, FinalPrice) VALUES (?, 'Offline', 'Unused', ?, 13, ?)",
                    (str(uuid.uuid4()), self.passengerId, final_price)
                )
                conn.commit()
                close_db(conn, cursor)

                # Khôi phục giao diện trong luồng chính
                self.root.after(0, self.reset_interface)
            except Exception as e:
                print(f"Lỗi trong handle_tickets: {e}")
                close_db(conn, cursor)
        
        # Chạy INSERT trong luồng riêng, không can thiệp vào update_video
        threading.Thread(target=db_insert, daemon=True).start()

    def reset_interface(self):
        # Xóa tất cả widget trong right_frame
        for widget in self.right_frame.winfo_children():
            widget.destroy()
            
        self.btn_buy_tickets = tk.Button(self.right_frame, text="Mua vé", command=self.buy_ticket, font=("Arial", 15), width=20)
        self.btn_buy_tickets.pack(pady=20)

        self.status_label = tk.Label(self.right_frame, text="Trạng thái: Đang chờ...", font=("Arial", 14))
        self.status_label.pack_forget() #pady=10, expand=True, fill=tk.X

    def buy_ticket(self):   
        print("Bắt đầu buy_ticket")

        self.btn_buy_tickets.pack_forget()
        self.status_label.pack(pady=10, expand=True, fill=tk.X) #pady=10, expand=True, fill=tk.X

        self.img = []

        result_queue = queue.Queue()
        thread = threading.Thread(target=self.check_and_capture, args=(result_queue,))
        thread.start()

        def check_thread():
            if thread.is_alive():
                self.root.after(100, check_thread)
            else:
                person_name = result_queue.get_nowait()
                if person_name != "unknown":
                    self.passengerId = person_name
                else:
                    embedding_queue = queue.Queue()
                    extract_embeddings_thread = threading.Thread(target=self.extract, args=(embedding_queue,))
                    extract_embeddings_thread.start()
                    
                    def check_extract_embeddings_thread():
                        if extract_embeddings_thread.is_alive():
                            self.root.after(50, check_extract_embeddings_thread)
                        else:
                            embedding = embedding_queue.get_nowait()

                            passengerId = str(uuid.uuid4())
                            self.passengerId = passengerId
                            
                            conn, cursor = connect_db()
                            if not conn:
                                print("Không thể kết nối để thực hiện INSERT")
                                return False

                            try:
                                cursor.execute("INSERT INTO PASSENGERS (PassengerId, FaceData) VALUES (?, ?)", (passengerId, embedding.tobytes()))
                                conn.commit()

                                print(f"Đã thêm passengerId: {passengerId}")

                                return True
                            except Exception as e:
                                print(f"Lỗi khi thực hiện INSERT: {e}")
                                return False
                            finally:
                                close_db(conn, cursor)

                    self.root.after(100, check_extract_embeddings_thread)
                
                self.root.after(0, lambda: self.status_label.configure(text=f"Chào mừng hành khách"))

                conn, cursor = connect_db()
                if not conn:
                    print("Không thể kết nối để thực hiện INSERT")
                    return False
                cursor.execute("SELECT * FROM TICKETCLASS WHERE TicketType = 'Single'")
                tickets = cursor.fetchall()
                for i in range (len(tickets)):
                    btn = tk.Button(self.right_frame, text=tickets[i][2], font=("Arial", 15), width=20, command=lambda t=tickets[i]: self.handle_tickets(t))
                    btn.pack(pady=5)
                    
        self.root.after(100, check_thread)
        
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()