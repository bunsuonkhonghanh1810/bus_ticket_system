import logging
import queue
import threading
import tkinter as tk
import cv2
import uuid
import numpy as np
from PIL import Image, ImageTk
from db import connect_db, close_db
from capture_face import capture_face, check_existed_faces
from process_faces import process_faces
from extract_embeddings import extract_embeddings

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1800x800")
        self.img = []
        self.processed_img = []
        self.passengerId = ""
        self.cap_lock = threading.Lock()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Không thể mở webcam!")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)
        self.video_label = tk.Label(self.left_frame, width=1280, height=700)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.btn_buy_tickets = tk.Button(self.right_frame, text="Mua vé", command=self.buy_ticket, font=("Arial", 15), width=20)
        self.btn_buy_tickets.pack(pady=20)
        self.status_label = tk.Label(self.right_frame, text="Trạng thái: Đang chờ...", font=("Arial", 14))
        self.status_label.pack_forget()
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (1280, 700))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk
        self.root.after(10, self.update_video)

    def check_and_capture(self, result_queue, processed_img_queue):
        logging.debug("Bắt đầu check_and_capture")
        try:
            with self.cap_lock:
                person_name = check_existed_faces(self.cap, self.status_label, self.root)
            logging.debug(f"check_existed_faces trả về: {person_name}")
            result_queue.put(person_name)
            if person_name == "unknown":
                with self.cap_lock:
                    capture_face(self.img, self.cap, self.status_label, self.root)
                self.root.after(0, lambda: self.status_label.configure(text="Đang xử lý"))
                logging.debug("Bắt đầu process_faces")
                processed_img = process_faces(self.img)
                logging.debug(f"process_faces hoàn thành, kết quả: {processed_img}")
                processed_img_queue.put(processed_img)
        except Exception as e:
            logging.error(f"Lỗi trong check_and_capture: {e}")
            result_queue.put("error")
            processed_img_queue.put([])

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
                self.root.after(0, self.reset_interface)
            except Exception as e:
                print(f"Lỗi trong handle_tickets: {e}")
                close_db(conn, cursor)
        threading.Thread(target=db_insert, daemon=True).start()

    def reset_interface(self):
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        self.btn_buy_tickets = tk.Button(self.right_frame, text="Mua vé", command=self.buy_ticket, font=("Arial", 15), width=20)
        self.btn_buy_tickets.pack(pady=20)
        self.status_label = tk.Label(self.right_frame, text="Trạng thái: Đang chờ...", font=("Arial", 14))
        self.status_label.pack_forget()

    def show_ticket_options(self):
        conn, cursor = connect_db()
        if not conn:
            print("Không thể kết nối để thực hiện INSERT")
            return
        cursor.execute("SELECT * FROM TICKETCLASS WHERE TicketType = 'Single'")
        tickets = cursor.fetchall()

        print('hehe')

        ticket_name_mapping = {
            "January": "Tháng 1",
            "February": "Tháng 2",
            "March": "Tháng 3",
            "April": "Tháng 4",
            "May": "Tháng 5",
            "June": "Tháng 6",
            "July": "Tháng 7",
            "August": "Tháng 8",
            "September": "Tháng 9",
            "October": "Tháng 10",
            "November": "Tháng 11",
            "December": "Tháng 12",
            "Single": "Vé lượt"
        }
        for i in range(len(tickets)):
            ticketName = ticket_name_mapping.get(tickets[i][2])

            print(ticketName)
            
            btn = tk.Button(self.right_frame, text=ticketName, font=("Arial", 15), width=20, command=lambda t=tickets[i]: self.handle_tickets(t))
            btn.pack(pady=5)
        close_db(conn, cursor)

    def buy_ticket(self):
        logging.debug("Bắt đầu buy_ticket")
        self.btn_buy_tickets.pack_forget()
        self.status_label.pack(pady=10, expand=True, fill=tk.X)
        self.img = []
        result_queue = queue.Queue()
        processed_img_queue = queue.Queue()
        thread = threading.Thread(target=self.check_and_capture, args=(result_queue, processed_img_queue))
        thread.start()

        def check_thread():
            if thread.is_alive():
                self.root.after(100, check_thread)
            else:
                try:
                    person_name = result_queue.get_nowait()
                    logging.debug(f"person_name: {person_name}")
                    if person_name != "unknown":
                        self.passengerId = person_name
                        self.root.after(0, lambda: self.status_label.configure(text=f"Chào mừng hành khách"))
                        self.show_ticket_options()
                    else:
                        try:
                            self.processed_img = processed_img_queue.get_nowait()
                            logging.debug(f"processed_img: {self.processed_img}")
                            embedding_queue = queue.Queue()
                            extract_embeddings_thread = threading.Thread(target=self.extract, args=(embedding_queue,))
                            extract_embeddings_thread.start()

                            def check_extract_embeddings_thread():
                                if extract_embeddings_thread.is_alive():
                                    self.root.after(50, check_extract_embeddings_thread)
                                else:
                                    embedding = embedding_queue.get_nowait()
                                    logging.debug(f"embedding: {embedding}")
                                    passengerId = str(uuid.uuid4())
                                    self.passengerId = passengerId
                                    conn, cursor = connect_db()
                                    if not conn:
                                        print("Không thể kết nối để thực hiện INSERT")
                                        return
                                    try:
                                        cursor.execute("INSERT INTO PASSENGERS (PassengerId, FaceData) VALUES (?, ?)", (passengerId, embedding.tobytes()))
                                        conn.commit()
                                        logging.debug(f"Đã thêm passengerId: {passengerId}")
                                        self.root.after(0, lambda: self.status_label.configure(text=f"Chào mừng hành khách"))
                                        self.show_ticket_options()
                                    except Exception as e:
                                        logging.error(f"Lỗi khi thực hiện INSERT: {e}")
                                    finally:
                                        close_db(conn, cursor)

                            self.root.after(100, check_extract_embeddings_thread)
                        except queue.Empty:
                            logging.debug("processed_img_queue chưa sẵn sàng, tiếp tục chờ")
                            self.root.after(100, check_thread)
                except queue.Empty:
                    logging.debug("result_queue chưa sẵn sàng, tiếp tục chờ")
                    self.root.after(100, check_thread)

        self.root.after(100, check_thread)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()