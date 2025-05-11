import math
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class HandDetector(metaclass=SingletonMeta):
    def __init__(self, static_mode: bool = False, max_hands: int = 1,
                 model_complexity: int = 1, detection_con: float = 0.5,
                 min_track_con: float = 0.5):
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.min_track_con = min_track_con

        # Khởi tạo MediaPipe Hands model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,     # False: tối ưu cho video, True: cho ảnh tĩnh
            max_num_hands=self.max_hands,           # Số lượng bàn tay tối đa cần phát hiện
            model_complexity=self.model_complexity,  # Độ phức tạp của mô hình (0-1)
            min_detection_confidence=self.detection_con,  # Ngưỡng tin cậy để phát hiện bàn tay
            min_tracking_confidence=self.min_track_con    # Ngưỡng tin cậy để theo dõi bàn tay
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img: np.ndarray, coeff1: Tuple[float, float, float], coeff2: Tuple[float, float, float],
                   draw: bool = True, flip_type: bool = True) -> Tuple[List[Dict], np.ndarray, float]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB cho MediaPipe
        results = self.hands.process(img_rgb)       # Xử lý ảnh để phát hiện bàn tay
        all_hands = []
        distance1 = 0.0
        distance2 = 0.0
        distance = 0.0
        pixel_distance_horizontal = 0.0
        pixel_distance_vertical = 0.0

        if results.multi_hand_landmarks:
            for hand_type, hand_lms in zip(results.multi_handedness, results.multi_hand_landmarks):
                my_hand = {}
                my_lm_list = []  # Danh sách các điểm mốc bàn tay
                x_list, y_list = [], []  # Danh sách tọa độ x, y của các điểm mốc
                h, w, _ = img.shape

                # Lưu tọa độ của từng điểm mốc
                for id, lm in enumerate(hand_lms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    my_lm_list.append([px, py, pz])
                    x_list.append(px)
                    y_list.append(py)

                # Tính toán hình chữ nhật bao quanh bàn tay
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                my_hand["lmList"] = my_lm_list
                my_hand["bbox"] = bbox
                my_hand["center"] = (cx, cy)
                # Xác định tay trái hay tay phải (đảo ngược do hiệu ứng gương của camera)
                my_hand["type"] = "Left" if hand_type.classification[0].label == "Right" else "Right"
                all_hands.append(my_hand)

                if len(my_lm_list) >= 21:
                    # Lấy tọa độ của các điểm mốc đã chọn để tính khoảng cách
                    x1, y1, _ = my_lm_list[5]   # Điểm gốc ngón cái
                    x2, y2, _ = my_lm_list[17]  # Điểm gốc ngón út
                    x3, y3, _ = my_lm_list[9]   # Điểm giữa lòng bàn tay
                    x4, y4, _ = my_lm_list[0]   # Điểm cổ tay

                    # Tính khoảng cách Euclidean giữa các điểm mốc đã chọn
                    pixel_distance_horizontal = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # Khoảng cách ngang (từ ngón cái đến ngón út)
                    pixel_distance_vertical = math.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)    # Khoảng cách dọc (từ lòng bàn tay đến cổ tay)

                    # Áp dụng hàm hồi quy bậc 2 để chuyển đổi khoảng cách pixel thành khoảng cách thực (cm)
                    a1, b1, c1 = coeff1  # Hệ số hồi quy cho khoảng cách ngang
                    a2, b2, c2 = coeff2  # Hệ số hồi quy cho khoảng cách dọc
                    distance1 = a1 * pixel_distance_horizontal ** 2 + b1 * pixel_distance_horizontal + c1  # Ước lượng khoảng cách theo chiều ngang
                    distance2 = a2 * pixel_distance_vertical ** 2 + b2 * pixel_distance_vertical + c2     # Ước lượng khoảng cách theo chiều dọc

                    # Lấy khoảng cách nhỏ nhất giữa hai kết quả làm kết quả cuối cùng
                    distance = round(min(distance1, distance2), 4)
                    
                if draw:
                    # Vẽ các điểm mốc và đường kết nối trên bàn tay
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    # Vẽ hình chữ nhật bao quanh bàn tay
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 255, 255), 2)
                    # Hiển thị thông tin: loại bàn tay (trái/phải) và khoảng cách ước lượng
                    cv2.putText(img, my_hand["type"], (bbox[0] - 30, bbox[1] - 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.putText(img, str(distance), (bbox[0] + 70, bbox[1] - 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return all_hands, img, distance, distance1, distance2, pixel_distance_horizontal, pixel_distance_vertical


def main(video_path: str = "video.mp4"):
    # Khởi tạo đầu đọc video từ file thay vì camera
    cap = cv2.VideoCapture(video_path)  # Đọc từ file video.mp4 thay vì camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Đặt độ phân giải chiều rộng
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Đặt độ phân giải chiều cao

    if not cap.isOpened():
        print("Lỗi: Không thể mở video.")
        return

    hand_detector = HandDetector()

    # Dữ liệu hiệu chuẩn: khoảng cách pixel và khoảng cách thực tế (cm) tương ứng
    raw_distances_1 = [240, 195, 174, 145, 130, 116, 105, 95, 86, 81, 75, 70, 65, 62, 58, 55, 53, 50, 48, 46]  # Khoảng cách pixel (điểm 5-17)
    cm_distances_1 = [22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 112, 117]     # Khoảng cách thực tế (cm)
    raw_distances_2 = [360, 295, 253, 222, 193, 173, 158, 148, 134, 126, 115, 109, 101, 96, 91, 87, 82, 78, 75, 72]  # Khoảng cách pixel (điểm 9-0)
    cm_distances_2 = [22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 112, 117]     # Khoảng cách thực tế (cm)

    # Tính toán hệ số hồi quy bậc 2 từ dữ liệu hiệu chuẩn
    coeff1 = np.polyfit(raw_distances_1, cm_distances_1, 2)  # Hệ số cho phương pháp đo 1 (điểm 5-17)
    coeff2 = np.polyfit(raw_distances_2, cm_distances_2, 2)  # Hệ số cho phương pháp đo 2 (điểm 9-0)

    # Chuẩn bị biểu đồ để hiển thị mối quan hệ pixel-cm theo thời gian thực
    plt.ion()  # Bật chế độ tương tác cho matplotlib
    fig, ax = plt.subplots()
    ax.set_xlabel('Pixel Distance')
    ax.set_ylabel('Actual Distance (cm)')

    # Lưu trữ dữ liệu để vẽ biểu đồ
    pixel_distances_1 = []
    pixel_distances_2 = []
    actual_distances_1 = []
    actual_distances_2 = []

    while True:
        success, image = cap.read()
        if not success:
            print("Error: Cannot read frame from camera.")
            break

        # Phát hiện bàn tay và tính toán khoảng cách
        hands, image, distance, distance1, distance2, pixel_distance_1, pixel_distance_2 = hand_detector.find_hands(image, coeff1, coeff2)

        # Cập nhật biểu đồ
        if distance > 0:
            pixel_distances_1.append(pixel_distance_1)
            pixel_distances_2.append(pixel_distance_2)
            actual_distances_1.append(distance1)
            actual_distances_2.append(distance2)

        ax.clear()

        # Tính và vẽ đường hồi quy bậc 2 từ dữ liệu đã thu thập
        if len(pixel_distances_1) > 0 and len(pixel_distances_2) > 0:
            x_fit_1 = np.linspace(min(pixel_distances_1), max(pixel_distances_1), 100)
            y_fit_1_cubic = np.polyval(coeff1, x_fit_1)

            x_fit_2 = np.linspace(min(pixel_distances_2), max(pixel_distances_2), 100)
            y_fit_2_cubic = np.polyval(coeff2, x_fit_2)

            # Vẽ dữ liệu và đường hồi quy
            plt.scatter(pixel_distances_1, actual_distances_1, color='blue', label='5 và 17')
            plt.scatter(pixel_distances_2, actual_distances_2, color='purple', label='9 và 0')
            plt.plot(x_fit_1, y_fit_1_cubic, color='blue', label='Fit line ước lượng 1')
            plt.plot(x_fit_2, y_fit_2_cubic, color='purple', label='Fit line ước lượng 2')

            # Vẽ đường dữ liệu hiệu chuẩn ban đầu để so sánh
            plt.plot(raw_distances_1, cm_distances_1, color='skyblue', label='Fit line dữ liệu ban đầu 1')
            plt.plot(raw_distances_2, cm_distances_2, color='red', label='Fit line dữ liệu ban đầu 2')

            ax.legend()
            plt.pause(0.001)  # Tạm dừng để cập nhật biểu đồ

        # Hiển thị hình ảnh từ video với kết quả phát hiện
        cv2.imshow('Camera Feed', image)

        # Nhấn ESC để thoát
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
