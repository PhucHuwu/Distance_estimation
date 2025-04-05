import math
import cv2
import mediapipe as mp
import numpy as np
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

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.min_track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img: np.ndarray,
                   draw: bool = True, flip_type: bool = True) -> Tuple[
            List[Dict], np.ndarray, float, float, float, float, float]:
        img_resize = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        all_hands = []
        pixel_distance_horizontal = 0.0
        pixel_distance_vertical = 0.0

        if results.multi_hand_landmarks:
            for hand_type, hand_lms in zip(results.multi_handedness, results.multi_hand_landmarks):
                my_hand = {}
                my_lm_list = []
                x_list, y_list = [], []
                h, w, _ = img.shape

                for id, lm in enumerate(hand_lms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    my_lm_list.append([px, py, pz])
                    x_list.append(px)
                    y_list.append(py)

                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                my_hand["lmList"] = my_lm_list
                my_hand["bbox"] = bbox
                my_hand["center"] = (cx, cy)
                my_hand["type"] = "Left" if hand_type.classification[0].label == "Right" else "Right"
                all_hands.append(my_hand)

                if len(my_lm_list) >= 21:
                    x1, y1, _ = my_lm_list[5]
                    x2, y2, _ = my_lm_list[17]
                    x3, y3, _ = my_lm_list[9]
                    x4, y4, _ = my_lm_list[0]

                    pixel_distance_horizontal = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # Euclidean distance
                    pixel_distance_vertical = math.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)  # Euclidean distance

                if draw:
                    # self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    # cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                    #               (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 255, 255), 2)
                    # cv2.putText(img, my_hand["type"], (bbox[0] - 30, bbox[1] - 30),
                    #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.putText(img, str(pixel_distance_horizontal), (bbox[0] + 70, bbox[1] - 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    cv2.putText(img, str(pixel_distance_vertical), (bbox[0] + 100, bbox[1] - 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return img


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        return

    hand_detector = HandDetector()

    while True:
        success, image = cap.read()
        if not success:
            break

        image = hand_detector.find_hands(image)

        cv2.imshow('Hand', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
