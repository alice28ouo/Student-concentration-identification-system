import cv2
import dlib
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading

class IntegratedRecognizer:
    def __init__(self, face_model_path, emotion_model_path, known_faces_path, camera_index):
        # 初始化人臉辨識相關組件
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_detector_options = dlib.detector_options()
        self.face_detector_options.upsampling_num_times = 2  # 增加這個值可以檢測更遠的人臉，但會降低速度
        self.shape_predictor = dlib.shape_predictor(face_model_path)
        self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        self.known_faces = self.load_known_faces(known_faces_path)

        # 初始化情緒辨識相關組件
        self.emotion_model = load_model(emotion_model_path, compile=False)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.frame_buffer = []

        self.camera_index = camera_index
        self.running = False

    def load_known_faces(self, filename):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        return []

    def recognize_face(self, frame, face):
        shape = self.shape_predictor(frame, face)
        face_descriptor = self.face_recognizer.compute_face_descriptor(frame, shape)
        face_descriptor = np.array(face_descriptor)

        min_distance = float('inf')
        recognized_name = "未知"
        for known_face in self.known_faces:
            for known_descriptor in known_face["descriptors"]:
                distance = np.linalg.norm(face_descriptor - known_descriptor)
                if distance < min_distance:
                    min_distance = distance
                    recognized_name = known_face["name"]

        if min_distance > 0.4:
            recognized_name = "未知"

        return recognized_name, min_distance
    
    def preprocess_face(self, face):
        if face.size == 0:
            print("警告：檢測到空的人臉區域")
            return None
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = face.astype("float") / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)
            return face
        except Exception as e:
            print(f"預處理人臉時發生錯誤：{e}")
            return None

    def enhance_image(self, image):
        # 圖像增強
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced

    def detect_and_recognize(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 設置更高的解析度
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        while self.running:
            ret, original_frame = cap.read()
            if not ret:
                print("無法獲取影像")
                break

            detected_faces = []
            for scale in [1.0, 0.75, 0.5, 0.25, 0.1]:  # 增加更多縮放級別
                frame = cv2.resize(original_frame, (0, 0), fx=scale, fy=scale)
                enhanced_frame = self.enhance_image(frame)
                rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                faces = self.face_detector(rgb_frame, 0, self.face_detector_options)
                
                if faces:
                    detected_faces = [(face, scale) for face in faces]
                    break  # 如果檢測到人臉，就停止縮放

            for face, scale in detected_faces:
                # 調整人臉座標以匹配原始圖像大小
                left = int(face.left() / scale)
                top = int(face.top() / scale)
                right = int(face.right() / scale)
                bottom = int(face.bottom() / scale)

                # 確保座標不超出原始圖像範圍
                left = max(0, left)
                top = max(0, top)
                right = min(original_frame.shape[1], right)
                bottom = min(original_frame.shape[0], bottom)

                # 人臉辨識
                name, distance = self.recognize_face(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB), 
                                                     dlib.rectangle(left, top, right, bottom))

                # 情緒辨識
                face_roi = original_frame[top:bottom, left:right]
                processed_face = self.preprocess_face(face_roi)
                
                if processed_face is not None:
                    try:
                        emotion_prediction = self.emotion_model.predict(processed_face)
                        self.frame_buffer.append(emotion_prediction)
                        if len(self.frame_buffer) > 5:
                            self.frame_buffer.pop(0)
                        
                        final_prediction = np.mean(self.frame_buffer, axis=0)
                        emotion_probability = np.max(final_prediction)
                        emotion_label_arg = np.argmax(final_prediction)
                        emotion_text = self.emotion_labels[emotion_label_arg]

                        # 在影像上繪製結果
                        cv2.rectangle(original_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(original_frame, f"{name} ({distance:.2f})", (left, top - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(original_frame, f"{emotion_text} ({emotion_probability:.2f})", (left, top - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"預測過程中發生錯誤：{e}")

            cv2.imshow('整合辨識', original_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def start(self):
        threading.Thread(target=self.detect_and_recognize).start()

    def stop(self):
        self.running = False

if __name__ == "__main__":
    face_model_path = "shape_predictor_68_face_landmarks.dat"
    emotion_model_path = "model.h5"
    known_faces_path = "known_faces.pkl"
    #recognizer = IntegratedRecognizer(face_model_path, emotion_model_path, known_faces_path)
    camera_index = 0  # 嘗試 0, 1, 2 等不同的值
    recognizer = IntegratedRecognizer(face_model_path, emotion_model_path, known_faces_path, camera_index)
    recognizer.start()

    input("按 Q 鍵停止程式")
    recognizer.stop()