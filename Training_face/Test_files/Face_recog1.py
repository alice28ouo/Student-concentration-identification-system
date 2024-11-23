import cv2
import dlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import pickle
import os

#這版本的距離可以到2.4公尺~2.8公尺 但2.4之後偶爾人名會辨識不出來

class IntegratedRecognizer:
    def __init__(self, face_model_path, emotion_model_path, known_faces_path, camera_index=0):
        # 初始化人臉辨識相關組件
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(face_model_path)
        self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        self.known_faces = self.load_known_faces(known_faces_path)

        # 初始化情緒辨識相關組件
        self.emotion_model = load_model(emotion_model_path, compile=False)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.camera_index = camera_index
        self.running = False
        self.frame_buffer = []

    def load_known_faces(self, filename):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        return []

    def preprocess_face(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        return face

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

    def detect_and_recognize(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("無法獲取影像")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 使用 dlib 檢測人臉
            dlib_faces = self.face_detector(rgb_frame)

            # 如果 dlib 沒有檢測到人臉，使用 OpenCV 的 Haar 級聯分類器
            if len(dlib_faces) == 0:
                cv_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                if len(cv_faces) == 0:
                    enhanced_gray = cv2.equalizeHist(gray)
                    cv_faces = self.face_cascade.detectMultiScale(enhanced_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            else:
                cv_faces = [(face.left(), face.top(), face.width(), face.height()) for face in dlib_faces]

            for (x, y, w, h) in cv_faces:
                dlib_rect = dlib.rectangle(x, y, x+w, y+h)
                
                # 人臉辨識
                name, distance = self.recognize_face(rgb_frame, dlib_rect)

                # 情緒辨識
                face_roi = frame[y:y+h, x:x+w]
                processed_face = self.preprocess_face(face_roi)
                
                try:
                    emotion_prediction = self.emotion_model.predict(processed_face)
                    
                    # 將結果添加到緩衝區
                    self.frame_buffer.append(emotion_prediction)
                    if len(self.frame_buffer) > 5:  # 保持最近 5 幀的結果
                        self.frame_buffer.pop(0)
                    
                    # 使用多幀結果進行最終預測
                    final_prediction = np.mean(self.frame_buffer, axis=0)
                    emotion_probability = np.max(final_prediction)
                    emotion_label_arg = np.argmax(final_prediction)
                    emotion_text = self.emotion_labels[emotion_label_arg]

                    # 在影像上繪製結果
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({distance:.2f})", (x, y-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, f"{emotion_text} ({emotion_probability:.2f})", 
                                (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"預測過程中發生錯誤：{e}")

            cv2.imshow('Face Recognition', frame)

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
    
    recognizer = IntegratedRecognizer(face_model_path, emotion_model_path, known_faces_path)
    recognizer.start()

    input("按 Q 鍵停止程式")
    recognizer.stop()