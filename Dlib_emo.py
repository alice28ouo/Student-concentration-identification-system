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

    def detect_and_recognize(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("無法獲取影像")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_detector(rgb_frame)

            for face in faces:
                # 人臉辨識
                name, distance = self.recognize_face(rgb_frame, face)

                # 確保人臉區域在圖像範圍內
                left = max(0, face.left())
                top = max(0, face.top())
                right = min(frame.shape[1], face.right())
                bottom = min(frame.shape[0], face.bottom())

                # 情緒辨識
                face_roi = frame[top:bottom, left:right]
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
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({distance:.2f})", (left, top - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"{emotion_text} ({emotion_probability:.2f})", (left, top - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"預測過程中發生錯誤：{e}")

            cv2.imshow('整合辨識', frame)

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