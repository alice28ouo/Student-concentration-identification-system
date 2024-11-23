import cv2
import dlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import pickle
import os
import time

# 設置 GPU 記憶體增長
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class IntegratedRecognizer:
    def __init__(self, face_model_path, emotion_model_path, known_faces_path, camera_index=0, frame_skip=2):
        # 初始化人臉辨識相關組件
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(face_model_path)
        self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        self.known_faces = self.load_known_faces(known_faces_path)

        # 初始化情緒辨識相關組件
        self.emotion_model = load_model(emotion_model_path, compile=False)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        self.camera_index = camera_index
        self.running = False
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.face_locations = []
        self.face_names = []
        self.face_emotions = []

    def load_known_faces(self, filename):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        return []

    def preprocess_face(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
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
    
    def detect_faces(self, image):
        faces = []
        for scale in [1, 0.5]:
            scaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            detected = self.face_detector(scaled_image, 0)
            for face in detected:
                left = int(face.left() / scale)
                top = int(face.top() / scale)
                right = int(face.right() / scale)
                bottom = int(face.bottom() / scale)
                faces.append((left, top, right, bottom))
        return self.non_max_suppression(faces)
    
    def non_max_suppression(self, boxes, overlap_thresh=0.3):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlap_thresh)[0])))

        return [dlib.rectangle(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]) for i in pick]
    
    @tf.function
    def predict_emotion(self, face):
        return self.emotion_model(face)

    def detect_and_recognize(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("無法獲取影像")
                break

            self.frame_count += 1

            if self.frame_count % self.frame_skip == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_locations = self.detect_faces(rgb_frame)
                self.face_locations = [(face.top(), face.right(), face.bottom(), face.left()) for face in face_locations]
                
                self.face_names = []
                self.face_emotions = []

                for face in face_locations:
                    name, _ = self.recognize_face(rgb_frame, face)
                    self.face_names.append(name)

                    top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()
                    face_roi = frame[top:bottom, left:right]
                    processed_face = self.preprocess_face(face_roi)
                    
                    try:
                        emotion_prediction = self.predict_emotion(processed_face)
                        emotion_text = self.emotion_labels[np.argmax(emotion_prediction)]
                        self.face_emotions.append(emotion_text)
                    except Exception as e:
                        print(f"預測過程中發生錯誤：{e}")
                        self.face_emotions.append("Unknown")

            for (top, right, bottom, left), name, emotion in zip(self.face_locations, self.face_names, self.face_emotions):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (left, top - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                cv2.putText(frame, f"{emotion}", (left, top - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

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
    
    recognizer = IntegratedRecognizer(face_model_path, emotion_model_path, known_faces_path, frame_skip=2)
    recognizer.start()

    input("按 Q 鍵停止程式")
    recognizer.stop()