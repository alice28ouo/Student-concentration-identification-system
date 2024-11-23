import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import time

# 載入模型
model_path = "C:/Users/User/Desktop/VS code/final(face with emotion)/Training_emo/emotion_cnn_model2.h5"
model = load_model(model_path, compile=False)

# 定義情緒標籤（確保順序與訓練時一致）
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 載入人臉檢測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face.astype("float") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

class EmotionDetector:
    def __init__(self):
        self.frame = None
        self.processed_frame = None
        self.running = False

    def process_frame(self):
        while self.running:
            if self.frame is not None:
                frame = self.frame.copy()
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    x, y, w, h = x*2, y*2, w*2, h*2  # 調整回原始尺寸
                    face_roi = frame[y:y+h, x:x+w]
                    processed_face = preprocess_face(face_roi)
                    
                    emotion_prediction = model.predict(processed_face)
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = emotion_labels[emotion_label_arg]

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{emotion_text} ({emotion_probability:.2f})", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                self.processed_frame = frame
            time.sleep(0.03)  # 限制處理頻率

    def detect_emotion(self):
        self.running = True
        processing_thread = threading.Thread(target=self.process_frame)
        processing_thread.start()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame = frame

            if self.processed_frame is not None:
                cv2.imshow('Emotion Recognition', self.processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.running = False
        processing_thread.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.detect_emotion()