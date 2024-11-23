import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading

class EmotionRecognizer:
    def __init__(self, model_path, camera_index=0):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.camera_index = camera_index
        self.running = False

    def preprocess_face(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        return face

    def detect_emotion(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("無法獲取影像")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                processed_face = self.preprocess_face(face_roi)
                
                try:
                    emotion_prediction = self.model.predict(processed_face)
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = self.emotion_labels[emotion_label_arg]

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{emotion_text} ({emotion_probability:.2f})", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"預測過程中發生錯誤：{e}")

            cv2.imshow('Emotion Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def start(self):
        threading.Thread(target=self.detect_emotion).start()

    def stop(self):
        self.running = False

if __name__ == "__main__":
    model_path = "C:/Users/User/Desktop/VS code/final(face with emotion)/Training_emo/model.h5"
    recognizer = EmotionRecognizer(model_path)
    recognizer.start()

    input("按 Q 鍵停止程式")
    recognizer.stop()