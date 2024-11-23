import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 載入模型
#model = load_model('emotion_model.h5')
model_path = "C:/Users/User/Desktop/VS code/final(face with emotion)/Training_emo/model.h5"
model = load_model(model_path, compile=False)

# 定義情緒標籤（確保順序與訓練時一致）
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 載入人臉檢測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face):
    # 將彩色圖像轉換為灰度圖像
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # 預處理人臉圖像
    face = cv2.resize(face, (48, 48))
    face = face.astype("float") / 255.0
    face = np.expand_dims(face, axis=-1)  # 添加通道維度
    face = np.expand_dims(face, axis=0)   # 添加批次維度
    return face

def detect_emotion():
    cap = cv2.VideoCapture(0)  # 使用默認攝像頭

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 轉換為灰度圖像進行人臉檢測
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 檢測人臉
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # 提取人臉區域
            face_roi = frame[y:y+h, x:x+w]
            
            # 預處理人臉
            processed_face = preprocess_face(face_roi)
            
            # 預測情緒
            emotion_prediction = model.predict(processed_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]

            # 在框架上繪製結果
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion_text} ({emotion_probability:.2f})", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 顯示結果
        cv2.imshow('Emotion Recognition', frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion()