import cv2
import dlib
import numpy as np
import pickle
import os

# 初始化 dlib 的人臉檢測器和特徵提取器
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# 載入已知人臉數據
def load_known_faces(filename="merged_6faces.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return []

known_faces = load_known_faces()

def recognize_face(frame, face):
    # 獲取人臉特徵點
    shape = shape_predictor(frame, face)
    
    # 提取人臉特徵
    face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)
    face_descriptor = np.array(face_descriptor)

    # 比較與已知人臉的相似度
    min_distance = float('inf')
    recognized_name = "未知"
    for known_face in known_faces:
        for known_descriptor in known_face["descriptors"]:
            distance = np.linalg.norm(face_descriptor - known_descriptor)
            if distance < min_distance:
                min_distance = distance
                recognized_name = known_face["name"]

    # 設定閾值，如果相似度太低，則視為未知人臉
    if min_distance > 0.4:
        recognized_name = "未知"

    return recognized_name, min_distance

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法獲取影像")
            break

        # 將 BGR 圖像轉換為 RGB 圖像
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 檢測人臉
        faces = face_detector(rgb_frame)

        for face in faces:
            # 辨識人臉
            name, distance = recognize_face(rgb_frame, face)

            # 在影像上繪製矩形和名字
            left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({distance:.2f})", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()