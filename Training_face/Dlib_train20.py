import cv2
import dlib
import numpy as np
import os
import pickle


print("當前工作目錄:", os.getcwd())
print("目錄內容:", os.listdir())

# 初始化 dlib 的人臉檢測器和特徵提取器
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def capture_and_learn_face(name, num_samples=20, sample_interval=30):
    cap = cv2.VideoCapture(0)
    face_descriptors = []
    frame_count = 0
    samples_captured = 0

    print(f"開始自動學習 {name} 的人臉。請面對攝像頭，並緩慢轉動頭部。")
    print(f"程式將捕獲 {num_samples} 個樣本。")

    while samples_captured < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("無法獲取影像")
            break

        frame_count += 1
        if frame_count % sample_interval != 0:
            continue

        # 轉換為灰度圖像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 檢測人臉
        faces = face_detector(gray)

        for face in faces:
            # 獲取人臉特徵點
            shape = shape_predictor(gray, face)
            
            # 提取人臉特徵
            face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)
            
            # 將特徵轉換為 numpy 數組並添加到列表中
            face_descriptors.append(np.array(face_descriptor))

            samples_captured += 1
            print(f"已捕獲樣本 {samples_captured}/{num_samples}")

            # 在影像上繪製矩形
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('自動學習人臉', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return face_descriptors

def save_face_data(name, face_descriptors, filename="known_faces.pkl"):
    data = {
        "name": name,
        "descriptors": face_descriptors
    }
    
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            known_faces = pickle.load(f)
    else:
        known_faces = []

    known_faces.append(data)

    with open(filename, "wb") as f:
        pickle.dump(known_faces, f)

def main():
    name = input("請輸入您的名字：")
    face_descriptors = capture_and_learn_face(name)
    
    if face_descriptors:
        save_face_data(name, face_descriptors)
        print(f"已成功學習並保存 {name} 的 {len(face_descriptors)} 個人臉樣本")
    else:
        print("未能捕獲到人臉樣本")

if __name__ == "__main__":
    main()