import cv2
import dlib
import numpy as np
import pickle
import os
from imgaug import augmenters as iaa

# 初始化 dlib 的人臉檢測器和特徵提取器
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def capture_and_augment_faces(name, num_samples=50, augmentations_per_sample=5):
    cap = cv2.VideoCapture(0)
    face_descriptors = []
    samples_captured = 0

    # 定義數據增強序列
    aug_seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-20, 20), scale=(0.8, 1.2)))
    ])

    while samples_captured < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("無法獲取影像")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            shape = shape_predictor(gray, face)
            face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)
            face_descriptors.append(np.array(face_descriptor))

            # 數據增強
            for _ in range(augmentations_per_sample):
                aug_frame = aug_seq.augment_image(frame)
                aug_shape = shape_predictor(aug_frame, face)
                aug_descriptor = face_recognizer.compute_face_descriptor(aug_frame, aug_shape)
                face_descriptors.append(np.array(aug_descriptor))

            samples_captured += 1
            print(f"已捕獲樣本 {samples_captured}/{num_samples}")

        cv2.imshow('捕獲人臉', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return face_descriptors

def save_face_data(name, face_descriptors, filename="improved_known_faces.pkl"):
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
    face_descriptors = capture_and_augment_faces(name)
    
    if face_descriptors:
        save_face_data(name, face_descriptors)
        print(f"已成功學習並保存 {name} 的 {len(face_descriptors)} 個人臉樣本（包括增強樣本）")
    else:
        print("未能捕獲到人臉樣本")

if __name__ == "__main__":
    main()