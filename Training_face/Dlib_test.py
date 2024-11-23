import cv2
import dlib
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 初始化 dlib 的人臉檢測器和特徵提取器
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def load_known_faces(filename="known_faces.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return []

def prepare_data(known_faces):
    X = []
    y = []
    for face_data in known_faces:
        X.extend(face_data["descriptors"])
        y.extend([face_data["name"]] * len(face_data["descriptors"]))
    return np.array(X), np.array(y)

def recognize_face(face_descriptor, known_faces):
    min_distance = float('inf')
    recognized_name = "未知"
    for known_face in known_faces:
        for known_descriptor in known_face["descriptors"]:
            distance = np.linalg.norm(face_descriptor - known_descriptor)
            if distance < min_distance:
                min_distance = distance
                recognized_name = known_face["name"]
    
    if min_distance > 0.6:
        recognized_name = "未知"
    
    return recognized_name

def evaluate_model(X_test, y_test, known_faces):
    y_pred = []
    for face_descriptor in X_test:
        predicted_name = recognize_face(face_descriptor, known_faces)
        y_pred.append(predicted_name)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, precision, recall, f1

def main():
    known_faces = load_known_faces()
    X, y = prepare_data(known_faces)
    
    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用訓練集重新構建 known_faces
    train_known_faces = []
    for name in set(y_train):
        train_known_faces.append({
            "name": name,
            "descriptors": X_train[y_train == name]
        })
    
    # 評估模型
    accuracy, precision, recall, f1 = evaluate_model(X_test, y_test, train_known_faces)
    
    print(f"準確率 (Accuracy): {accuracy:.4f}")
    print(f"精確率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分數: {f1:.4f}")

if __name__ == "__main__":
    main()