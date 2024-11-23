import cv2
import mediapipe as mp
import numpy as np
import time

# 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化攝像頭
cap = cv2.VideoCapture(0)

# 設置參數
EAR_THRESHOLD = 0.2  # 眼睛縱橫比閾值
DROWSY_TIME = 1.0    # 判定為打瞌睡的時間閾值（秒）

# 用於計算 EAR 的眼睛關鍵點索引
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_landmarks):
    # 計算眼睛縱橫比
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

drowsy_start_time = None

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("無法獲取攝像頭畫面")
        continue

    # 將圖像轉換為 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 處理圖像
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        mesh_points = np.array([np.multiply([p.x, p.y], [image.shape[1], image.shape[0]]).astype(int) 
                                for p in results.multi_face_landmarks[0].landmark])
        
        # 計算左右眼的 EAR
        left_ear = calculate_ear(mesh_points[LEFT_EYE])
        right_ear = calculate_ear(mesh_points[RIGHT_EYE])
        
        # 計算平均 EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        # 在畫面上顯示 EAR 值
        cv2.putText(image, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 檢測打瞌睡
        if avg_ear < EAR_THRESHOLD:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()
            elif time.time() - drowsy_start_time > DROWSY_TIME:
                cv2.putText(image, "Doze Alarm！", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # 這裡可以添加其他警告操作，如聲音提醒等
        else:
            drowsy_start_time = None

    # 顯示圖像
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:  # 按 ESC 鍵退出
        break

cap.release()
cv2.destroyAllWindows()