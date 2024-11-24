from flask import Flask, render_template, request, jsonify,url_for,redirect,flash,Response
import pyodbc
import openai
import cv2
import numpy as np
import mediapipe as mp
from datetime import date
import os 
import datetime
import json
import time
from keras.models import load_model
import pandas as pd
import pickle
import dlib
import sys
import joblib

focus_model = joblib.load('C:/Users/teohz/uni_project/File/focus_level_model.pkl')


today = date.today()
app = Flask(__name__)
app.secret_key = 'your_secret_key'

#openAi api
openai.api_key = ''

#動作辨識
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#人臉辨識模型
face_cascade = cv2.CascadeClassifier('C:/Users/teohz/uni_project/model/haarcascade_frontalface_default.xml')
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("C:/Users/teohz/uni_project/model/shape_predictor_68_face_landmarks (1).dat")
face_recognizer = dlib.face_recognition_model_v1("C:/Users/teohz/uni_project/model/dlib_face_recognition_resnet_model_v1.dat")



#眼睛偵測
eye_cascPath = 'C:/Users/teohz/uni_project/model/haarcascade_eye_tree_eyeglasses.xml'  # Eye detect model
face_cascPath = 'C:/Users/teohz/uni_project/model/haarcascade_frontalface_alt.xml'  # Face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

#頭部動作出現錯誤
error_start_times = {
    "not_at_seat": None,
    "head_shaking_or_tilting": None,
    "head_moving_to_corner": None,
    "head_turned": None
}

error_durations = {
    "not_at_seat": 0,
    "head_shaking_or_tilting": 0,
    "head_moving_to_corner": 0,
    "head_turned": 0
}

# 定义情绪标签
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_model = load_model("C:/Users/teohz/uni_project/model/model.h5")

#記錄學生上課資料
data_records_list = []

# 載入多個已知人臉數據
def load_all_known_faces(directory='C:/Users/teohz/uni_project/Face_recongnition/dlib/'):
    known_faces = []
    
    # 遍歷目錄中的所有 .pkl 文件
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "rb") as f:
                known_faces.extend(pickle.load(f))
    
    return known_faces

known_faces = load_all_known_faces()

def recognize_face(frame, face):
    # 将 frame 转为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 确保 `face` 是 dlib.rectangle 对象
    if isinstance(face, dlib.rectangle):
        # 获取人脸特征点
        shape = shape_predictor(gray, face)
        
        # 提取人脸特征
        face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)
        face_descriptor = np.array(face_descriptor)

        # 比较与已知人脸的相似度
        min_distance = float('inf')
        recognized_name = "未知"
        for known_face in known_faces:
            for known_descriptor in known_face["descriptors"]:
                distance = np.linalg.norm(face_descriptor - known_descriptor)
                if distance < min_distance:
                    min_distance = distance
                    recognized_name = known_face["name"]

        # 设置阈值，如果相似度太低，则视为未知人脸
        if min_distance > 0.5:
            recognized_name = "未知"

        return recognized_name, min_distance
    else:
        return "未知", float('inf')

#sql資料庫的環境設置
def get_user_data():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=DESKTOP-CGJ6AVJ;'
        'DATABASE=專題;'
        'Trusted_Connection=yes;'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dbo.學生老師賬號資料")
    data = cursor.fetchall()
    conn.close()
    return data

def format_data_for_prompt(data):
    formatted_data = "以下是点名记录:\n"
    for row in data:
        # 假设数据行是 (name, attendance, date)，根据实际情况修改
        name, attendance_date, attendance = row
        formatted_data += f"Name: {name}, Date: {attendance_date}, Attendance: {attendance}\n"
    return formatted_data

#建立資料庫點名連接
def get_attendance_data():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=DESKTOP-CGJ6AVJ;'
        'DATABASE=專題;'
        'Trusted_Connection=yes;'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dbo.點名")
    data = cursor.fetchall()
    conn.close()
    return data

#判斷是否在座位上
def is_at_seat(landmarks, image_height):
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height

    threshold = 300
    left_shoulder_distance = image_height - left_shoulder_y
    right_shoulder_distance = image_height - right_shoulder_y

    seated_condition = (left_shoulder_distance < threshold) and (right_shoulder_distance < threshold)
    
    return seated_condition

def get_head_angle(landmarks, image_width):
    if not landmarks:
        return None

    nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x * image_width
    left_ear_x = landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * image_width
    right_ear_x = landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x * image_width

    head_angle = abs(nose_x - (left_ear_x + right_ear_x) / 2)
    return head_angle

def get_head_position(landmarks, image_width, image_height):
    if not landmarks:
        return None, None

    nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x * image_width
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * image_height

    return nose_x, nose_y

def is_head_shaking_or_tilting(landmarks, image_width):
    if not landmarks:
        return False

    head_angle = get_head_angle(landmarks, image_width)
    return head_angle > 15  # 增加閾值

def is_head_moving_to_corner(start_pos, current_pos, threshold=15):  # 增加閾值
    if start_pos is None:
        return False

    start_x, start_y = start_pos
    current_x, current_y = current_pos

    x_movement = current_x - start_x
    y_movement = current_y - start_y

    moved_to_left_bottom = (x_movement < -threshold) and (y_movement > threshold)
    moved_to_right_bottom = (x_movement > threshold) and (y_movement > threshold)

    return moved_to_left_bottom or moved_to_right_bottom

def is_head_turned(landmarks, image_width, angle_threshold=15):
    """簡單判斷頭部是否轉向側方，並加入平滑處理"""
    if not landmarks:
        return "Center"

    nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x * image_width
    left_ear_x = landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * image_width
    right_ear_x = landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x * image_width

    # 計算頭部偏移量
    head_offset = nose_x - (left_ear_x + right_ear_x) / 2

    if head_offset < -angle_threshold:
        return "Right"
    elif head_offset > angle_threshold:
        return "Left"
    else:
        return "Center"


#人臉辨識相機調用
def generate_frames():

    global stop_video
    stop_video = False
    cap = cv2.VideoCapture(1)  # 使用默认设备摄像头
    cap.set(cv2.CAP_PROP_FPS, 30)
    start_head_pos = None
    HEAD_TURN_CONSECUTIVE_FRAMES = 3
    head_turn_counter = {"Left": 0, "Right": 0, "Center": 0}

    # 初始化变量
    smooth_face = None
    alpha = 0.5  # 平滑系数
    eye_closed_count = 0
    eye_state = 'open'  # 初始眼睛状态
    previously_at_seat = True
    leave_count = 0
    left_turn_count = 0
    right_turn_count = 0
    start_open_time = time.time()
    total_open_time = 0.0  # 初始化为 0.0
    total_closed_time = 0
    attendance = {"zekai": False, "song": False, "shinyeh": False, "Andrea": False, "shin": False, "yixuan":False}

    # 设置情绪检测的时间间隔（秒）
    emotion_detection_interval = 0.5
    last_emotion_detection_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        start_time = time.time()
        face_detection_time = 0

        while True:
            if stop_video:
                break
            ret, frame = cap.read()
            if not ret:
                break
            
            # 调用人脸检测器
            faces = face_detector(frame)
            frame = cv2.resize(frame, (800, 500))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.equalizeHist(img_gray)

            # 姿势检测
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                image_height, image_width = frame.shape[:2]

                current_time = time.time()
                face_detection_time += current_time - start_time
                start_time = current_time

                current_head_pos = get_head_position(landmarks, image_width, image_height)
                
                if start_head_pos is None:
                    start_head_pos = current_head_pos

                # 判断是否在座位上
                if is_at_seat(landmarks, image_height):
                    cv2.putText(frame, "At Seat", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    previously_at_seat = True
                else:
                    if previously_at_seat:
                        leave_count += 1
                        previously_at_seat = False
                    cv2.putText(frame, "Not At Seat", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Head shaking or tilting detection
                if is_head_shaking_or_tilting(landmarks, image_width):
                    cv2.putText(frame, "Head Shaking or Tilting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    if error_start_times["head_shaking_or_tilting"] is None:
                        error_start_times["head_shaking_or_tilting"] = current_time
                else:
                    if error_start_times["head_shaking_or_tilting"] is not None:
                        error_durations["head_shaking_or_tilting"] += current_time - error_start_times[
                            "head_shaking_or_tilting"]
                        error_start_times["head_shaking_or_tilting"] = None

                # Check for head moving to corner
                if is_head_moving_to_corner(start_head_pos, current_head_pos):
                    cv2.putText(frame, "Head Moving to Corner", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    if error_start_times["head_moving_to_corner"] is None:
                        error_start_times["head_moving_to_corner"] = current_time
                else:
                    if error_start_times["head_moving_to_corner"] is not None:
                        error_durations["head_moving_to_corner"] += current_time - error_start_times[
                            "head_moving_to_corner"]
                        error_start_times["head_moving_to_corner"] = None

                # Head turn detection
                head_turn_direction = is_head_turned(landmarks, image_width)

                # Reset other direction counters
                for direction in head_turn_counter:
                    if direction != head_turn_direction:
                        head_turn_counter[direction] = 0
                
                # Increase current direction count
                head_turn_counter[head_turn_direction] += 1

                # Check if consecutive frames threshold is reached
                if head_turn_counter["Left"] >= HEAD_TURN_CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "Head Turned: Left", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    if error_start_times["head_turned"] is None:
                        error_start_times["head_turned"] = current_time

                    left_turn_count += 1  # Increment left turn count

                elif head_turn_counter["Right"] >= HEAD_TURN_CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "Head Turned: Right", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    if error_start_times["head_turned"] is None:
                        error_start_times["head_turned"] = current_time

                    right_turn_count += 1  # Increment right turn count

                else:
                    # When no longer turning, reset the timer and counter
                    if error_start_times["head_turned"] is not None:
                        error_durations["head_turned"] += current_time - error_start_times["head_turned"]
                        error_start_times["head_turned"] = None
            
            # 人脸识别
            faces = face_detector(frame)
            if len(faces) > 0:
                dlib_rect = faces[0]

                # 使用 dlib.rectangle 的方法获取坐标
                x = dlib_rect.left()
                y = dlib_rect.top()
                w = dlib_rect.right() - x
                h = dlib_rect.bottom() - y
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                 # Eye detection
                face_gray = img_gray[y:y+h, x:x+w]
                eyes = eyeCascade.detectMultiScale(face_gray, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30))
                if len(eyes) == 0 and eye_state == 'open':
                    eye_closed_count += 1  # Eye closed count
                    eye_state = 'closed'
                else:
                    eye_state = 'open'

                # 假设 recognize_face 返回名字和距离
                name, distance = recognize_face(frame, dlib_rect)
                 # 提取面部区域并进行情绪预测
                face_region = img_gray[y:y + h, x:x + w]  # 提取灰度人脸区域
                face_resized = cv2.resize(face_region, (48, 48))  # 调整到48x48，模型输入大小
                face_resized = face_resized.astype('float32') / 255  # 归一化
                face_resized = np.expand_dims(face_resized, axis=0)  # 增加批次维度
                face_resized = np.expand_dims(face_resized, axis=-1)  # 增加通道维度（灰度图）
                
                 # 进行情绪预测
                emotion_prediction = emotion_model.predict(face_resized)
                emotion_label_idx = np.argmax(emotion_prediction)
                emotion_label = emotion_labels[emotion_label_idx]

                # 更新出勤状态
                if name in attendance:
                    attendance[name] = True  # 标记为已出席
                
                # 绘制名字和情绪标签
                cv2.putText(frame, f"{name} {emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 返回视频帧
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

    # 最后数据处理
    total_count_head = left_turn_count + right_turn_count
    date_today = datetime.datetime.now().strftime("%Y-%m-%d")
    average_open_time = total_open_time / eye_closed_count if eye_closed_count > 0 else 0.0

    attendance_data = {
        "date": date_today,
        "attendance": attendance
    }

    data_records = {
        "Name":name,
        "eye_gaze_duration":round(average_open_time,  2),#平均睜眼時間
        "emotion_label":emotion_label, #情緒變化
        "total_closed_time":round(total_closed_time, 2), #閉眼時間
        "total_not_at_seat":round(error_durations["not_at_seat"], 2), #離開座位時間
        "total_count_head":total_count_head #頭部轉動次數
    }

    # 将当前学生的数据添加到列表中
    data_records_list.append(data_records)

    save_attendance_to_database(attendance_data)#記錄點名

    #save_data_to_database(attendance_data)#記錄課堂表現

    #將記錄數據轉成csv文件
    df = pd.DataFrame(data_records_list)
    csv_path = f'C:/Users/teohz/uni_project/File/{date_today}.csv'
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8')
    except Exception as e:
        print(f"保存数据到 CSV 失败: {e}")
    

def save_attendance_to_database(attendance_data):
    """保存出勤数据到 SQL 数据库"""
    try:
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=DESKTOP-CGJ6AVJ;'
            'DATABASE=專題;'
            'Trusted_Connection=yes;'
        )
        cursor = conn.cursor()

        for student, present in attendance_data['attendance'].items():
            attendance_status = 'Present' if present else 'Absent'
            cursor.execute("""
                INSERT INTO dbo.點名 (Name, attendance_date, attendance)
                VALUES (?, ?, ?)
            """, (student, attendance_data['date'], attendance_status))
        conn.commit()
        print("Attendance data saved successfully.")
    except Exception as e:
        print(f"Failed to save attendance data to database: {e}")
    finally:
        conn.close()

#分析學生上課狀況並判斷出分數
def focus_level_recognize(model):
    new_data = pd.DataFrame({
    'eye_gaze_duration': [0.2, 2.5],
    'emotion_label': [4, 4],  # 数值化的情绪标签
    'total_closed_time': [0, 1],
    'total_not_at_seat': [4.0, 6.0],
    'total_count_head':[4.0,10]
    })

    predictions = model.predict(new_data)

    # 输出预测结果
    print(predictions)

#------------------------------------------------FLASK----------------------------------------------------#
#html界面各個
@app.route("/")
def first():
    return redirect(url_for('detection'))

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("Shome.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        
        try:
            conn = pyodbc.connect(
                'DRIVER={ODBC Driver 17 for SQL Server};'
                'SERVER=DESKTOP-CGJ6AVJ;'
                'DATABASE=專題;'
                'Trusted_Connection=yes;'
            )
            cursor = conn.cursor()
            cursor.execute("SELECT [Role] FROM dbo.學生老師賬號資料 WHERE ID = ? AND Password = ?", (username, password))
            user = cursor.fetchone()
            conn.close()
            
            if user:
                role = user[0]  # Extract role from the result
                if role == 'student':
                    return redirect(url_for('Shome'))  # Redirect to student home page
                elif role == 'teacher':
                    return redirect(url_for('Thome'))  # Redirect to teacher home page
                else:
                    flash('Unknown role, please contact admin.', 'error')
            else:
                flash('帳號或密碼錯誤', 'error')
        except Exception as e:
            print(f"Error: {e}")  # Print error information to console
            flash('賬號或密碼錯誤，請重新再試', 'error')
    
    return render_template('login.html')  # Render login page on GET request

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm-password")
        role = request.form.get("role")

        if not username or not password or not confirm_password or not role:
            flash("所有字段都需要填写。")
            return redirect(url_for("register"))
        
        if password != confirm_password:
            flash("两次输入的密码不一致。")
            return redirect(url_for("register"))
        
        try:
            conn = pyodbc.connect(
                'DRIVER={ODBC Driver 17 for SQL Server};'
                'SERVER=DESKTOP-CGJ6AVJ;'
                'DATABASE=專題;'
                'Trusted_Connection=yes;'
            )
            cursor = conn.cursor()

            # 检查用户名是否已存在
            cursor.execute("SELECT [ID] FROM dbo.學生老師賬號資料 WHERE ID = ?", (username,))
            existing_user = cursor.fetchone()
            
            if existing_user:
                flash("用户名已存在。")
                return redirect(url_for("register"))

            # 插入新用户数据
            cursor.execute("INSERT INTO dbo.學生老師賬號資料 ([ID], [Password], [Role]) VALUES (?, ?, ?)", 
                           (username, password, role))
            conn.commit()
            conn.close()

            flash("注册成功！")
            return redirect(url_for("login"))

        except Exception as e:
            flash(f"发生错误: {str(e)}")
            return redirect(url_for("register"))
    
    return render_template("register.html")

@app.route("/Thome")
def Thome():
    return render_template("Thome.html")

@app.route("/Shome")
def Shome():
    return render_template("Shome.html")

@app.route("/wait")
def wait():
    return render_template("wait.html")

@app.route("/chatbot")
def chatbot():
    data = get_attendance_data()
    formatted_data = []
    for row in data:
        Name, attendance_date, attendance = row
        formatted_data.append(f"{Name}, {attendance_date}, {attendance}")
    return render_template('chatbot.html', attendance_data=formatted_data)

# NOTE: 这个函数假设输入是整数
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    # 获取数据库中的点名记录
    data = get_attendance_data()
    if not data:
        return jsonify({'reply': '无法获取数据。'})
    
    # 格式化数据库中的数据
    formatted_data = format_data_for_prompt(data)
    
    # 初始机器人消息和指令
    initial_instruction = (
        "我是人工智慧，同時能够处理用户的问题和数据库数据。\n"
        "以下是我可以帮助你做的事情：\n"
        "1. 解析和分析你的点名记录。\n"
        "2. 提供有关学生专注度的反馈。\n"
        "3. 給予這些缺席的學生一些鼓勵。\n"
        "请告诉我你的问题或选择其中一项功能。"
    )
    # 初始机器人消息
    if not user_message:
        bot_reply = initial_instruction

    # 将用户消息和数据库数据一起发送到聊天模型
    else:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": initial_instruction},
                    {"role": "user", "content": 
                        f"以下是我的点名记录：\n{formatted_data}\n\n用户消息: {user_message}"
                    }                
                ],
                max_tokens=150,
                temperature=0.7,
                top_p=0.9
            )
            bot_reply = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            bot_reply = f"请求失败: {e}"

    return jsonify({'reply': bot_reply})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/choose_class')
def choose_class():
    return render_template('choose_class.html')

@app.route('/stop_video', methods=['POST'])
def stop_video_feed():
    global stop_video
    stop_video = True
    return jsonify(success=True)

if __name__ == "__main__":
    app.run()