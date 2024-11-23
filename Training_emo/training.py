import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# 設定情緒目標資料夾路徑
emotions_dir = 'C:/Users/User/Desktop/archive/images/images/train'

# 設定預期的情緒類別
expected_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 設定圖像大小
img_size = (64, 64)

# 讀取數據和標籤
X = []
y = []

for emotion_folder in os.listdir(emotions_dir):
    if emotion_folder in expected_emotions:
        emotion_path = os.path.join(emotions_dir, emotion_folder)
        if os.path.isdir(emotion_path):
            print(f"處理 {emotion_folder} 類別的圖片...")
            for file in os.listdir(emotion_path):
                if file.lower().endswith('.jpg'):
                    img_path = os.path.join(emotion_path, file)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(img_size)
                    img_array = np.array(img) / 255.0  # 標準化像素值
                    X.append(img_array)
                    y.append(emotion_folder)
    else:
        print(f"警告：找不到預期的情緒類別 {emotion_folder}")

X = np.array(X)
y = np.array(y)

# 將標籤編碼為數字
le = LabelEncoder()
y = le.fit_transform(y)

# 獲取情緒類別數量
num_emotions = len(le.classes_)
print(f"檢測到的情緒類別數量: {num_emotions}")
print("情緒類別映射:")
for i, emotion in enumerate(le.classes_):
    print(f"{emotion}: {i}")

# 將數據分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 將標籤轉換為one-hot編碼
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_emotions)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_emotions)

# 創建數據增強器
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2  # 用於分割驗證集
)

# 創建自定義CNN模型
def create_model(input_shape, num_classes):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # 凍結基礎模型的層
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=output)
    return model

# 創建模型
model = create_model((64, 64, 3), num_emotions)

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 創建學習率調度器
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# 訓練模型
batch_size = 128
epochs = 50

train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='validation')

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) * 0.8 // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(X_train) * 0.2 // batch_size,
    callbacks=[reduce_lr],
    workers=4,
    use_multiprocessing=True
)

# 評估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'測試準確率: {test_acc}')

# 保存模型
model.save('emotion_cnn_model.h5')

# 輸出每個情緒的預測準確率
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

for i, emotion in enumerate(le.classes_):
    emotion_accuracy = np.mean((y_pred_classes == i) & (y_true_classes == i))
    print(f'{emotion} 的預測準確率: {emotion_accuracy:.2f}')

# 可選：微調模型
print("開始微調模型...")
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) * 0.8 // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(X_train) * 0.2 // batch_size,
    callbacks=[reduce_lr],
    workers=4,
    use_multiprocessing=True
)

# 再次評估微調後的模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'微調後的測試準確率: {test_acc}')

# 保存微調後的模型
model.save('emotion_cnn_model_fine_tuned.h5')

# 輸出微調後每個情緒的預測準確率
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

for i, emotion in enumerate(le.classes_):
    emotion_accuracy = np.mean((y_pred_classes == i) & (y_true_classes == i))
    print(f'微調後 {emotion} 的預測準確率: {emotion_accuracy:.2f}')