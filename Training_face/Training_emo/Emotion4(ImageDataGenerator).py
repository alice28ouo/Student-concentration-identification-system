import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import kerastuner as kt

def main():
    # 設定情緒目標資料夾路徑
    emotions_dir = 'C:/Users/User/Desktop/archive/images/images/train'

    # 設定預期的情緒類別
    expected_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # 設定圖像大小
    img_size = (48, 48)

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
    def build_model(hp):
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
        
        for layer in base_model.layers:
            layer.trainable = False
        
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(
            units=hp.Int('units', min_value=256, max_value=1024, step=128),
            activation='relu',
            kernel_regularizer=regularizers.l2(hp.Choice('l2', values=[1e-4, 1e-3, 1e-2]))
        )(x)
        x = layers.Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1))(x)
        output = layers.Dense(num_emotions, activation='softmax')(x)
        
        model = models.Model(inputs=base_model.input, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='my_dir',
        project_name='emotion_recognition'
    )

    train_generator = datagen.flow(X_train, y_train, batch_size=128, subset='training')
    validation_generator = datagen.flow(X_train, y_train, batch_size=128, subset='validation')

    tuner.search(train_generator, epochs=50, validation_data=validation_generator)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    最佳超參數:
    - 單元數量: {best_hps.get('units')}
    - L2 正則化: {best_hps.get('l2')}
    - Dropout: {best_hps.get('dropout')}
    - 學習率: {best_hps.get('learning_rate')}
    """)

    # 使用最佳超參數訓練最終模型
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) * 0.8 // 128,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=len(X_train) * 0.2 // 128,
        callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)],
        workers=1,
        use_multiprocessing=False
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

if __name__ == '__main__':
    main()