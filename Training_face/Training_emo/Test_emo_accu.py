import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 載入模型
model = load_model('model.h5')

# 假設你已經有了預處理好的測試數據
# test_images 應該是一個 numpy 數組，形狀為 (n_samples, 48, 48, 1)
# true_labels 應該是一個包含真實標籤的數組

# 進行預測
predictions = model.predict(test_images)

# 將預測結果轉換為類別標籤
predicted_labels = np.argmax(predictions, axis=1)

# 計算準確率
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"準確率: {accuracy:.2f}")

# 輸出詳細的分類報告
print(classification_report(true_labels, predicted_labels))

# 輸出混淆矩陣
print(confusion_matrix(true_labels, predicted_labels))