import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_model(model_path):
    try:
        # 打印 TensorFlow 版本
        print(f"TensorFlow version: {tf.__version__}")
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 尝试加载模型
        print("Loading model...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        
        # 获取模型结构
        model.summary()
        
        # 获取每层的权重
        weights = []
        layer_names = []
        for layer in model.layers:
            if len(layer.get_weights()) > 0:
                weights.append(np.concatenate([w.flatten() for w in layer.get_weights()]))
                layer_names.append(layer.name)
        
        # 绘制权重分布图
        plt.figure(figsize=(12, 6))
        plt.title("Weight Distribution Across Layers")
        plt.boxplot(weights, labels=layer_names)
        plt.ylabel("Weight Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("weight_distribution.png")
        plt.close()
        
        # 绘制模型结构
        tf.keras.utils.plot_model(model, to_file="model_structure.png", show_shapes=True, show_layer_names=True)
        
        print("分析完成。权重分布图保存为 'weight_distribution.png'，模型结构图保存为 'model_structure.png'。")
    
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print("错误详情:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model_path = "model.h5"  # 请确保这个路径是正确的
    analyze_model(model_path)