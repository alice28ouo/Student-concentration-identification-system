import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 設置中文字體
def set_chinese_font():
    # 嘗試設置不同的中文字體，直到成功
    chinese_fonts = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'STFangsong']
    for font in chinese_fonts:
        try:
            test_font = FontProperties(fname=mpl.font_manager.findfont(font))
            mpl.rcParams['font.family'] = test_font.get_name()
            print(f"成功設置字體: {font}")
            return
        except:
            continue
    print("警告：無法設置中文字體，可能會導致中文顯示異常")

set_chinese_font()
plt.rcParams['axes.unicode_minus'] = False  # 用來正常顯示負號

def load_and_process_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    processed_data = {}
    for person in data:
        name = person['name']
        descriptors = np.array(person['descriptors'])
        processed_data[name] = descriptors
    
    return processed_data

def visualize_descriptors(data):
    plt.figure(figsize=(12, 8))
    
    # 使用 PCA 將 128 維降到 2 維
    all_descriptors = np.vstack([desc.mean(axis=0) for desc in data.values()])
    pca = PCA(n_components=2)
    reduced_descriptors = pca.fit_transform(all_descriptors)
    
    # 繪製每個人的平均描述符
    for i, (name, _) in enumerate(data.items()):
        plt.scatter(reduced_descriptors[i, 0], reduced_descriptors[i, 1], label=name)
        plt.annotate(name, (reduced_descriptors[i, 0], reduced_descriptors[i, 1]))
    
    plt.title('臉部特徵描述符分布 (PCA降維後)')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('face_descriptors_visualization.png')
    plt.show()

def analyze_descriptors(data):
    print("臉部特徵描述符分析:")
    for name, descriptors in data.items():
        print(f"\n{name}:")
        print(f"  - 描述符數量: {len(descriptors)}")
        print(f"  - 每個描述符的維度: {descriptors[0].shape}")
        print(f"  - 平均值: {descriptors.mean():.4f}")
        print(f"  - 標準差: {descriptors.std():.4f}")
        print(f"  - 最小值: {descriptors.min():.4f}")
        print(f"  - 最大值: {descriptors.max():.4f}")

# 主程序
file_path = 'merged_6faces.pkl'
data = load_and_process_data(file_path)

analyze_descriptors(data)
visualize_descriptors(data)

print("\n分析完成。可視化圖表已保存為 'face_descriptors_visualization.png'")