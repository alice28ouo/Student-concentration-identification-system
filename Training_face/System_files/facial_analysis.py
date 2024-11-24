import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他支持中文的字體
plt.rcParams['axes.unicode_minus'] = False  # 用來正確顯示負號

def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_descriptors(data):
    analysis_results = {}
    for face_data in data:
        name = face_data['name']
        descriptors = np.array([np.array(d) for d in face_data['descriptors']])
        analysis_results[name] = {
            "描述符數量": len(descriptors),
            "描述符維度": descriptors.shape[1],
            "平均值": np.mean(descriptors),
            "標準差": np.std(descriptors),
            "最小值": np.min(descriptors),
            "最大值": np.max(descriptors)
        }
    return analysis_results

def plot_descriptors(data, analysis_results):
    st.subheader("臉部特徵描述符分析圖表")
    
    # 條形圖
    fig, ax = plt.subplots(figsize=(12, 6))
    df = pd.DataFrame(analysis_results).T
    df[["平均值", "標準差", "最小值", "最大值"]].plot(kind="bar", ax=ax)
    plt.title("臉部特徵描述符統計", fontsize=16)
    plt.ylabel("值", fontsize=12)
    plt.xlabel("人名", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    
    # 熱力圖
    st.subheader("描述符熱力圖")
    for face_data in data:
        name = face_data['name']
        descriptors = np.array([np.array(d) for d in face_data['descriptors']])
        if len(descriptors) > 0:
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.heatmap(descriptors[:10], cmap="viridis", ax=ax)
            plt.title(f"{name} 的前10個描述符", fontsize=16)
            st.pyplot(fig)
        else:
            st.write(f"{name}: 無法生成熱力圖 (沒有描述符數據)")

def main():
    st.set_page_config(page_title="臉部特徵描述符分析", page_icon="👤", layout="wide")
    st.title("臉部特徵描述符分析")
    
    file_path = "merged_6faces.pkl"
    data = load_pkl_file(file_path)
    
    # 打印數據結構信息
    st.write("## 數據結構信息")
    st.write("數據類型:", type(data))
    st.write("數據長度:", len(data))
    if len(data) > 0:
        st.write("第一個元素類型:", type(data[0]))
        st.write("第一個元素鍵:", list(data[0].keys()))
        st.write("描述符數量:", len(data[0]['descriptors']))
    
    analysis_results = analyze_descriptors(data)
    
    st.write("## 數據分析結果")
    col1, col2 = st.columns(2)
    for i, (name, results) in enumerate(analysis_results.items()):
        if i % 2 == 0:
            with col1:
                st.write(f"### {name}")
                for key, value in results.items():
                    st.write(f"- {key}: {value}")
                st.write("---")
        else:
            with col2:
                st.write(f"### {name}")
                for key, value in results.items():
                    st.write(f"- {key}: {value}")
                st.write("---")
    
    plot_descriptors(data, analysis_results)

if __name__ == "__main__":
    main()