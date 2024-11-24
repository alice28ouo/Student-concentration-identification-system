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
    for i, face_data in enumerate(data):
        name = f"Face {i+1}"
        if isinstance(face_data, np.ndarray):
            analysis_results[name] = {
                "描述符數量": len(face_data),
                "描述符維度": face_data.shape[1] if len(face_data.shape) > 1 else 1,
                "平均值": np.mean(face_data),
                "標準差": np.std(face_data),
                "最小值": np.min(face_data),
                "最大值": np.max(face_data)
            }
        else:
            st.write(f"Face {i+1}: 數據不是 numpy array")
            analysis_results[name] = {
                "描述符數量": "未知",
                "描述符維度": "未知",
                "平均值": "未知",
                "標準差": "未知",
                "最小值": "未知",
                "最大值": "未知"
            }
    return analysis_results

def plot_descriptors(data, analysis_results):
    st.subheader("臉部特徵描述符分析圖表")
    
    # 條形圖
    fig, ax = plt.subplots(figsize=(12, 6))
    df = pd.DataFrame(analysis_results).T
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if not numeric_columns.empty:
        df[numeric_columns].plot(kind="bar", ax=ax)
        plt.title("臉部特徵描述符統計", fontsize=16)
        plt.ylabel("值", fontsize=12)
        plt.xlabel("臉部", fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    else:
        st.write("無法生成條形圖：數據不包含數值型列")
    
    # 熱力圖
    st.subheader("描述符熱力圖")
    for i, face_data in enumerate(data):
        if isinstance(face_data, np.ndarray) and len(face_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.heatmap(face_data[:10].reshape(1, -1), cmap="viridis", ax=ax)
            plt.title(f"Face {i+1} 的前10個描述符", fontsize=16)
            st.pyplot(fig)
        else:
            st.write(f"Face {i+1}: 無法生成熱力圖 (描述符數據不可用或格式不正確)")

def main():

    st.set_page_config(page_title="臉部特徵描述符分析", page_icon="👤", layout="wide")
    st.title("臉部特徵描述符分析")
    
    file_path = "merged_6faces.pkl"
    data = load_pkl_file(file_path)
    
    data = load_pkl_file(file_path)
    st.write("## 原始數據檢查")
    for i, face_data in enumerate(data):
        st.write(f"Face {i+1} 類型:", type(face_data))
        st.write(f"Face {i+1} 數據:", face_data)
    
    # 打印數據結構信息
    st.write("## 數據結構信息")
    st.write("數據類型:", type(data))
    st.write("數據長度:", len(data))
    if len(data) > 0:
        st.write("第一個元素類型:", type(data[0]))
        if isinstance(data[0], np.ndarray):
            st.write("第一個元素形狀:", data[0].shape)
    
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