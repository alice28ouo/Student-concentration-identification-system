import pickle
import os

def load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def merge_pkl_files(input_files, output_file):
    merged_data = []
    for file in input_files:
        data = load_pkl(file)
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)
    
    save_pkl(merged_data, output_file)
    print(f"Merged data saved to {output_file}")

# 使用示例
input_files = ['shinyeh.pkl', 'shin.pkl', 'song.pkl', 'zekai.pkl', 'yixuan.pkl', 'Andrea.pkl']
output_file = 'merged_6faces.pkl'

merge_pkl_files(input_files, output_file)