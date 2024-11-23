import h5py

with h5py.File("model.h5", "r") as f:
    # 列出文件中的所有組
    print(list(f.keys()))
    
    # 遍歷並打印組結構
    def print_structure(name, obj):
        print(name)
        if isinstance(obj, h5py.Dataset):
            print("    Shape:", obj.shape)
            print("    Type:", obj.dtype)
    
    f.visititems(print_structure)