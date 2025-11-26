import os
import zipfile

def download_kaggle_dataset(dataset="vbmokin/mnist-models-testing-handwritten-digits", data_dir="data"):
    """
    使用 Kaggle API 下載並解壓縮資料集。
    需事先在系統環境中設定好 Kaggle Token（~/.kaggle/kaggle.json）。
    """
    os.makedirs(data_dir, exist_ok=True)
    cmd = f"kaggle datasets download -d {dataset} -p {data_dir}"
    print(f"執行：{cmd}")
    os.system(cmd)

    for fname in os.listdir(data_dir):
        if fname.endswith(".zip"):
            zip_path = os.path.join(data_dir, fname)
            print(f"找到 zip：{zip_path}，開始解壓縮...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            print("解壓縮完成。")
            return True

    print("未找到 zip 檔，請確認 Kaggle 下載是否成功。")
    return False

if __name__ == "__main__":
    download_kaggle_dataset()
