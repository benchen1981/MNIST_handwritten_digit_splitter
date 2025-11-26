MNIST資料處理+模型訓練+視覺化系統
手寫數位影像的自動切割與標準化，將每個數字裁切並標準化為28×28
依檔名推斷標籤，分到output/0~9資料夾
提供精細預覽每像素字串框位置(投影、強化、框選)，單顆數字放大 + 對比強化
使用 LeNet 類型的小型(快速) CNN 模型訓練與預測
Grad-CAM熱度圖顯示模型
可將output/中的影像進行即時訓練資料

使用說明
1.安裝套件：
    pip install -r requirements.txt

2.啟動系統：
    streamlit run app.py

    Run 指令保持：
    streamlit run app.py --server.port=8000 --server.address=0.0.0.0

    如果 Replit 撐不住 TensorFlow：
    可以先把 requirements.txt 裡的 tensorflow 刪掉，CNN / Grad-CAM 暫時關掉，保留切割 + 預覽 + PDF 部分。

    # Project Structure

# Project Structure
mnist_crispdm_full/
│── app.py
│── kaggle_downloader.py
│── requirements.txt
│── splitter/
│   ├── __init__.py
│   ├── file_utils.py
│   ├── save_output.py
│   ├── big_image_split.py
│   ├── classify_single.py
│   ├── stats.py
│   └── preview_utils.py
│── models/
│   ├── __init__.py
│   ├── cnn_digit_classifier.py
│   └── gradcam_utils.py
│── report/
│   ├── __init__.py
│   └── report_generator.py
│── docs/
│   ├── ReadMe.md
│   ├── ToDo.md
│   ├── AllDone.md
│   ├── DevelopLog.md
│   ├── ReplitDeploy.md
│   └── ProjectStructure.md
│── upload/           # 使用時自動建立 / 存放上傳檔案
│── output/           # 使用時自動建立 / 切割與分類後影像
│── reports/          # 使用時自動建立 / PDF 報告輸出


