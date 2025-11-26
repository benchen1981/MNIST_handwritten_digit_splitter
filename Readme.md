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
