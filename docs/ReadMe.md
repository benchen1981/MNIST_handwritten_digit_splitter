# MNIST 手寫數字切割 + CNN + Grad-CAM ‧ CRISP-DM 旗艦教學版

這是一個以 **CRISP-DM 流程** 為核心設計的 Streamlit 專案，用於：

- 處理類似 MNIST 的手寫數字影像資料
- 自動偵測並切割大圖中的多顆手寫數字
- 將每顆數字標準化為 28×28 MNIST 風格
- 依照檔名推斷標籤並分到 `output/0 ~ output/9` 資料夾
- 即時顯示統計圖表與資料特性
- 使用 CNN 進行分類訓練與預測
- 利用 Grad-CAM 顯示模型「關注」區域
- 一鍵生成 PDF 報告（使用 ReportLab）

## 主要功能

1. **Data Understanding**
   - 上傳影像、預覽影像與灰階分佈直方圖

2. **Data Preparation**
   - 智慧切割：透過輪廓偵測自動找出每顆數字
   - 將數字標準化為 28×28 並分標籤資料夾
   - 精細預覽：Bounding Boxes、強化視覺、投影曲線

3. **Modeling**
   - 使用小型 CNN（類 LeNet）快速訓練
   - 適合教學、實驗 Demo

4. **Evaluation**
   - 單顆數字的預測結果（Top-1 / Top-3 機率）
   - Grad-CAM 熱度圖疊加顯示模型關注區域
   - 類別樣本數長條圖

5. **Deployment**
   - 一鍵產生 CRISP-DM PDF 報告
   - 提供 Replit 部署設定
   - 提供 Kaggle 資料集下載腳本

## 安裝

```bash
pip install -r requirements.txt
```

> ⚠ 注意：TensorFlow 在某些環境（例如 Replit 免費版）可能較重，請視情況調整。
> 若環境無法安裝 TensorFlow，可暫時註解掉 CNN 相關功能。

## 執行

```bash
streamlit run app.py
```

## Kaggle 資料集下載（選用）

若要透過 Kaggle API 下載 dataset，請先設定好 `~/.kaggle/kaggle.json`，
之後可執行：

```bash
python kaggle_downloader.py
```

下載完成後，請依實際資料檔名，將影像放入 `upload/` 或整合到你的前處理流程中。
