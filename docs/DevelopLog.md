# DevelopLog

## v0.2.0
- 加入 `models/` 模組：
  - `cnn_digit_classifier.py`：小型 CNN 訓練與預測
  - `gradcam_utils.py`：Grad-CAM 熱度圖產生
- 新增 `splitter/preview_utils.py` 精細預覽工具
- app.py：
  - Data Preparation 加入互動精細預覽
  - Modeling 加入 CNN 訓練
  - Evaluation 加入 CNN 預測 + Grad-CAM
- 文件更新：新增 TensorFlow 相關說明

## v0.1.0
- 建立專案骨架與模組化結構
- 新增 `splitter/` 模組：
  - `file_utils.py`
  - `save_output.py`
  - `big_image_split.py`
  - `classify_single.py`
  - `stats.py`
- 新增 `report/` 模組：
  - `report_generator.py`，用 ReportLab 產 PDF
- 完成 `app.py` CRISP-DM 分頁與互動介面
- 撰寫 Replit 部署設定與文件
