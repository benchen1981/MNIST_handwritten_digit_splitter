# Project Structure

```text
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
```

此專案可視為一個「MNIST 教學 / 研究系統」的核心骨架，
可再依需要擴充更多模型、實驗與視覺化功能。
