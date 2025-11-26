# Replit 部署說明

> 注意：TensorFlow 在 Replit 上可能需要較高資源，若遇到安裝問題，
> 可在 `requirements.txt` 中暫時移除 `tensorflow`，只保留影像切割 + 報告等功能。

1. 建立新的 Replit 專案（Python 或自訂 Nix 環境）。
2. 將本專案所有檔案上傳到 Replit。
3. 確認 `replit.nix` 與 `.replit` 存在且設定正確。
4. 在 Shell 中執行：

```bash
pip install -r requirements.txt
```

5. 在 Replit 的 Run 按鈕設定中，確保執行指令為：

```bash
streamlit run app.py --server.port=8000 --server.address=0.0.0.0
```

6. 按 Run 後，即可透過 Replit 提供的 Web URL 存取此應用。

若 Replit 無法安裝 TensorFlow，可：

- 移除 `requirements.txt` 內的 `tensorflow`
- 在 `app.py` 中註解掉 `models` 相關 import 與使用處
- 保留資料切割、精細預覽與 PDF 報告等功能
