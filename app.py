import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from splitter.file_utils import ensure_folder
from splitter.save_output import clear_output
from splitter.big_image_split import auto_split_digits, grid_split_image
from splitter.classify_single import classify_image
from splitter.stats import compute_label_stats
from splitter.preview_utils import draw_bounding_boxes, enhance_digit, gray_profile
from models.cnn_digit_classifier import train_quick_cnn, predict_digit
from models.gradcam_utils import compute_gradcam_overlay
from report.report_generator import generate_crispdm_pdf

# ==== 基本路徑設定 ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "upload")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

ensure_folder(UPLOAD_DIR)
ensure_folder(OUTPUT_DIR)
ensure_folder(REPORT_DIR)

st.set_page_config(
    page_title="MNIST 手寫數字切割 + CNN + Grad-CAM 旗艦教學版",
    layout="wide"
)

st.title("🧩 MNIST 手寫數字切割 + CNN + Grad-CAM 旗艦教學版")
st.caption("CRISP-DM 流程 + 智慧切割 + 精細預覽 + 強化 CNN + Grad-CAM 可視化")

# ==== CRISP-DM 進度條 ====
steps = ["Business Understanding", "Data Understanding", "Data Preparation",
         "Modeling", "Evaluation", "Deployment"]
current_step_index = 3  # 本系統涵蓋到 Modeling / Evaluation / Deployment
progress = int((current_step_index + 1) / len(steps) * 100)
st.progress(progress)
st.write(f"CRISP-DM 目前階段：**{steps[current_step_index]}**（{progress}%）")

tab_bu, tab_du, tab_dp, tab_model, tab_eval, tab_deploy = st.tabs(steps)

# ====== Business Understanding ======
with tab_bu:
    st.header("Business Understanding")
    st.markdown(
        """
        本專案的目標：

        - 建立一套 **完整可教學的 MNIST 資料處理 + 模型訓練 + 可視化系統**
        - 適合：
            - 課堂示範「從原始影像到模型部署」的全流程
            - 研究 / 作業報告中的實驗平台
        - 功能包含：
            - 手寫數字影像的自動切割與標準化（MNIST 風格 28×28）
            - 精細預覽每一顆數字（投影、強化、框選）
            - 快速 CNN 模型訓練與預測
            - Grad-CAM 熱度圖顯示模型「看哪裡」
            - 一鍵產生 CRISP-DM PDF 報告

        你可以把它當成：**專題 / 實驗室教學 / 企業 PoC demo 的核心骨架**。
        """
    )

# ====== Data Understanding ======
with tab_du:
    st.header("Data Understanding")
    st.markdown(
        """
        在這裡你可以：

        - 上傳原始手寫數字影像（單顆或多顆合併的大圖皆可）
        - 觀察：
            - 圖像尺寸
            - 灰階分佈直方圖
        """
    )

    uploaded_files = st.file_uploader(
        "上傳手寫數字影像（可多檔）",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_files:
        ensure_folder(UPLOAD_DIR)
        for f in uploaded_files:
            save_path = os.path.join(UPLOAD_DIR, f.name)
            with open(save_path, "wb") as fp:
                fp.write(f.getbuffer())
        st.success("✔ 檔案已上傳到伺服器 upload/ 目錄")

        for filename in os.listdir(UPLOAD_DIR):
            path = os.path.join(UPLOAD_DIR, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape
            st.subheader(f"📌 {filename} （{w} x {h}）")

            fig = px.imshow(img, color_continuous_scale="gray")
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

            hist_values, bin_edges = np.histogram(img.flatten(), bins=25, range=(0, 255))
            fig_hist = px.bar(x=bin_edges[:-1], y=hist_values)
            fig_hist.update_layout(
                xaxis_title="灰階值",
                yaxis_title="像素數量",
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# ====== Data Preparation ======
with tab_dp:
    st.header("Data Preparation")
    st.markdown(
        """
        這裡負責：

        - 自動偵測大圖中的每一顆數字（使用輪廓 / connected components）
        - 將每顆數字裁切並標準化為 28×28 MNIST 風格
        - 依檔名推斷標籤，分到 `output/0~9` 資料夾
        - 提供 **精細預覽**：
            - 括號框位置
            - 單顆數字放大 + 對比強化
            - 水平 / 垂直投影曲線
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        grid_cols = st.number_input("（備援）格子切割欄數", min_value=1, max_value=50, value=10, step=1)
    with col2:
        grid_rows = st.number_input("（備援）格子切割列數", min_value=1, max_value=50, value=1, step=1)

    if st.button("開始智慧切割 + 分類"):
        clear_output(OUTPUT_DIR)
        ensure_folder(OUTPUT_DIR)

        progress_text = st.empty()
        progress_bar = st.progress(0)
        files = os.listdir(UPLOAD_DIR)
        total = len(files)

        # 用於精細預覽（只示範第一張大圖）
        preview_done = False
        session_digits = []

        for i, filename in enumerate(files, start=1):
            progress_text.write(f"處理中：{filename} ({i}/{total})")
            path = os.path.join(UPLOAD_DIR, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape
            if (w > 28 or h > 28):
                split_dir = os.path.join(OUTPUT_DIR, "split_raw")
                ensure_folder(split_dir)

                count, bboxes, digits = auto_split_digits(path, split_dir, return_boxes=True)

                if count == 0:
                    # 退回格子切割
                    count = grid_split_image(path, split_dir, grid_cols=int(grid_cols), grid_rows=int(grid_rows))
                    digits = []
                    bboxes = []

                for sub_name in os.listdir(split_dir):
                    classify_image(sub_name, split_dir, OUTPUT_DIR)

                st.write(f"✂ {filename} 已切出 {count} 顆數字")

                # 只拿第一張大圖來做精細預覽 demo
                if (not preview_done) and count > 0 and len(digits) > 0:
                    preview_done = True
                    st.subheader("🔍 精細預覽示範（來自第一張大圖）")

                    boxed = draw_bounding_boxes(img, bboxes)
                    st.caption("原圖 + Bounding Boxes")
                    st.image(boxed, use_container_width=True)

                    st.session_state["preview_digits"] = digits

            else:
                classify_image(filename, UPLOAD_DIR, OUTPUT_DIR)

            progress_bar.progress(i / total)

        st.success("✅ 資料切割與分類完成！")

        stats = compute_label_stats(OUTPUT_DIR)
        if stats:
            labels = list(stats.keys())
            counts = [stats[k] for k in labels]
            st.subheader("類別數量統計")
            fig_bar = px.bar(x=labels, y=counts)
            fig_bar.update_layout(
                xaxis_title="標籤 (digit)",
                yaxis_title="樣本數量"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("目前尚未產生任何輸出影像。")

    # 互動精細預覽區（若已切出 digits）
    if "preview_digits" in st.session_state and len(st.session_state["preview_digits"]) > 0:
        st.subheader("🎯 互動精細預覽")
        digits = st.session_state["preview_digits"]
        idx = st.slider("選擇數字編號", 0, len(digits) - 1, 0)
        digit = digits[idx]

        colA, colB, colC = st.columns(3)
        with colA:
            st.caption("原始 28×28")
            st.image(digit, width=150)
        with colB:
            st.caption("對比 + 邊緣強化")
            st.image(enhance_digit(digit), width=150)
        with colC:
            h_prof, v_prof = gray_profile(digit)
            st.caption("水平投影")
            st.line_chart(h_prof)

        st.caption("垂直投影")
        st.line_chart(gray_profile(digit)[1])

# ====== Modeling ======
with tab_model:
    st.header("Modeling - 強化 CNN 訓練")
    st.markdown(
        """
        本頁示範：

        - 將 `output/` 中的影像當作訓練資料
        - 使用一個 LeNet 類型的小型 CNN
        - 快速訓練（少量 epoch），用於教學與 Demo
        """
    )

    if st.button("⚡ 使用 output/ 影像快速訓練 CNN"):
        X = []
        y_list = []

        for label in os.listdir(OUTPUT_DIR):
            if not label.isdigit():
                continue
            folder = os.path.join(OUTPUT_DIR, label)
            for f in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                X.append(img)
                y_list.append(int(label))

        if len(X) == 0:
            st.error("找不到可用的 output/ 影像，請先在 Data Preparation 分頁執行切割。")
        else:
            X = np.array(X)
            y_arr = np.array(y_list)
            st.write(f"訓練資料：{len(X)} 張")

            model = train_quick_cnn(X, y_arr, epochs=3)
            st.session_state["cnn_model"] = model

            st.success("✅ CNN 訓練完成，模型已儲存在 session_state['cnn_model']")

# ====== Evaluation ======
with tab_eval:
    st.header("Evaluation - CNN 預測 + Grad-CAM")

    if "cnn_model" not in st.session_state:
        st.info("請先到 Modeling 分頁訓練 CNN。")
    else:
        model = st.session_state["cnn_model"]

        all_digit_paths = []
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    all_digit_paths.append(os.path.join(root, f))

        if not all_digit_paths:
            st.warning("目前 output/ 資料夾沒有可用影像。")
        else:
            test_img_path = st.selectbox("選擇一張切割後的數字影像進行預測＋Grad-CAM：", all_digit_paths)
            test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

            pred = predict_digit(model, test_img)
            top3_idx = pred.argsort()[-3:][::-1]

            st.subheader("📌 預測結果")
            st.write(f"Top-1 預測：**{top3_idx[0]}**，機率 {pred[top3_idx[0]]:.3f}")

            df_top3 = pd.DataFrame({
                "digit": top3_idx,
                "probability": pred[top3_idx]
            })
            st.dataframe(df_top3, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.caption("原始 28×28 影像")
                st.image(test_img, width=200)

            with col2:
                st.caption("Grad-CAM 熱度圖疊加")
                overlay = compute_gradcam_overlay(model, test_img)
                st.image(overlay, width=200)

        # 類別統計重複提供
        st.subheader("類別樣本數統計")
        stats = compute_label_stats(OUTPUT_DIR)
        if stats:
            labels = list(stats.keys())
            counts = [stats[k] for k in labels]
            fig_bar = px.bar(x=labels, y=counts)
            fig_bar.update_layout(xaxis_title="標籤", yaxis_title="樣本數量")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("目前沒有統計資料。")

# ====== Deployment ======
with tab_deploy:
    st.header("Deployment - 報告輸出與 Replit / Kaggle")

    st.markdown(
        """
        在這個頁面，你可以：

        - 一鍵產生 **CRISP-DM PDF 報告**（包含：流程、說明、類別統計）
        - 參考 Replit 部署說明，把整個系統變成雲端 Web App
        - 利用 `kaggle_downloader.py` 自動下載 Kaggle 資料集（需先設定 token）
        """
    )

    report_name = st.text_input("報告檔名（不含副檔名）", value="mnist_crispdm_full_report")
    if st.button("📄 生成 CRISP-DM PDF 報告"):
        stats = compute_label_stats(OUTPUT_DIR)
        pdf_path = os.path.join(REPORT_DIR, f"{report_name}.pdf")

        generate_crispdm_pdf(
            pdf_path=pdf_path,
            label_stats=stats,
            project_title="MNIST 手寫數字切割 + CNN + Grad-CAM 旗艦教學版",
            description=(
                "本報告由 Streamlit + ReportLab 自動產生，"
                "內容包含 CRISP-DM 各階段簡述、資料前處理與類別統計結果，"
                "適合課堂說明、實驗紀錄或專案繳交附件。"
            )
        )

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇ 下載 PDF 報告",
                    data=f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )
            st.success(f"PDF 報告已生成：{pdf_path}")
        else:
            st.error("報告生成失敗，請檢查伺服器端 log。")
