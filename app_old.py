import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms


# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "Alzheimer MRI Classifier Pro"
APP_SUBTITLE = "ResNet18 asosidagi 4-klass MRI demo"
MODEL_PATH = Path("model_resnet18_4class.pth")
SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png"]

# train_resnet.py bilan aynan bir xil preprocessing
IMAGE_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #94a3b8;
        margin-bottom: 1rem;
    }
    .soft-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .small-muted {
        color: #94a3b8;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# session state
if "history" not in st.session_state:
    st.session_state.history = []

if "last_batch_results" not in st.session_state:
    st.session_state.last_batch_results = []


# =========================================================
# MODEL
# =========================================================
def build_model(num_classes: int):
    """
    train_resnet.py dagi modelga aynan mos:
    resnet18(weights=IMAGENET1K_V1) bilan train qilingan,
    app tarafda weights=None bilan skeleton qurib, state_dict yuklaymiz.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_transform():
    """
    train_resnet.py dagi test_tf bilan aynan bir xil.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


@st.cache_resource
def load_bundle_cached(model_path_str: str, model_mtime: float):
    """
    model fayli o'zgarsa cache yangilanadi.
    """
    model_path = Path(model_path_str)

    if not model_path.exists():
        return None, None, None, f"Model fayli topilmadi: {model_path}"

    try:
        bundle = torch.load(model_path, map_location="cpu")
    except Exception as e:
        return None, None, None, f"Model faylini o‘qishda xato: {e}"

    if not isinstance(bundle, dict):
        return None, None, None, "Model formati noto‘g‘ri: dict emas."

    if "model_state" not in bundle:
        return None, None, None, "Model ichida 'model_state' topilmadi."

    if "class_to_idx" not in bundle:
        return None, None, None, "Model ichida 'class_to_idx' topilmadi."

    class_to_idx = bundle["class_to_idx"]
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        return None, None, None, "class_to_idx noto‘g‘ri yoki bo‘sh."

    # indekslarni tekshirish
    try:
        idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
    except Exception:
        return None, None, None, "class_to_idx ichidagi qiymatlar noto‘g‘ri formatda."

    sorted_indices = sorted(idx_to_class.keys())
    expected_indices = list(range(len(idx_to_class)))
    if sorted_indices != expected_indices:
        return None, None, None, (
            f"class_to_idx indekslari ketma-ket emas. "
            f"Topildi: {sorted_indices}, kutilgani: {expected_indices}"
        )

    model = build_model(num_classes=len(idx_to_class))
    try:
        model.load_state_dict(bundle["model_state"], strict=True)
    except Exception as e:
        return None, None, None, f"State dict yuklashda xato: {e}"

    model.eval()

    meta = {
        "model_path": str(model_path.resolve()),
        "num_classes": len(idx_to_class),
        "classes": [idx_to_class[i] for i in sorted(idx_to_class)],
        "image_size": IMAGE_SIZE,
        "normalize_mean": NORM_MEAN,
        "normalize_std": NORM_STD,
        "device": "CPU",
    }

    return model, class_to_idx, idx_to_class, meta


def open_uploaded_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img, None
    except UnidentifiedImageError:
        return None, f"Rasm formatini o‘qib bo‘lmadi: {uploaded_file.name}"
    except Exception as e:
        return None, f"Rasmni ochishda xato ({uploaded_file.name}): {e}"


def predict_single_image(model, idx_to_class, image: Image.Image):
    tfm = get_transform()
    x = tfm(image).unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    dt = time.perf_counter() - t0

    pred_idx = int(torch.argmax(probs).item())
    pred_name = idx_to_class[pred_idx]

    prob_items = []
    for i in range(len(probs)):
        prob_items.append({
            "Class": idx_to_class[i],
            "Probability": float(probs[i].item()),
        })

    prob_df = pd.DataFrame(prob_items).sort_values("Probability", ascending=False)

    result = {
        "predicted_class": pred_name,
        "confidence": float(probs[pred_idx].item()),
        "inference_time_sec": float(dt),
        "probabilities": {row["Class"]: row["Probability"] for _, row in prob_df.iterrows()},
        "top_class_order": prob_df["Class"].tolist(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result, prob_df


def build_single_report(case_id, age, sex, notes, filename, result):
    payload = {
        "timestamp": result["timestamp"],
        "source_file": filename,
        "case_id": case_id or "N/A",
        "age": age,
        "sex": sex,
        "prediction": result["predicted_class"],
        "confidence": round(result["confidence"], 6),
        "inference_time_sec": round(result["inference_time_sec"], 6),
        "notes": notes,
        "probabilities": result["probabilities"],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def add_to_history(source_file, case_id, age, sex, notes, result):
    record = {
        "timestamp": result["timestamp"],
        "source_file": source_file,
        "case_id": case_id or "N/A",
        "age": age,
        "sex": sex,
        "prediction": result["predicted_class"],
        "confidence": round(result["confidence"], 4),
        "inference_time_sec": round(result["inference_time_sec"], 4),
        "notes": notes,
    }
    st.session_state.history.append(record)


# =========================================================
# LOAD MODEL
# =========================================================
model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
loaded = load_bundle_cached(str(MODEL_PATH), model_mtime)

# MODEL_PATH yo'q bo'lsa 4 ta emas, xato string qaytadi; mavjud bo'lsa meta dict
if len(loaded) == 4 and isinstance(loaded[3], str):
    model, class_to_idx, idx_to_class, load_error = loaded
    model_meta = None
else:
    model, class_to_idx, idx_to_class, model_meta = loaded
    load_error = None


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("⚙️ Panel")

    if load_error:
        st.error(load_error)
    else:
        st.success("Model muvaffaqiyatli yuklandi")
        st.write({
            "Model": "ResNet18",
            "Classes": model_meta["classes"],
            "Input size": f'{model_meta["image_size"]}x{model_meta["image_size"]}',
            "Device": model_meta["device"],
        })

    st.markdown("### Sozlamalar")
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.60, 0.01)
    show_prob_table = st.checkbox("Probability jadvalini ko‘rsatish", value=True)
    show_debug = st.checkbox("Debug ma'lumotlarini ko‘rsatish", value=False)

    st.markdown("### Eslatma")
    st.info("Bu demo klinik tashxis o‘rnini bosmaydi.")

    if model_meta and show_debug:
        st.markdown("### Model diagnostikasi")
        st.json(model_meta)


# =========================================================
# HEADER
# =========================================================
st.markdown(f'<div class="main-title">🧠 {APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{APP_SUBTITLE}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="soft-card">
        Bu ilova sizning <b>train_resnet.py</b> va <b>model_resnet18_4class.pth</b> fayllaringizga
        aynan moslab yozilgan. Prediction, batch tahlil, history va eksport funksiyalari qo‘shilgan.
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Single prediction",
    "Batch prediction",
    "History & Export",
    "System info"
])


# =========================================================
# TAB 1 — SINGLE
# =========================================================
with tab1:
    left, right = st.columns([1.05, 1])

    with left:
        uploaded = st.file_uploader(
            "MRI rasm yuklang",
            type=SUPPORTED_EXTENSIONS,
            key="single_uploader",
        )

        image = None
        image_error = None
        if uploaded is not None:
            image, image_error = open_uploaded_image(uploaded)
            if image_error:
                st.error(image_error)
            else:
                st.image(image, caption=f"Yuklangan rasm: {uploaded.name}", use_container_width=True)
        else:
            st.info("Boshlash uchun rasm yuklang.")

    with right:
        st.markdown("### Case ma'lumotlari")
        case_id = st.text_input("Case ID", placeholder="CASE-001")
        age = st.number_input("Yosh", min_value=1, max_value=120, value=65)
        sex = st.selectbox("Jins", ["Unknown", "Male", "Female"])
        notes = st.text_area("Izoh", placeholder="Qo‘shimcha ma'lumot...")

        run_btn = st.button("🔍 Tahlilni boshlash", use_container_width=True, type="primary")

        if run_btn:
            if uploaded is None:
                st.warning("Avval rasm yuklang.")
            elif image_error:
                st.error(image_error)
            else:
                try:
                    result, prob_df = predict_single_image(model, idx_to_class, image)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted class", result["predicted_class"])
                    c2.metric("Confidence", f"{result['confidence']:.2%}")
                    c3.metric("Inference time", f"{result['inference_time_sec']:.4f} s")

                    if result["confidence"] >= confidence_threshold:
                        st.warning("Natija ko‘rib chiqilishi tavsiya etiladi.")
                    else:
                        st.success("Natija nisbatan past noaniqlik bilan qaytdi.")

                    st.markdown("### Ehtimollar taqsimoti")
                    st.bar_chart(prob_df.set_index("Class"))

                    if show_prob_table:
                        st.dataframe(prob_df, use_container_width=True)

                    add_to_history(
                        source_file=uploaded.name,
                        case_id=case_id,
                        age=age,
                        sex=sex,
                        notes=notes,
                        result=result,
                    )

                    report_json = build_single_report(
                        case_id=case_id,
                        age=age,
                        sex=sex,
                        notes=notes,
                        filename=uploaded.name,
                        result=result,
                    )

                    st.download_button(
                        "⬇️ JSON report yuklab olish",
                        data=report_json,
                        file_name=f"{Path(uploaded.name).stem}_prediction.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                    if show_debug:
                        st.markdown("### Debug")
                        st.json(result)

                except Exception as e:
                    st.error(f"Prediction paytida xato: {e}")


# =========================================================
# TAB 2 — BATCH
# =========================================================
with tab2:
    st.markdown("### Batch prediction")
    st.caption("Bir nechta MRI rasmni bir yo‘la tahlil qilish.")

    batch_files = st.file_uploader(
        "Bir nechta rasm yuklang",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        key="batch_uploader",
    )

    batch_case_prefix = st.text_input("Batch case prefix", value="BATCH")
    batch_run = st.button("📦 Batch tahlilni boshlash", use_container_width=True)

    if batch_run:
        if not batch_files:
            st.warning("Avval batch fayllarni yuklang.")
        else:
            results = []
            errors = []

            progress = st.progress(0)
            total = len(batch_files)

            for idx, file in enumerate(batch_files, start=1):
                image, err = open_uploaded_image(file)
                if err:
                    errors.append({"file": file.name, "error": err})
                else:
                    try:
                        result, _ = predict_single_image(model, idx_to_class, image)

                        row = {
                            "timestamp": result["timestamp"],
                            "source_file": file.name,
                            "case_id": f"{batch_case_prefix}-{idx:03d}",
                            "prediction": result["predicted_class"],
                            "confidence": round(result["confidence"], 4),
                            "inference_time_sec": round(result["inference_time_sec"], 4),
                        }
                        results.append(row)
                        st.session_state.history.append({
                            **row,
                            "age": "",
                            "sex": "",
                            "notes": "batch prediction",
                        })
                    except Exception as e:
                        errors.append({"file": file.name, "error": str(e)})

                progress.progress(idx / total)

            st.session_state.last_batch_results = results

            if results:
                df = pd.DataFrame(results).sort_values(
                    by=["confidence", "prediction"],
                    ascending=[False, True]
                )
                st.success(f"{len(results)} ta fayl tahlil qilindi.")
                st.dataframe(df, use_container_width=True)

                class_counts = Counter(df["prediction"].tolist())
                count_df = pd.DataFrame({
                    "Class": list(class_counts.keys()),
                    "Count": list(class_counts.values()),
                }).sort_values("Count", ascending=False)

                st.markdown("### Klasslar bo‘yicha taqsimot")
                st.bar_chart(count_df.set_index("Class"))

                st.download_button(
                    "⬇️ Batch CSV yuklab olish",
                    data=df.to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.download_button(
                    "⬇️ Batch JSON yuklab olish",
                    data=json.dumps(results, indent=2, ensure_ascii=False),
                    file_name="batch_predictions.json",
                    mime="application/json",
                    use_container_width=True,
                )

            if errors:
                err_df = pd.DataFrame(errors)
                st.error(f"{len(errors)} ta faylda xato yuz berdi.")
                st.dataframe(err_df, use_container_width=True)


# =========================================================
# TAB 3 — HISTORY
# =========================================================
with tab3:
    st.markdown("### Prediction history")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        st.download_button(
            "⬇️ History CSV yuklab olish",
            data=hist_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "⬇️ History JSON yuklab olish",
            data=json.dumps(st.session_state.history, indent=2, ensure_ascii=False),
            file_name="prediction_history.json",
            mime="application/json",
            use_container_width=True,
        )

        if st.button("🗑️ History ni tozalash", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_batch_results = []
            st.rerun()
    else:
        st.info("Hozircha history bo‘sh.")


# =========================================================
# TAB 4 — SYSTEM INFO
# =========================================================
with tab4:
    st.markdown("### Tizim haqida")
    st.write(
        """
        Ushbu ilova aynan sizning quyidagi fayllarga mos yozilgan:

        - `train_resnet.py`
        - `model_resnet18_4class.pth`

        App ichidagi texnik qarorlar:
        - model: `resnet18`
        - `fc` qatlam klasslar soniga mos qayta quriladi
        - `strict=True` bilan state_dict yuklanadi
        - resize: `224x224`
        - normalize:
          - mean = `[0.485, 0.456, 0.406]`
          - std  = `[0.229, 0.224, 0.225]`

        Bu qiymatlar train tarafdagi preprocessing bilan aynan bir xil.
        """
    )

    st.markdown("### Cheklovlar")
    st.write(
        """
        - Bu demo klinik tashxis o‘rnini bosmaydi.
        - Model sifati dataset sifati va trening natijasiga bog‘liq.
        - Agar siz keyin boshqa arxitektura bilan qayta train qilsangiz,
          app ichidagi model skeleton ham yangilanadi.
        """
    )

    st.markdown("### Model meta")
    st.json(model_meta)
