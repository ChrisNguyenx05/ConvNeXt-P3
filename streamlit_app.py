import streamlit as st
import cv2
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from preprocessing import DRPredictor, full_preprocess_pipeline

# Page configuration
st.set_page_config(
    page_title="Chẩn đoán Bệnh võng mạc tiểu đường",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Chẩn Đoán Bệnh Võng Mạc Tiểu Đường")
st.write(
    "Hệ thống AI hỗ trợ chẩn đoán mức độ Bệnh võng mạc tiểu đường (Diabetic Retinopathy - DR) "
    "sử dụng mô hình Ensemble tối ưu ngưỡng giữa ConvNeXt-Tiny và DeiT-ViT-Tiny."
)

# Cache model loader to avoid downloading and reloading on every user interaction
@st.cache_resource
def load_models():
    # Fetch Hugging Face token from Streamlit secrets (configured in dashboard)
    if "HF_TOKEN" not in st.secrets:
        st.error("Lỗi: Chưa cấu hình HF_TOKEN trong mục Secrets của Streamlit Cloud. Vui lòng thêm HF_TOKEN để truy cập mô hình private.")
        st.stop()
    hf_token = st.secrets["HF_TOKEN"]
    
    with st.spinner("Đang nạp mô hình từ Hugging Face Hub (quá trình này chỉ thực hiện 1 lần)..."):
        convnext_path = hf_hub_download(
            repo_id="chrisnguyenx/ConvNeXt-P3",
            filename="convnext_inference.pt",
            token=hf_token
        )
        vit_path = hf_hub_download(
            repo_id="chrisnguyenx/DeiT-ViT-P3",
            filename="vit_inference.pt",
            token=hf_token
        )
        predictor = DRPredictor(convnext_path=convnext_path, vit_path=vit_path)
    return predictor

try:
    predictor = load_models()
    st.success("Nạp mô hình AI thành công!")
except Exception as e:
    st.error(f"Lỗi nạp mô hình từ Hugging Face: {e}")
    st.stop()

# Image uploader
uploaded_file = st.file_uploader("Tải lên ảnh chụp đáy mắt màu (JPEG/PNG)", type=["jpg", "jpeg", "png"])

CLASS_NAMES = {
    0: "No DR (Bình thường)",
    1: "Mild (Nhẹ)",
    2: "Moderate (Trung bình)",
    3: "Severe (Nặng)",
    4: "Proliferative DR (Tăng sinh nguy hiểm)",
}

CLINICAL_ADVICE = {
    0: "Chưa phát hiện tổn thương võng mạc tiểu đường. Khuyến nghị khám mắt định kỳ 12 tháng/lần và kiểm soát chỉ số đường huyết tốt.",
    1: "Xuất hiện các vi phình mạch nhỏ. Khuyến nghị tái khám theo dõi chuyên khoa mắt sau 6 - 12 tháng và kiểm soát nghiêm ngặt đường huyết, huyết áp.",
    2: "Tổn thương xuất huyết/xuất tiết mức độ vừa. Cần thăm khám bác sĩ nhãn khoa trong 3 - 6 tháng để đánh giá hoàng điểm và can thiệp kịp thời.",
    3: "Tổn thương nghiêm trọng ở nhiều góc phần tư võng mạc. CẦN THIẾT chuyển khám chuyên khoa mắt gấp trong 2 - 4 tuần để xét can thiệp Laser/OCT.",
    4: "Tăng sinh tân mạch nguy cơ gây mờ mắt vĩnh viễn hoặc bong võng mạc! CẦN ĐIỀU TRỊ KHẨN CẤP tại trung tâm nhãn khoa chuyên sâu.",
}

if uploaded_file is not None:
    # Read image
    file_bytes = uploaded_file.read()
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc đã tải lên")
        st.image(image, use_column_width=True)
        
    # Run prediction
    with st.spinner("AI đang thực hiện chẩn đoán..."):
        # Convert PIL image to BGR numpy array
        img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Get preprocessed image for visualization
        processed_bgr = full_preprocess_pipeline(img_bgr, target_size=(224, 224), use_ben_graham=True)
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        
        # Call predictor
        result = predictor.predict(file_bytes, use_ben_graham=True)
        
    with col2:
        st.subheader("Ảnh đã tiền xử lý (Ben Graham)")
        st.image(processed_rgb, use_column_width=True)
        
    st.divider()
    
    # Display results
    class_id = result["class_id"]
    confidence = result["confidence"]
    advice = CLINICAL_ADVICE[class_id]
    
    st.header(f"Chẩn đoán: **{CLASS_NAMES[class_id]}**")
    st.subheader(f"Độ tin cậy: **{confidence:.2%}**")
    
    # Colored alert box depending on severity
    if class_id == 0:
        st.success(f"**Lời khuyên lâm sàng**: {advice}")
    elif class_id in [1, 2]:
        st.warning(f"**Lời khuyên lâm sàng**: {advice}")
    else:
        st.error(f"**CẢNH BÁO LÂM SÀNG**: {advice}")
        
    # Probabilities chart
    st.subheader("Phân bố xác suất các lớp bệnh")
    chart_data = {k: float(v) for k, v in result["probabilities"].items()}
    st.bar_chart(chart_data)
    
    # Show Raw JSON response for developers
    st.divider()
    with st.expander("🔍 Xem phản hồi JSON gốc (dành cho lập trình viên Backend/Frontend)"):
        st.json(result)
