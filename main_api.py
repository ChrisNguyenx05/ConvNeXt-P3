import os
import sys
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Thêm đường dẫn package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import DRPredictor, load_image, full_preprocess_pipeline

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Diabetic Retinopathy Classification API",
    description="Hệ thống AI Chẩn đoán Mức độ Bệnh Võng mạc Tiểu đường (Threshold-Optimized Ensemble: ConvNeXt-Tiny + ViT-Tiny)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Cấu hình CORS để Web Frontend (React / Vue / Flutter / Angular) truy cập được
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lời khuyên y tế lâm sàng theo tiêu chuẩn ICDR 5 mức độ
CLINICAL_ADVICE = {
    0: {
        "title": "Mắt Bình Thường (No DR)",
        "badge_color": "#10b981", # Green
        "advice": "Chưa phát hiện tổn thương võng mạc tiểu đường. Khuyến nghị khám mắt định kỳ 12 tháng/lần và kiểm soát chỉ số đường huyết tốt.",
        "urgency": "Bình thường"
    },
    1: {
        "title": "Bệnh Nhẹ (Mild DR)",
        "badge_color": "#3b82f6", # Blue
        "advice": "Xuất hiện các vi phình mạch nhỏ. Khuyến nghị tái khám theo dõi chuyên khoa mắt sau 6 - 12 tháng và kiểm soát nghiêm ngặt đường huyết, huyết áp.",
        "urgency": "Theo dõi định kỳ"
    },
    2: {
        "title": "Bệnh Trung Bình (Moderate DR)",
        "badge_color": "#f59e0b", # Orange/Yellow
        "advice": "Tổn thương xuất huyết/xuất tiết mức độ vừa. Cần thăm khám bác sĩ nhãn khoa trong 3 - 6 tháng để đánh giá hoàng điểm và can thiệp kịp thời.",
        "urgency": "Khám chuyên khoa"
    },
    3: {
        "title": "Bệnh Nặng (Severe DR)",
        "badge_color": "#ef4444", # Red
        "advice": "Tổn thương nghiêm trọng ở nhiều góc phần tư võng mạc. CẦN THIẾT chuyển khám chuyên khoa mắt gấp trong 2 - 4 tuần để xét can thiệp Laser/OCT.",
        "urgency": "Cần can thiệp sớm"
    },
    4: {
        "title": "Tăng Sinh Nguy Hiểm (Proliferative DR)",
        "badge_color": "#8b5cf6", # Purple
        "advice": "Tăng sinh tân mạch nguy cơ gây mờ mắt vĩnh viễn hoặc bong võng mạc! CẦN ĐIỀU TRỊ KHẨN CẤP tại trung tâm nhãn khoa chuyên sâu.",
        "urgency": "KHẨN CẤP"
    }
}

# Biến toàn cục lưu trữ DRPredictor instance
predictor: DRPredictor = None


def get_predictor() -> DRPredictor:
    """Tải lazy DRPredictor nếu chưa được khởi tạo thành công."""
    global predictor
    if predictor is None:
        print("[INFO] Dang nap mo hinh AI vao RAM/GPU...")
        try:
            convnext_path = "convnext_results/convnext_inference.pt"
            
            # Nếu không tìm thấy file cục bộ, tự động tải về từ Hugging Face
            if not os.path.exists(convnext_path):
                print("[INFO] Không tìm thấy tệp mô hình cục bộ. Đang tải tự động từ Hugging Face Hub...")
                from huggingface_hub import hf_hub_download
                hf_token = os.getenv("HF_TOKEN")
                
                convnext_path = hf_hub_download(
                    repo_id="chrisnguyenx/ConvNeXt-P3",
                    filename="convnext_inference.pt",
                    token=hf_token
                )
                
            predictor = DRPredictor(convnext_path=convnext_path)
            print(f"[SUCCESS] Da nap thanh cong mo hinh AI tren thiet bi: {predictor.device}")
        except Exception as e:
            print(f"[ERROR] Loi khoi tao mo hinh AI: {e}")
            raise e
    return predictor


@app.on_event("startup")
def startup_event():
    try:
        get_predictor()
    except Exception as e:
        print(f"[WARNING] Startup predictor deferred: {e}")


@app.get("/")
def root():
    """Endpoint gốc kiểm tra trạng thái dịch vụ."""
    return {
        "message": "AI Diabetic Retinopathy API Service (ConvNeXt) is running.",
        "docs_url": "/docs",
        "health_check": "/api/info",
        "predict_endpoint": "POST /api/predict"
    }


@app.get("/api/info")
def get_info():
    """Lấy thông tin hệ thống và trạng thái mô hình."""
    try:
        pred_instance = get_predictor()
        return {
            "status": "online",
            "model_name": "ConvNeXt-Tiny",
            "num_classes": 5,
            "device": str(pred_instance.device),
            "classes": pred_instance.class_names
        }
    except Exception as e:
        return {
            "status": "error_loading_model",
            "error": str(e)
        }


@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint chính tiếp nhận ảnh đáy mắt (Fundus Image) và trả về kết quả chẩn đoán JSON.
    """
    try:
        pred_instance = get_predictor()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi khoi tao mo hinh AI: {str(e)}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File tải lên không phải là định dạng ảnh hợp lệ.")

    try:
        # Đọc dữ liệu ảnh từ request upload
        image_bytes = await file.read()

        # 1. Chạy dự đoán AI
        result = pred_instance.predict(image_bytes, use_ben_graham=True)

        # 2. Xử lý ảnh Ben Graham để trả về dạng Base64 (phục vụ hiển thị nếu cần)
        img_bgr = load_image(image_bytes)
        processed_bgr = full_preprocess_pipeline(img_bgr, target_size=(224, 224), use_ben_graham=True)
        _, encoded_img = cv2.imencode(".jpg", processed_bgr)
        base64_preprocessed = base64.b64encode(encoded_img).decode("utf-8")

        # 3. Lấy thông tin y tế bổ sung
        class_id = result["class_id"]
        clinical_info = CLINICAL_ADVICE.get(class_id, CLINICAL_ADVICE[0])

        return {
            "success": True,
            "filename": file.filename,
            "prediction": {
                "class_id": class_id,
                "class_name": result["class_name"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            },
            "clinical_guidance": clinical_info,
            "preprocessed_image_base64": f"data:image/jpeg;base64,{base64_preprocessed}",
            "gradcam_image_base64": result.get("gradcam_image_base64", ""),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình xử lý ảnh: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
