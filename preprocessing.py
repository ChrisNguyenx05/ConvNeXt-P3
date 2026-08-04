"""
preprocessing.py
================
Pipeline tiền xử lý ảnh đáy mắt (fundus) cho khâu suy luận (Inference Pipeline).
Hỗ trợ loại bỏ viền đen, resize giữ tỷ lệ (letterbox), lọc nhiễu Ben Graham và chuẩn hóa Tensor PyTorch.
"""

from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_MAIN_FREE"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"

import io
from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
# Disable multi-threading memory overhead in PyTorch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
from torchvision import transforms

TARGET_SIZE = (224, 224)
CROP_TOLERANCE = 12
BEN_SIGMA = 10
BLACK_BORDER_RATIO_THRESH = 0.05
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_fundus_circle(img: np.ndarray, tolerance: int = CROP_TOLERANCE) -> np.ndarray:
    """Crop bounding box của vùng sáng trên ảnh BGR (loại viền đen)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, mask = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y: y + h, x: x + w]


def auto_detect_border(img: np.ndarray, thresh: float = BLACK_BORDER_RATIO_THRESH) -> bool:
    """Tự động phát hiện xem ảnh có viền đen xung quanh hay không dựa trên tỷ lệ pixel tối."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float((gray < CROP_TOLERANCE).mean()) > thresh


def letterbox_resize(
    img: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Resize ảnh về kích thước target_size giữ nguyên aspect ratio (đệm viền đen)."""
    h, w = img.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=interpolation)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    pad_y = (th - nh) // 2
    pad_x = (tw - nw) // 2
    canvas[pad_y: pad_y + nh, pad_x: pad_x + nw] = resized
    return canvas


def ben_graham_transform(img: np.ndarray, sigma_x: int = BEN_SIGMA) -> np.ndarray:
    """Xử lý tăng cường tương phản Ben Graham: output = 4*img - 4*Blur + 128."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_x)
    enhanced = cv2.addWeighted(img, 4, blur, -4, 128)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def full_preprocess_pipeline(
    img: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE,
    use_ben_graham: bool = True,
    force_crop: bool | None = None,
) -> np.ndarray:
    """
    Pipeline xử lý ảnh OpenCV đầy đủ:
    1. Tự động kiểm tra & Crop viền đen
    2. Resize letterbox về target_size
    3. Ben Graham contrast enhancement (nếu use_ben_graham=True)
    Returns: numpy array BGR
    """
    do_crop = auto_detect_border(img) if force_crop is None else force_crop
    if do_crop:
        img = crop_fundus_circle(img)
    img = letterbox_resize(img, target_size)
    if use_ben_graham:
        img = ben_graham_transform(img)
    return img


def load_image(image_input: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Chuyển đổi các định dạng đầu vào (Path, Bytes, PIL Image, BGR Numpy Array) -> OpenCV BGR array.
    """
    if isinstance(image_input, (str, bytes, bytearray)):
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Không thể đọc file ảnh từ đường dẫn: {image_input}")
            return img
        else:
            buf = np.frombuffer(image_input, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Không thể giải mã dữ liệu bytes thành ảnh.")
            return img
    elif isinstance(image_input, Image.Image):
        # PIL (RGB) -> OpenCV (BGR)
        img_rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:  # Grayscale
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        return image_input.copy()
    else:
        raise TypeError(f"Kiểu dữ liệu đầu vào không được hỗ trợ: {type(image_input)}")


def prepare_image_tensor(
    image_input: Union[str, bytes, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = TARGET_SIZE,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    use_ben_graham: bool = True,
) -> torch.Tensor:
    """
    Nhận đầu vào linh hoạt -> Tiền xử lý OpenCV -> Chuyển thành PyTorch Tensor (1, C, H, W).
    """
    img_bgr = load_image(image_input)
    processed_bgr = full_preprocess_pipeline(
        img_bgr, target_size=target_size, use_ben_graham=use_ben_graham
    )
    # OpenCV BGR -> PIL RGB -> PyTorch Tensor
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(processed_rgb)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tensor = transform(pil_img)  # shape: (3, H, W)
    return tensor


import base64
import timm
from typing import Optional, Tuple
import torch.nn.functional as F
import torch.nn as nn

class AuthenticConvNeXtGradCAM:
    """
    Authentic Grad-CAM for ConvNeXt architectures using PyTorch forward and full backward hooks.
    Target layer: ConvNeXt Stage 4 (model.stages[-1]).
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "stages"):
                target_layer = model.stages[-1]
            elif hasattr(model, "features"):
                target_layer = model.features[-1]
            else:
                for name, module in reversed(list(model.named_modules())):
                    if isinstance(module, (nn.Conv2d, nn.Sequential)):
                        target_layer = module
                        break

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.handles = []
        if self.target_layer is not None:
            h_fwd = self.target_layer.register_forward_hook(self._forward_hook)
            h_bwd = self.target_layer.register_full_backward_hook(self._backward_hook)
            self.handles.extend([h_fwd, h_bwd])

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        self.model.zero_grad()
        x = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            cam = np.ones((x.shape[2], x.shape[3]), dtype=np.float32)
            self.last_probs = probs[0].detach().cpu().numpy()
            return cam, target_class, float(probs[0, target_class].item())

        grads = self.gradients.cpu().data.numpy()[0]  # [C, H, W]
        acts = self.activations.cpu().data.numpy()[0]  # [C, H, W]

        weights = np.mean(grads, axis=(1, 2))  # [C]
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]

        cam = np.maximum(cam, 0)  # ReLU
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        h, w = x.shape[2], x.shape[3]
        cam = cv2.resize(cam, (w, h))
        self.last_probs = probs[0].detach().cpu().numpy()
        return cam, target_class, float(probs[0, target_class].item())

    def remove_hooks(self):
        for h in self.handles:
            h.remove()


class DRPredictor:
    def __init__(self, convnext_path: str | None = None, device: str | None = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Candidate paths for ConvNeXt
        convnext_candidates = [
            convnext_path,
            "convnext_inference.pt",
            "convnext_results/convnext_inference.pt",
            os.path.join(os.path.dirname(__file__), "convnext_inference.pt"),
            os.path.join(os.path.dirname(__file__), "convnext_results", "convnext_inference.pt"),
        ]
        self.convnext_path = None
        for cand in convnext_candidates:
            if cand and os.path.exists(cand):
                self.convnext_path = cand
                break
        if self.convnext_path is None:
            self.convnext_path = "convnext_results/convnext_inference.pt"

        print(f"[DRPredictor] Loading ConvNeXt JIT model from {self.convnext_path} on {self.device}...")
        m_jit = torch.jit.load(self.convnext_path, map_location=self.device)
        
        # Shift state dict references key-by-key to avoid RAM spikes
        jit_sd = m_jit.state_dict()
        sd = {}
        for k in list(jit_sd.keys()):
            if k == 'log_prior':
                continue
            new_k = k.replace('base_model.', '')
            sd[new_k] = jit_sd[k]
            del jit_sd[k]
            
        del jit_sd
        del m_jit
        import gc
        gc.collect()
        
        # Recreate PyTorch timm model to support hooks for Grad-CAM
        self.model = timm.create_model("convnext_tiny", pretrained=False, num_classes=5)
        self.model.load_state_dict(sd)
        del sd
        gc.collect()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Grad-CAM engine
        self.cam_engine = AuthenticConvNeXtGradCAM(self.model)
        
        self.class_names = {
            0: "No DR",
            1: "Mild",
            2: "Moderate",
            3: "Severe",
            4: "Proliferative DR",
        }

    def predict(self, image_input: Union[str, bytes, Image.Image, np.ndarray], use_ben_graham: bool = True) -> dict:
        # Prepare input tensor using existing prepare_image_tensor function
        tensor = prepare_image_tensor(image_input, use_ben_graham=use_ben_graham).unsqueeze(0).to(self.device)
        
        # Run Grad-CAM directly with gradients enabled locally
        with torch.enable_grad():
            cam, class_id, confidence = self.cam_engine.generate_cam(tensor, target_class=None)
            probs = self.cam_engine.last_probs

        # Free tensor and gradients immediately to prevent RAM spikes / OOM
        del tensor
        self.model.zero_grad(set_to_none=True)
        import gc
        gc.collect()
            
        # Generate BGR image for overlay
        img_bgr = load_image(image_input)
        processed_bgr = full_preprocess_pipeline(
            img_bgr, target_size=(224, 224), use_ben_graham=use_ben_graham
        )
        
        # Apply JET color map to CAM heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # Overlay heatmap with the preprocessed image (both are 224x224 BGR)
        overlay = cv2.addWeighted(processed_bgr, 0.6, heatmap, 0.4, 0)
        
        # Encode overlay to base64
        _, encoded_img = cv2.imencode(".jpg", overlay)
        base64_gradcam = base64.b64encode(encoded_img).decode("utf-8")
        gradcam_base64_str = f"data:image/jpeg;base64,{base64_gradcam}"
        
        return {
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "confidence": float(confidence),
            "probabilities": {
                "No DR": float(probs[0]),
                "Mild": float(probs[1]),
                "Moderate": float(probs[2]),
                "Severe": float(probs[3]),
                "Proliferative DR": float(probs[4]),
            },
            "gradcam_image_base64": gradcam_base64_str,
        }


