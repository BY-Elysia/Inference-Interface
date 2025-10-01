# app.py —— 单张在线推理（FastAPI）
import io, base64, time
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2

# === 来自你的工程 ===
from nets.basicUnet_new import UNetTaskAligWeight               # 你的模型
from util.data_utils import CDDataAugmentation                  # 你的预处理

# ====== 配置（按你的实际修改） ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "checkpoint/Zhou/best_model_epoch153.pt"            # ← 用你的权重路径
IMG_SIZE = 224                                                  # 与训练一致
THRESH = 0.5                                                    # 分割阈值
ALPHA = 0.35                                                    # 叠加透明度
CLASS_NAMES = None                                              # 如 ["0-健康","1-翼状胬肉"]，不填则返回 "0/1"

app = FastAPI(title="Seg+Cls Inference API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
# 健康检查端点 —— 供 K8s 探针使用
@app.get("/healthz")
def healthz():
    return {"ok": True}

# === 你原来的：CDDataAugmentation（复用） ===
aug = CDDataAugmentation(
    img_size=IMG_SIZE, ori_size=IMG_SIZE, crop=None,
    p_hflip=0.0, p_vflip=0.0, color_jitter_params=None, long_mask=True
)

def correct_dims(img: np.ndarray) -> np.ndarray:
    """来自你的 TestImageDataset.correct_dims"""
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img

def pil_to_base64_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def overlay_mask_on_image(orig_bgr: np.ndarray, mask: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """把 0/1 mask 以红色半透明叠到原图（替代你原先逐像素 putpixel + 保存 png 的做法）"""
    color = np.array([0, 0, 255], dtype=np.uint8)  # BGR 红
    layer = np.zeros_like(orig_bgr, dtype=np.uint8)
    layer[mask == 1] = color
    return cv2.addWeighted(orig_bgr, 1.0, layer, alpha, 0.0)

@torch.no_grad()
def load_model() -> UNetTaskAligWeight:
    """来自你的主程序：服务启动时只加载一次（避免每次请求都 load_state_dict）"""
    model = UNetTaskAligWeight(n_channels=3, n_classes=1).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt["net"] if "net" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

MODEL = load_model()

def preprocess_for_model(img_bgr: np.ndarray) -> torch.Tensor:
    """
    来自你的 Dataset.__getitem__：
    - correct_dims + aug.transform
    - 输出 [1,3,224,224]（与你训练一致）
    """
    h, w = img_bgr.shape[:2]
    dummy_mask = np.zeros((h, w), dtype=np.uint8)       # 你原来 transform 需要 mask 协同
    img_bgr = correct_dims(img_bgr)                     # HWC
    img_t, _ = aug.transform(img_bgr, dummy_mask)       # 你的增强输出通常是 tensor(C,H,W) 或 np.ndarray
    if isinstance(img_t, np.ndarray):
        img_t = torch.from_numpy(img_t)
    if img_t.ndim == 3:
        img_t = img_t.unsqueeze(0)                      # [C,H,W] -> [1,C,H,W]
    return img_t.float().to(DEVICE)

@torch.no_grad()
def infer_one(img_bytes: bytes):
    """
    等价于你原来的 inference_all 里每张图的那一段：
    se_out/cl_out -> sigmoid/softmax -> mask阈值 -> resize回原图 -> 画红mask -> 返回类别+结果图
    """
    t0 = time.time()

    # 读取原图（保持原始尺寸用于可视化）
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    orig_rgb = np.array(pil)                      # (H,W,3) RGB
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    # 预处理（与你训练一致）
    x = preprocess_for_model(orig_bgr)            # [1,3,224,224]
    # 模型前向：=== 来自你旧代码 ===
    se_out, cl_out = MODEL(x)

    # 分割：sigmoid + 阈值（=== 来自你旧代码 ===）
    se_prob = torch.sigmoid(se_out)               # [1,1,h,w]
    se_mask = (se_prob > THRESH).float()[0, 0].cpu().numpy().astype(np.uint8)

    # 把 224×224 的 mask 拉回原图大小（在线可视化的关键）
    H, W = orig_bgr.shape[:2]
    mask_full = cv2.resize(se_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # 分类：softmax + argmax（=== 来自你旧代码 ===）
    probs = F.softmax(cl_out, dim=1)
    cls_idx = int(torch.argmax(probs, dim=1).item())
    classType = CLASS_NAMES[cls_idx] if CLASS_NAMES else str(cls_idx)

    # 可视化叠加（替代你以前逐像素画红 + 保存 png）
    blended_bgr = overlay_mask_on_image(orig_bgr, mask_full, alpha=ALPHA)
    label = f"class: {classType}"
    cv2.rectangle(blended_bgr, (12, 12), (12 + 16*max(6, len(label)), 44), (0,0,0), thickness=-1)
    cv2.putText(blended_bgr, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # 返回 base64（Node 再决定是否落盘）
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(blended_rgb)
    image_b64 = pil_to_base64_png(out_pil)

    return {"classType": classType, "imageBase64": image_b64, "timeMs": int((time.time()-t0)*1000)}

# === FastAPI 路由：接收单张图 ===
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    raw = await file.read()
    return infer_one(raw)
# app.py
@app.on_event("startup")
def warmup():
    import torch
    from time import perf_counter
    t0 = perf_counter()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        _ = MODEL(dummy)            # 走一遍前向，初始化算子/显卡
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    print(f"[warmup] done in {perf_counter()-t0:.3f}s")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
