from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageOps
from torchvision import transforms


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "models" / "basic_cnn_final.pt"
OUTPUT_DIR = APP_DIR / "static" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

MAX_OUTPUT_SIDE = 1100


class ConvBN(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class BasicCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBN(3, 32), nn.MaxPool2d(2),
            ConvBN(32, 64), nn.MaxPool2d(2),
            ConvBN(64, 128), nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            ConvBN(128, 192), nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
            ConvBN(192, 256)
        )
        self.cam_layer = self.features[-1]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.a = None
        self.g = None
        self.h1 = layer.register_forward_hook(self.f_hook)
        self.h2 = layer.register_full_backward_hook(self.b_hook)

    def f_hook(self, m, i, o):
        self.a = o

    def b_hook(self, m, gi, go):
        self.g = go[0]

    def __call__(self, x, target):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        out[:, target].sum().backward()
        w = self.g.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.a).sum(1).relu()
        cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def close(self):
        self.h1.remove()
        self.h2.remove()


def load_package():
    package = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = BasicCNN(package["num_classes"]).to(device)
    model.load_state_dict(package["model_state_dict"])
    model.eval()
    idx_to_class = {int(k): v for k, v in package["idx_to_class"].items()}
    return model, package, idx_to_class


model, package, idx_to_class = load_package()


def preprocess(img):
    tf = transforms.Compose([
        transforms.Resize((package["img_size"], package["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(package["mean"], package["std"])
    ])
    return tf(img).unsqueeze(0).to(device)


def load_rgb_image(file_path):
    img = Image.open(file_path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def shrink_only(img):
    img = img.copy()
    img.thumbnail((MAX_OUTPUT_SIDE, MAX_OUTPUT_SIDE), Image.Resampling.LANCZOS)
    return img


def overlay_gradcam(img, cam):
    display_img = shrink_only(img)
    w, h = display_img.size

    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    img_np = np.array(display_img).astype(np.float32) / 255

    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    overlay = np.clip(0.62 * img_np + 0.38 * heat, 0, 1)
    return np.uint8(overlay * 255)


def get_image_info(file_path):
    img = Image.open(file_path)
    img = ImageOps.exif_transpose(img)

    return {
        "name": Path(file_path).name,
        "format": img.format,
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "size_kb": round(Path(file_path).stat().st_size / 1024, 2)
    }


def clean_old_outputs(limit=80):
    files = sorted(OUTPUT_DIR.glob("gradcam_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[limit:]:
        p.unlink(missing_ok=True)


def predict_image(file_path):
    img = load_rgb_image(file_path)
    x = preprocess(img)

    with torch.inference_mode():
        prob = torch.softmax(model(x), dim=1)[0].detach().cpu().numpy()

    pred_id = int(prob.argmax())
    pred_class = idx_to_class[pred_id]
    confidence = float(prob[pred_id])

    cam_tool = GradCAM(model, model.cam_layer)

    try:
        cam = cam_tool(x, pred_id)
    finally:
        cam_tool.close()

    overlay = overlay_gradcam(img, cam)

    overlay_name = f"gradcam_{uuid4().hex}.jpg"
    overlay_path = OUTPUT_DIR / overlay_name

    Image.fromarray(overlay).save(overlay_path, quality=88, optimize=True)
    clean_old_outputs()

    return {
        "class": pred_class,
        "confidence": confidence,
        "probabilities": {idx_to_class[i]: float(prob[i]) for i in range(len(prob))},
        "overlay_url": f"/static/outputs/{overlay_name}",
        "overlay_width": int(overlay.shape[1]),
        "overlay_height": int(overlay.shape[0]),
        "device": str(device)
    }