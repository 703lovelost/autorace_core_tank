import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ament_index_python.packages import get_package_share_directory

from .shufflenet_v1 import ShuffleNet


TURN_RIGHT_CLASS_ID = 33
TURN_LEFT_CLASS_ID = 34

ID2LABEL = {
    TURN_RIGHT_CLASS_ID: "Turn right ahead",
    TURN_LEFT_CLASS_ID: "Turn left ahead",
}


def get_default_model_path():
    pkg_share = get_package_share_directory("autorace_core_tank")
    return os.path.join(pkg_share, "models", "pytorch_model.bin")


class SignClassifier:
    def __init__(self, model_path=None, device="cpu", input_size=32):
        self.device = torch.device(device)
        self.input_size = int(input_size)
        self.model_path = model_path or get_default_model_path()
        self.model = ShuffleNet(num_classes=43, groups=3)
        state = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def _to_tensor(self, bgr):
        if bgr is None:
            return None
        if bgr.size == 0:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0)
        return x

    @torch.inference_mode()
    def predict(self, bgr):
        x = self._to_tensor(bgr)
        if x is None:
            return None, None, 0.0
        x = x.to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        cls = int(torch.argmax(probs).item())
        score = float(probs[cls].item())
        label = ID2LABEL.get(cls, f"CLASS_{cls}")
        return cls, label, score
