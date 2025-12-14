# -*- coding: utf-8 -*-
"""
Paddle 文本识别示例（OpenVINO™）

参考：https://docs.openvino.ai/2024/notebooks/paddle-ocr-webcam-with-output.html#select-inference-device
本文件为自包含示例，不依赖仓库其他模块，演示如何加载 Paddle 文本识别
模型（已转换为 OpenVINO IR）并在指定设备上推理。
"""
import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

try:
    from openvino.runtime import Core, Layout
except Exception as exc:  # pragma: no cover - 便于在未安装 openvino 时给出友好提示
    Core = None
    Layout = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

# 官方字符字典下载地址（与 OpenVINO 教程一致）
DICT_URL = "https://raw.githubusercontent.com/WenmuZhou/PytorchOCR/master/torchocr/datasets/alphabets/ppocr_keys_v1.txt"
DEFAULT_DICT_PATH = Path(__file__).resolve().parent / "fonts" / "ppocr_keys_v1.txt"


def check_openvino_installed() -> bool:
    """检测 OpenVINO Runtime 是否已安装。"""
    return Core is not None


def download_file(url: str, target_path: Path) -> Path:
    """下载文件到指定路径（若已存在则直接返回）。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return target_path

    import urllib.request

    logging.info("下载资源：%s -> %s", url, target_path)
    with urllib.request.urlopen(url) as resp, open(target_path, "wb") as f:
        f.write(resp.read())
    return target_path


def load_charset(dict_path: Optional[Path]) -> List[str]:
    """加载字符字典，若缺失则退回默认数字+英文字母集合。"""
    if dict_path and dict_path.exists():
        with dict_path.open("r", encoding="utf-8") as f:
            chars = [line.strip() for line in f if line.strip()]
        if chars:
            return chars
    logging.warning("未找到字典文件，将使用默认字符集（数字+英文小写）。")
    return list("0123456789abcdefghijklmnopqrstuvwxyz")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax。"""
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def ctc_greedy_decode(
    probs: np.ndarray, charset: Sequence[str], blank_idx: int
) -> List[dict]:
    """简易 CTC 贪心解码，返回 [{'rec_text': str, 'rec_score': float}, ...]。"""
    indices = probs.argmax(axis=2)
    confs = probs.max(axis=2)
    results: List[dict] = []
    for idx_seq, conf_seq in zip(indices, confs):
        last = -1
        chars: List[str] = []
        scores: List[float] = []
        for idx, conf in zip(idx_seq, conf_seq):
            if idx == blank_idx or idx == last:
                last = idx
                continue
            if 0 <= idx < len(charset):
                chars.append(charset[idx])
                scores.append(float(conf))
            last = idx
        results.append(
            {
                "rec_text": "".join(chars),
                "rec_score": float(np.mean(scores)) if scores else 0.0,
            }
        )
    return results


def preprocess_image(img: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    """
    将原图按高度缩放并填充到固定宽度。
    image_shape 形如 (C, H, W)。
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("输入图像为空或格式不正确")

    if img.ndim == 2:  # 灰度转 BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_c, img_h, img_w = image_shape
    h, w = img.shape[:2]
    ratio = w / float(h + 1e-6)
    max_wh_ratio = img_w / img_h
    img_w_target = int(img_h * max_wh_ratio)
    resized_w = img_w_target if math.ceil(img_h * ratio) > img_w_target else int(
        math.ceil(img_h * ratio)
    )

    resized_image = cv2.resize(img, (resized_w, img_h))
    resized_image = resized_image.astype("float32")
    resized_image = resized_image.transpose((2, 0, 1)) / 255.0
    resized_image -= 0.5
    resized_image /= 0.5

    padding = np.zeros((img_c, img_h, img_w_target), dtype=np.float32)
    padding[:, :, :resized_w] = resized_image
    return np.expand_dims(padding, axis=0)


class PaddleTextRecognizer:
    """基于 OpenVINO 的 Paddle 文本识别推理器（自包含实现）。"""

    def __init__(
        self,
        model_xml: Path,
        device: str = "AUTO",
        dict_path: Optional[Path] = None,
    ) -> None:
        if Core is None:
            raise ImportError(
                "未检测到 openvino runtime，请先安装：pip install openvino"
            ) from _IMPORT_ERROR

        if not model_xml.exists():
            raise FileNotFoundError(
                f"未找到模型文件: {model_xml}\n请先将 Paddle 文本识别模型转换为 OpenVINO IR (xml/bin)"
            )

        self.core = Core()
        self.device = self._select_device(device)

        model = self.core.read_model(model=model_xml)
        if Layout is not None and model.inputs and model.inputs[0].get_layout().empty():
            model.inputs[0].set_layout(Layout("NCHW"))

        self.compiled_model = self.core.compile_model(model, device_name=self.device)
        self.output = self.compiled_model.output(0)
        self.input = self.compiled_model.input(0)

        # 解析输入 shape，若动态则回退到常见的 (3, 48, 320)
        try:
            partial = self.input.get_partial_shape()
            if partial.rank.is_static and partial.rank.get_length() >= 3:
                shape = partial.to_shape()
                if len(shape) >= 4:
                    self.image_shape = (shape[1], shape[2], shape[3])
                else:
                    self.image_shape = (3, 48, 320)
            else:
                self.image_shape = (3, 48, 320)
        except Exception:
            self.image_shape = (3, 48, 320)

        self.charset = load_charset(dict_path)
        self.blank_idx = len(self.charset)

        logging.info(
            "OpenVINO 模型已加载：%s | 设备=%s | 输入形状=%s",
            model_xml.name,
            self.device,
            self.image_shape,
        )

    def _select_device(self, preferred: str) -> str:
        """依据可用设备选择最优设备，逻辑参考官方教程的 AUTO 选择。"""
        preferred = (preferred or "AUTO").upper()
        available = {d.upper() for d in self.core.available_devices}
        if preferred == "AUTO" or preferred.startswith("AUTO:"):
            return preferred
        if preferred in available:
            return preferred
        if "GPU" in available:
            logging.warning("设备 %s 不可用，回退到 GPU", preferred)
            return "GPU"
        logging.warning("设备 %s 不可用，回退到 CPU", preferred)
        return "CPU"

    def predict(self, image: np.ndarray) -> List[dict]:
        """单张图片推理，返回 [{'rec_text': ..., 'rec_score': ...}]。"""
        inputs = preprocess_image(image, self.image_shape)
        outputs = self.compiled_model({self.input.any_name: inputs})[self.output]
        if outputs.ndim == 2:  # 兼容部分模型输出为 [T, C]
            outputs = np.expand_dims(outputs, axis=0)

        probs = softmax(outputs, axis=2)
        return ctc_greedy_decode(probs, self.charset, self.blank_idx)


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Paddle 文本识别 (OpenVINO) 示例"
    )
    parser.add_argument(
        "--model_xml",
        default=str(
            base_dir / "models" / "ch_PP-OCRv4_rec_infer" / "ch_PP-OCRv4_rec_infer.xml"
        ),
        help="OpenVINO IR 模型路径（*.xml），需与 *.bin 同目录",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="待识别的图片路径（建议为单行文字或裁剪后的文本区域）",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("OPENVINO_DEVICE", "AUTO"),
        help="推理设备，如 AUTO / CPU / GPU / NPU 等",
    )
    parser.add_argument(
        "--dict_path",
        default=str(DEFAULT_DICT_PATH),
        help="字符字典文件路径，若不存在会自动尝试下载官方字典",
    )
    parser.add_argument(
        "--save_vis",
        default=None,
        help="可选，保存可视化图片的路径（将在左上角写入识别文本）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not check_openvino_installed():
        logging.error("OpenVINO™ 未安装，请先执行: pip install openvino")
        sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        logging.error("未找到图片：%s", image_path)
        sys.exit(1)

    dict_path = Path(args.dict_path)
    try:
        dict_path = download_file(DICT_URL, dict_path)
    except Exception as exc:  # pragma: no cover - 网络不可用时退回默认字典
        logging.warning("字典下载失败，使用默认字典，原因：%s", exc)
        dict_path = None

    recognizer = PaddleTextRecognizer(
        model_xml=Path(args.model_xml), device=args.device, dict_path=dict_path
    )

    image = cv2.imread(str(image_path))
    if image is None:
        logging.error("无法读取图片：%s", image_path)
        sys.exit(1)

    start = time.time()
    results = recognizer.predict(image)
    elapsed = (time.time() - start) * 1000
    if not results:
        logging.warning("未获得识别结果")
        sys.exit(0)

    rec_text = results[0]["rec_text"]
    rec_score = results[0]["rec_score"]
    logging.info("识别结果：%s | score=%.4f | 用时=%.1f ms", rec_text, rec_score, elapsed)

    if args.save_vis:
        vis = image.copy()
        cv2.putText(
            vis,
            f"{rec_text} ({rec_score:.2f})",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        out_path = Path(args.save_vis)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
        logging.info("可视化结果已保存到：%s", out_path)


if __name__ == "__main__":
    main()