from ultralytics import YOLO
import cv2
import os
from PIL import Image
from loguru import logger

import config
from util.tool import get_amount, get_date, get_num, get_page, get_qrcode_data

# 药品入库单类别映射（与模型标签保持一致）
converter = {
    'title': 'title',
    'qrcode': 'qrcode',
    'supplier': 'supplier',
    'total': 'total',
    'idate': 'idate',
    'doc_number': 'doc_number',
    'total2': 'total2',
    'verified_by': 'verified_by',
    'accountant': 'accountant',
    'page': 'page',
    'line': 'line',
    'director': 'director',
    'purchaser': 'purchaser',
    'note': 'note',
    'rk_way': 'rk_way',
}

# 模型权重与输入尺寸
pub_weights = os.getenv("STOCK_V2_WEIGHTS", "models/stock_2/11n/best.onnx")
pub_img_size = getattr(config, "STOCK_V2_IMGSZ", 640)

# 初始化 YOLO 模型
device = config.GPUID if getattr(config, "GPU", False) else "cpu"
model = YOLO(pub_weights, task='detect')

# 仅检测不做 OCR 的标签
SKIP_OCR_LABELS = {'qrcode', 'director', 'purchaser', 'verified_by', 'accountant'}


def _process_label_text(label: str, text: str) -> str:
    """根据标签类型做简单后处理。"""
    if label in ('total', 'total2'):
        return get_amount(text)
    if label == 'idate':
        return get_date(text)
    if label == 'doc_number':
        return get_num(text)
    if label == 'page':
        return get_page(text)
    return text.strip()


def stock_detection_v2(img_numpy, stock=None, context=None, saveImage=False):
    """
    入库单检测与识别（西药/中药版本）。
    参考 vat_detect 的推理流程，使用 batch_ocr 提升吞吐。
    """
    if stock is None:
        stock = {}
    if context is None:
        return stock

    results = model.predict(
        source=img_numpy,
        imgsz=pub_img_size,
        device=device,
        verbose=False,
        half=False,
    )

    names = model.names
    im0 = img_numpy
    batch_ocr = context.batch_ocr

    detected = False

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            labels = {}
            label_confidences = {}  # 保存每个标签的置信度
            converter_keys = set(converter.keys())

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())  # 获取置信度
                label = names[cls]
                if label not in converter_keys:
                    continue

                x1, y1, x2, y2 = [int(v) for v in xyxy]
                if label == 'line':
                    # line标签边界容错处理
                    img_h, img_w = im0.shape[:2]
                    
                    # x方向：扩展到图像边界，但添加容错边距
                    margin_x = 5  # 左右边距容错（像素）
                    x1 = max(0, x1 - margin_x)  # 左边界，确保不小于0
                    x2 = min(img_w, x2 + margin_x)  # 右边界，确保不超过图像宽度
                    
                    # 如果检测框宽度太小，扩展到全宽（容错处理）
                    box_width = x2 - x1
                    min_width_ratio = 0.3  # 最小宽度比例阈值
                    if box_width < img_w * min_width_ratio:
                        # 检测框太小，扩展到全宽
                        x1 = 0
                        x2 = img_w
                    
                    # y方向：添加容错边距，但确保不超出图像范围
                    margin_y = 5  # 上下边距容错（像素）
                    y1 = max(0, y1 - margin_y)  # 上边界
                    y2 = min(img_h, y2 + margin_y)  # 下边界
                    
                    # 确保边界值有效
                    x1 = max(0, min(x1, img_w - 1))
                    x2 = max(x1 + 1, min(x2, img_w))
                    y1 = max(0, min(y1, img_h - 1))
                    y2 = max(y1 + 1, min(y2, img_h))

                if saveImage:
                    stock_fp = os.path.join('images', 'stock_v2')
                    os.makedirs(stock_fp, exist_ok=True)
                    path = os.path.join(stock_fp, f'{label}.png')
                    cv2.imwrite(path, im0[y1:y2, x1:x2])

                labels[label] = [y1, y2, x1, x2]
                label_confidences[label] = conf  # 保存置信度

            if labels:
                labels = {key: im0[v[0]:v[1], v[2]:v[3]] for key, v in labels.items()}

                # 批量 OCR，跳过无需 OCR 的标签
                ocr_label_keys = []
                ocr_images = []
                for label, img_region in labels.items():
                    if label in SKIP_OCR_LABELS:
                        continue
                    ocr_label_keys.append(label)
                    ocr_images.append(img_region)

                ocr_results_dict = {}
                if ocr_images:
                    batch_results = batch_ocr(ocr_images)
                    for label, text in zip(ocr_label_keys, batch_results):
                        ocr_results_dict[label] = text

                # 处理二维码
                if 'qrcode' in labels:
                    try:
                        qr_text = get_qrcode_data(Image.fromarray(labels['qrcode']))
                        if qr_text:
                            stock['qrcode'] = qr_text
                            stock['qrcode_conf'] = label_confidences.get('qrcode', 0.0)
                            detected = True
                    except Exception:
                        pass

                # 处理 OCR 结果
                for label, text in ocr_results_dict.items():
                    processed = _process_label_text(label, text)
                    stock[converter[label]] = processed
                    detected = True

                # SKIP_OCR_LABELS 中的标签：保存置信度
                for label in SKIP_OCR_LABELS:
                    if label in labels:
                        if label == 'qrcode':
                            # qrcode 已在上面处理
                            continue
                        stock[converter[label]] = 'detected'
                        stock[f'{converter[label]}_conf'] = label_confidences.get(label, 0.0)
                        detected = True

                # 处理 line 标签：进行表格识别
                if 'line' in labels:
                    try:
                        # 对line区域进行表格识别
                        line_img = labels['line']
                        table_result = context.table_recognize(line_img)

                        stock['line_conf'] = label_confidences.get('line', 0.0)
                        
                        # 从table_ocr_pred中提取行数据
                        if isinstance(table_result, dict):
                            rows = table_result.get('rows', [])
                            if rows:
                                # 提取行文本列表（便捷格式）
                                row_texts = context.extract_table_rows(table_result)
                                stock['line'] = row_texts

                        
                        detected = True
                    except Exception as e:
                        logger.error(f"表格识别错误: {e}")
                        # 如果表格识别失败，回退到简单标记
                        stock['line'] = 'detected'
                        stock['line_conf'] = label_confidences.get('line', 0.0)
                        detected = True

    if detected:
        # 补齐关键字段，方便上层消费
        for key in converter.values():
            if key in ('total', 'total2'):
                stock.setdefault(key, '¥ 0.00')
            else:
                stock.setdefault(key, "")
        stock.setdefault('total_amount', stock.get('total') or stock.get('total2') or '¥ 0.00')
        stock.setdefault('page', '-1/-1')
        stock['_stock_detected'] = True
        title = stock.get('title') or ''
        if '药' in title:
            stock['_stock_v2_detected'] = True
        else:
            stock['_stock_v2_detected'] = False
    return stock
