from ultralytics import YOLO
import numpy as np
import cv2
import os
import config
from loguru import logger
from typing import List, Dict, Tuple, Optional

# 模型配置
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "table", "best.pt")
CLASS_NAMES = ['table row', 'table column', 'table spanning cell']
IMG_SIZE = getattr(config, "TABLE_IMG_SIZE", 640)

# 初始化模型
device = config.GPUID if getattr(config, "GPU", False) else "cpu"
model = YOLO(MODEL_PATH, task='detect') if os.path.exists(MODEL_PATH) else None


def extract_table(img: np.ndarray,
                  detect_angle: bool = False,
                  angle_threshold: float = 0.5) -> Dict:
    """
    从图像中抽取表格结构
    
    Args:
        img: 输入图像 (numpy数组)
        detect_angle: 是否检测并校正角度，默认 False
        angle_threshold: 角度阈值（度），小于此值才进行校正，默认 0.5
        
    Returns:
        字典，包含以下字段：
            - rows: 行列表，每行包含该行的单元格
            - columns: 列列表，每列包含该列的单元格
            - cells: 所有单元格列表，每个单元格包含位置和内容信息
            - structure: 表格结构矩阵 (行x列)
            - processed_img: 预处理后的图像（如果进行了角度校正）
    """
    if model is None:
        logger.warning("表格模型未找到，返回空结构")
        return _empty_table_structure()

    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        logger.warning("输入图像无效")
        return _empty_table_structure()

    try:
        # 预处理图像（仅角度检测和校正）
        processed_img = img.copy()
        if detect_angle:
            processed_img = preprocess_table_image(
                processed_img,
                detect_angle=detect_angle,
                angle_threshold=angle_threshold
            )

        # 执行检测（使用预处理后的图像）
        results = model.predict(
            source=processed_img,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False,
            half=False,
        )

        if not results or len(results) == 0:
            return _empty_table_structure()

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return _empty_table_structure()

        # 分类检测结果
        rows, columns, cells = _classify_detections(boxes, result.names)
        
        # 调试信息：打印检测到的类别
        logger.debug(f"检测到: {len(rows)} 行, {len(columns)} 列, {len(cells)} 单元格")
        if len(cells) == 0 and len(rows) > 0 and len(columns) > 0:
            logger.debug(f"类别名称映射: {result.names}")
            # 如果没有检测到单元格，根据行和列的交集生成单元格
            cells = _generate_cells_from_rows_columns(rows, columns)

        # 组织表格结构
        table_structure = _organize_table_structure(rows, columns, cells)
        
        # 添加预处理后的图像到返回结果（如果进行了角度校正）
        if detect_angle:
            table_structure['processed_img'] = processed_img

        return table_structure

    except Exception as e:
        logger.error(f"表格抽取错误: {e}", exc_info=True)
        return _empty_table_structure()


def _classify_detections(boxes, names: Dict[int, str]) -> Tuple[List, List, List]:
    """
    将检测结果分类为行、列和单元格
    
    Args:
        boxes: YOLO检测结果
        names: 类别名称映射
        
    Returns:
        (rows, columns, cells) 三个列表
    """
    rows = []
    columns = []
    cells = []
    
    # 收集所有检测到的类别名称用于调试
    detected_classes = set()
    
    # 从 CLASS_NAMES 获取类别名称
    row_class = CLASS_NAMES[0] if len(CLASS_NAMES) > 0 else 'table row'
    column_class = CLASS_NAMES[1] if len(CLASS_NAMES) > 1 else 'table column'
    cell_class = CLASS_NAMES[2] if len(CLASS_NAMES) > 2 else 'table spanning cell'

    for box in boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = [int(v) for v in xyxy]

        class_name = names.get(cls, '')
        detected_classes.add(class_name)
        
        box_info = {
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
        }

        # 使用 CLASS_NAMES 进行精确匹配
        if class_name == row_class:
            rows.append(box_info)
        elif class_name == column_class:
            columns.append(box_info)
        elif class_name == cell_class:
            cells.append(box_info)
    
    if detected_classes:
        logger.debug(f"检测到的类别: {detected_classes}")
        logger.debug(f"期望的类别: row={row_class}, column={column_class}, cell={cell_class}")

    return rows, columns, cells


def _organize_table_structure(rows: List, columns: List, cells: List) -> Dict:
    """
    组织表格结构
    
    Args:
        rows: 行检测框列表
        columns: 列检测框列表
        cells: 单元格检测框列表（可选，如果为空则根据行列计算）
        
    Returns:
        表格结构字典
    """
    # 按位置排序
    rows_sorted = sorted(rows, key=lambda x: x['center'][1])
    columns_sorted = sorted(columns, key=lambda x: x['center'][0])

    # 如果没有检测到单元格，根据行列交集生成
    if not cells:
        cells = _generate_cells_from_rows_columns(rows_sorted, columns_sorted)

    # 构建行列表：每行的单元格根据该行与所有列的交集计算
    table_rows = []
    for row_idx, row_box in enumerate(rows_sorted):
        row_cells = []
        for col_idx, col_box in enumerate(columns_sorted):
            cell = _calculate_cell_from_row_column(row_box, col_box, row_idx, col_idx)
            if cell:
                row_cells.append(cell)
        table_rows.append({
            'bbox': row_box['bbox'],
            'cells': row_cells
        })

    # 构建列列表：每列的单元格根据该列与所有行的交集计算
    table_columns = []
    for col_idx, col_box in enumerate(columns_sorted):
        col_cells = []
        for row_idx, row_box in enumerate(rows_sorted):
            cell = _calculate_cell_from_row_column(row_box, col_box, row_idx, col_idx)
            if cell:
                col_cells.append(cell)
        table_columns.append({
            'bbox': col_box['bbox'],
            'cells': col_cells
        })

    # 构建单元格列表（带位置信息）
    cells_with_position = []
    for row_idx, row_box in enumerate(rows_sorted):
        for col_idx, col_box in enumerate(columns_sorted):
            cell = _calculate_cell_from_row_column(row_box, col_box, row_idx, col_idx)
            if cell:
                cells_with_position.append({
                    'bbox': cell['bbox'],
                    'confidence': cell.get('confidence', 1.0),
                    'row': row_idx,
                    'col': col_idx
                })

    # 构建结构矩阵
    structure = _build_structure_matrix(rows_sorted, columns_sorted, cells_with_position)

    return {
        'rows': table_rows,
        'columns': table_columns,
        'cells': cells_with_position,
        'structure': structure
    }


def _calculate_cell_from_row_column(row_box: Dict, col_box: Dict, row_idx: int, col_idx: int) -> Optional[Dict]:
    """
    根据行和列的交集计算单元格
    
    Args:
        row_box: 行检测框
        col_box: 列检测框
        row_idx: 行索引
        col_idx: 列索引
        
    Returns:
        单元格信息字典，如果无效则返回 None
    """
    # 计算交集区域
    cell_x1 = max(row_box['bbox'][0], col_box['bbox'][0])
    cell_y1 = max(row_box['bbox'][1], col_box['bbox'][1])
    cell_x2 = min(row_box['bbox'][2], col_box['bbox'][2])
    cell_y2 = min(row_box['bbox'][3], col_box['bbox'][3])
    
    # 确保是有效的交集
    if cell_x1 < cell_x2 and cell_y1 < cell_y2:
        return {
            'bbox': [cell_x1, cell_y1, cell_x2, cell_y2],
            'confidence': min(row_box.get('confidence', 1.0), col_box.get('confidence', 1.0)),
            'center': [(cell_x1 + cell_x2) / 2, (cell_y1 + cell_y2) / 2],
            'row': row_idx,
            'col': col_idx
        }
    
    return None


def _get_cell_position(cell: Dict, rows: List, columns: List) -> Tuple[int, int]:
    """
    确定单元格在表格中的位置（行索引和列索引）
    
    Returns:
        (row_idx, col_idx) 如果无法确定则返回 (-1, -1)
    """
    cell_center = cell['center']

    # 找到最接近的行
    row_idx = -1
    min_row_dist = float('inf')
    for idx, row in enumerate(rows):
        row_center_y = row['center'][1]
        dist = abs(cell_center[1] - row_center_y)
        if dist < min_row_dist:
            min_row_dist = dist
            row_idx = idx

    # 找到最接近的列
    col_idx = -1
    min_col_dist = float('inf')
    for idx, col in enumerate(columns):
        col_center_x = col['center'][0]
        dist = abs(cell_center[0] - col_center_x)
        if dist < min_col_dist:
            min_col_dist = dist
            col_idx = idx

    return row_idx, col_idx


def _build_structure_matrix(rows: List, columns: List, cells: List) -> List[List]:
    """
    构建表格结构矩阵
    
    Returns:
        二维列表，表示表格结构
    """
    if not rows or not columns:
        return []

    # 初始化矩阵
    structure = [[None for _ in range(len(columns))] for _ in range(len(rows))]

    # 填充单元格
    for cell in cells:
        row_idx = cell.get('row', -1)
        col_idx = cell.get('col', -1)
        if 0 <= row_idx < len(rows) and 0 <= col_idx < len(columns):
            structure[row_idx][col_idx] = {
                'bbox': cell['bbox'],
                'confidence': cell.get('confidence', 0.0)
            }

    return structure


def _boxes_overlap_vertically(box1: List, box2: List) -> bool:
    """检查两个框是否垂直重叠"""
    y1_min, y1_max = box1[0], box1[1]
    y2_min, y2_max = box2[0], box2[1]
    return not (y1_max < y2_min or y2_max < y1_min)


def _boxes_overlap_horizontally(box1: List, box2: List) -> bool:
    """检查两个框是否水平重叠"""
    x1_min, x1_max = box1[0], box1[1]
    x2_min, x2_max = box2[0], box2[1]
    return not (x1_max < x2_min or x2_max < x1_min)


def preprocess_table_image(img: np.ndarray,
                           detect_angle: bool = False,
                           angle_threshold: float = 0.5) -> np.ndarray:
    """
    预处理表格图像，进行角度检测和校正
    
    Args:
        img: 输入图像 (BGR格式或灰度图)
        detect_angle: 是否检测并校正角度，默认 False
        angle_threshold: 角度阈值（度），小于此值才进行校正，默认 0.5
        
    Returns:
        预处理后的图像
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return img

    # 角度检测和校正
    if detect_angle:
        angle = _detect_table_angle(img)
        if abs(angle) > angle_threshold:
            img = _rotate_image(img, -angle)
            logger.debug(f"检测到表格倾斜角度: {angle:.2f}度，已自动校正")

    return img


def _detect_table_angle(img: np.ndarray) -> float:
    """
    检测表格图像的倾斜角度
    
    Args:
        img: 输入图像
        
    Returns:
        倾斜角度（度），正值表示逆时针旋转
    """
    try:
        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 使用霍夫线变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) == 0:
            return 0.0

        # 计算角度
        angles = []
        for line in lines:
            rho, theta = line[0]
            # 转换为角度（度）
            angle = np.degrees(theta) - 90
            # 只考虑接近水平或垂直的线（-45到45度范围）
            if -45 <= angle <= 45:
                angles.append(angle)

        if not angles:
            return 0.0

        # 计算平均角度
        avg_angle = np.mean(angles)

        # 如果角度接近90度，可能是垂直线，需要调整
        if abs(avg_angle) > 45:
            avg_angle = avg_angle - 90 if avg_angle > 0 else avg_angle + 90

        return avg_angle

    except Exception as e:
        logger.error(f"角度检测错误: {e}", exc_info=True)
        return 0.0


def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    旋转图像
    
    Args:
        img: 输入图像
        angle: 旋转角度（度），正值表示逆时针旋转
        
    Returns:
        旋转后的图像
    """
    try:
        if abs(angle) < 0.1:
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 计算新的图像尺寸
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # 调整旋转矩阵的平移部分
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # 执行旋转
        rotated = cv2.warpAffine(img, rotation_matrix, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255) if len(img.shape) == 3 else 255)

        return rotated

    except Exception as e:
        logger.error(f"图像旋转错误: {e}", exc_info=True)
        return img


def _generate_cells_from_rows_columns(rows: List, columns: List) -> List:
    """
    根据行和列的交集生成单元格
    
    Args:
        rows: 行检测框列表
        columns: 列检测框列表
        
    Returns:
        单元格列表
    """
    cells = []
    
    if not rows or not columns:
        return cells
    
    # 按位置排序
    rows_sorted = sorted(rows, key=lambda x: x['center'][1])
    columns_sorted = sorted(columns, key=lambda x: x['center'][0])
    
    for row in rows_sorted:
        row_y1, row_y2 = row['bbox'][1], row['bbox'][3]
        for col in columns_sorted:
            col_x1, col_x2 = col['bbox'][0], col['bbox'][2]
            
            # 计算交集区域
            cell_x1 = max(row['bbox'][0], col_x1)
            cell_y1 = max(row_y1, col['bbox'][1])
            cell_x2 = min(row['bbox'][2], col_x2)
            cell_y2 = min(row_y2, col['bbox'][3])
            
            # 确保是有效的交集
            if cell_x1 < cell_x2 and cell_y1 < cell_y2:
                cell_info = {
                    'bbox': [cell_x1, cell_y1, cell_x2, cell_y2],
                    'confidence': min(row.get('confidence', 1.0), col.get('confidence', 1.0)),
                    'center': [(cell_x1 + cell_x2) / 2, (cell_y1 + cell_y2) / 2]
                }
                cells.append(cell_info)
    
    logger.debug(f"根据行和列生成了 {len(cells)} 个单元格")
    return cells


def draw_detection_result(img: np.ndarray, result: Dict,
                          row_color: Tuple[int, int, int] = (0, 255, 0),
                          column_color: Tuple[int, int, int] = (255, 0, 0),
                          thickness: int = 3,
                          show_labels: bool = True,
                          shadow_offset: int = 2,
                          inner_border: bool = True) -> np.ndarray:
    """
    在图像上绘制检测结果，只标记行和列（增强边框效果）
    
    Args:
        img: 输入图像
        result: extract_table 返回的结果字典
        row_color: 行的颜色 (B, G, R)，默认绿色
        column_color: 列的颜色 (B, G, R)，默认红色
        thickness: 线条粗细，默认 3
        show_labels: 是否显示标签，默认 True
        shadow_offset: 阴影偏移量，默认 2
        inner_border: 是否绘制内边框，默认 True
        
    Returns:
        绘制了检测结果的图像
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return img
    
    # 创建副本
    result_img = img.copy()
    
    # 如果是灰度图，转换为彩色
    if len(result_img.shape) == 2:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    
    rows = result.get('rows', [])
    columns = result.get('columns', [])
    
    def _draw_enhanced_rectangle(img, x1, y1, x2, y2, color, thickness, shadow_offset, inner_border):
        """绘制增强边框的矩形"""
        # 绘制阴影（外边框）
        shadow_color = (0, 0, 0)  # 黑色阴影
        cv2.rectangle(
            img,
            (x1 + shadow_offset, y1 + shadow_offset),
            (x2 + shadow_offset, y2 + shadow_offset),
            shadow_color,
            thickness + 2
        )
        
        # 绘制主边框（外边框，更粗）
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness + 1)
        
        # 绘制内边框（高亮效果）
        if inner_border:
            # 计算内边框位置（向内缩进）
            inner_offset = max(1, thickness // 2)
            inner_color = tuple(min(255, c + 50) for c in color)  # 更亮的颜色
            cv2.rectangle(
                img,
                (x1 + inner_offset, y1 + inner_offset),
                (x2 - inner_offset, y2 - inner_offset),
                inner_color,
                1
            )
    
    # 绘制行
    for idx, row in enumerate(rows):
        bbox = row.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            _draw_enhanced_rectangle(
                result_img, x1, y1, x2, y2,
                row_color, thickness, shadow_offset, inner_border
            )
            
            if show_labels:
                label = f"Row {idx}"
                # 计算文字大小
                font_scale = 0.7
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                # 在框的上方绘制文字背景（带阴影）
                label_y = y1 - text_height - baseline - 8
                # 文字阴影
                cv2.rectangle(
                    result_img,
                    (x1 + shadow_offset, label_y + shadow_offset),
                    (x1 + text_width + shadow_offset, y1 + shadow_offset),
                    (0, 0, 0),
                    -1
                )
                # 文字背景
                cv2.rectangle(
                    result_img,
                    (x1, label_y),
                    (x1 + text_width + 10, y1),
                    row_color,
                    -1
                )
                # 绘制文字
                cv2.putText(
                    result_img,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness
                )
    
    # 绘制列
    for idx, col in enumerate(columns):
        bbox = col.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            _draw_enhanced_rectangle(
                result_img, x1, y1, x2, y2,
                column_color, thickness, shadow_offset, inner_border
            )
            
            if show_labels:
                label = f"Col {idx}"
                # 计算文字大小
                font_scale = 0.7
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                # 在框的左侧绘制文字背景（带阴影）
                label_x = x1 + text_width + 10
                # 文字阴影
                cv2.rectangle(
                    result_img,
                    (x1 + shadow_offset, y1 + shadow_offset),
                    (label_x + shadow_offset, y1 + text_height + baseline + 5 + shadow_offset),
                    (0, 0, 0),
                    -1
                )
                # 文字背景
                cv2.rectangle(
                    result_img,
                    (x1, y1),
                    (label_x, y1 + text_height + baseline + 8),
                    column_color,
                    -1
                )
                # 绘制文字
                cv2.putText(
                    result_img,
                    label,
                    (x1 + 5, y1 + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness
                )
    
    return result_img


def crop_rows_and_columns(img: np.ndarray, result: Dict,
                          use_processed_img: bool = True,
                          padding: int = 0) -> Dict:
    """
    裁剪检测到的行和列
    
    Args:
        img: 原始输入图像
        result: extract_table 返回的结果字典
        use_processed_img: 是否使用预处理后的图像进行裁剪，默认 True
        padding: 裁剪时的边距（像素），默认 0
        
    Returns:
        字典，包含以下字段：
            - rows: 行图像列表，每个元素是裁剪后的行图像
            - columns: 列图像列表，每个元素是裁剪后的列图像
            - row_bboxes: 行边界框列表
            - column_bboxes: 列边界框列表
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return {
            'rows': [],
            'columns': [],
            'row_bboxes': [],
            'column_bboxes': []
        }
    
    # 选择使用的图像（预处理后的或原始的）
    crop_img = result.get('processed_img', img) if use_processed_img else img
    
    # 确保图像是有效的
    if crop_img is None or not isinstance(crop_img, np.ndarray) or crop_img.size == 0:
        crop_img = img
    
    rows_data = result.get('rows', [])
    columns_data = result.get('columns', [])
    
    cropped_rows = []
    cropped_columns = []
    row_bboxes = []
    column_bboxes = []
    
    # 裁剪行
    for idx, row in enumerate(rows_data):
        bbox = row.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            
            # 添加边距
            h, w = crop_img.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # 确保边界有效
            if x1 < x2 and y1 < y2:
                cropped_row = crop_img[y1:y2, x1:x2]
                if cropped_row.size > 0:
                    cropped_rows.append(cropped_row)
                    row_bboxes.append([x1, y1, x2, y2])
    
    # 裁剪列
    for idx, col in enumerate(columns_data):
        bbox = col.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            
            # 添加边距
            h, w = crop_img.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # 确保边界有效
            if x1 < x2 and y1 < y2:
                cropped_col = crop_img[y1:y2, x1:x2]
                if cropped_col.size > 0:
                    cropped_columns.append(cropped_col)
                    column_bboxes.append([x1, y1, x2, y2])
    
    return {
        'rows': cropped_rows,
        'columns': cropped_columns,
        'row_bboxes': row_bboxes,
        'column_bboxes': column_bboxes
    }


def save_cropped_images(cropped_data: Dict, output_dir: str, prefix: str = "table") -> List[str]:
    """
    保存裁剪后的行和列图像
    
    Args:
        cropped_data: crop_rows_and_columns 返回的字典
        output_dir: 输出目录路径
        prefix: 文件名前缀，默认 "table"
        
    Returns:
        保存的文件路径列表
    """
    import os
    from pathlib import Path
    
    saved_paths = []
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存行图像
    rows = cropped_data.get('rows', [])
    for idx, row_img in enumerate(rows):
        if row_img.size > 0:
            filename = f"{prefix}_row_{idx:03d}.png"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), row_img)
            saved_paths.append(str(filepath))
    
    # 保存列图像
    columns = cropped_data.get('columns', [])
    for idx, col_img in enumerate(columns):
        if col_img.size > 0:
            filename = f"{prefix}_column_{idx:03d}.png"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), col_img)
            saved_paths.append(str(filepath))
    
    return saved_paths


def _empty_table_structure() -> Dict:
    """返回空的表格结构"""
    return {
        'rows': [],
        'columns': [],
        'cells': [],
        'structure': []
    }


if __name__ == "__main__":
    img = cv2.imread(r"D:\ocr\ocrv5\ocr\code\images\stock_v1\line.png")
    r = extract_table(img, detect_angle=True)
    print(r)
    
    # 获取预处理后的图像（如果有）
    draw_img = r.get('processed_img', img)
    
    # 在预处理后的图像上绘制检测结果
    result_img = draw_detection_result(draw_img, r)
    
    # 保存结果图像
    output_path = r"D:\ocr\ocrv5\ocr\code\images\stock_v1\line_detected.png"
    cv2.imwrite(output_path, result_img)
    print(f"检测结果已保存到: {output_path}")
    
    # 裁剪行和列
    cropped_data = crop_rows_and_columns(img, r, use_processed_img=True, padding=5)
    print(f"裁剪到 {len(cropped_data['rows'])} 行, {len(cropped_data['columns'])} 列")
    
    # 保存裁剪后的图像
    output_dir = r"D:\ocr\ocrv5\ocr\code\images\stock_v1\cropped"
    saved_paths = save_cropped_images(cropped_data, output_dir, prefix="table")
    print(f"已保存 {len(saved_paths)} 张裁剪图像到: {output_dir}")
