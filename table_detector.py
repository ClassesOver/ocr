#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
表格检测和处理模块
使用OpenCV检测图片中的有框表格，并对表格进行处理，用于后续OCR识别
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import math


class TableDetector:
    """表格检测器类"""
    
    def __init__(self, 
                 min_table_area: int = 5000,
                 horizontal_kernel_size: Tuple[int, int] = (40, 1),
                 vertical_kernel_size: Tuple[int, int] = (1, 40),
                 line_thickness: int = 3):
        """
        初始化表格检测器
        
        Args:
            min_table_area: 最小表格面积阈值，小于此值的区域将被忽略
            horizontal_kernel_size: 水平线检测的核大小 (height, width)
            vertical_kernel_size: 垂直线检测的核大小 (height, width)
            line_thickness: 线条厚度，用于形态学操作
        """
        self.min_table_area = min_table_area
        self.horizontal_kernel_size = horizontal_kernel_size
        self.vertical_kernel_size = vertical_kernel_size
        self.line_thickness = line_thickness
        
    def detect_tables(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的所有表格
        
        Args:
            image: 输入的BGR图像
            
        Returns:
            包含表格信息的列表，每个元素包含：
            - 'bbox': (x, y, w, h) 表格边界框
            - 'corners': 表格四个角点坐标
            - 'contour': 表格轮廓
            - 'area': 表格面积
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 预处理：二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 检测水平线
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.horizontal_kernel_size)
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
        
        # 检测垂直线
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.vertical_kernel_size)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=2)
        
        # 合并水平和垂直线
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        table_mask = cv2.erode(table_mask, (2, 2), iterations=1)
        table_mask = cv2.dilate(table_mask, (2, 2), iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_table_area:
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 使用approxPolyDP简化轮廓，找到表格的角点
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果找到的角点接近4个，认为是矩形表格
            if len(approx) >= 4:
                corners = self._order_points(approx.reshape(-1, 2))
                tables.append({
                    'bbox': (x, y, w, h),
                    'corners': corners,
                    'contour': contour,
                    'area': area
                })
            else:
                # 如果不是四边形，使用边界框
                corners = np.array([
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ], dtype=np.float32)
                tables.append({
                    'bbox': (x, y, w, h),
                    'corners': corners,
                    'contour': contour,
                    'area': area
                })
        
        # 按面积从大到小排序
        tables.sort(key=lambda x: x['area'], reverse=True)
        
        return tables
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        对四个点进行排序：左上、右上、右下、左下
        
        Args:
            pts: 形状为(4, 2)的点集
            
        Returns:
            排序后的点集
        """
        # 初始化排序后的点坐标
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # 计算点的和与差
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # 左上角点的和最小，右下角点的和最大
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # 右上角点的差最小，左下角点的差最大
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _calculate_skew_angle(self, corners: np.ndarray) -> float:
        """
        计算表格的倾斜角度
        
        Args:
            corners: 表格的四个角点坐标 (左上、右上、右下、左下)
            
        Returns:
            倾斜角度（度），正值表示逆时针倾斜
        """
        # 计算上边和下边的角度
        top_angle = np.arctan2(
            corners[1][1] - corners[0][1],
            corners[1][0] - corners[0][0]
        ) * 180 / np.pi
        
        bottom_angle = np.arctan2(
            corners[2][1] - corners[3][1],
            corners[2][0] - corners[3][0]
        ) * 180 / np.pi
        
        # 返回平均角度
        return (top_angle + bottom_angle) / 2
    
    def _detect_line_angle(self, image: np.ndarray) -> float:
        """
        通过霍夫变换检测表格线条的倾斜角度
        
        Args:
            image: 输入图像
            
        Returns:
            倾斜角度（度）
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # 收集接近水平的线条角度
        horizontal_angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            
            # 只考虑接近水平的线条（0度或180度附近）
            if angle < 10 or angle > 170:
                if angle > 90:
                    angle = angle - 180
                horizontal_angles.append(angle)
        
        if len(horizontal_angles) == 0:
            return 0.0
        
        # 返回中位数角度（更鲁棒）
        return np.median(horizontal_angles)
    
    def rotate_image(self, image: np.ndarray, angle: float, 
                    keep_size: bool = False) -> np.ndarray:
        """
        旋转图像以矫正倾斜
        
        Args:
            image: 输入图像
            angle: 旋转角度（度），正值表示逆时针旋转
            keep_size: 是否保持原始尺寸（True会裁剪，False会扩展画布）
            
        Returns:
            旋转后的图像
        """
        h, w = image.shape[:2]
        
        # 计算旋转中心
        center = (w // 2, h // 2)
        
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        if keep_size:
            # 保持原始尺寸
            rotated = cv2.warpAffine(image, M, (w, h), 
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
        else:
            # 计算新的图像尺寸以容纳完整的旋转图像
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # 调整旋转矩阵以考虑平移
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
        
        return rotated
    
    def straighten_table_borders(self, image: np.ndarray, 
                                method: str = 'hough',
                                angle_threshold: float = 0.5) -> Tuple[np.ndarray, float]:
        """
        矫正表格边框，使其保持水平和垂直
        
        Args:
            image: 输入图像
            method: 检测方法 ('hough' 霍夫变换, 'corners' 角点检测)
            angle_threshold: 角度阈值，小于此值则不旋转
            
        Returns:
            (矫正后的图像, 检测到的倾斜角度)
        """
        if method == 'hough':
            # 使用霍夫变换检测线条角度
            angle = self._detect_line_angle(image)
        else:
            # 使用角点检测
            tables = self.detect_tables(image)
            if len(tables) == 0:
                return image, 0.0
            angle = self._calculate_skew_angle(tables[0]['corners'])
        
        # 如果角度很小，不需要旋转
        if abs(angle) < angle_threshold:
            return image, angle
        
        # 旋转图像矫正倾斜
        straightened = self.rotate_image(image, -angle, keep_size=False)
        
        return straightened, angle
    
    def crop_table_borders(self, image: np.ndarray, 
                          border_size: int = 5,
                          auto_detect: bool = True) -> np.ndarray:
        """
        裁剪表格边缘多余的边框线条
        
        Args:
            image: 输入图像
            border_size: 裁剪边框大小（像素），仅在auto_detect=False时使用
            auto_detect: 是否自动检测并裁剪到内容区域
            
        Returns:
            裁剪后的图像
        """
        if image is None or image.size == 0:
            return image
        
        h, w = image.shape[:2]
        
        if not auto_detect:
            # 简单裁剪固定边框
            x1 = min(border_size, w // 2)
            y1 = min(border_size, h // 2)
            x2 = max(w - border_size, w // 2)
            y2 = max(h - border_size, h // 2)
            return image[y1:y2, x1:x2]
        
        # 自动检测内容区域
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 形态学操作，连接内容区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return image
        
        # 找到最大的轮廓（假设这是主要内容）
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        
        # 添加小的边距以保留边框
        margin = 3
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + w_box + margin)
        y2 = min(h, y + h_box + margin)
        
        # 裁剪图像
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    def remove_outer_borders(self, image: np.ndarray, 
                            border_thickness: int = 3) -> np.ndarray:
        """
        移除表格外围的边框线条，保留内部内容
        
        Args:
            image: 输入图像
            border_thickness: 要移除的边框线条厚度（像素）
            
        Returns:
            移除边框后的图像
        """
        if image is None or image.size == 0:
            return image
        
        h, w = image.shape[:2]
        
        # 创建掩码，标记边缘区域
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # 将边缘区域设为0（要移除的区域）
        mask[0:border_thickness, :] = 0  # 上边
        mask[h-border_thickness:h, :] = 0  # 下边
        mask[:, 0:border_thickness] = 0  # 左边
        mask[:, w-border_thickness:w] = 0  # 右边
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 将边缘区域设为白色
        result = image.copy()
        if len(result.shape) == 3:
            # 彩色图像
            result[mask == 0] = [255, 255, 255]
        else:
            # 灰度图像
            result[mask == 0] = 255
        
        # 裁剪掉边缘的空白
        return self.crop_table_borders(result, border_size=border_thickness, auto_detect=False)
    
    def clean_table_image(self, image: np.ndarray,
                         remove_borders: bool = True,
                         crop_content: bool = True,
                         border_thickness: int = 3) -> np.ndarray:
        """
        清理表格图像：移除外围边框并裁剪到内容区域
        
        Args:
            image: 输入图像
            remove_borders: 是否移除外围边框线条
            crop_content: 是否裁剪到内容区域
            border_thickness: 边框线条厚度
            
        Returns:
            清理后的图像
        """
        cleaned = image.copy()
        
        # 步骤1: 移除外围边框
        if remove_borders:
            cleaned = self.remove_outer_borders(cleaned, border_thickness)
        
        # 步骤2: 裁剪到内容区域
        if crop_content:
            cleaned = self.crop_table_borders(cleaned, auto_detect=True)
        
        return cleaned
    
    def extract_table(self, image: np.ndarray, corners: np.ndarray, 
                     output_size: Optional[Tuple[int, int]] = None,
                     auto_correct: bool = True,
                     angle_threshold: float = 2.0,
                     clean_borders: bool = False,
                     border_thickness: int = 3) -> np.ndarray:
        """
        提取并矫正表格区域
        
        Args:
            image: 原始图像
            corners: 表格的四个角点坐标 (左上、右上、右下、左下)
            output_size: 输出图像的尺寸 (width, height)，如果为None则自动计算
            auto_correct: 是否自动判断是否需要透视变换
            angle_threshold: 角度阈值（度），小于此值则认为表格已水平，直接裁剪
            clean_borders: 是否清理边缘的边框线条
            border_thickness: 边框线条厚度（像素）
            
        Returns:
            矫正后的表格图像
        """
        # 计算表格倾斜角度
        skew_angle = self._calculate_skew_angle(corners)
        
        # 如果表格基本水平，直接使用边界框裁剪
        if auto_correct and abs(skew_angle) < angle_threshold:
            # 获取最小外接矩形
            x_min = int(min(corners[:, 0]))
            x_max = int(max(corners[:, 0]))
            y_min = int(min(corners[:, 1]))
            y_max = int(max(corners[:, 1]))
            
            # 确保坐标在图像范围内
            h, w = image.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            # 直接裁剪
            cropped = image[y_min:y_max, x_min:x_max]
            
            # 清理边框（如果需要）
            if clean_borders:
                cropped = self.clean_table_image(cropped, border_thickness=border_thickness)
            
            return cropped
        
        # 表格明显倾斜，使用透视变换
        # 计算输出尺寸
        if output_size is None:
            # 计算表格的宽度和高度
            width = max(
                np.linalg.norm(corners[0] - corners[1]),
                np.linalg.norm(corners[2] - corners[3])
            )
            height = max(
                np.linalg.norm(corners[0] - corners[3]),
                np.linalg.norm(corners[1] - corners[2])
            )
            output_size = (int(width), int(height))
        
        # 定义目标点
        dst = np.array([
            [0, 0],
            [output_size[0] - 1, 0],
            [output_size[0] - 1, output_size[1] - 1],
            [0, output_size[1] - 1]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(corners, dst)
        
        # 应用透视变换
        warped = cv2.warpPerspective(image, M, output_size, flags=cv2.INTER_LINEAR)
        
        # 清理边框（如果需要）
        if clean_borders:
            warped = self.clean_table_image(warped, border_thickness=border_thickness)
        
        return warped
    
    def remove_table_lines(self, image: np.ndarray, 
                          horizontal_kernel_length: int = 30,
                          vertical_kernel_length: int = 30) -> np.ndarray:
        """
        去除表格线条，保留文字内容
        
        Args:
            image: 输入图像（灰度图或彩色图）
            horizontal_kernel_length: 水平线检测核长度
            vertical_kernel_length: 垂直线检测核长度
            
        Returns:
            去除表格线后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 检测水平线
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_length, 1))
        detected_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        
        # 检测垂直线
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_length))
        detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        
        # 合并检测到的线条
        table_lines = cv2.add(detected_horizontal, detected_vertical)
        
        # 膨胀线条使其完全覆盖原始线条
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_lines = cv2.dilate(table_lines, kernel_dilate, iterations=2)
        
        # 从原图中移除线条（将线条部分设为白色）
        result = gray.copy()
        result[table_lines == 255] = 255
        
        return result
    
    def enhance_text(self, image: np.ndarray, 
                    sharpen: bool = True,
                    denoise: bool = True) -> np.ndarray:
        """
        增强文字清晰度
        
        Args:
            image: 输入图像（灰度图）
            sharpen: 是否锐化
            denoise: 是否降噪
            
        Returns:
            增强后的图像
        """
        result = image.copy()
        
        # 降噪
        if denoise:
            result = cv2.fastNlMeansDenoising(result, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 锐化文字
        if sharpen:
            # 使用USM锐化（Unsharp Mask）
            gaussian = cv2.GaussianBlur(result, (0, 0), 2.0)
            result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)
        
        return result
    
    def preprocess_for_ocr(self, table_image: np.ndarray, 
                          method: str = 'enhanced',
                          remove_lines: bool = True,
                          enhance_contrast: bool = True,
                          sharpen_text: bool = True) -> np.ndarray:
        """
        对表格图像进行预处理，优化OCR识别效果（针对PaddleOCR优化）
        
        Args:
            table_image: 表格图像
            method: 预处理方法
                - 'enhanced': 增强模式（推荐用于PaddleOCR）- 去线+增强+二值化
                - 'adaptive': 自适应阈值
                - 'otsu': OTSU阈值
                - 'grayscale': 仅灰度化
                - 'simple': 简单二值化
            remove_lines: 是否移除表格线条
            enhance_contrast: 是否增强对比度
            sharpen_text: 是否锐化文字
                
        Returns:
            预处理后的图像
        """
        # 转换为灰度图
        if len(table_image.shape) == 3:
            gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = table_image.copy()
        
        # 增强模式（推荐用于PaddleOCR）
        if method == 'enhanced':
            # 步骤1: 去除表格线
            if remove_lines:
                gray = self.remove_table_lines(gray, 
                                              horizontal_kernel_length=40, 
                                              vertical_kernel_length=40)
            
            # 步骤2: 对比度增强
            if enhance_contrast:
                # 使用CLAHE增强对比度
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            
            # 步骤3: 文字增强
            if sharpen_text:
                gray = self.enhance_text(gray, sharpen=True, denoise=True)
            
            # 步骤4: 自适应二值化
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, 8
            )
            
        elif method == 'adaptive':
            # 自适应阈值
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
        elif method == 'otsu':
            # OTSU阈值
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == 'simple':
            # 简单二值化
            _, processed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        elif method == 'grayscale':
            # 仅灰度化，不二值化（PaddleOCR也支持灰度图）
            if enhance_contrast:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(gray)
            else:
                processed = gray
        else:
            processed = gray
        
        return processed
    
    def visualize_detection(self, image: np.ndarray, tables: List[Dict], 
                          show_angle: bool = True) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            tables: 检测到的表格列表
            show_angle: 是否显示倾斜角度
            
        Returns:
            绘制了检测框的图像
        """
        vis_image = image.copy()
        
        for i, table in enumerate(tables):
            # 绘制边界框
            x, y, w, h = table['bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制角点
            for j, corner in enumerate(table['corners']):
                cv2.circle(vis_image, tuple(corner.astype(int)), 5, (255, 0, 0), -1)
            
            # 计算并显示倾斜角度
            angle = self._calculate_skew_angle(table['corners'])
            
            # 标注表格编号和角度
            label = f'Table {i+1}'
            if show_angle:
                label += f' (angle: {angle:.2f}deg)'
            
            cv2.putText(vis_image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image


def detect_and_process_tables(image: np.ndarray, 
                              output_dir: Optional[str] = None,
                              preprocess_method: str = 'enhanced',
                              detector: Optional[TableDetector] = None,
                              straighten_borders: bool = False,
                              clean_borders: bool = False,
                              border_thickness: int = 3,
                              remove_lines: bool = True,
                              enhance_contrast: bool = True,
                              sharpen_text: bool = True) -> List[Dict]:
    """
    检测并处理图像中的表格（便捷函数）
    
    Args:
        image: 输入图像（numpy数组）或图像路径（字符串）
        output_dir: 输出目录，如果指定则保存处理后的表格图像
        preprocess_method: 预处理方法 ('enhanced', 'adaptive', 'otsu', 'grayscale', 'simple')
        detector: 可选的表格检测器实例，如果为None则创建新实例
        straighten_borders: 是否先矫正表格边框使其水平垂直
        clean_borders: 是否清理边缘的边框线条
        border_thickness: 边框线条厚度（像素）
        remove_lines: 是否移除表格线条（推荐用于OCR）
        enhance_contrast: 是否增强对比度
        sharpen_text: 是否锐化文字
        
    Returns:
        包含处理结果的列表，每个元素包含：
        - 'original_table': 原始提取的表格图像
        - 'processed_table': 预处理后的表格图像
        - 'bbox': 表格边界框 (x, y, w, h)
        - 'corners': 表格角点坐标
        - 'area': 表格面积
        - 'rotation_angle': 矫正的旋转角度（如果进行了矫正）
    """
    # 如果输入是字符串，则作为路径读取图像
    if isinstance(image, str):
        image_path = image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
    
    # 创建或使用提供的检测器
    if detector is None:
        detector = TableDetector()
    
    # 如果需要，先矫正边框
    rotation_angle = 0.0
    if straighten_borders:
        image, rotation_angle = detector.straighten_table_borders(image, method='hough')
    
    # 检测表格
    tables = detector.detect_tables(image)
    
    # 处理和保存表格
    results = []
    for i, table in enumerate(tables):
        # 提取表格
        table_image = detector.extract_table(
            image, 
            table['corners'],
            clean_borders=clean_borders,
            border_thickness=border_thickness
        )
        
        # 预处理
        processed_table = detector.preprocess_for_ocr(
            table_image, 
            method=preprocess_method,
            remove_lines=remove_lines,
            enhance_contrast=enhance_contrast,
            sharpen_text=sharpen_text
        )
        
        results.append({
            'original_table': table_image,
            'processed_table': processed_table,
            'bbox': table['bbox'],
            'corners': table['corners'],
            'area': table['area'],
            'rotation_angle': rotation_angle
        })
        
        # 保存结果
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, f'table_{i+1}_original.png'), table_image)
            cv2.imwrite(os.path.join(output_dir, f'table_{i+1}_processed.png'), processed_table)
    
    return results


def process_image_bytes(content: bytes,
                       preprocess_method: str = 'enhanced',
                       detector: Optional[TableDetector] = None,
                       straighten_borders: bool = False,
                       clean_borders: bool = False,
                       border_thickness: int = 3,
                       remove_lines: bool = True,
                       enhance_contrast: bool = True,
                       sharpen_text: bool = True) -> List[Dict]:
    """
    处理字节流图像中的表格（适配项目中的图像解码方式）
    
    Args:
        content: 图像字节流
        preprocess_method: 预处理方法
        detector: 可选的表格检测器实例
        straighten_borders: 是否先矫正表格边框使其水平垂直
        clean_borders: 是否清理边缘的边框线条
        border_thickness: 边框线条厚度（像素）
        remove_lines: 是否移除表格线条
        enhance_contrast: 是否增强对比度
        sharpen_text: 是否锐化文字
        
    Returns:
        包含处理结果的列表
    """
    # 解码图像（使用项目中常见的方式）
    np_arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("无法解码图像字节流")
    
    # 使用通用的检测和处理函数
    return detect_and_process_tables(
        image, 
        preprocess_method=preprocess_method, 
        detector=detector, 
        straighten_borders=straighten_borders,
        clean_borders=clean_borders, 
        border_thickness=border_thickness,
        remove_lines=remove_lines,
        enhance_contrast=enhance_contrast,
        sharpen_text=sharpen_text
    )


if __name__ == '__main__':
    # 示例用法
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # 检测并处理表格
        results = detect_and_process_tables(image_path, output_dir='output_tables', straighten_borders=True, clean_borders=True)
        
        print(f"检测到 {len(results)} 个表格")
        for i, result in enumerate(results):
            print(f"表格 {i+1}: 面积={result['area']}, 边界框={result['bbox']}")
    else:
        print("使用方法: python table_detector.py <image_path>")

