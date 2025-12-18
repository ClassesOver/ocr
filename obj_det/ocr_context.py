from obj_det.vat_detect import invoice_detection as vat
from obj_det.stock_detect_v2 import stock_detection_v2 as stock_v2
from obj_det.stock_detect import stock_detection as stock_v1
from obj_det.bill_detect import bill_detection as bill
from settings import ocr_predict
from paddleocr import TextRecognition as _TextRecognition
from paddleocr import TableRecognitionPipelineV2 as _TableRecognitionPipelineV2
from paddleocr import PaddleOCRVL
from loguru import logger
import cv2
import numpy as np
import config
import platform
import os

class TextRecognition(_TextRecognition):

    def _get_extra_paddlex_predictor_init_args(self):
        res =  super()._get_extra_paddlex_predictor_init_args()
        res.update({
            'hpi_config': {
                'backend': 'onnxruntime',
            }
        })
        return res

class TableRecognitionPipelineV2(_TableRecognitionPipelineV2):

    def _get_extra_paddlex_predictor_init_args(self):
        res = super()._get_extra_paddlex_predictor_init_args()
        res.update({
            'hpi_config': {
                'backend': 'onnxruntime',
            }
        })
        return res



class TextOcrModel(object):
    def __init__(self):
        self._chinese_ocr_instance = ocr_predict
        
        # 优化配置：启用 GPU 和性能优化参数
        use_gpu = config.GPU if hasattr(config, 'GPU') else False
        
        # 根据操作系统决定是否启用 HPI（High Performance Inference）
        is_linux = platform.system().lower() == 'linux'

        # PaddleOCR 识别模型配置：支持通过配置/环境变量切换
        model_name = getattr(config, "PADDLE_REC_MODEL_NAME", "ch_SVTRv2_rec")
        
        # HPI 配置：通过环境变量控制（默认启用以获得更好性能）
        enable_hpi_env = os.getenv("PADDLE_ENABLE_HPI", "").strip().lower()
        enable_hpi = is_linux and enable_hpi_env not in ["0", "false", "no"]

        # 批处理配置：批量大小（可在 config 中覆盖）
        self._batch_size = getattr(config, "OCR_BATCH_SIZE", 16)  # 默认批量大小
        
        # MKLDNN 配置：Linux 平台下通过环境变量控制（默认：CPU 模式下启用）
        enable_mkldnn_env = os.getenv("PADDLE_ENABLE_MKLDNN", "").strip().lower()
        if enable_mkldnn_env:
            # 如果设置了环境变量，使用环境变量的值
            enable_mkldnn = is_linux and enable_mkldnn_env in ["1", "true", "yes"]
        else:
            # 默认：CPU 模式下启用，GPU 模式下禁用
            enable_mkldnn = not use_gpu and is_linux

        try:
            self._paddle_ocr_instance = TextRecognition(
                model_name=model_name,
                device='gpu' if use_gpu else 'cpu',
                enable_mkldnn=enable_mkldnn,
                enable_hpi=enable_hpi,
                cpu_threads = max(self._batch_size, 10),
            )
            logger.info(f"PaddleOCR 初始化成功: model={model_name}, device={'gpu' if use_gpu else 'cpu'}, hpi={enable_hpi}, mkldnn={enable_mkldnn}")
        except Exception as e:
            logger.error(f"PaddleOCR 初始化失败: {e}")
            # 如果失败，尝试禁用 HPI 和 MKLDNN 重新初始化
            self._paddle_ocr_instance = TextRecognition(
                model_name=model_name,
                device='gpu' if use_gpu else 'cpu',
                enable_mkldnn=False,
                enable_hpi=False,  # 失败后禁用 HPI
                cpu_threads = max(self._batch_size, 10)
            )
            logger.warning("已回退到默认配置（禁用 HPI 和 MKLDNN）")

        # 预热：避免首次调用的长延迟（不影响后续性能）
        try:
            warmup_img = np.zeros((32, 32, 3), dtype=np.uint8)
            _ = self._paddle_ocr_instance.predict(warmup_img)
        except Exception as e:
            logger.debug(f"PaddleOCR 预热失败: {e}")
        # self.table = TableRecognitionPipelineV2(device='gpu' if use_gpu else 'cpu',
        #                                         enable_mkldnn=enable_mkldnn,
        #                                         use_doc_unwarping=False,
        #                                         use_layout_detection=False,
        #                                         text_detection_model_name='PP-OCRv4_mobile_det',
        #                                         text_recognition_model_name='PP-OCRv4_mobile_rec',
        #                                         use_doc_orientation_classify=False,
        #                                         enable_hpi=enable_hpi, )

        self.ocr_vl = PaddleOCRVL(vl_rec_backend="native")

        self.ocr = self._ocr
        self.vat = vat
        self.stock_v1 = stock_v1
        self.stock_v2 = stock_v2
        # 默认入库单检测使用新版（药品），但仍保留别名以兼容调用
        self.stock = self.stock_v2
        self.bill = bill
        
        # 性能优化：图像预处理参数（可在 config 中覆盖）
        self._max_img_size = getattr(config, "OCR_MAX_IMG", 960)  # 最大尺寸
        self._min_img_size = getattr(config, "OCR_MIN_IMG", 32)   # 最小尺寸（过小不缩放）



    def _ocr(self, img, use_paddle_first=True, fallback=True):
        """
        混合使用 PaddleOCR 和 ChineseOCR
        
        Args:
            img: 输入图像
            use_paddle_first: 是否优先使用 PaddleOCR (默认 True)
            fallback: 如果第一个模型失败，是否回退到另一个模型 (默认 True)
            
        Returns:
            识别的文本字符串
        """
        try:
            if use_paddle_first:
                # 先尝试 PaddleOCR
                paddle_result = self._get_paddle_text(img)
                if paddle_result and paddle_result.strip():
                    return paddle_result

                # 如果 PaddleOCR 结果为空且允许回退，使用 ChineseOCR
                if fallback:
                    logger.debug("PaddleOCR 结果为空，回退到 ChineseOCR")
                    return self._chinese_ocr_instance(img)
                return paddle_result
            else:
                # 先尝试 ChineseOCR
                chinese_result = self._chinese_ocr_instance(img)
                if chinese_result and chinese_result.strip():
                    return chinese_result

                # 如果 ChineseOCR 结果为空且允许回退，使用 PaddleOCR
                if fallback:
                    logger.debug("ChineseOCR 结果为空，回退到 PaddleOCR")
                    return self._get_paddle_text(img)
                return chinese_result

        except Exception as e:
            logger.error(f"混合 OCR 识别错误: {e}")
            # 出错时尝试使用备用模型
            if fallback:
                try:
                    if use_paddle_first:
                        return self._chinese_ocr_instance(img)
                    else:
                        return self._get_paddle_text(img)
                except:
                    return ""
            return ""

    def _preprocess_image(self, img):
        """
        图像预处理，优化识别速度
        
        Args:
            img: 输入图像
            
        Returns:
            预处理后的图像
        """
        if img is None:
            return None
        
        # 检查是否为空数组
        if not isinstance(img, np.ndarray) or img.size == 0:
            return None
        
        h, w = img.shape[:2]
        
        # 如果图像过小，不处理，避免过度放大
        if h < self._min_img_size or w < self._min_img_size:
            return img

        # 如果图像过大，缩放以提升速度（默认限制 960，可通过 config 调整）
        max_dim = max(h, w)
        if max_dim > self._max_img_size:
            scale = self._max_img_size / max_dim
            new_h, new_w = int(h * scale), int(w * scale)
            # 使用 INTER_AREA 进行缩小，速度快且质量好
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 保证内存连续，提高底层推理效率
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        
        return img
    
    def _get_paddle_text(self, img):
        """
        PaddleOCR 辅助函数，提取文本内容（加速优化版）
        
        优化项：
        1. 图像预处理（缩放大图）
        2. 移除冗余日志
        3. 快速返回
        4. 简化异常处理
        """
        try:
            # 快速空值检查
            if img is None:
                return ""
            
            if not isinstance(img, np.ndarray) or img.size == 0:
                return ""
            
            # 图像预处理（加速）
            img = self._preprocess_image(img)
            if img is None:
                return ""
            
            # 执行识别
            result = self._paddle_ocr_instance.predict(img)
            
            # 快速返回结果（无冗余日志）
            if result and len(result) > 0 and 'rec_text' in result[0]:
                return result[0]['rec_text']
            
            return ""
            
        except Exception as e:
            return ""
    
    def _get_paddle_text_batch(self, images):
        """
        PaddleOCR 批量识别辅助函数（性能优化版）
        
        Args:
            images: 图像列表
            
        Returns:
            识别结果列表
            
        优化项：
        1. 使用 PaddleOCR 原生批处理能力
        2. 列表推导式优化
        3. 减少函数调用和检查
        4. 优化内存使用
        """
        # 快速空值检查
        if not images:
            return []
        
        try:
            # 批量预处理图像 - 使用列表推导式
            processed_data = [
                (idx, self._preprocess_image(img))
                for idx, img in enumerate(images)
                if img is not None and isinstance(img, np.ndarray) and img.size > 0
            ]
            
            # 如果没有有效图像，快速返回
            if not processed_data:
                return [""] * len(images)
            
            # 分离索引和处理后的图像
            valid_indices, processed_images = zip(*[
                (idx, img) for idx, img in processed_data if img is not None
            ])
            
            # 批量执行识别（PaddleOCR 原生支持）
            batch_results = self._paddle_ocr_instance.predict(list(processed_images))
            
            # 构建结果列表 - 使用字典映射优化查找
            result_map = {
                idx: batch_results[i].get('rec_text', '')
                for i, idx in enumerate(valid_indices)
                if i < len(batch_results)
            }
            
            # 返回完整结果列表
            return [result_map.get(i, '') for i in range(len(images))]
            
        except Exception as e:
            return [""] * len(images)
    
    def batch_ocr(self, images, use_paddle_first=True, batch_size=None):
        """
        批量 OCR 识别
        
        Args:
            images: 图像列表
            use_paddle_first: 是否优先使用 PaddleOCR (默认 True)
            batch_size: 批量大小，None 则使用默认配置（仅PaddleOCR使用）
            
        Returns:
            识别结果列表
            
        优化说明：
        - PaddleOCR: 使用原生批处理，显著提升吞吐量
        - ChineseOCR: 逐个处理图像
        """
        if not images:
            return []
        
        try:
            if use_paddle_first:
                # 使用 PaddleOCR 批处理
                effective_batch_size = batch_size if batch_size is not None else self._batch_size
                all_results = []
                
                for i in range(0, len(images), effective_batch_size):
                    batch = images[i:i + effective_batch_size]
                    batch_results = self._get_paddle_text_batch(batch)
                    all_results.extend(batch_results)
                
                return all_results
            else:
                # 使用 ChineseOCR（逐个处理）
                results = []
                for img in images:
                    try:
                        results.append(self._chinese_ocr_instance(img) if img is not None else "")
                    except:
                        results.append("")
                return results
                
        except Exception as e:
            logger.error(f"批量 OCR 识别错误: {e}")
            return [""] * len(images)
    
    def table_recognize(self, img, use_vl=True):
        """
        表格识别推理
        
        Args:
            img: 输入图像（numpy数组）
            use_vl: 是否使用OCR_VL模型识别（默认False，使用传统表格识别）
            
        Returns:
            返回字典，包含以下字段：
                - table_ocr_pred: 表格OCR预测结果
                - rows: 提取的行数据列表
                - raw_result: 原始识别结果
        """
        # 如果使用OCR_VL，调用VL识别方法
        if use_vl:
            return self.table_recognize_vl(img)
        
        try:
            # 快速空值检查
            if img is None:
                logger.warning("table_recognize: 输入图像为空")
                return {}
            
            if not isinstance(img, np.ndarray) or img.size == 0:
                logger.warning("table_recognize: 输入图像无效")
                return {}
            
            # 图像预处理（可选，根据需要调整）
            processed_img = self._preprocess_image(img)
            if processed_img is None:
                return {}
            
            # 执行表格识别
            logger.debug(f"开始表格识别，图像尺寸: {processed_img.shape}")
            result = self.table.predict(processed_img,
                                        use_doc_orientation_classify=False,
                                        use_e2e_wired_table_rec_model=True,
                                        use_doc_unwarping=False,
                                        use_layout_detection=False,
                                        use_table_orientation_classify=False)
            
            if not result or len(result) == 0:
                logger.warning("table_recognize: 表格识别结果为空")
                return {}
            
            # 提取table_ocr_pred
            table_res = result[0].get('table_res_list', [])
            if not table_res or len(table_res) == 0:
                logger.warning("table_recognize: table_res_list为空")
                return {}
            
            # 提取table_ocr_pred和行数据
            table_ocr_pred = table_res[0].get('table_ocr_pred', {})
            if not table_ocr_pred:
                logger.warning("table_recognize: table_ocr_pred为空")
                return {}
            
            # 返回结构化数据，包含table_ocr_pred和提取的行数据
            rows = self._extract_rows_from_table_ocr_pred(table_ocr_pred)
            return {
                'table_ocr_pred': table_ocr_pred,
                'rows': rows,
                'raw_result': result
            }
            
        except Exception as e:
            logger.error(f"表格识别错误: {e}", exc_info=True)
            return {}
    
    def table_recognize_vl(self, img):
        """
        使用OCR_VL模型进行表格识别
        
        Args:
            img: 输入图像（numpy数组）
            
        Returns:
            返回字典，包含以下字段：
                - text: OCR_VL识别的文本结果
                - rows: 提取的行数据列表
                - raw_result: 原始识别结果
        """
        try:
            # 快速空值检查
            if img is None:
                logger.warning("table_recognize_vl: 输入图像为空")
                return {}
            
            if not isinstance(img, np.ndarray) or img.size == 0:
                logger.warning("table_recognize_vl: 输入图像无效")
                return {}
            
            # 图像预处理
            processed_img = self._preprocess_image(img)
            if processed_img is None:
                return {}
            
            # 使用OCR_VL进行表格识别
            logger.debug(f"开始OCR_VL表格识别，图像尺寸: {processed_img.shape}")
            
            # OCR_VL通常使用prompt来指定任务
            prompt = "请识别这个表格，并以结构化格式返回表格内容。"
            
            # 调用OCR_VL模型（尝试多种可能的API调用方式）
            result = self.ocr_vl.predict(processed_img, prompt=prompt)
            
            if not result:
                logger.warning("table_recognize_vl: OCR_VL识别结果为空")
                return {}
            
            # 处理OCR_VL返回的结果
            # OCR_VL通常返回文本或结构化数据
            if isinstance(result, str):
                # 如果是字符串，尝试解析为表格结构
                text = result
                rows = self._parse_vl_table_text(text)
            elif isinstance(result, dict):
                # 如果是字典，直接使用
                text = result.get('text', '')
                rows = result.get('rows', [])
                if not rows and text:
                    rows = self._parse_vl_table_text(text)
            else:
                text = str(result)
                rows = self._parse_vl_table_text(text)
            
            # 返回结构化数据
            return {
                'text': text,
                'rows': rows,
                'raw_result': result
            }
            
        except Exception as e:
            logger.error(f"OCR_VL表格识别错误: {e}", exc_info=True)
            # 如果OCR_VL失败，可以回退到传统方法
            logger.warning("OCR_VL识别失败，尝试使用传统表格识别方法")
            try:
                return self.table_recognize(img, use_vl=False)
            except:
                return {}
    
    def _parse_vl_table_text(self, text):
        """
        解析OCR_VL返回的表格文本，提取行数据
        
        Args:
            text: OCR_VL返回的文本
            
        Returns:
            行数据列表
        """
        try:
            rows = []
            if not text:
                return rows
            
            # 按行分割
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 尝试按制表符或空格分割单元格
                # 可以根据实际OCR_VL返回格式调整
                cells = []
                if '\t' in line:
                    # 制表符分隔
                    cell_texts = line.split('\t')
                elif '|' in line:
                    # 管道符分隔（Markdown表格格式）
                    cell_texts = [cell.strip() for cell in line.split('|') if cell.strip()]
                else:
                    # 尝试按多个空格分割
                    cell_texts = [cell.strip() for cell in line.split('  ') if cell.strip()]
                    if len(cell_texts) == 1:
                        # 如果只有一个单元格，按单个空格分割
                        cell_texts = line.split(' ')
                
                # 构建单元格字典
                for idx, cell_text in enumerate(cell_texts):
                    cells.append({
                        'text': cell_text.strip(),
                        'col': idx
                    })
                
                if cells:
                    rows.append(cells)
            
            return rows
            
        except Exception as e:
            logger.error(f"解析OCR_VL表格文本错误: {e}", exc_info=True)
            return []
    
    def _convert_rows_to_html(self, rows):
        """
        将行数据转换为HTML表格格式
        
        Args:
            rows: 行数据列表
            
        Returns:
            HTML字符串
        """
        try:
            if not rows:
                return ""
            
            html_parts = ['<table>']
            for row in rows:
                html_parts.append('<tr>')
                for cell in row:
                    text = cell.get('text', '') if isinstance(cell, dict) else str(cell)
                    html_parts.append(f'<td>{text}</td>')
                html_parts.append('</tr>')
            html_parts.append('</table>')
            
            return ''.join(html_parts)
            
        except Exception as e:
            logger.error(f"转换行数据为HTML错误: {e}", exc_info=True)
            return ""
    
    def _extract_rows_from_table_ocr_pred(self, table_ocr_pred):
        """
        从table_ocr_pred中提取行数据
        
        Args:
            table_ocr_pred: 表格OCR预测结果，包含cells、rec_texts、bbox等信息
            
        Returns:
            行数据列表，每行是一个字典列表，包含该行的所有单元格信息
        """
        try:
            rows = []
            
            # 方法1: 如果有cells信息，直接使用
            if 'cells' in table_ocr_pred and table_ocr_pred['cells']:
                cells = table_ocr_pred['cells']
                # 按行索引分组
                row_dict = {}
                for cell in cells:
                    row_idx = cell.get('row', -1)
                    if row_idx not in row_dict:
                        row_dict[row_idx] = []
                    row_dict[row_idx].append(cell)
                
                # 按行索引排序，每行内按列索引排序
                for row_idx in sorted(row_dict.keys()):
                    row_cells = sorted(row_dict[row_idx], key=lambda x: x.get('col', 0))
                    rows.append(row_cells)
                return rows
            
            # 方法2: 如果有rec_texts和bbox，根据bbox的y坐标分组为行
            if 'rec_texts' in table_ocr_pred and 'bbox' in table_ocr_pred:
                rec_texts = table_ocr_pred['rec_texts']
                bboxes = table_ocr_pred['bbox']
                
                if not rec_texts or not bboxes or len(rec_texts) != len(bboxes):
                    return []
                
                # 创建单元格列表，包含文本和bbox
                cells = []
                for i, (text, bbox) in enumerate(zip(rec_texts, bboxes)):
                    # bbox格式可能是 [x1, y1, x2, y2] 或 [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    if isinstance(bbox, list) and len(bbox) > 0:
                        if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 2:
                            # 多边形格式，取最小y和最大y
                            y_coords = [point[1] for point in bbox if len(point) >= 2]
                            y_center = (min(y_coords) + max(y_coords)) / 2 if y_coords else 0
                        else:
                            # 矩形格式 [x1, y1, x2, y2]
                            if len(bbox) >= 4:
                                y_center = (bbox[1] + bbox[3]) / 2
                            else:
                                y_center = 0
                    else:
                        y_center = 0
                    
                    cells.append({
                        'text': text,
                        'bbox': bbox,
                        'y_center': y_center,
                        'index': i
                    })
                
                # 按y_center分组为行（允许一定误差）
                if not cells:
                    return []
                
                # 对y_center进行聚类分组
                cells_sorted = sorted(cells, key=lambda x: x['y_center'])
                rows_grouped = []
                current_row = [cells_sorted[0]]
                y_threshold = 10  # y坐标差异阈值，用于判断是否同一行
                
                for i in range(1, len(cells_sorted)):
                    cell = cells_sorted[i]
                    prev_cell = cells_sorted[i-1]
                    # 如果y坐标差异小于阈值，认为是同一行
                    if abs(cell['y_center'] - prev_cell['y_center']) < y_threshold:
                        current_row.append(cell)
                    else:
                        # 新行，先对当前行按x坐标排序
                        current_row_sorted = sorted(current_row, key=lambda x: self._get_bbox_x_center(x.get('bbox', [])))
                        rows_grouped.append(current_row_sorted)
                        current_row = [cell]
                
                # 处理最后一行
                if current_row:
                    current_row_sorted = sorted(current_row, key=lambda x: self._get_bbox_x_center(x.get('bbox', [])))
                    rows_grouped.append(current_row_sorted)
                
                return rows_grouped
            
            # 方法3: 如果只有rec_texts，每行一个文本（简单情况）
            if 'rec_texts' in table_ocr_pred:
                rec_texts = table_ocr_pred['rec_texts']
                if rec_texts:
                    return [[{'text': text} for text in rec_texts]]
            
            return []
            
        except Exception as e:
            logger.error(f"提取表格行数据错误: {e}", exc_info=True)
            return []
    
    def _get_bbox_x_center(self, bbox):
        """获取bbox的x中心坐标"""
        try:
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 1:
                    # 多边形格式，取最小x和最大x
                    x_coords = [point[0] for point in bbox if len(point) >= 1]
                    return (min(x_coords) + max(x_coords)) / 2 if x_coords else 0
                else:
                    # 矩形格式 [x1, y1, x2, y2]
                    if len(bbox) >= 4:
                        return (bbox[0] + bbox[2]) / 2
            return 0
        except:
            return 0
    
    def extract_table_rows(self, table_result):
        """
        从表格识别结果中提取行文本列表（便捷方法）
        
        Args:
            table_result: table_recognize返回的结果字典
            
        Returns:
            行文本列表，每行是一个文本列表（单元格文本）
        """
        try:
            if not isinstance(table_result, dict):
                return []
            
            rows = table_result.get('rows', [])
            if not rows:
                return []
            
            # 提取每行的文本
            row_texts = []
            for row in rows:
                if isinstance(row, list):
                    # 如果row是单元格列表
                    texts = []
                    for cell in row:
                        if isinstance(cell, dict):
                            text = cell.get('text', '')
                        elif isinstance(cell, str):
                            text = cell
                        else:
                            text = str(cell)
                        texts.append(text)
                    row_texts.append(texts)
                elif isinstance(row, (list, tuple)):
                    # 如果row直接是文本列表
                    row_texts.append([str(item) for item in row])
                else:
                    row_texts.append([str(row)])
            
            return row_texts
            
        except Exception as e:
            logger.error(f"提取表格行文本错误: {e}", exc_info=True)
            return []
    
    def batch_table_recognize(self, images, return_html=False):
        """
        批量表格识别推理

        Args:
            images: 图像列表
            return_html: 是否返回HTML格式（默认False）

        Returns:
            识别结果列表
        """
        if not images:
            return []

        results = []
        for idx, img in enumerate(images):
            try:
                logger.debug(f"处理第 {idx+1}/{len(images)} 张表格图像")
                result = self.table_recognize(img, return_html=return_html)
                results.append(result)
            except Exception as e:
                logger.error(f"批量表格识别第 {idx} 张图像失败: {e}")
                results.append("" if return_html else {})
        
        return results


context = TextOcrModel()
