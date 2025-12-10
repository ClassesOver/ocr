from obj_det.vat_detect import invoice_detection as vat
# from obj_det.v1.stock_detect import stock_detection_image as stock
from settings import ocr_predict
from paddleocr import TextRecognition
from loguru import logger
import cv2
import numpy as np
import config
import platform
import os



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
        enable_hpi_env = os.getenv("PADDLE_ENABLE_HPI", "1").strip().lower()
        enable_hpi = is_linux and enable_hpi_env not in ["0", "false", "no"]
        
        # MKLDNN 配置：Linux 平台下通过环境变量控制（默认：CPU 模式下启用）
        enable_mkldnn_env = os.getenv("PADDLE_ENABLE_MKLDNN", "1").strip().lower()
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
            )
            logger.warning("已回退到默认配置（禁用 HPI 和 MKLDNN）")

        # 预热：避免首次调用的长延迟（不影响后续性能）
        try:
            warmup_img = np.zeros((32, 32, 3), dtype=np.uint8)
            _ = self._paddle_ocr_instance.predict(warmup_img)
        except Exception as e:
            logger.debug(f"PaddleOCR 预热失败: {e}")
        
        self.ocr = self._ocr
        self.vat = vat
        # self.stock = stock
        
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
        if img is None or img.size == 0:
            return None
        
        h, w = img.shape[:2]
        
        
        # 如果图像过小，不处理，避免过度放大
        if h < self._min_img_size or w < self._min_img_size:
            return img

        # 如果图像过大，缩放以提升速度（默认限制 960，可通过 config 调整）
        if max(h, w) > self._max_img_size:
            scale = self._max_img_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
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
            if img is None or img.size == 0:
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
            # 精简异常处理，仅在调试模式记录详细信息
            if logger.level <= 10:  # DEBUG 级别
                logger.debug(f"PaddleOCR 识别错误: {e}")
            return ""


context = TextOcrModel()
