"""
推理引擎 - 负责模型推理和结果处理
扩展功能：GPU任务调度优化、内存管理优化、混合精度推理
"""
import os
import numpy as np
import time
from collections import deque
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QFont, QPolygonF
from PyQt5.QtCore import Qt, QPointF, QMutex, QWaitCondition
from typing import Optional, List, Dict, Any, Tuple, Callable

from utils.logger import LogManager
from utils.image_utils import ImageProcessor
from utils.error_handler import safe_execute
from config.settings import AppConfig


class GPUResourceManager:
    """GPU资源管理器 - 负责GPU内存管理和任务调度"""
    
    def __init__(self, logger: Optional[LogManager] = None):
        import torch
        self.logger = logger or LogManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPU内存状态
        self.memory_used = 0
        self.memory_peak = 0
        self.memory_limit = self._get_memory_limit()
        
        # 任务队列
        self.task_queue = deque()
        self.is_running = False
        self.current_task = None
        
        # 锁定机制
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # 统计信息
        self.total_tasks = 0
        self.completed_tasks = 0
        self.total_inference_time = 0
        
        # 自适应批大小
        self.optimal_batch_size = 8
        self.min_batch_size = 1
        self.max_batch_size = 32
        
        # 混合精度设置
        self.use_mixed_precision = False
        self.amp_enabled = False
        
        self.logger.info(f"GPU资源管理器初始化 - 设备: {self.device}")
        if self.device.type == 'cuda':
            self.logger.info(f"GPU内存限制: {self.memory_limit / 1024**3:.2f} GB")
    
    def _get_memory_limit(self) -> int:
        """获取GPU内存限制"""
        import torch
        if self.device.type == 'cuda':
            return torch.cuda.get_device_properties(self.device).total_memory
        return 8 * 1024**3  # 默认8GB

    def get_available_memory(self) -> int:
        """获取可用内存"""
        import torch
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            return torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)
        return self.memory_limit

    def update_memory_usage(self):
        """更新内存使用状态"""
        import torch
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            self.memory_used = torch.cuda.memory_allocated(self.device)
            self.memory_peak = max(self.memory_peak, self.memory_used)
    
    def estimate_batch_memory(self, batch_size: int, image_size: Tuple[int, int] = (640, 640)) -> int:
        """
        估算批处理所需内存
        
        Args:
            batch_size: 批大小
            image_size: 图像尺寸
            
        Returns:
            所需内存(字节)
        """
        # 粗略估算：每张640x640图像约需要10MB
        base_memory = batch_size * 10 * 1024**2
        # 添加额外开销
        overhead = base_memory * 0.5
        return int(base_memory + overhead)
    
    def calculate_optimal_batch_size(self, image_size: Tuple[int, int] = (640, 640), 
                                     safety_margin: float = 0.7) -> int:
        """
        计算最优批大小
        
        Args:
            image_size: 图像尺寸
            safety_margin: 安全边际(0-1)
            
        Returns:
            最优批大小
        """
        if self.device.type != 'cuda':
            return self.min_batch_size
        
        available_memory = self.get_available_memory() * safety_margin
        single_image_memory = self.estimate_batch_memory(1, image_size)
        
        if single_image_memory == 0:
            return self.min_batch_size
        
        max_possible = int(available_memory / single_image_memory)
        optimal_size = min(max(max_possible, self.min_batch_size), self.max_batch_size)
        
        # 动态调整
        self.optimal_batch_size = optimal_size
        return optimal_size
    
    def clear_cache(self):
        """清理GPU缓存"""
        import torch
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self.update_memory_usage()
            self.logger.info(f"GPU缓存已清理，当前使用: {self.memory_used / 1024**3:.2f} GB")
    
    def enable_mixed_precision(self):
        """启用混合精度推理"""
        if self.device.type == 'cuda':
            self.use_mixed_precision = True
            self.amp_enabled = True
            self.logger.info("已启用混合精度推理")
    
    def disable_mixed_precision(self):
        """禁用混合精度推理"""
        self.use_mixed_precision = False
        self.amp_enabled = False
        self.logger.info("已禁用混合精度推理")
    


class InferenceTask:
    """推理任务封装"""
    
    def __init__(self, task_id: str, model, inputs: List[str], 
                 callback: Optional[Callable] = None, **kwargs):
        self.task_id = task_id
        self.model = model
        self.inputs = inputs
        self.callback = callback
        self.kwargs = kwargs
        self.status = 'queued'  # queued, running, completed, failed
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """标记任务开始"""
        self.status = 'running'
        self.start_time = time.time()
    
    
    


class InferenceResult:
    """推理结果封装"""

    def __init__(self, raw_result, image_path: str, task: str):
        """
        初始化推理结果

        Args:
            raw_result: YOLO原始结果
            image_path: 图像路径
            task: 任务类型
        """
        self.raw_result = raw_result[0] if isinstance(raw_result, list) else raw_result
        self.image_path = image_path
        self.task = task

        # 提取结果组件
        self.boxes = self.raw_result.boxes if hasattr(self.raw_result, 'boxes') else None
        self.masks = self.raw_result.masks if hasattr(self.raw_result, 'masks') else None
        self.probs = self.raw_result.probs if hasattr(self.raw_result, 'probs') else None
        self.keypoints = self.raw_result.keypoints if hasattr(self.raw_result, 'keypoints') else None
        self.obb = self.raw_result.obb if hasattr(self.raw_result, 'obb') else None
        # YOLO26 新增任务结果
        self.semantic_mask = self.raw_result.semantic_mask if hasattr(self.raw_result, 'semantic_mask') else None
        self.depth = self.raw_result.depth if hasattr(self.raw_result, 'depth') else None

        # 类别名称
        self.names = self.raw_result.names if hasattr(self.raw_result, 'names') else {}

        # 原始图像尺寸
        self.orig_shape = self.raw_result.orig_shape if hasattr(self.raw_result, 'orig_shape') else (0, 0)

    def get_detection_summary(self) -> str:
        """
        获取检测摘要

        Returns:
            格式化的摘要文本
        """
        if self.task == 'classify':
            return self._get_classification_summary()
        elif self.task == 'pose':
            return self._get_pose_summary()
        elif self.task == 'obb':
            return self._get_obb_summary()
        elif self.task == 'semantic':
            return self._get_semantic_summary()
        elif self.task == 'depth':
            return self._get_depth_summary()
        else:
            return self._get_detection_summary()

    def _get_detection_summary(self) -> str:
        """获取检测/分割摘要"""
        if not self.boxes or len(self.boxes) == 0:
            return "未检测到任何目标"

        lines = []
        for i, box in enumerate(self.boxes):
            class_id = int(box.cls)
            class_name = self.names.get(class_id, f"Class_{class_id}")
            conf = box.conf.item()
            lines.append(f"  {i+1}. {class_name}: {conf:.3f}")

        task_name = "目标" if self.task == 'detect' else "实例"
        return f"检测到 {len(self.boxes)} 个{task_name}:\n" + "\n".join(lines)

    def _get_classification_summary(self) -> str:
        """获取分类摘要"""
        if not self.probs:
            return "无分类结果"

        probs_data = self.probs.data.cpu().numpy()
        top5_idx = np.argsort(probs_data)[-5:][::-1]

        lines = []
        for idx in top5_idx:
            class_name = self.names.get(idx, f"Class_{idx}")
            conf = probs_data[idx]
            lines.append(f"  {class_name}: {conf:.4f}")

        return "Top-5 分类结果:\n" + "\n".join(lines)

    def _get_pose_summary(self) -> str:
        """获取姿态估计摘要"""
        if not self.keypoints or len(self.keypoints) == 0:
            return "未检测到姿态"

        num_kpts = len(self.keypoints[0].data[0]) if hasattr(self.keypoints[0], 'data') else 0
        return f"检测到 {len(self.keypoints)} 个人体姿态，每个 {num_kpts} 个关键点"

    def _get_obb_summary(self) -> str:
        """获取旋转框检测摘要"""
        if not self.obb or len(self.obb) == 0:
            return "未检测到旋转目标"

        return f"检测到 {len(self.obb)} 个旋转目标"

    def _get_semantic_summary(self) -> str:
        """获取语义分割摘要"""
        import torch
        if self.semantic_mask is None or self.semantic_mask.data is None:
            return "无语义分割结果"

        mask = self.semantic_mask.data
        # 统计各类别像素数量
        unique, counts = torch.unique(mask, return_counts=True)
        total = mask.numel()
        lines = []
        for cls_id, count in zip(unique.cpu().numpy(), counts.cpu().numpy()):
            cls_name = self.names.get(int(cls_id), f"Class_{int(cls_id)}")
            ratio = count / total * 100
            lines.append(f"  {cls_name}({int(cls_id)}): {ratio:.1f}%")
        return f"语义分割完成，共 {len(unique)} 个类别:\n" + "\n".join(lines)

    def _get_depth_summary(self) -> str:
        """获取深度估计摘要"""
        if self.depth is None or self.depth.data is None:
            return "无深度估计结果"

        depth_data = self.depth.data
        valid = depth_data[depth_data > 0]
        if valid.numel() == 0:
            return "深度估计完成，但无有效深度值"

        return (f"深度估计完成，范围: {valid.min().item():.3f} ~ {valid.max().item():.3f} 米, "
                f"平均: {valid.mean().item():.3f} 米")

    def get_topk_classification(self, k: int = 5) -> List[Tuple[str, float, int]]:
        """
        获取Top-K分类结果

        Args:
            k: Top-K数量

        Returns:
            [(类别名, 置信度, 索引)] 列表
        """
        if not self.probs:
            return []

        probs_data = self.probs.data.cpu().numpy()
        topk_idx = np.argsort(probs_data)[-k:][::-1]

        results = []
        for idx in topk_idx:
            class_name = self.names.get(idx, f"Class_{idx}")
            conf = probs_data[idx]
            results.append((class_name, conf, idx))

        return results


class InferenceEngine:
    """推理引擎 - 支持GPU任务调度优化和内存管理优化"""

    def __init__(self, logger: Optional[LogManager] = None):
        """
        初始化推理引擎

        Args:
            logger: 日志管理器
        """
        self.logger = logger or LogManager()
        self.current_result: Optional[InferenceResult] = None

        # GPU资源管理器延迟创建（避免启动时导入 torch）
        self._gpu_manager = None
        
        # 批处理统计
        self.batch_stats = {
            'total_batches': 0,
            'total_images': 0,
            'total_time': 0.0,
            'avg_batch_time': 0.0,
            'avg_images_per_sec': 0.0
        }

        # 默认可视化配置
        self.vis_config = {
            'show_bbox': True,
            'show_label': True,
            'show_mask': True,
            'line_width': AppConfig.DEFAULT_LINE_WIDTH,
            'font_size': AppConfig.DEFAULT_FONT_SIZE,
            'bbox_color': AppConfig.DEFAULT_COLORS['bbox'],
            'label_color': AppConfig.DEFAULT_COLORS['label_text'],
            'mask_color': AppConfig.DEFAULT_COLORS['mask'],
            'mask_opacity': AppConfig.DEFAULT_MASK_OPACITY,
            'conf_threshold': AppConfig.DEFAULT_CONF_THRESHOLD
        }

    @property
    def gpu_manager(self) -> GPUResourceManager:
        """惰性创建 GPU 资源管理器，避免启动时导入 torch 造成卡顿"""
        if self._gpu_manager is None:
            self._gpu_manager = GPUResourceManager(self.logger)
            # 启用混合精度推理（如果GPU可用）
            if self._gpu_manager.device.type == 'cuda':
                self._gpu_manager.enable_mixed_precision()
                self.logger.info("推理引擎初始化完成，已启用混合精度推理")
        return self._gpu_manager

    @safe_execute("执行推理失败")
    def run_inference(self, model, image_path: str, **kwargs) -> Optional[InferenceResult]:
        """
        执行推理

        Args:
            model: 模型对象
            image_path: 图像路径
            **kwargs: 推理参数（conf, iou等）

        Returns:
            推理结果
        """
        self.logger.info(f"推理: {os.path.basename(image_path)}")

        # 合并配置
        infer_kwargs = {
            'verbose': False,
            'conf': self.vis_config['conf_threshold'],
            **kwargs
        }

        # 执行推理
        results = model(image_path, **infer_kwargs)

        # 获取任务类型
        task = 'detect'
        if hasattr(model, 'task') and model.task:
            task = model.task

        # 封装结果
        result = InferenceResult(results, image_path, task)
        self.current_result = result

        self.logger.info(f"推理完成，{result.get_detection_summary()}")
        return result

    @safe_execute("执行批处理推理失败")
    def run_batch_inference(self, model, image_paths: List[str], batch_size: int = None, 
                           enable_auto_batch: bool = True, **kwargs) -> List[Optional[InferenceResult]]:
        """
        执行批处理推理（优化版本）

        Args:
            model: 模型对象
            image_paths: 图像路径列表
            batch_size: 批处理大小，None表示自动计算
            enable_auto_batch: 是否启用自适应批大小
            **kwargs: 推理参数（conf, iou等）

        Returns:
            推理结果列表
        """
        import torch
        if not image_paths:
            self.logger.warning("没有图像需要处理")
            return []

        # 记录开始时间
        total_start_time = time.time()
        
        # 计算最优批大小
        if enable_auto_batch or batch_size is None:
            target_batch_size = self.gpu_manager.calculate_optimal_batch_size()
            if batch_size is not None:
                # 用户指定了批大小，但仍需要确保不超过可用内存
                target_batch_size = min(batch_size, target_batch_size)
        else:
            target_batch_size = batch_size if batch_size else 8
        
        self.logger.info(f"批处理推理: {len(image_paths)} 张图像，批大小: {target_batch_size}")
        self.logger.info(f"GPU设备: {self.gpu_manager.device}")
        if self.gpu_manager.device.type == 'cuda':
            self.logger.info(f"可用内存: {self.gpu_manager.get_available_memory() / 1024**3:.2f} GB")
        
        # 合并配置
        infer_kwargs = {
            'verbose': False,
            'conf': self.vis_config['conf_threshold'],
            **kwargs
        }

        results = []
        total_batches = 0
        total_processed = 0
        
        # 分批处理（带自适应调整）
        i = 0
        while i < len(image_paths):
            # 更新内存状态
            self.gpu_manager.update_memory_usage()
            
            # 根据内存使用动态调整批大小
            if enable_auto_batch and i > 0:
                available_memory = self.gpu_manager.get_available_memory()
                if available_memory < self.gpu_manager.memory_limit * 0.2:
                    # 内存不足，减小批大小
                    target_batch_size = max(self.gpu_manager.min_batch_size, target_batch_size // 2)
                    self.logger.warning(f"内存不足，调整批大小为: {target_batch_size}")
                    # 清理缓存
                    self.gpu_manager.clear_cache()
            
            batch = image_paths[i:i+target_batch_size]
            batch_start_time = time.time()
            
            try:
                # 执行批处理推理
                with torch.cuda.amp.autocast(enabled=self.gpu_manager.use_mixed_precision):
                    batch_results = model(batch, **infer_kwargs)
                
                # 封装结果
                for j, (img_path, res) in enumerate(zip(batch, batch_results)):
                    task = 'detect'
                    if hasattr(model, 'task') and model.task:
                        task = model.task
                    
                    result = InferenceResult([res], img_path, task)
                    results.append(result)
                
                batch_time = time.time() - batch_start_time
                images_per_sec = len(batch) / batch_time
                
                self.logger.info(f"批次 {total_batches + 1}: {len(batch)} 张图像, "
                                f"耗时 {batch_time:.2f}s, "
                                f"{images_per_sec:.2f} img/s")
                
                total_batches += 1
                total_processed += len(batch)
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    # OOM错误，减小批大小并重试
                    self.logger.error(f"GPU内存不足，尝试减小批大小: {e}")
                    self.gpu_manager.clear_cache()
                    
                    if target_batch_size > 1:
                        target_batch_size = max(1, target_batch_size // 2)
                        self.logger.info(f"调整批大小为: {target_batch_size}")
                        continue  # 重试当前批次
                    else:
                        # 批大小已为1仍失败
                        self.logger.error("批大小已为1，仍无法处理")
                        for img_path in batch:
                            results.append(None)
                else:
                    raise
            
            i += target_batch_size
        
        # 清理缓存
        self.gpu_manager.clear_cache()
        
        # 更新统计信息
        total_time = time.time() - total_start_time
        avg_images_per_sec = len(image_paths) / total_time if total_time > 0 else 0
        
        self.batch_stats.update({
            'total_batches': total_batches,
            'total_images': len(image_paths),
            'total_time': total_time,
            'avg_batch_time': total_time / total_batches if total_batches > 0 else 0,
            'avg_images_per_sec': avg_images_per_sec
        })
        
        # 更新GPU管理器统计
        self.gpu_manager.total_tasks += 1
        self.gpu_manager.completed_tasks += 1
        self.gpu_manager.total_inference_time += total_time
        
        self.logger.info(f"批处理推理完成，共处理 {len(results)} 张图像")
        self.logger.info(f"总耗时: {total_time:.2f}s, 平均 {avg_images_per_sec:.2f} img/s")
        
        return results


    @safe_execute("后处理结果失败")
    def postprocess_results(self, result: InferenceResult, **postprocess_kwargs) -> InferenceResult:
        """
        后处理推理结果

        Args:
            result: 推理结果
            **postprocess_kwargs: 后处理参数

        Returns:
            后处理后的推理结果
        """
        # 支持的后处理选项
        nms_threshold = postprocess_kwargs.get('nms_threshold', 0.45)
        
        # 对于检测、分割和旋转框任务，应用NMS
        if result.task in ['detect', 'segment', 'obb'] and result.boxes:
            # 提取边界框、置信度和类别
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            # 应用NMS
            keep_indices = self.non_max_suppression(boxes, confs, classes, nms_threshold)
            
            # 过滤结果
            if len(keep_indices) > 0:
                # 这里我们返回原始结果，实际应用中可能需要创建新的结果对象
                # 由于Ultralytics YOLO的结果对象结构复杂，这里仅记录日志
                self.logger.info(f"NMS后保留 {len(keep_indices)} 个检测结果")
            else:
                self.logger.info("NMS后无检测结果")
        
        return result

    def non_max_suppression(self, boxes, confs, classes, nms_threshold):
        """
        非极大值抑制

        Args:
            boxes: 边界框 [N, 4]
            confs: 置信度 [N]
            classes: 类别 [N]
            nms_threshold: NMS阈值

        Returns:
            保留的索引
        """
        import numpy as np
        
        # 按类别分组
        unique_classes = np.unique(classes)
        keep_indices = []
        
        for cls in unique_classes:
            # 获取当前类别的所有边界框
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            cls_confs = confs[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            if len(cls_boxes) == 0:
                continue
            
            # 按置信度排序
            sorted_indices = np.argsort(-cls_confs)
            cls_boxes = cls_boxes[sorted_indices]
            cls_indices = cls_indices[sorted_indices]
            
            # 应用NMS
            while len(cls_boxes) > 0:
                # 保留置信度最高的框
                keep_idx = cls_indices[0]
                keep_indices.append(keep_idx)
                
                # 计算与其他框的IoU
                iou = self.calculate_iou(cls_boxes[0], cls_boxes[1:])
                
                # 保留IoU小于阈值的框
                mask = iou < nms_threshold
                cls_boxes = cls_boxes[1:][mask]
                cls_indices = cls_indices[1:][mask]
        
        return keep_indices

    def calculate_iou(self, box, boxes):
        """
        计算IoU

        Args:
            box: 单个边界框 [4]
            boxes: 多个边界框 [N, 4]

        Returns:
            IoU值 [N]
        """
        import numpy as np
        
        # 计算交集
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算并集
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        # 计算IoU
        iou = intersection / union
        return iou

    @safe_execute("绘制结果失败")
    def draw_results(self, result: Optional[InferenceResult] = None,
                     config: Optional[dict] = None) -> Optional[QPixmap]:
        """
        绘制推理结果

        Args:
            result: 推理结果，None则使用当前结果
            config: 可视化配置，None则使用默认配置

        Returns:
            绘制后的QPixmap
        """
        if result is None:
            result = self.current_result

        if result is None:
            self.logger.warning("没有可绘制的推理结果")
            return None

        # 合并配置
        draw_config = {**self.vis_config, **(config or {})}

        # 语义分割 / 深度估计：直接使用 Ultralytics 的 plot() 渲染
        if result.task in ('semantic', 'depth'):
            try:
                if hasattr(result.raw_result, 'plot') and result.raw_result.plot:
                    plotted = result.raw_result.plot()
                    pixmap = ImageProcessor.cv_to_qpixmap(plotted)
                    if pixmap is not None and not pixmap.isNull():
                        return pixmap
            except Exception as e:
                self.logger.warning(f"语义/深度结果渲染失败，回退到原图: {e}")

        # 加载原始图像
        pixmap = ImageProcessor.load_image_safe(result.image_path)
        if pixmap is None or pixmap.isNull():
            self.logger.error(f"无法加载图像: {result.image_path}")
            return None

        painter = QPainter(pixmap)

        try:
            # 设置抗锯齿
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)

            # 根据任务类型绘制
            if result.task == 'classify':
                self._draw_classification_result(painter, result, draw_config)
            elif result.task == 'pose':
                self._draw_pose_result(painter, result, draw_config)
            elif result.task == 'obb':
                self._draw_obb_result(painter, result, draw_config)
            else:
                self._draw_detection_result(painter, result, draw_config)

        finally:
            painter.end()

        return pixmap

    def _draw_detection_result(self, painter: QPainter, result: InferenceResult,
                               config: dict):
        """绘制检测/分割结果"""
        if not result.boxes:
            return

        # 设置字体
        font = QFont()
        font_size = config.get('font_size', AppConfig.DEFAULT_FONT_SIZE)
        font_pt = max(8, int(painter.device().height() * (font_size / 1500.0)))
        font.setPointSize(font_pt)
        painter.setFont(font)

        fm = painter.fontMetrics()

        # 获取配置
        show_bbox = config.get('show_bbox', True)
        show_label = config.get('show_label', True)
        show_mask = config.get('show_mask', True) and result.masks is not None
        line_width = config.get('line_width', AppConfig.DEFAULT_LINE_WIDTH)
        bbox_color = config.get('bbox_color', QColor("red"))
        label_color = config.get('label_color', QColor("white"))
        mask_color = config.get('mask_color', QColor("red"))
        mask_opacity = config.get('mask_opacity', AppConfig.DEFAULT_MASK_OPACITY)

        for i in range(len(result.boxes)):
            try:
                class_id = int(result.boxes.cls[i])
                confidence = result.boxes.conf[i].item()

                # 置信度过滤
                if confidence < config.get('conf_threshold', 0.25):
                    continue

                # 绘制蒙版
                if show_mask:
                    self._draw_mask(painter, result, i, mask_color, mask_opacity)

                # 获取边界框
                x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy()

                # 绘制边界框
                if show_bbox:
                    pen = QPen(bbox_color, line_width)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                # 绘制标签
                if show_label:
                    label = f"{result.names[class_id]} {confidence:.2f}"
                    text_width = fm.horizontalAdvance(label) + 8
                    text_height = fm.height() + 4

                    # 标签位置
                    text_x = int(x1)
                    text_y = int(y1) - text_height
                    if text_y < 0:
                        text_y = int(y1)

                    # 标签背景
                    painter.setBrush(bbox_color)
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(text_x, text_y, text_width, text_height)

                    # 标签文本
                    painter.setPen(label_color)
                    painter.drawText(text_x + 4, text_y + fm.ascent() + 2, label)

            except Exception as e:
                self.logger.error(f"绘制第 {i} 个目标时出错: {e}")
                continue

    def _draw_mask(self, painter: QPainter, result: InferenceResult, idx: int, mask_color: QColor, opacity: int):
        """绘制蒙版"""
        try:
            import cv2
            if not result.masks:
                return

            # 调整蒙版大小
            img_width = painter.device().width()
            img_height = painter.device().height()

            # 尝试多种方式获取蒙版数据
            if hasattr(result.masks, 'xy') and result.masks.xy is not None:
                # YOLOv8/11 格式：masks.xy 包含多边形坐标
                if idx < len(result.masks.xy):
                    polygons = result.masks.xy[idx]
                    if polygons is not None and len(polygons) > 0:
                        polygon = QPolygonF()
                        for point in polygons:
                            # 转换坐标到图像尺寸
                            x = float(point[0]) * (img_width / result.orig_shape[1])
                            y = float(point[1]) * (img_height / result.orig_shape[0])
                            polygon.append(QPointF(x, y))

                        if polygon.size() >= 3:
                            painter.setBrush(QColor(
                                mask_color.red(), mask_color.green(),
                                mask_color.blue(), opacity
                            ))
                            painter.setPen(Qt.NoPen)
                            painter.drawPolygon(polygon)
            elif hasattr(result.masks, 'data') and result.masks.data is not None:
                # 原始格式：使用mask.data
                mask_tensor = result.masks.data[idx]

                if hasattr(mask_tensor, 'cpu'):
                    mask_np = mask_tensor.cpu().numpy()
                else:
                    mask_np = mask_tensor.numpy()

                if len(mask_np.shape) > 2:
                    mask_np = mask_np.squeeze()

                mask_resized = cv2.resize(
                    mask_np, (img_width, img_height),
                    interpolation=cv2.INTER_LINEAR
                )

                # 二值化
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

                # 形态学操作
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
                mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)

                # 查找轮廓
                contours, _ = cv2.findContours(
                    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

                    polygon = QPolygonF()
                    for point in simplified:
                        polygon.append(QPointF(float(point[0][0]), float(point[0][1])))

                    if polygon.size() >= 3:
                        painter.setBrush(QColor(
                            mask_color.red(), mask_color.green(),
                            mask_color.blue(), opacity
                        ))
                        painter.setPen(Qt.NoPen)
                        painter.drawPolygon(polygon)

        except Exception as e:
            self.logger.error(f"绘制蒙版出错: {e}")

    def _draw_classification_result(self, painter: QPainter, result: InferenceResult,
                                    config: dict):
        """绘制分类结果"""
        topk_results = result.get_topk_classification(k=config.get('topk', 5))

        if not topk_results:
            return

        # 设置字体
        font = QFont("Arial", config.get('label_font_size', 12))
        font.setBold(True)
        painter.setFont(font)

        fm = painter.fontMetrics()
        line_height = fm.height() + 8

        # 颜色设置
        label_color = config.get('cls_label_color', QColor("yellow"))
        bg_opacity = config.get('bg_opacity', 150)

        y_offset = 20

        for i, (class_name, conf, _) in enumerate(topk_results):
            # 构建标签文本
            if config.get('show_confidences', True):
                text = f"{i+1}. {class_name}: {conf:.3f}"
            else:
                text = f"{i+1}. {class_name}"

            # 计算文本宽度
            text_width = fm.horizontalAdvance(text) + 20

            # 设置背景颜色（根据排名渐变）
            intensity = 200 - i * 30
            bg_color = QColor(
                min(255, label_color.red() + intensity),
                min(255, label_color.green() + intensity),
                min(255, label_color.blue() + intensity),
                bg_opacity
            )

            # 绘制背景
            bg_rect = painter.window()  # 修正：使用QRect
            from PyQt5.QtCore import QRect
            bg_rect = QRect(10, y_offset, text_width, line_height)
            painter.setBrush(bg_color)
            painter.setPen(QPen(QColor(50, 50, 50), 1))
            painter.drawRect(bg_rect)

            # 绘制文本
            text_color = QColor("black") if bg_color.lightness() > 128 else QColor("white")
            painter.setPen(text_color)
            painter.drawText(15, y_offset + fm.ascent() + 4, text)

            y_offset += line_height + 5

            if y_offset > painter.device().height() - 50:
                break

    def _draw_pose_result(self, painter: QPainter, result: InferenceResult,
                          config: dict):
        """绘制姿态估计结果"""
        # 简化实现，实际需要根据关键点绘制骨架
        self._draw_detection_result(painter, result, config)

    def _draw_obb_result(self, painter: QPainter, result: InferenceResult,
                         config: dict):
        """绘制旋转框检测结果"""
        if not result.obb:
            return

        # 设置字体
        font = QFont()
        font_size = config.get('font_size', AppConfig.DEFAULT_FONT_SIZE)
        font_pt = max(8, int(painter.device().height() * (font_size / 1500.0)))
        font.setPointSize(font_pt)
        painter.setFont(font)

        fm = painter.fontMetrics()

        # 获取配置
        show_bbox = config.get('show_bbox', True)
        show_label = config.get('show_label', True)
        line_width = config.get('line_width', AppConfig.DEFAULT_LINE_WIDTH)
        bbox_color = config.get('bbox_color', QColor(0, 162, 232))  # 蓝色
        label_color = config.get('label_color', QColor(255, 255, 255))

        for i in range(len(result.obb)):
            try:
                # 获取旋转框数据
                obb_data = result.obb[i]
                class_id = int(obb_data.cls)
                confidence = obb_data.conf.item()

                # 置信度过滤
                if confidence < config.get('conf_threshold', 0.25):
                    continue

                # 获取旋转框的四个顶点
                if hasattr(obb_data, 'xyxyxyxy'):
                    # YOLOv8/11 OBB格式: 8个坐标点 (x1,y1,x2,y2,x3,y3,x4,y4)
                    points = obb_data.xyxyxyxy[0].cpu().numpy()
                    if len(points) == 8:
                        # 转换为QPointF列表
                        qpoints = []
                        for j in range(0, 8, 2):
                            x = float(points[j])
                            y = float(points[j+1])
                            qpoints.append(QPointF(x, y))

                        if len(qpoints) == 4:
                            # 绘制旋转框
                            if show_bbox:
                                pen = QPen(bbox_color, line_width)
                                painter.setPen(pen)
                                painter.setBrush(Qt.NoBrush)
                                
                                # 绘制四边形
                                for j in range(4):
                                    start_point = qpoints[j]
                                    end_point = qpoints[(j + 1) % 4]
                                    painter.drawLine(start_point, end_point)

                            # 绘制标签
                            if show_label:
                                class_name = result.names.get(class_id, f"Class_{class_id}")
                                label = f"{class_name} {confidence:.2f}"
                                text_width = fm.horizontalAdvance(label) + 8
                                text_height = fm.height() + 4

                                # 计算标签位置（使用第一个点作为起点）
                                text_x = int(qpoints[0].x())
                                text_y = int(qpoints[0].y()) - text_height
                                if text_y < 0:
                                    text_y = int(qpoints[0].y())

                                # 标签背景
                                painter.setBrush(bbox_color)
                                painter.setPen(Qt.NoPen)
                                painter.drawRect(text_x, text_y, text_width, text_height)

                                # 标签文本
                                painter.setPen(label_color)
                                painter.drawText(text_x + 4, text_y + fm.ascent() + 2, label)

            except Exception as e:
                self.logger.error(f"绘制第 {i} 个旋转目标时出错: {e}")
                continue


    
    # ========== GPU管理方法 ==========
    
    
    
    
    def enable_mixed_precision(self):
        """启用混合精度推理"""
        self.gpu_manager.enable_mixed_precision()
    
    def disable_mixed_precision(self):
        """禁用混合精度推理"""
        self.gpu_manager.disable_mixed_precision()
    
    
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        import torch
        device = self.gpu_manager.device
        info = {
            'device_type': device.type,
            'is_cuda': device.type == 'cuda',
            'device_name': '',
            'total_memory_gb': 0.0,
            'available_memory_gb': 0.0
        }
        
        if device.type == 'cuda':
            props = torch.cuda.get_device_properties(device)
            info['device_name'] = props.name
            info['total_memory_gb'] = props.total_memory / 1024**3
            info['available_memory_gb'] = self.gpu_manager.get_available_memory() / 1024**3
        
        return info
