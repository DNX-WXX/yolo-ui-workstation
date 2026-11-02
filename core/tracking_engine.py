"""
跟踪引擎 - 支持视频对象跟踪
基于Ultralytics YOLO的跟踪模式
支持ByteTrack、DeepSORT等跟踪算法
"""
import os
import cv2
import time
from typing import Optional, List, Dict, Any
from collections import deque
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QFont

from utils.logger import LogManager
from utils.error_handler import safe_execute


class TrackingResult:
    """跟踪结果封装"""

    def __init__(self, frame_index: int, boxes_data: List[Dict], frame_time: float):
        """
        Args:
            frame_index: 帧索引
            boxes_data: 边界框数据列表，包含track_id
            frame_time: 处理时间
        """
        self.frame_index = frame_index
        self.boxes_data = boxes_data
        self.frame_time = frame_time
        self.tracked_objects = {}  # track_id -> object info
        self.frame_image = None    # 原始帧图像 (numpy BGR)，用于绘制
        self.image_path = None     # 图像来源路径

    def get_track_summary(self) -> str:
        """获取跟踪摘要"""
        track_ids = set(box.get('track_id', 'N/A') for box in self.boxes_data)
        return f"帧 {self.frame_index}: 跟踪到 {len(track_ids)} 个目标"


class TrackingEngine:
    """跟踪引擎 - 支持视频对象跟踪"""

    def __init__(self, logger: Optional[LogManager] = None):
        self.logger = logger or LogManager()
        self.tracker = None
        self.tracking_history = {}  # track_id -> [(x, y, time), ...]
        self.max_history_length = 50
        self.current_result: Optional[TrackingResult] = None

        # 跟踪配置
        self.track_config = {
            'tracker': 'bytetrack.yaml',  # 跟踪器类型
            'confidence': 0.25,
            'iou_threshold': 0.45,
            'max_frame_skip': 5,
            'show_labels': True,
            'show_track_id': True,
            'show_trajectory': True,
            'trajectory_length': 20,
            'color_by_track': True
        }

        # 性能统计
        self.track_stats = {
            'total_frames': 0,
            'total_time': 0.0,
            'avg_fps': 0.0,
            'max_tracked_objects': 0,
            'total_objects_tracked': 0
        }

        self.logger.info("跟踪引擎初始化完成")

    def set_tracker_config(self, **kwargs):
        """设置跟踪器配置"""
        self.track_config.update(kwargs)
        self.logger.info(f"跟踪配置已更新: {self.track_config}")

    @safe_execute("执行单帧跟踪失败")
    def track_frame(self, model, image_path: str, **kwargs) -> Optional[TrackingResult]:
        """
        执行单帧跟踪

        Args:
            model: YOLO模型对象
            image_path: 图像路径
            **kwargs: 跟踪参数

        Returns:
            跟踪结果
        """
        self.logger.info(f"跟踪帧: {os.path.basename(image_path)}")

        # 合并配置
        track_kwargs = {
            'verbose': False,
            'conf': self.track_config['confidence'],
            'iou': self.track_config['iou_threshold'],
            'tracker': self.track_config['tracker'],
            **kwargs
        }

        # 执行跟踪
        results = model.track(image_path, **track_kwargs)

        # 解析结果
        result = results[0] if isinstance(results, list) else results
        boxes_data = []

        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                track_id = int(box.id) if box.id is not None else -1
                class_id = int(box.cls)
                class_name = result.names.get(class_id, f"Class_{class_id}")
                confidence = box.conf.item() if hasattr(box.conf, 'item') else float(box.conf)

                boxes_data.append({
                    'track_id': track_id,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': box.xyxy[0].cpu().numpy().tolist() if hasattr(box, 'xyxy') else []
                })

                # 更新跟踪历史
                if track_id != -1:
                    if track_id not in self.tracking_history:
                        self.tracking_history[track_id] = deque(maxlen=self.max_history_length)
                    bbox = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else []
                    if len(bbox) >= 4:
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        self.tracking_history[track_id].append((center_x, center_y, time.time()))

        # 更新统计
        self.track_stats['total_frames'] += 1
        self.track_stats['total_objects_tracked'] += len(boxes_data)
        self.track_stats['max_tracked_objects'] = max(
            self.track_stats['max_tracked_objects'],
            len(set(b['track_id'] for b in boxes_data if b['track_id'] != -1))
        )

        # 创建跟踪结果
        tracking_result = TrackingResult(
            frame_index=self.track_stats['total_frames'],
            boxes_data=boxes_data,
            frame_time=0
        )
        tracking_result.image_path = image_path
        # 保留原始帧用于结果绘制（ultralytics Result.orig_img 为 numpy BGR 图像）
        if result is not None and hasattr(result, 'orig_img') and result.orig_img is not None:
            tracking_result.frame_image = result.orig_img
        self.current_result = tracking_result

        self.logger.info(tracking_result.get_track_summary())
        return tracking_result


    # track_id 着色调色板（RGB）
    TRACK_COLORS = [
        (0, 120, 212),    # 蓝
        (216, 59, 1),     # 橙红
        (16, 124, 16),    # 绿
        (92, 45, 145),    # 紫
        (202, 80, 16),    # 橙
        (3, 131, 135),    # 青
        (135, 100, 184),  # 淡紫
        (152, 111, 11),   # 金
        (209, 52, 56),    # 红
        (0, 153, 188),    # 天蓝
    ]

    @safe_execute("绘制跟踪结果失败")
    def draw_tracking_results(self, result: Optional[TrackingResult] = None,
                              config: Optional[dict] = None) -> Optional[QPixmap]:
        """
        绘制跟踪结果：边界框（按track_id着色）+ track_id/类别标注 + 运动轨迹

        Args:
            result: 跟踪结果，None则使用当前结果
            config: 可视化配置

        Returns:
            绘制后的QPixmap
        """
        if result is None:
            result = self.current_result

        if result is None:
            self.logger.warning("没有可绘制的跟踪结果")
            return None

        # 合并配置
        draw_config = {**self.track_config, **(config or {})}

        # 获取原始帧图像
        frame_image = getattr(result, 'frame_image', None)
        if frame_image is None:
            image_path = getattr(result, 'image_path', None)
            if not image_path or not os.path.exists(image_path):
                self.logger.warning("跟踪结果中没有可用的原始图像")
                return None
            frame_image = cv2.imread(image_path)
            if frame_image is None:
                self.logger.warning(f"无法读取图像: {image_path}")
                return None

        # numpy BGR -> QImage -> QPixmap
        height, width, channel = frame_image.shape
        bytes_per_line = 3 * width
        from PyQt5.QtGui import QImage
        q_image = QImage(frame_image.data, width, height, bytes_per_line,
                         QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Arial", max(10, width // 80))
        painter.setFont(font)

        show_labels = draw_config.get('show_labels', True)
        show_track_id = draw_config.get('show_track_id', True)
        show_trajectory = draw_config.get('show_trajectory', True)
        trajectory_length = draw_config.get('trajectory_length', 20)
        color_by_track = draw_config.get('color_by_track', True)

        for box in result.boxes_data:
            bbox = box.get('bbox') or []
            if len(bbox) < 4:
                continue

            track_id = box.get('track_id', -1)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # 按 track_id 着色
            if color_by_track and track_id != -1:
                rgb = self.TRACK_COLORS[track_id % len(self.TRACK_COLORS)]
            else:
                rgb = (0, 120, 212)
            color = QColor(rgb[0], rgb[1], rgb[2])

            # 绘制边界框
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # 绘制标签（track_id + 类别 + 置信度）
            label_parts = []
            if show_track_id and track_id != -1:
                label_parts.append(f"ID:{track_id}")
            if show_labels:
                label_parts.append(str(box.get('class_name', '')))
                conf = box.get('confidence')
                if conf is not None:
                    label_parts.append(f"{conf:.2f}")

            if label_parts:
                label = ' '.join(part for part in label_parts if part)
                fm = painter.fontMetrics()
                text_width = fm.horizontalAdvance(label) + 6
                text_height = fm.height()
                label_y = int(y1) - text_height if int(y1) - text_height > 0 else int(y1)
                painter.fillRect(int(x1), label_y, text_width, text_height, color)
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(int(x1) + 3, label_y + fm.ascent(), label)

            # 绘制运动轨迹
            if show_trajectory and track_id != -1:
                history = self.tracking_history.get(track_id)
                if history and len(history) >= 2:
                    points = list(history)[-trajectory_length:]
                    pen = QPen(color, 2, Qt.DashLine)
                    painter.setPen(pen)
                    for i in range(1, len(points)):
                        painter.drawLine(int(points[i - 1][0]), int(points[i - 1][1]),
                                         int(points[i][0]), int(points[i][1]))
                    # 轨迹端点
                    painter.setPen(QPen(color, 4))
                    painter.drawPoint(int(points[-1][0]), int(points[-1][1]))

        painter.end()
        return pixmap

    def get_track_stats(self) -> Dict[str, Any]:
        """获取跟踪统计信息"""
        return self.track_stats.copy()


    def clear_tracking_history(self):
        """清除跟踪历史"""
        self.tracking_history.clear()
        self.logger.info("跟踪历史已清除")

    def get_tracked_objects(self) -> Dict[int, Dict]:
        """获取当前跟踪的所有对象"""
        objects = {}
        if self.current_result:
            for box in self.current_result.boxes_data:
                track_id = box['track_id']
                if track_id != -1 and track_id not in objects:
                    objects[track_id] = {
                        'class_name': box['class_name'],
                        'class_id': box['class_id'],
                        'confidence': box['confidence'],
                        'bbox': box['bbox']
                    }
        return objects
