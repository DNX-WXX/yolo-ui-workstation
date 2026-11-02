"""
综合模型推理标签页 - 支持检测、分割、姿态和分类任务的推理
"""
import os
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QTextEdit,
    QFileDialog, QMessageBox, QGroupBox, QLabel, QComboBox,
    QCheckBox, QProgressBar, QScrollArea,
    QSplitter, QGridLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage

from utils.logger import LogManager, QTextEditLogger
from core.inference_engine import InferenceEngine, InferenceResult
from core.tracking_engine import TrackingEngine
from config.settings import AppConfig

from ui.widgets import DropLineEdit


class InferenceWorker(QThread):
    """推理工作线程"""

    # 信号定义
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    image_updated = pyqtSignal(QPixmap)

    def __init__(self, engine: InferenceEngine, model, image_path: str, **kwargs):
        super().__init__()
        self.engine = engine
        self.model = model
        self.image_path = image_path
        self.kwargs = kwargs
        self.is_stopped = False

    def run(self):
        """执行推理"""
        import time
        t0 = time.time()
        try:
            result = self.engine.run_inference(
                self.model, self.image_path, **self.kwargs
            )
            self.log.emit(f"推理完成 ({time.time()-t0:.1f}s)")

            if not self.is_stopped:
                self.finished.emit(result)
        except Exception as e:
            if not self.is_stopped:
                self.error.emit(str(e))

    def stop(self):
        """停止推理"""
        self.is_stopped = True


class TrackingWorker(QThread):
    """单帧跟踪工作线程"""

    finished = pyqtSignal(object, QPixmap)  # TrackingResult, 绘制后的图像
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, tracking_engine: TrackingEngine, model, image_path: str, **kwargs):
        super().__init__()
        self.tracking_engine = tracking_engine
        self.model = model
        self.image_path = image_path
        self.kwargs = kwargs
        self.is_stopped = False

    def run(self):
        """执行单帧跟踪"""
        import time
        t0 = time.time()
        try:
            result = self.tracking_engine.track_frame(
                self.model, self.image_path, **self.kwargs
            )

            if result is None:
                if not self.is_stopped:
                    self.error.emit("跟踪失败，请查看日志")
                return

            pixmap = self.tracking_engine.draw_tracking_results(result)
            self.log.emit(f"跟踪完成 ({time.time()-t0:.1f}s)")

            if not self.is_stopped:
                self.finished.emit(result, pixmap)

        except Exception as e:
            if not self.is_stopped:
                self.error.emit(str(e))

    def stop(self):
        """停止跟踪"""
        self.is_stopped = True


class ClassificationInferenceWorker(QThread):
    """分类推理工作线程"""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, model, image_path: str, topk: int = 5):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.topk = topk
        self.is_stopped = False

    def run(self):
        """执行推理"""
        import time
        t0 = time.time()
        try:
            results = self.model(self.image_path, verbose=False)
            self.log.emit(f"分类推理完成 ({time.time()-t0:.1f}s)")

            if not self.is_stopped:
                self.finished.emit(results)
        except Exception as e:
            if not self.is_stopped:
                self.error.emit(str(e))

    def stop(self):
        """停止推理"""
        self.is_stopped = True


class CombinedInferenceTab(QWidget):
    """综合模型推理标签页"""

    def __init__(
            self,
            logger: Optional[LogManager] = None,
            model_manager=None,
            image_display=None,
            image_label=None
    ):
        """
        初始化综合推理标签页

        Args:
            logger: 日志管理器
            model_manager: 模型管理器
            image_display: 图像显示区域
            image_label: 图像标签
        """
        super().__init__()

        self.logger = logger or LogManager()
        self.model_manager = model_manager
        self.image_display = image_display
        self.image_label = image_label

        # 推理引擎
        self.engine = InferenceEngine(self.logger)

        # 跟踪引擎
        self.tracking_engine = TrackingEngine(self.logger)

        # 工作线程
        self.worker = None
        self.batch_worker = None
        self.video_worker = None

        # 模型和图像路径
        self.current_model = None
        self.current_image = None
        self.model_path = ""
        self.task_type = "detect"

        # 批量推理结果存储
        self.batch_results = []
        self.batch_image_files = []
        self.current_batch_index = -1

        # 设置UI
        self.setup_ui()

        # 加载上次的配置
        self.load_config()

    def setup_ui(self):
        """设置UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)

        # ========== 1. 任务类型选择 ==========
        task_group = QGroupBox("1. 任务类型")
        task_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        task_layout = QHBoxLayout(task_group)

        self.task_combo = QComboBox()
        self.task_combo.addItem("目标检测", "detect")
        self.task_combo.addItem("实例分割", "segment")
        self.task_combo.addItem("语义分割", "semantic")
        self.task_combo.addItem("深度估计", "depth")
        self.task_combo.addItem("姿态估计", "pose")
        self.task_combo.addItem("图像分类", "classify")
        self.task_combo.addItem("旋转框检测", "obb")
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        task_layout.addWidget(QLabel("选择任务:"))
        task_layout.addWidget(self.task_combo)
        task_layout.addStretch()

        scroll_layout.addWidget(task_group)

        # ========== 2. 模型配置 ==========
        model_group = QGroupBox("2. 模型配置")
        model_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        model_layout = QFormLayout(model_group)

        # 模型路径（支持拖拽）
        self.model_path_edit = DropLineEdit(file_filter='model')
        self.model_path_edit.setPlaceholderText("拖拽模型文件到这里，或点击浏览...")
        self.model_path_edit.textChanged.connect(self.update_ui_state)

        browse_model_btn = QPushButton("浏览...")
        browse_model_btn.clicked.connect(self.browse_model)
        browse_model_btn.setFixedWidth(80)

        model_hbox = QHBoxLayout()
        model_hbox.addWidget(self.model_path_edit, 1)
        model_hbox.addWidget(browse_model_btn)
        model_layout.addRow("模型文件:", model_hbox)

        # 模型信息
        self.model_info_label = QLabel("未加载模型")
        self.model_info_label.setStyleSheet("color: #666666; font-size: 11px;")
        model_layout.addRow("模型信息:", self.model_info_label)

        scroll_layout.addWidget(model_group)

        # ========== 3. 推理参数 ==========
        param_group = QGroupBox("3. 推理参数")
        param_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        param_layout = QGridLayout(param_group)
        # 调整间距，使标签和输入控件更紧凑
        param_layout.setHorizontalSpacing(10)  # 减小水平间距
        param_layout.setVerticalSpacing(10)  # 调整垂直间距
        # 设置列的拉伸比例，使标签和输入控件对齐
        param_layout.setColumnStretch(0, 0)
        param_layout.setColumnStretch(1, 1)
        param_layout.setColumnStretch(2, 0)
        param_layout.setColumnStretch(3, 1)

        # 检测/分割/姿态参数
        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.01, 1.0)
        self.conf_threshold.setValue(AppConfig.DEFAULT_CONF_THRESHOLD)
        self.conf_threshold.setSingleStep(0.05)
        self.conf_threshold.setFixedWidth(100)
        self.conf_threshold_label = QLabel("置信度阈值:")
        param_layout.addWidget(self.conf_threshold_label, 0, 0)
        param_layout.addWidget(self.conf_threshold, 0, 1, alignment=Qt.AlignLeft)

        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setRange(0.01, 1.0)
        self.iou_threshold.setValue(0.45)
        self.iou_threshold.setSingleStep(0.05)
        self.iou_threshold.setFixedWidth(100)
        self.iou_threshold_label = QLabel("IOU阈值:")
        param_layout.addWidget(self.iou_threshold_label, 0, 2)
        param_layout.addWidget(self.iou_threshold, 0, 3, alignment=Qt.AlignLeft)

        self.max_det = QSpinBox()
        self.max_det.setRange(1, 1000)
        self.max_det.setValue(100)
        self.max_det.setFixedWidth(100)
        self.max_det_label = QLabel("最大检测数:")
        param_layout.addWidget(self.max_det_label, 1, 0)
        param_layout.addWidget(self.max_det, 1, 1, alignment=Qt.AlignLeft)

        # 分类参数
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 10)
        self.topk_spin.setValue(5)
        self.topk_spin.setFixedWidth(100)
        self.topk_spin_label = QLabel("Top-K预测:")
        param_layout.addWidget(self.topk_spin_label, 1, 2)
        param_layout.addWidget(self.topk_spin, 1, 3, alignment=Qt.AlignLeft)

        # 图像大小
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(AppConfig.MIN_IMAGE_SIZE, AppConfig.MAX_IMAGE_SIZE)
        self.imgsz_spin.setValue(AppConfig.DEFAULT_IMAGE_SIZE)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setFixedWidth(100)
        param_layout.addWidget(QLabel("图像大小:"), 2, 0)
        param_layout.addWidget(self.imgsz_spin, 2, 1, alignment=Qt.AlignLeft)

        # 设备选择
        self.device_combo = QComboBox()
        
        # 批量推理参数
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 32)
        self.batch_size.setValue(8)
        self.batch_size.setFixedWidth(100)
        param_layout.addWidget(QLabel("批处理大小:"), 3, 0)
        param_layout.addWidget(self.batch_size, 3, 1, alignment=Qt.AlignLeft)
        

        
        # 后处理参数
        self.nms_threshold = QDoubleSpinBox()
        self.nms_threshold.setRange(0.01, 1.0)
        self.nms_threshold.setValue(0.45)
        self.nms_threshold.setSingleStep(0.05)
        self.nms_threshold.setFixedWidth(100)
        param_layout.addWidget(QLabel("NMS阈值:"), 4, 0)
        param_layout.addWidget(self.nms_threshold, 4, 1, alignment=Qt.AlignLeft)
        self.device_combo.addItems(["自动", "CPU", "GPU"])
        self.device_combo.setFixedWidth(100)
        param_layout.addWidget(QLabel("设备:"), 2, 2)
        param_layout.addWidget(self.device_combo, 2, 3, alignment=Qt.AlignLeft)

        scroll_layout.addWidget(param_group)

        # ========== 3.5 跟踪模式 ==========
        tracking_group = QGroupBox("跟踪模式")
        tracking_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        tracking_layout = QHBoxLayout(tracking_group)

        self.track_check = QCheckBox("启用跟踪（目标ID + 轨迹）")
        self.track_check.setToolTip(
            "启用后使用多目标跟踪算法，单张图像显示ID/轨迹，视频模式逐帧跟踪")
        tracking_layout.addWidget(self.track_check)

        tracking_layout.addWidget(QLabel("跟踪器:"))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItem("ByteTrack", "bytetrack.yaml")
        self.tracker_combo.addItem("BotSORT", "botsort.yaml")
        self.tracker_combo.addItem("DeepSORT", "deepsort.yaml")
        self.tracker_combo.setFixedWidth(120)
        self.tracker_combo.currentIndexChanged.connect(self.on_tracker_changed)
        tracking_layout.addWidget(self.tracker_combo)

        tracking_layout.addWidget(QLabel("轨迹长度:"))
        self.trajectory_spin = QSpinBox()
        self.trajectory_spin.setRange(2, 100)
        self.trajectory_spin.setValue(20)
        self.trajectory_spin.setFixedWidth(70)
        self.trajectory_spin.setToolTip("绘制的运动轨迹点数")
        tracking_layout.addWidget(self.trajectory_spin)

        tracking_layout.addStretch()
        scroll_layout.addWidget(tracking_group)

        # ========== 4. 图像选择 ==========
        self.input_group = QGroupBox("4. 输入选择")
        self.input_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        input_layout = QVBoxLayout(self.input_group)

        # 输入类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("输入类型:"))
        self.input_type_combo = QComboBox()
        self.input_type_combo.addItem("单张图像", "image")
        self.input_type_combo.addItem("视频文件", "video")
        self.input_type_combo.addItem("摄像头", "camera")
        self.input_type_combo.addItem("RTSP 流", "rtsp")
        self.input_type_combo.currentIndexChanged.connect(self.on_input_type_changed)
        type_layout.addWidget(self.input_type_combo)
        type_layout.addStretch()
        input_layout.addLayout(type_layout)

        # 单张图像（支持拖拽）
        self.image_layout = QHBoxLayout()
        self.image_path = DropLineEdit(file_filter='image')
        self.image_path.setPlaceholderText("拖拽图像文件到这里，或点击浏览...")
        self.image_path.textChanged.connect(self.update_ui_state)
        self.image_layout.addWidget(self.image_path, 1)

        browse_image_btn = QPushButton("浏览...")
        browse_image_btn.clicked.connect(self.browse_image)
        browse_image_btn.setFixedWidth(80)
        self.image_layout.addWidget(browse_image_btn)

        # 视频文件
        self.video_layout = QHBoxLayout()
        self.video_path = DropLineEdit(file_filter='file')
        self.video_path.setPlaceholderText("拖拽视频文件到这里，或点击浏览...")
        self.video_path.textChanged.connect(self.update_ui_state)
        self.video_layout.addWidget(self.video_path, 1)

        browse_video_btn = QPushButton("浏览...")
        browse_video_btn.clicked.connect(self.browse_video)
        browse_video_btn.setFixedWidth(80)
        self.video_layout.addWidget(browse_video_btn)

        # 摄像头选择
        self.camera_layout = QHBoxLayout()
        self.camera_layout.addWidget(QLabel("摄像头:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("自动选择", 0)
        self.camera_combo.addItem("摄像头 1", 1)
        self.camera_combo.addItem("摄像头 2", 2)
        self.camera_layout.addWidget(self.camera_combo)
        self.camera_layout.addStretch()

        # RTSP 流地址输入
        self.rtsp_url = QLineEdit()
        self.rtsp_url.setPlaceholderText("输入 RTSP 流地址，例如: rtsp://192.168.1.100:554/live")
        self.rtsp_url.textChanged.connect(self.update_ui_state)

        # 视频控制按钮
        self.video_control_layout = QHBoxLayout()
        self.start_video_btn = QPushButton("开始")
        self.start_video_btn.clicked.connect(self.start_video_inference)
        self.start_video_btn.setEnabled(False)

        self.pause_video_btn = QPushButton("暂停")
        self.pause_video_btn.clicked.connect(self.pause_video_inference)
        self.pause_video_btn.setEnabled(False)

        self.stop_video_btn = QPushButton("停止")
        self.stop_video_btn.clicked.connect(self.stop_video_inference)
        self.stop_video_btn.setEnabled(False)

        self.video_control_layout.addWidget(self.start_video_btn)
        self.video_control_layout.addWidget(self.pause_video_btn)
        self.video_control_layout.addWidget(self.stop_video_btn)
        self.video_control_layout.addStretch()

        # 批量处理（支持拖拽）
        self.batch_layout = QHBoxLayout()
        self.batch_dir = DropLineEdit(file_filter='folder')
        self.batch_dir.setPlaceholderText("拖拽图像文件夹到这里，或点击浏览...")
        self.batch_layout.addWidget(self.batch_dir, 1)

        browse_batch_btn = QPushButton("浏览...")
        browse_batch_btn.clicked.connect(self.browse_batch)
        browse_batch_btn.setFixedWidth(80)
        self.batch_layout.addWidget(browse_batch_btn)

        # 批量处理选项
        self.batch_check = QCheckBox("批量处理")
        self.batch_check.stateChanged.connect(self.on_batch_mode_changed)

        # 初始显示图像输入选项
        input_layout.addLayout(self.image_layout)
        input_layout.addLayout(self.batch_layout)
        input_layout.addWidget(self.batch_check)

        scroll_layout.addWidget(self.input_group)

        # ========== 5. 输出配置 ==========
        output_group = QGroupBox("5. 输出配置")
        output_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        output_layout = QFormLayout(output_group)

        # 保存结果
        self.save_results_check = QCheckBox("保存推理结果")
        self.save_results_check.setChecked(True)
        output_layout.addRow("保存结果:", self.save_results_check)

        # 输出目录
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("选择输出目录...")

        browse_output_btn = QPushButton("浏览...")
        browse_output_btn.clicked.connect(self.browse_output)
        browse_output_btn.setFixedWidth(80)

        output_hbox = QHBoxLayout()
        output_hbox.addWidget(self.output_dir, 1)
        output_hbox.addWidget(browse_output_btn)
        output_layout.addRow("输出目录:", output_hbox)

        scroll_layout.addWidget(output_group)

        # ========== 6. 操作按钮 ==========
        button_layout = QHBoxLayout()

        self.infer_btn = QPushButton("开始推理")
        self.infer_btn.clicked.connect(self.start_inference)
        self.infer_btn.setEnabled(False)
        self.infer_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        button_layout.addWidget(self.infer_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #d83b01;
                color: white;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #a02c01;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_btn)

        button_layout.addStretch()

        # 恢复默认值按钮
        self.reset_btn = QPushButton("恢复默认值")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        button_layout.addWidget(self.reset_btn)

        scroll_layout.addLayout(button_layout)

        # ========== 7. 进度条 ==========
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        scroll_layout.addWidget(self.progress_bar)

        # ========== 8. 批量结果导航 ==========
        self.batch_nav_group = QGroupBox("批量结果导航")
        self.batch_nav_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.batch_nav_group.setVisible(False)
        nav_layout = QHBoxLayout(self.batch_nav_group)

        self.prev_btn = QPushButton("◀ 上一张")
        self.prev_btn.clicked.connect(self.show_previous_batch_result)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)

        self.batch_info_label = QLabel("0 / 0")
        self.batch_info_label.setAlignment(Qt.AlignCenter)
        self.batch_info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        nav_layout.addWidget(self.batch_info_label, 1)

        self.next_btn = QPushButton("下一张 ▶")
        self.next_btn.clicked.connect(self.show_next_batch_result)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)

        # 批量结果导出与统计
        self.export_summary_btn = QPushButton("导出汇总")
        self.export_summary_btn.clicked.connect(self.export_batch_summary)
        self.export_summary_btn.setEnabled(False)
        nav_layout.addWidget(self.export_summary_btn)

        self.stats_btn = QPushButton("统计图表")
        self.stats_btn.clicked.connect(self.show_batch_statistics)
        self.stats_btn.setEnabled(False)
        nav_layout.addWidget(self.stats_btn)

        scroll_layout.addWidget(self.batch_nav_group)

        # 添加伸缩项
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_widget)

        # 分类结果表格
        self.classification_table = QTableWidget()
        self.classification_table.setColumnCount(3)
        self.classification_table.setHorizontalHeaderLabels(["排名", "类别", "置信度"])
        self.classification_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.classification_table.setVisible(False)
        
        # 创建垂直分割器
        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #d0d0d0;
                height: 8px;
                border: 1px solid #a0a0a0;
                border-radius: 4px;
            }
            QSplitter::handle:hover {
                background-color: #b0b0b0;
            }
        """)

        # 添加内容区域到分割器
        splitter.addWidget(scroll_area)

        # 添加分类结果表格到分割器
        splitter.addWidget(self.classification_table)

        # 日志显示
        log_group = QGroupBox("推理日志")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QVBoxLayout(log_group)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(100)
        self.log_display.setFont(QFont("Consolas", 9))

        # 设置日志处理器
        log_handler = QTextEditLogger(self.log_display)
        self.logger.add_handler(log_handler)

        log_layout.addWidget(self.log_display)

        # 添加日志区域到分割器
        splitter.addWidget(log_group)

        # 设置初始大小比例（70% 上面，10% 表格，20% 日志）
        splitter.setSizes([700, 100, 200])

        main_layout.addWidget(splitter)

        # 初始化UI状态
        self.update_ui_state()
        self.on_task_changed(0)  # 初始化任务类型

    def on_task_changed(self, index):
        """任务类型改变回调"""
        self.task_type = self.task_combo.currentData()
        self.logger.info(f"任务类型已切换为: {self.task_type}")

        # 调整UI显示
        if self.task_type == "classify":
            # 隐藏检测参数，显示分类参数
            self.conf_threshold.setVisible(False)
            self.conf_threshold_label.setVisible(False)
            self.iou_threshold.setVisible(False)
            self.iou_threshold_label.setVisible(False)
            self.max_det.setVisible(False)
            self.max_det_label.setVisible(False)
            self.topk_spin.setVisible(True)
            self.topk_spin_label.setVisible(True)
            self.classification_table.setVisible(True)
            # 设置分类任务默认参数
            self.imgsz_spin.setValue(224)  # 分类任务默认图像大小
            self.topk_spin.setValue(5)     # 默认Top-5预测
        elif self.task_type in ("semantic", "depth"):
            # 语义分割/深度估计：不需要检测类参数，显示为像素级输出
            self.conf_threshold.setVisible(False)
            self.conf_threshold_label.setVisible(False)
            self.iou_threshold.setVisible(False)
            self.iou_threshold_label.setVisible(False)
            self.max_det.setVisible(False)
            self.max_det_label.setVisible(False)
            self.topk_spin.setVisible(False)
            self.topk_spin_label.setVisible(False)
            self.classification_table.setVisible(False)
            # 语义分割/深度估计默认参数
            self.imgsz_spin.setValue(640)
        else:
            # 显示检测参数，隐藏分类参数
            self.conf_threshold.setVisible(True)
            self.conf_threshold_label.setVisible(True)
            self.iou_threshold.setVisible(True)
            self.iou_threshold_label.setVisible(True)
            self.max_det.setVisible(True)
            self.max_det_label.setVisible(True)
            self.topk_spin.setVisible(False)
            self.topk_spin_label.setVisible(False)
            self.classification_table.setVisible(False)
            
            # 设置不同任务的默认参数
            if self.task_type == "obb":
                # 旋转框检测默认参数
                self.imgsz_spin.setValue(640)  # OBB任务默认图像大小
                self.conf_threshold.setValue(0.25)  # 默认置信度阈值
                self.iou_threshold.setValue(0.45)   # 默认IOU阈值
                self.max_det.setValue(100)          # 默认最大检测数
            else:
                # 检测/分割/姿态任务默认参数
                self.imgsz_spin.setValue(640)  # 检测任务默认图像大小
                self.conf_threshold.setValue(0.25)  # 默认置信度阈值
                self.iou_threshold.setValue(0.45)   # 默认IOU阈值
                self.max_det.setValue(100)          # 默认最大检测数

        # 清空之前的结果
        self.clear_results()

    def on_batch_mode_changed(self, state):
        """批量处理模式改变"""
        is_batch = (state == Qt.Checked)
        self.image_path.setEnabled(not is_batch)
        self.batch_dir.setEnabled(is_batch)

    def browse_model(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "模型文件 (*.pt *.onnx);;PyTorch模型 (*.pt);;ONNX模型 (*.onnx)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            self.load_model(file_path)

    def load_model(self, model_path: str):
        """加载模型"""
        # 显示加载进度
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不定进度条
        self.model_info_label.setText("正在加载模型...")
        self.infer_btn.setEnabled(False)
        
        # 创建模型加载线程
        self.model_load_worker = ModelLoadWorker(model_path)
        
        # 连接信号
        self.model_load_worker.finished.connect(self.on_model_load_finished)
        self.model_load_worker.error.connect(self.on_model_load_error)
        self.model_load_worker.log.connect(self.logger.info)

        # 启动线程
        self.model_load_worker.start()

    def on_model_load_finished(self, model, task):
        """模型加载完成回调"""
        try:
            self.current_model = model
            self.model_path = self.model_path_edit.text()

            # 获取模型信息
            info_text = f"任务类型: {task}"
            self.model_info_label.setText(info_text)
            self.logger.info(f"模型加载成功: {info_text}")

            # 更新任务类型
            if task in ['detect', 'segment', 'semantic', 'depth', 'pose', 'classify', 'obb']:
                self.task_type = task
                index = self.task_combo.findData(task)
                if index >= 0:
                    self.task_combo.blockSignals(True)
                    self.task_combo.setCurrentIndex(index)
                    self.task_combo.blockSignals(False)
                    
                    # 手动触发UI更新
                    if task == "classify":
                        self.conf_threshold.setVisible(False)
                        self.conf_threshold_label.setVisible(False)
                        self.iou_threshold.setVisible(False)
                        self.iou_threshold_label.setVisible(False)
                        self.max_det.setVisible(False)
                        self.max_det_label.setVisible(False)
                        self.topk_spin.setVisible(True)
                        self.topk_spin_label.setVisible(True)
                        self.classification_table.setVisible(True)
                        self.imgsz_spin.setValue(224)
                        self.topk_spin.setValue(5)
                    elif task in ("semantic", "depth"):
                        # 语义分割/深度估计：隐藏检测类参数
                        self.conf_threshold.setVisible(False)
                        self.conf_threshold_label.setVisible(False)
                        self.iou_threshold.setVisible(False)
                        self.iou_threshold_label.setVisible(False)
                        self.max_det.setVisible(False)
                        self.max_det_label.setVisible(False)
                        self.topk_spin.setVisible(False)
                        self.topk_spin_label.setVisible(False)
                        self.classification_table.setVisible(False)
                        self.imgsz_spin.setValue(640)
                    else:
                        self.conf_threshold.setVisible(True)
                        self.conf_threshold_label.setVisible(True)
                        self.iou_threshold.setVisible(True)
                        self.iou_threshold_label.setVisible(True)
                        self.max_det.setVisible(True)
                        self.max_det_label.setVisible(True)
                        self.topk_spin.setVisible(False)
                        self.topk_spin_label.setVisible(False)
                        self.classification_table.setVisible(False)
                        
                        if task == "obb":
                            # OBB任务默认参数
                            self.imgsz_spin.setValue(640)
                            self.conf_threshold.setValue(0.25)
                            self.iou_threshold.setValue(0.45)
                            self.max_det.setValue(100)
                        else:
                            # 检测/分割/姿态任务默认参数
                            self.imgsz_spin.setValue(640)
                            self.conf_threshold.setValue(0.25)
                            self.iou_threshold.setValue(0.45)
                            self.max_det.setValue(100)

            # 更新UI状态
            self.update_ui_state()
            
            # 隐藏进度条
            self.progress_bar.setVisible(False)
        except Exception as e:
            self.logger.error(f"模型加载后处理失败: {e}")
            self.model_info_label.setText("模型加载失败")
            self.progress_bar.setVisible(False)

    def on_model_load_error(self, error):
        """模型加载错误回调"""
        self.logger.error(f"模型加载失败: {error}")
        QMessageBox.critical(self, "加载失败", f"模型加载失败: {error}")
        self.model_info_label.setText("模型加载失败")
        self.progress_bar.setVisible(False)
        self.update_ui_state()

    def browse_image(self):
        """浏览图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.image_path.setText(file_path)
            self.current_image = file_path
            # 显示图像
            if self.image_label:
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    self.image_label.setPixmap(pixmap)
                    self.image_label.setScaledContents(True)

    def browse_batch(self):
        """浏览批量图像文件夹"""
        folder = QFileDialog.getExistingDirectory(
            self, "选择图像文件夹"
        )
        if folder:
            self.batch_dir.setText(folder)

    def browse_output(self):
        """浏览输出目录"""
        folder = QFileDialog.getExistingDirectory(
            self, "选择输出目录"
        )
        if folder:
            self.output_dir.setText(folder)



    def update_ui_state(self):
        """更新UI状态"""
        has_model = bool(self.model_path_edit.text())
        input_type = self.input_type_combo.currentData()
        
        # 根据输入类型检查是否有有效的输入
        has_input = False
        if input_type == "image":
            has_input = bool(self.image_path.text()) or bool(self.batch_dir.text())
        elif input_type == "video":
            has_input = bool(self.video_path.text())
        elif input_type == "camera":
            has_input = True  # 摄像头不需要路径
        elif input_type == "rtsp":
            has_input = bool(self.rtsp_url.text().strip())

        self.infer_btn.setEnabled(has_model and has_input)

        # 视频控制按钮状态
        if input_type in ["video", "camera", "rtsp"]:
            self.start_video_btn.setEnabled(has_model)
        else:
            self.start_video_btn.setEnabled(False)
        self.pause_video_btn.setEnabled(False)
        self.stop_video_btn.setEnabled(False)

    def start_inference(self):
        """开始推理"""
        # 验证输入
        # 单模型推理模式
        model_path = self.model_path_edit.text()
        if not model_path:
            QMessageBox.warning(self, "警告", "请选择模型文件")
            return

        if self.batch_check.isChecked():
            input_path = self.batch_dir.text()
            if not input_path:
                QMessageBox.warning(self, "警告", "请选择图像文件夹")
                return
        else:
            input_path = self.image_path.text()
            if not input_path:
                QMessageBox.warning(self, "警告", "请选择图像文件")
                return

        # 加载模型
        if not self.current_model:
            self.load_model(model_path)
            if not self.current_model:
                return

        # 收集参数
        imgsz = self.imgsz_spin.value()
        device = self.device_combo.currentText()
        if device == "自动":
            device = None
        elif device == "CPU":
            device = "cpu"
        else:  # GPU
            device = "0"

        # 清空日志
        self.log_display.clear()
        self.logger.info("=" * 60)
        self.logger.info("开始模型推理")

        # 更新UI
        self.infer_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不定进度条

        # 根据推理模式执行
        if self.batch_check.isChecked():
            # 批量处理模式
            self.start_batch_inference(input_path, imgsz, device)
        elif self.track_check.isChecked() and self.task_type != "classify":
            # 跟踪模式（单张图像）
            self.start_tracking_inference(input_path, imgsz, device)
        elif self.task_type == "classify":
            self.start_classification_inference(input_path, imgsz, device)
        else:
            self.start_detection_inference(input_path, imgsz, device)

    def start_batch_inference(self, folder_path: str, imgsz: int, device: str):
        """开始批量推理"""
        # 收集参数
        conf = self.conf_threshold.value()
        iou = self.iou_threshold.value()
        max_det = self.max_det.value()
        batch_size = self.batch_size.value()
        nms_threshold = self.nms_threshold.value()

        self.logger.info(f"置信度阈值: {conf}")
        self.logger.info(f"IOU阈值: {iou}")
        self.logger.info(f"最大检测数: {max_det}")
        self.logger.info(f"批处理大小: {batch_size}")
        self.logger.info(f"NMS阈值: {nms_threshold}")
        self.logger.info("=" * 60)

        # 获取输出目录
        output_dir = self.output_dir.text() or os.path.join(os.path.dirname(self.model_path), "batch_results")

        # 创建工作线程
        self.batch_worker = BatchInferenceWorker(
            engine=self.engine,
            model=self.current_model,
            folder_path=folder_path,
            output_dir=output_dir,
            batch_size=batch_size,
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=imgsz,
            device=device
        )

        # 连接信号
        self.batch_worker.finished.connect(self.on_batch_inference_finished)
        self.batch_worker.error.connect(self.on_inference_error)
        self.batch_worker.log.connect(self.logger.info)
        self.batch_worker.progress.connect(self.on_batch_progress)
        self.batch_worker.image_updated.connect(self.on_batch_image_updated)

        # 批量推理期间禁用自动适应视图（避免每张图都重算缩放）
        if self.image_display:
            self.image_display.set_fit_to_view_enabled(False)

        # 启动线程
        self.batch_worker.start()

    def on_batch_progress(self, current: int, total: int):
        """批量推理进度回调"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)

    def on_batch_image_updated(self, pixmap: QPixmap, current: int, total: int):
        """批量推理图像更新回调"""
        if self.image_display:
            self.image_display.set_pixmap(pixmap)

    def on_batch_inference_finished(self, results: list, image_files: list):
        """批量推理完成回调"""
        self.logger.info("=" * 60)
        self.logger.info(f"批量推理完成！共处理 {len(results)} 张图像")
        self.logger.info("=" * 60)

        # 保存结果和图像文件列表
        self.batch_results = results
        self.batch_image_files = image_files
        self.current_batch_index = len(results) - 1 if results else -1

        # 显示导航控件
        if len(results) > 0:
            self.batch_nav_group.setVisible(True)
            self.export_summary_btn.setEnabled(True)
            self.stats_btn.setEnabled(True)
            self.update_batch_navigation()

        # 更新UI
        self.infer_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        # 批量推理结束，恢复图像自动适应视图
        if self.image_display:
            self.image_display.set_fit_to_view_enabled(True)

    def update_batch_navigation(self):
        """更新批量结果导航状态"""
        total = len(self.batch_results)
        if total > 0 and self.current_batch_index >= 0:
            self.batch_info_label.setText(f"{self.current_batch_index + 1} / {total}")
            self.prev_btn.setEnabled(self.current_batch_index > 0)
            self.next_btn.setEnabled(self.current_batch_index < total - 1)

            # 显示当前结果
            self.show_batch_result(self.current_batch_index)
        else:
            self.batch_info_label.setText("0 / 0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

    def show_batch_result(self, index: int):
        """显示指定索引的批量结果"""
        if 0 <= index < len(self.batch_results):
            result = self.batch_results[index]
            image_path = self.batch_image_files[index]

            # 绘制结果
            pixmap = self.engine.draw_results(result)
            if pixmap and not pixmap.isNull():
                self.image_display.set_pixmap(pixmap)
                self.image_display.fit_to_view()

            self.logger.info(f"显示结果 [{index + 1}/{len(self.batch_results)}]: {os.path.basename(image_path)}")

    def show_previous_batch_result(self):
        """显示上一张批量结果"""
        if self.current_batch_index > 0:
            self.current_batch_index -= 1
            self.update_batch_navigation()

    def show_next_batch_result(self):
        """显示下一张批量结果"""
        if self.current_batch_index < len(self.batch_results) - 1:
            self.current_batch_index += 1
            self.update_batch_navigation()

    def keyPressEvent(self, event):
        """键盘事件处理"""
        if self.batch_nav_group.isVisible() and len(self.batch_results) > 0:
            if event.key() == Qt.Key_Left:
                # 左箭头：上一张
                self.show_previous_batch_result()
            elif event.key() == Qt.Key_Right:
                # 右箭头：下一张
                self.show_next_batch_result()
            elif event.key() == Qt.Key_Home:
                # Home键：第一张
                if self.current_batch_index > 0:
                    self.current_batch_index = 0
                    self.update_batch_navigation()
            elif event.key() == Qt.Key_End:
                # End键：最后一张
                last_index = len(self.batch_results) - 1
                if self.current_batch_index < last_index:
                    self.current_batch_index = last_index
                    self.update_batch_navigation()
        super().keyPressEvent(event)

    def _collect_batch_summary(self) -> List[Dict[str, Any]]:
        """收集批量推理结果的汇总记录（检测框 / 分类 Top-K）"""
        import torch
        records = []
        if not getattr(self, 'batch_results', None):
            return records
        for result, img_path in zip(self.batch_results, self.batch_image_files):
            img_name = os.path.basename(img_path)
            boxes = getattr(result, 'boxes', None)
            probs = getattr(result, 'probs', None)
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls)
                    xyxy = box.xyxy[0]
                    coords = xyxy.tolist() if hasattr(xyxy, 'tolist') else list(xyxy)
                    records.append({
                        'image': img_name,
                        'class_id': cls_id,
                        'class_name': result.names.get(cls_id, str(cls_id)),
                        'confidence': round(float(box.conf), 4),
                        'x1': round(float(coords[0]), 2),
                        'y1': round(float(coords[1]), 2),
                        'x2': round(float(coords[2]), 2),
                        'y2': round(float(coords[3]), 2),
                    })
            elif probs is not None:
                data = probs.data
                topk = min(5, len(data))
                top_indices = torch.topk(data, topk).indices.tolist()
                top_values = torch.topk(data, topk).values.tolist()
                for rank, (idx, conf) in enumerate(zip(top_indices, top_values), 1):
                    records.append({
                        'image': img_name,
                        'class_id': idx,
                        'class_name': result.names.get(idx, str(idx)),
                        'confidence': round(conf, 4),
                        'rank': rank,
                    })
        return records

    def export_batch_summary(self):
        """导出批量推理结果汇总（CSV / JSON）"""
        records = self._collect_batch_summary()
        if not records:
            QMessageBox.information(self, "提示", "暂无批量推理结果可导出")
            return

        default_dir = self.output_dir.text() or os.getcwd()
        path, _ = QFileDialog.getSaveFileName(
            self, "导出结果汇总", os.path.join(default_dir, "batch_summary"),
            "CSV 文件 (*.csv);;JSON 文件 (*.json)"
        )
        if not path:
            return
        try:
            if path.lower().endswith('.json'):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
            else:
                if not path.lower().endswith('.csv'):
                    path += '.csv'
                # utf-8-sig 保证 Excel 正确识别中文表头
                with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                    writer.writeheader()
                    writer.writerows(records)
            self.logger.info(f"结果汇总已导出: {path}（共 {len(records)} 条记录）")
        except Exception as e:
            self.logger.error(f"导出失败: {e}")
            QMessageBox.critical(self, "导出失败", str(e))

    def show_batch_statistics(self):
        """显示批量推理结果统计图表（类别分布 + 置信度直方图）"""
        try:
            from utils.plot_style import setup_matplotlib
            setup_matplotlib()
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        except ImportError:
            QMessageBox.warning(self, "提示", "matplotlib 未安装，无法显示统计图表")
            return
        records = self._collect_batch_summary()
        if not records:
            QMessageBox.information(self, "提示", "暂无批量推理结果")
            return

        from collections import Counter
        class_counts = Counter(r['class_name'] for r in records)
        confidences = [r['confidence'] for r in records]

        dialog = QDialog(self)
        dialog.setWindowTitle("批量推理统计")
        dialog.resize(960, 460)
        layout = QVBoxLayout(dialog)

        fig = Figure(figsize=(9, 4))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # 类别分布
        ax1 = fig.add_subplot(121)
        names = list(class_counts.keys())
        counts = list(class_counts.values())
        ax1.bar(range(len(names)), counts, color='steelblue')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_title("类别分布")
        ax1.set_ylabel("数量")

        # 置信度直方图
        ax2 = fig.add_subplot(122)
        ax2.hist(confidences, bins=20, range=(0.0, 1.0), color='darkorange', edgecolor='black')
        ax2.set_title("置信度分布")
        ax2.set_xlabel("置信度")
        ax2.set_ylabel("数量")

        fig.tight_layout()
        canvas.draw()
        dialog.exec_()

    def start_detection_inference(self, input_path: str, imgsz: int, device: str):
        """开始检测/分割/姿态推理"""
        # 收集参数
        conf = self.conf_threshold.value()
        iou = self.iou_threshold.value()
        max_det = self.max_det.value()
        nms_threshold = self.nms_threshold.value()

        self.logger.info(f"置信度阈值: {conf}")
        self.logger.info(f"IOU阈值: {iou}")
        self.logger.info(f"最大检测数: {max_det}")
        self.logger.info(f"NMS阈值: {nms_threshold}")
        self.logger.info("=" * 60)

        # 创建工作线程
        self.worker = InferenceWorker(
            engine=self.engine,
            model=self.current_model,
            image_path=input_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=imgsz,
            device=device
        )

        # 连接信号
        self.worker.finished.connect(self.on_detection_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.log.connect(self.logger.info)
        self.worker.image_updated.connect(self.update_image)

        # 启动线程
        self.worker.start()

    def start_tracking_inference(self, input_path: str, imgsz: int, device: str):
        """开始单帧跟踪"""
        conf = self.conf_threshold.value()
        iou = self.iou_threshold.value()

        # 同步跟踪引擎配置
        self.tracking_engine.set_tracker_config(
            tracker=self.tracker_combo.currentData(),
            confidence=conf,
            iou_threshold=iou,
            trajectory_length=self.trajectory_spin.value()
        )
        self.tracking_engine.clear_tracking_history()

        self.logger.info(f"跟踪模式: {self.tracker_combo.currentText()}, "
                         f"轨迹长度: {self.trajectory_spin.value()}")
        self.logger.info("=" * 60)

        # 创建跟踪工作线程
        self.worker = TrackingWorker(
            tracking_engine=self.tracking_engine,
            model=self.current_model,
            image_path=input_path,
            imgsz=imgsz,
            device=device,
            persist=True
        )

        # 连接信号
        self.worker.finished.connect(self.on_tracking_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.log.connect(self.logger.info)

        # 启动线程
        self.worker.start()

    def on_tracking_inference_finished(self, tracking_result, pixmap):
        """跟踪完成回调"""
        self.logger.info("=" * 60)
        self.logger.info("跟踪完成！")

        # 显示跟踪结果图像
        if pixmap and not pixmap.isNull() and self.image_display:
            self.image_display.set_pixmap(pixmap)
            self.image_display.fit_to_view()

        # 输出统计信息
        self.tracking_engine.get_track_stats()
        objects = self.tracking_engine.get_tracked_objects()
        self.logger.info(f"当前帧跟踪目标数: {len(objects)}")
        if objects:
            for track_id, obj in list(objects.items())[:10]:
                self.logger.info(
                    f"  ID:{track_id} - {obj['class_name']} ({obj['confidence']:.2f})")

        # 保存结果
        if self.save_results_check.isChecked() and pixmap and not pixmap.isNull():
            output_dir = self.output_dir.text() or os.path.join(
                os.path.dirname(self.model_path), "tracking_results")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, f"track_{os.path.basename(self.current_image or 'frame.png')}")
            pixmap.save(output_path)
            self.logger.info(f"结果保存到: {output_path}")

        # 更新UI
        self.infer_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.worker = None

    def on_tracker_changed(self, index):
        """跟踪器切换回调"""
        tracker_name = self.tracker_combo.currentData()
        if hasattr(self, 'tracking_engine'):
            self.tracking_engine.set_tracker_config(tracker=tracker_name)

    def start_classification_inference(self, image_path: str, imgsz: int, device: str):
        """开始分类推理"""
        # 收集参数
        topk = self.topk_spin.value()

        self.logger.info(f"Top-K: {topk}")
        self.logger.info("=" * 60)

        # 创建工作线程
        self.worker = ClassificationInferenceWorker(
            model=self.current_model,
            image_path=image_path,
            topk=topk
        )

        # 连接信号
        self.worker.finished.connect(self.on_classification_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.log.connect(self.logger.info)

        # 启动线程
        self.worker.start()

    def stop_inference(self):
        """停止推理"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            # 先等待线程自行结束，超时才强制终止
            if not self.worker.wait(3000):
                self.worker.terminate()
                self.worker.wait()
            self.logger.warning("推理已停止")
        elif self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.stop()
            if not self.batch_worker.wait(3000):
                self.batch_worker.terminate()
                self.batch_worker.wait()
            self.logger.warning("批量推理已停止")
            # 批量中断，恢复图像自动适应视图
            if self.image_display:
                self.image_display.set_fit_to_view_enabled(True)

        self.infer_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    def on_detection_inference_finished(self, result: InferenceResult):
        """检测推理完成回调"""
        self.logger.info("=" * 60)
        self.logger.info("推理完成！")
        
        # 检查结果是否为None
        if result is None:
            self.logger.error("推理结果为None")
            # 更新UI
            self.infer_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            return
        
        # 应用后处理
        nms_threshold = self.nms_threshold.value()
        processed_result = self.engine.postprocess_results(
            result,
            nms_threshold=nms_threshold
        )
            
        self.logger.info(f"检测到目标: {len(processed_result.boxes) if processed_result.boxes else 0} 个")
        self.logger.info("=" * 60)

        # 显示结果
        if self.image_display:
            # 使用推理引擎绘制结果
            pixmap = self.engine.draw_results(processed_result)
            if pixmap and not pixmap.isNull():
                self.image_display.set_pixmap(pixmap)
                self.image_display.fit_to_view()

        # 保存结果
        if self.save_results_check.isChecked():
            output_dir = self.output_dir.text() or os.path.join(os.path.dirname(self.model_path), "inference_results")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存可视化图像
            pixmap = self.engine.draw_results(processed_result)
            if pixmap and not pixmap.isNull():
                output_path = os.path.join(output_dir, f"result_{os.path.basename(self.current_image)}")
                pixmap.save(output_path)
                self.logger.info(f"结果保存到: {output_path}")

        # 更新UI
        self.infer_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    def on_classification_inference_finished(self, results):
        """分类推理完成回调"""
        import torch
        self.logger.info("=" * 60)
        self.logger.info("推理完成！")

        # 处理结果
        if results:
            result = results[0]
            probs = result.probs
            topk = min(self.topk_spin.value(), len(probs.data))
            top_indices = torch.topk(probs.data, topk).indices.tolist()
            top_probs = torch.topk(probs.data, topk).values.tolist()

            # 显示结果
            self.classification_table.setRowCount(topk)
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                class_name = result.names[idx]
                self.classification_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
                self.classification_table.setItem(i, 1, QTableWidgetItem(class_name))
                self.classification_table.setItem(i, 2, QTableWidgetItem(f"{prob:.4f}"))

            # 显示图像
            if self.image_display:
                pixmap = QPixmap(self.image_path.text())
                if not pixmap.isNull():
                    self.image_display.set_pixmap(pixmap)
                    self.image_display.fit_to_view()

            # 保存结果
            if self.save_results_check.isChecked():
                output_dir = self.output_dir.text() or os.path.join(os.path.dirname(self.model_path), "inference_results")
                os.makedirs(output_dir, exist_ok=True)
                
                # 保存结果到文本文件
                output_path = os.path.join(output_dir, f"result_{os.path.splitext(os.path.basename(self.current_image))[0]}.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Image: {self.current_image}\n")
                    f.write("Top-K Predictions:\n")
                    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                        class_name = result.names[idx]
                        f.write(f"{i+1}. {class_name}: {prob:.4f}\n")
                self.logger.info(f"结果保存到: {output_path}")

        # 更新UI
        self.infer_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    def on_inference_error(self, error_msg: str):
        """推理错误回调"""
        self.logger.error(f"推理失败: {error_msg}")
        QMessageBox.critical(self, "推理失败", f"错误信息:\n{error_msg}")

        # 更新UI
        self.infer_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        # 出错时也恢复图像自动适应视图（批量推理可能已禁用）
        if self.image_display:
            self.image_display.set_fit_to_view_enabled(True)

    def update_image(self, pixmap: QPixmap):
        """更新图像显示"""
        if self.image_display:
            self.image_display.set_pixmap(pixmap)
            self.image_display.fit_to_view()

    def clear_results(self):
        """清空结果"""
        # 清空图像显示
        if self.image_display:
            self.image_display.clear()

        # 清空分类结果表格
        self.classification_table.setRowCount(0)

        # 重置批量结果状态
        self.batch_results = []
        self.batch_image_files = []
        self.export_summary_btn.setEnabled(False)
        self.stats_btn.setEnabled(False)

    def reset_to_defaults(self):
        """恢复默认参数设置"""
        reply = QMessageBox.question(
            self, '确认恢复默认值',
            '确定要将所有推理参数恢复为默认值吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # 重置任务类型
                self.task_combo.setCurrentIndex(0)

                # 重置参数
                self.conf_threshold.setValue(AppConfig.DEFAULT_CONF_THRESHOLD)
                self.iou_threshold.setValue(0.45)
                self.max_det.setValue(100)
                self.topk_spin.setValue(5)
                self.imgsz_spin.setValue(AppConfig.DEFAULT_IMAGE_SIZE)
                self.device_combo.setCurrentIndex(0)

                # 重置输出配置
                self.save_results_check.setChecked(True)
                self.output_dir.clear()

                # 重置跟踪配置
                self.track_check.setChecked(False)
                self.tracker_combo.setCurrentIndex(0)
                self.trajectory_spin.setValue(20)

                # 清空路径
                self.model_path_edit.clear()
                self.image_path.clear()
                self.batch_dir.clear()
                self.batch_check.setChecked(False)

                # 清空结果
                self.clear_results()

                # 更新UI状态
                self.update_ui_state()

                self.logger.info("已恢复推理默认参数设置")
                QMessageBox.information(self, '恢复成功', '所有参数已恢复为默认值！')

            except Exception as e:
                self.logger.error(f"恢复默认值失败: {e}")
                QMessageBox.warning(self, '恢复失败', f'恢复默认值时发生错误：{e}')

    def save_config(self):
        """保存配置"""
        config = {
            'task': self.task_type,
            'model_path': self.model_path_edit.text(),
            'conf_threshold': self.conf_threshold.value(),
            'iou_threshold': self.iou_threshold.value(),
            'max_det': self.max_det.value(),
            'topk': self.topk_spin.value(),
            'imgsz': self.imgsz_spin.value(),
            'device': self.device_combo.currentText(),
            'save_results': self.save_results_check.isChecked(),
            'output_dir': self.output_dir.text(),
            'batch_mode': self.batch_check.isChecked(),
            'batch_dir': self.batch_dir.text(),
            'rtsp_url': self.rtsp_url.text()
        }

        # 保存到文件
        config_dir = Path.home() / ".yolo_ui_workstation"
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "combined_inference_config.json"
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")

    def load_config(self):
        """加载配置（现在不加载，每次启动使用默认值）"""
        # 不加载之前的配置，确保每次启动都使用默认值
        pass



    def on_tab_selected(self):
        """标签页被选中时的回调"""
        self.logger.info("模型推理标签页已激活")

        # 更新图像显示区域
        if self.image_label:
            self.image_label.setText("请在这里预览推理结果")

        # 不自动保存配置，避免覆盖默认值
        # self.save_config()

    def on_input_type_changed(self, index):
        """输入类型改变回调"""
        input_type = self.input_type_combo.currentData()
        self.logger.info(f"输入类型已切换为: {input_type}")

        # 清空输入布局
        input_layout = self.input_group.layout()
        if input_layout:
            # 移除所有子元素（除了输入类型选择）
            while input_layout.count() > 1:  # 保留输入类型选择行
                item = input_layout.takeAt(1)
                if item.widget():
                    item.widget().setParent(None)
                elif item.layout():
                    # 移除布局中的所有元素
                    sub_layout = item.layout()
                    while sub_layout.count() > 0:
                        sub_item = sub_layout.takeAt(0)
                        if sub_item.widget():
                            sub_item.widget().setParent(None)
                    sub_layout.setParent(None)

        # 根据输入类型显示相应的选项
        if input_layout:
            if input_type == "image":
                # 创建并显示图像输入选项
                image_layout = QHBoxLayout()
                image_layout.addWidget(self.image_path, 1)
                browse_image_btn = QPushButton("浏览...")
                browse_image_btn.clicked.connect(self.browse_image)
                browse_image_btn.setFixedWidth(80)
                image_layout.addWidget(browse_image_btn)
                input_layout.addLayout(image_layout)

                batch_layout = QHBoxLayout()
                batch_layout.addWidget(self.batch_dir, 1)
                browse_batch_btn = QPushButton("浏览...")
                browse_batch_btn.clicked.connect(self.browse_batch)
                browse_batch_btn.setFixedWidth(80)
                batch_layout.addWidget(browse_batch_btn)
                input_layout.addLayout(batch_layout)

                input_layout.addWidget(self.batch_check)
            elif input_type == "video":
                # 创建并显示视频输入选项
                video_layout = QHBoxLayout()
                video_layout.addWidget(self.video_path, 1)
                browse_video_btn = QPushButton("浏览...")
                browse_video_btn.clicked.connect(self.browse_video)
                browse_video_btn.setFixedWidth(80)
                video_layout.addWidget(browse_video_btn)
                input_layout.addLayout(video_layout)

                video_control_layout = QHBoxLayout()
                video_control_layout.addWidget(self.start_video_btn)
                video_control_layout.addWidget(self.pause_video_btn)
                video_control_layout.addWidget(self.stop_video_btn)
                video_control_layout.addStretch()
                input_layout.addLayout(video_control_layout)
            elif input_type == "camera":
                # 创建并显示摄像头输入选项
                camera_layout = QHBoxLayout()
                camera_layout.addWidget(QLabel("摄像头:"))
                camera_layout.addWidget(self.camera_combo)
                camera_layout.addStretch()
                input_layout.addLayout(camera_layout)

                video_control_layout = QHBoxLayout()
                video_control_layout.addWidget(self.start_video_btn)
                video_control_layout.addWidget(self.pause_video_btn)
                video_control_layout.addWidget(self.stop_video_btn)
                video_control_layout.addStretch()
                input_layout.addLayout(video_control_layout)
            elif input_type == "rtsp":
                # 创建并显示 RTSP 流输入选项
                rtsp_layout = QHBoxLayout()
                rtsp_layout.addWidget(QLabel("RTSP 地址:"))
                rtsp_layout.addWidget(self.rtsp_url, 1)
                input_layout.addLayout(rtsp_layout)

                video_control_layout = QHBoxLayout()
                video_control_layout.addWidget(self.start_video_btn)
                video_control_layout.addWidget(self.pause_video_btn)
                video_control_layout.addWidget(self.stop_video_btn)
                video_control_layout.addStretch()
                input_layout.addLayout(video_control_layout)

        # 更新UI状态
        self.update_ui_state()

    def browse_video(self):
        """浏览视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.video_path.setText(file_path)

    def start_video_inference(self):
        """开始视频推理"""
        if not self.current_model:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        input_type = self.input_type_combo.currentData()
        if input_type == "video":
            video_path = self.video_path.text()
            if not video_path:
                QMessageBox.warning(self, "警告", "请选择视频文件")
                return
            if not os.path.exists(video_path):
                QMessageBox.warning(self, "警告", "视频文件不存在")
                return
        elif input_type == "camera":
            video_path = self.camera_combo.currentData()
        elif input_type == "rtsp":
            video_path = self.rtsp_url.text().strip()
            if not video_path:
                QMessageBox.warning(self, "警告", "请输入 RTSP 流地址")
                return
            if not video_path.lower().startswith("rtsp://"):
                QMessageBox.warning(self, "警告", "RTSP 地址应以 rtsp:// 开头")
                return
        else:
            return

        # 禁用按钮
        self.start_video_btn.setEnabled(False)
        self.pause_video_btn.setEnabled(True)
        self.stop_video_btn.setEnabled(True)

        # 启动视频推理线程
        # 转换参数名称为YOLO模型接受的格式
        yolo_kwargs = {
            'conf': self.conf_threshold.value(),
            'iou': self.iou_threshold.value(),
            'max_det': self.max_det.value(),
            'imgsz': self.imgsz_spin.value()
        }
        
        # 分类任务特殊参数
        if self.task_type == 'classify':
            yolo_kwargs['topk'] = self.topk_spin.value()

        self.video_worker = VideoInferenceWorker(
            self.current_model,
            video_path,
            self.task_type,
            tracking=self.track_check.isChecked(),
            is_stream=(input_type == "rtsp"),
            **yolo_kwargs
        )

        # 连接信号
        self.video_worker.image_updated.connect(self.update_video_frame)
        self.video_worker.finished.connect(self.on_video_inference_finished)
        self.video_worker.error.connect(self.on_video_inference_error)

        # 视频模式禁用自动适应视图，避免每帧重算缩放
        if self.image_display:
            self.image_display.set_fit_to_view_enabled(False)

        # 启动线程
        self.video_worker.start()

    def pause_video_inference(self):
        """暂停视频推理"""
        if self.video_worker:
            self.video_worker.pause()
            self.pause_video_btn.setText("继续" if self.video_worker.is_paused else "暂停")

    def stop_video_inference(self):
        """停止视频推理"""
        if self.video_worker:
            self.video_worker.stop()
            self.video_worker.wait()
            self.video_worker = None

        # 恢复自动适应视图
        if self.image_display:
            self.image_display.set_fit_to_view_enabled(True)

        # 重置按钮状态
        self.start_video_btn.setEnabled(True)
        self.pause_video_btn.setEnabled(False)
        self.pause_video_btn.setText("暂停")
        self.stop_video_btn.setEnabled(False)

    def update_video_frame(self, pixmap):
        """更新视频帧（不重置缩放，保持用户缩放状态）"""
        if self.image_display:
            self.image_display.set_pixmap(pixmap)

    def on_video_inference_finished(self):
        """视频推理完成回调"""
        self.logger.info("视频推理完成")
        self.stop_video_inference()

    def on_video_inference_error(self, error_msg):
        """视频推理错误回调"""
        self.logger.error(f"视频推理失败: {error_msg}")
        QMessageBox.critical(self, "推理失败", f"错误信息:\n{error_msg}")
        self.stop_video_inference()


class VideoInferenceWorker(QThread):
    """视频推理工作线程"""

    # 信号定义
    image_updated = pyqtSignal(QPixmap)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, video_source, task_type, tracking=False, is_stream=False, **kwargs):
        super().__init__()
        self.model = model
        self.video_source = video_source
        self.task_type = task_type
        self.tracking = tracking
        self.is_stream = is_stream  # 是否为 RTSP 等网络流（断流时自动重连）
        self.kwargs = kwargs
        self.is_stopped = False
        self.is_paused = False

    def run(self):
        """执行视频推理"""
        try:
            import cv2
            import torch

            # 打开视频源
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                self.error.emit("无法打开视频源")
                return

            # 网络流断流重连计数
            retry_count = 0
            max_retries = 30 if self.is_stream else 0

            while not self.is_stopped:
                if not self.is_paused:
                    # 读取帧
                    ret, frame = cap.read()
                    if not ret:
                        if self.is_stream and retry_count < max_retries:
                            # RTSP 流断流，等待后重连
                            retry_count += 1
                            self.msleep(500)
                            cap.release()
                            cap = cv2.VideoCapture(self.video_source)
                            continue
                        break
                    retry_count = 0

                    # 执行推理
                    if self.tracking and self.task_type != "classify":
                        # 多目标跟踪模式：persist=True 保持跨帧跟踪状态
                        results = self.model.track(frame, persist=True, **self.kwargs)
                    else:
                        results = self.model(frame, **self.kwargs)

                    # 处理结果
                    if self.task_type == "classify":
                        # 分类任务处理
                        result = results[0]
                        probs = result.probs
                        topk = min(self.kwargs.get('topk', 5), len(probs.data))
                        top_indices = torch.topk(probs.data, topk).indices.tolist()
                        top_probs = torch.topk(probs.data, topk).values.tolist()

                        # 在帧上绘制结果
                        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                            class_name = result.names[idx]
                            text = f"{class_name}: {prob:.4f}"
                            cv2.putText(frame, text, (10, 30 + i*30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        # 检测/分割/姿态任务处理
                        annotated_frame = results[0].plot()
                        frame = annotated_frame

                    # 转换为Qt图像
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)

                    # 发送信号
                    self.image_updated.emit(pixmap)

                    # 控制帧率
                    self.msleep(33)  # 约30fps

            # 释放资源
            cap.release()
            self.finished.emit()

        except Exception as e:
            if not self.is_stopped:
                self.error.emit(str(e))

    def stop(self):
        """停止推理"""
        self.is_stopped = True

    def pause(self):
        """暂停/继续推理"""
        self.is_paused = not self.is_paused


class ModelLoadWorker(QThread):
    """模型加载工作线程"""

    # 信号定义
    finished = pyqtSignal(object, str)  # 模型对象, 任务类型
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.is_stopped = False

    def run(self):
        """执行模型加载"""
        import time
        t0 = time.time()
        try:
            from ultralytics import YOLO

            self.log.emit(f"开始加载模型: {os.path.basename(self.model_path)}")
            model = YOLO(self.model_path)
            task = model.task
            self.log.emit(f"模型加载完成 ({time.time()-t0:.1f}s), 任务: {task}")

            self.finished.emit(model, task)
        except Exception as e:
            if not self.is_stopped:
                self.error.emit(str(e))

    def stop(self):
        """停止加载"""
        self.is_stopped = True


class BatchInferenceWorker(QThread):
    """批量推理工作线程"""

    # 信号定义
    finished = pyqtSignal(list, list)  # 结果列表, 图像文件列表
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    image_updated = pyqtSignal(QPixmap, int, int)

    def __init__(self, engine: InferenceEngine, model, folder_path: str, output_dir: str, batch_size: int = 8, **kwargs):
        super().__init__()
        self.engine = engine
        self.model = model
        self.folder_path = folder_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.is_stopped = False
        self.results = []
        self.image_files = []

    def run(self):
        """执行批量推理"""
        try:
            import glob
            
            # 获取所有图像文件（使用不区分大小写的匹配，但去重）
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif', '*.webp']
            image_files_set = set()
            for ext in image_extensions:
                image_files_set.update(glob.glob(os.path.join(self.folder_path, ext), recursive=False))
                image_files_set.update(glob.glob(os.path.join(self.folder_path, ext.upper()), recursive=False))

            # 转换为列表并排序
            self.image_files = sorted(list(image_files_set))

            total = len(self.image_files)
            if total == 0:
                self.error.emit("文件夹中没有找到图像文件")
                return

            self.log.emit(f"找到 {total} 个图像文件")
            self.log.emit(f"批处理大小: {self.batch_size}")

            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)

            # 执行批处理推理
            results = self.engine.run_batch_inference(
                self.model, self.image_files, batch_size=self.batch_size, **self.kwargs
            )

            # 处理结果
            for i, (result, image_path) in enumerate(zip(results, self.image_files)):
                if self.is_stopped:
                    break

                if result:
                    self.results.append(result)

                    # 绘制结果
                    pixmap = self.engine.draw_results(result)
                    if pixmap and not pixmap.isNull():
                        # 保存结果
                        output_path = os.path.join(self.output_dir, f"result_{os.path.basename(image_path)}")
                        pixmap.save(output_path)

                        # 发送图像更新信号
                        self.image_updated.emit(pixmap, i + 1, total)

                # 发送进度信号
                self.progress.emit(i + 1, total)

            if not self.is_stopped:
                self.finished.emit(self.results, self.image_files)
        
        except Exception as e:
            if not self.is_stopped:
                self.error.emit(str(e))

    def stop(self):
        """停止推理"""
        self.is_stopped = True
