"""
综合模型训练标签页 - 支持检测、分割、姿态和分类任务的训练
"""
import os
from pathlib import Path
from typing import Optional
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QTextEdit,
    QFileDialog, QMessageBox, QGroupBox, QLabel, QComboBox,
    QCheckBox, QProgressBar, QScrollArea, QGridLayout, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from utils.logger import LogManager, QTextEditLogger
from core.trainer import ModelTrainer, TrainingCallback
from config.settings import AppConfig
from ui.widgets import DropLineEdit


class TrainingProgressCallback(TrainingCallback):
    """训练进度回调 - 通过 worker 的信号转发给界面"""

    def __init__(self, worker):
        super().__init__()
        self.worker = worker

    def on_epoch_end(self, trainer):
        """每个epoch结束"""
        epoch_done = trainer.epoch + 1
        progress = int(epoch_done / trainer.total_epochs * 100)
        self.worker.update_progress(progress)
        self.worker.update_epoch_info(epoch_done, trainer.total_epochs)

    def on_validation_end(self, trainer, metrics):
        """验证结束"""
        if not isinstance(metrics, dict) or not metrics:
            return
        if 'metrics/mAP50(B)' in metrics:
            self.worker.update_metric("mAP50", metrics['metrics/mAP50(B)'])
        elif 'metrics/accuracy_top1' in metrics:
            self.worker.update_metric("Top-1 Acc", metrics['metrics/accuracy_top1'])
        elif 'metrics/accuracy_top5' in metrics:
            self.worker.update_metric("Top-5 Acc", metrics['metrics/accuracy_top5'])


class TrainingWorker(QThread):
    """训练工作线程"""

    # 信号定义
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    epoch_update = pyqtSignal(int, int)
    metric_update = pyqtSignal(str, float)

    def __init__(self, trainer: ModelTrainer, **kwargs):
        super().__init__()
        self.trainer = trainer
        self.kwargs = kwargs
        self.is_stopped = False

        # 添加进度回调
        self.trainer.add_callback(TrainingProgressCallback(self))

    def run(self):
        """执行训练"""
        try:
            # 执行训练
            results = self.trainer.train(**self.kwargs)

            # 无论正常完成还是用户停止，都通知主线程清理UI
            self.finished.emit(results)
        except KeyboardInterrupt:
            # 训练被用户停止，作为正常结束处理
            self.finished.emit(None)
        except Exception as e:
            if not self.is_stopped:
                self.error.emit(str(e))
            else:
                # 停止过程中抛出的异常，按正常结束处理
                self.finished.emit(None)

    def stop(self):
        """停止训练"""
        self.is_stopped = True
        self.trainer.stop_training()

    # 供回调使用的方法
    def update_progress(self, value):
        self.progress.emit(value)

    def update_epoch_info(self, current, total):
        self.epoch_update.emit(current, total)

    def update_metric(self, name, value):
        self.metric_update.emit(name, value)


class CombinedTrainTab(QWidget):
    """综合模型训练标签页"""

    def __init__(
            self,
            logger: Optional[LogManager] = None,
            model_manager=None,
            trainer=None,
            image_display=None,
            image_label=None
    ):
        """
        初始化综合训练标签页

        Args:
            logger: 日志管理器
            model_manager: 模型管理器
            trainer: 训练器实例
            image_display: 图像显示区域
            image_label: 图像标签
        """
        super().__init__()

        self.logger = logger or LogManager()
        self.model_manager = model_manager
        self.trainer = trainer or ModelTrainer(self.logger)
        self.image_display = image_display
        self.image_label = image_label

        # 工作线程
        self.worker = None

        # 训练状态
        self.is_training = False
        self.best_model_path = None

        # 任务类型
        self.task_type = "segment"

        # 设置UI
        self.setup_ui()

        # 加载上次的配置
        self.load_config()

    def setup_ui(self):
        """设置UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # 训练配置区（占满主布局）
        self.config_tab = QWidget()
        config_layout = QVBoxLayout(self.config_tab)
        config_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.config_tab)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)
        
        config_layout.addWidget(scroll_area)

        # ========== 1. 任务类型选择 ==========
        task_group = QGroupBox("1. 任务类型")
        task_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        task_layout = QHBoxLayout(task_group)

        self.task_combo = QComboBox()
        for task_id, task_name in AppConfig.SUPPORTED_TASKS.items():
            self.task_combo.addItem(task_name, task_id)
        self.task_combo.setCurrentText("实例分割")
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        task_layout.addWidget(QLabel("选择任务:"))
        task_layout.addWidget(self.task_combo)
        task_layout.addStretch()

        scroll_layout.addWidget(task_group)

        # ========== 2. 模型配置 ==========
        model_group = QGroupBox("2. 模型配置")
        model_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        model_layout = QFormLayout(model_group)

        # 模型选择方式
        self.model_choice = QComboBox()
        self.model_choice.addItems(["加载本地模型文件", "使用预训练模型（自动下载）"])
        self.model_choice.currentTextChanged.connect(self.on_model_choice_changed)
        model_layout.addRow("模型来源:", self.model_choice)

        # 预训练模型类型
        self.pretrained_combo = QComboBox()
        self.update_pretrained_models()
        model_layout.addRow("预训练模型:", self.pretrained_combo)

        # 本地模型路径
        self.local_model_path = DropLineEdit(self, file_filter='model')
        self.local_model_path.setPlaceholderText("拖拽模型文件(.pt/.onnx)到这里，或点击浏览...")
        self.local_model_path.setEnabled(False)

        browse_local_btn = QPushButton("浏览...")
        browse_local_btn.clicked.connect(self.browse_local_model)
        browse_local_btn.setFixedWidth(80)

        local_hbox = QHBoxLayout()
        local_hbox.addWidget(self.local_model_path, 1)
        local_hbox.addWidget(browse_local_btn)
        model_layout.addRow("本地模型:", local_hbox)

        scroll_layout.addWidget(model_group)

        # ========== 3. 数据集配置 ==========
        data_group = QGroupBox("3. 数据集配置")
        data_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        data_layout = QFormLayout(data_group)

        # 数据集路径（接受文件夹和 yaml 文件，按任务类型在 browse_dataset 中区分）
        self.data_path = DropLineEdit(self, file_filter=None)
        self.data_path.setPlaceholderText("拖拽数据集文件夹或data.yaml到这里，或点击浏览...")
        self.data_path.textChanged.connect(self.update_ui_state)
        self.data_path.pathDropped.connect(self.load_dataset_info)

        browse_data_btn = QPushButton("浏览...")
        browse_data_btn.clicked.connect(self.browse_dataset)
        browse_data_btn.setFixedWidth(80)

        data_hbox = QHBoxLayout()
        data_hbox.addWidget(self.data_path, 1)
        data_hbox.addWidget(browse_data_btn)
        data_layout.addRow("数据集:", data_hbox)

        # 数据集信息显示
        self.data_info_label = QLabel("未加载数据集信息")
        self.data_info_label.setStyleSheet("color: #666666; font-size: 11px;")
        data_layout.addRow("", self.data_info_label)

        scroll_layout.addWidget(data_group)

        # ========== 4. 训练参数 ==========
        param_group = QGroupBox("4. 训练参数")
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

        # 第一行
        param_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(AppConfig.DEFAULT_EPOCHS)
        self.epochs_spin.setFixedWidth(100)
        param_layout.addWidget(self.epochs_spin, 0, 1, alignment=Qt.AlignLeft)

        param_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(AppConfig.DEFAULT_BATCH_SIZE)
        self.batch_spin.setFixedWidth(100)
        param_layout.addWidget(self.batch_spin, 0, 3, alignment=Qt.AlignLeft)

        # 第二行
        param_layout.addWidget(QLabel("Image Size:"), 1, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(AppConfig.MIN_IMAGE_SIZE, AppConfig.MAX_IMAGE_SIZE)
        self.imgsz_spin.setValue(AppConfig.DEFAULT_IMAGE_SIZE)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setFixedWidth(100)
        param_layout.addWidget(self.imgsz_spin, 1, 1, alignment=Qt.AlignLeft)

        param_layout.addWidget(QLabel("Workers:"), 1, 2)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(AppConfig.DEFAULT_WORKERS)
        self.workers_spin.setFixedWidth(100)
        param_layout.addWidget(self.workers_spin, 1, 3, alignment=Qt.AlignLeft)

        # 第三行
        param_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(AppConfig.DEFAULT_LR)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setFixedWidth(100)
        param_layout.addWidget(self.lr_spin, 2, 1, alignment=Qt.AlignLeft)

        param_layout.addWidget(QLabel("Patience:"), 2, 2)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 200)
        self.patience_spin.setValue(AppConfig.DEFAULT_PATIENCE)
        self.patience_spin.setToolTip("早停耐心值，0表示禁用早停")
        self.patience_spin.setFixedWidth(100)
        param_layout.addWidget(self.patience_spin, 2, 3, alignment=Qt.AlignLeft)

        # 第四行 - 训练策略
        param_layout.addWidget(QLabel("学习率调度:"), 3, 0)
        self.lr_scheduler_combo = QComboBox()
        self.lr_scheduler_combo.addItems(["cos", "linear", "step", "plateau"])
        self.lr_scheduler_combo.setFixedWidth(100)
        param_layout.addWidget(self.lr_scheduler_combo, 3, 1, alignment=Qt.AlignLeft)

        param_layout.addWidget(QLabel("混合精度:"), 3, 2)
        self.amp_check = QCheckBox()
        self.amp_check.setChecked(True)
        self.amp_check.setToolTip("启用混合精度训练，加速训练过程")
        param_layout.addWidget(self.amp_check, 3, 3, alignment=Qt.AlignLeft)

        scroll_layout.addWidget(param_group)

        # ========== 5. 优化器配置 ==========
        optimizer_group = QGroupBox("5. 优化器配置")
        optimizer_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        optimizer_layout = QGridLayout(optimizer_group)
        # 调整间距，使标签和输入控件更紧凑
        optimizer_layout.setHorizontalSpacing(10)  # 减小水平间距
        optimizer_layout.setVerticalSpacing(10)  # 调整垂直间距
        # 设置列的拉伸比例，使标签和输入控件对齐
        optimizer_layout.setColumnStretch(0, 0)
        optimizer_layout.setColumnStretch(1, 1)
        optimizer_layout.setColumnStretch(2, 0)
        optimizer_layout.setColumnStretch(3, 1)

        # 优化器选择
        optimizer_layout.addWidget(QLabel("Optimizer:"), 0, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "RMSProp"])
        self.optimizer_combo.setFixedWidth(100)
        optimizer_layout.addWidget(self.optimizer_combo, 0, 1, alignment=Qt.AlignLeft)

        # 学习率调度
        optimizer_layout.addWidget(QLabel("LR Schedule:"), 0, 2)
        self.lr_schedule_combo = QComboBox()
        self.lr_schedule_combo.addItems(["cos", "linear", "constant"])
        self.lr_schedule_combo.setFixedWidth(100)
        optimizer_layout.addWidget(self.lr_schedule_combo, 0, 3, alignment=Qt.AlignLeft)

        # 预热
        optimizer_layout.addWidget(QLabel("Warmup Epochs:"), 1, 0)
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 10)
        self.warmup_spin.setValue(3)
        self.warmup_spin.setFixedWidth(100)
        optimizer_layout.addWidget(self.warmup_spin, 1, 1, alignment=Qt.AlignLeft)

        optimizer_layout.addWidget(QLabel("Warmup LR:"), 1, 2)
        self.warmup_lr_spin = QDoubleSpinBox()
        self.warmup_lr_spin.setRange(0.00001, 0.1)
        self.warmup_lr_spin.setValue(0.01)
        self.warmup_lr_spin.setDecimals(4)
        self.warmup_lr_spin.setFixedWidth(100)
        optimizer_layout.addWidget(self.warmup_lr_spin, 1, 3, alignment=Qt.AlignLeft)

        # 权重衰减和动量（分类任务常用）
        optimizer_layout.addWidget(QLabel("Weight Decay:"), 2, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0, 0.1)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setDecimals(4)
        self.weight_decay_spin.setFixedWidth(100)
        optimizer_layout.addWidget(self.weight_decay_spin, 2, 1, alignment=Qt.AlignLeft)

        optimizer_layout.addWidget(QLabel("Momentum:"), 2, 2)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0, 1)
        self.momentum_spin.setValue(0.937)
        self.momentum_spin.setDecimals(3)
        self.momentum_spin.setFixedWidth(100)
        optimizer_layout.addWidget(self.momentum_spin, 2, 3, alignment=Qt.AlignLeft)

        scroll_layout.addWidget(optimizer_group)

        # ========== 6. 数据增强配置 ==========
        augment_group = QGroupBox("6. 数据增强")
        augment_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        augment_group.setCheckable(True)
        augment_group.setChecked(True)
        augment_layout = QGridLayout(augment_group)
        # 调整间距，使标签和输入控件更紧凑
        augment_layout.setHorizontalSpacing(10)  # 减小水平间距
        augment_layout.setVerticalSpacing(10)  # 调整垂直间距
        # 设置列的拉伸比例，使标签和输入控件对齐
        augment_layout.setColumnStretch(0, 0)
        augment_layout.setColumnStretch(1, 1)
        augment_layout.setColumnStretch(2, 0)
        augment_layout.setColumnStretch(3, 1)

        # 第一行
        augment_layout.addWidget(QLabel("Mosaic:"), 0, 0)
        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0, 1)
        self.mosaic_spin.setValue(1.0)
        self.mosaic_spin.setSingleStep(0.1)
        self.mosaic_spin.setFixedWidth(100)
        augment_layout.addWidget(self.mosaic_spin, 0, 1, alignment=Qt.AlignLeft)

        augment_layout.addWidget(QLabel("MixUp:"), 0, 2)
        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0, 1)
        self.mixup_spin.setValue(0.0)
        self.mixup_spin.setSingleStep(0.1)
        self.mixup_spin.setFixedWidth(100)
        augment_layout.addWidget(self.mixup_spin, 0, 3, alignment=Qt.AlignLeft)

        # 第二行
        augment_layout.addWidget(QLabel("Copy-Paste:"), 1, 0)
        self.copypaste_spin = QDoubleSpinBox()
        self.copypaste_spin.setRange(0, 1)
        self.copypaste_spin.setValue(0.0)
        self.copypaste_spin.setSingleStep(0.1)
        self.copypaste_spin.setFixedWidth(100)
        augment_layout.addWidget(self.copypaste_spin, 1, 1, alignment=Qt.AlignLeft)

        augment_layout.addWidget(QLabel("Flip LR:"), 1, 2)
        self.flip_lr_spin = QDoubleSpinBox()
        self.flip_lr_spin.setRange(0, 1)
        self.flip_lr_spin.setValue(0.5)
        self.flip_lr_spin.setSingleStep(0.1)
        self.flip_lr_spin.setFixedWidth(100)
        augment_layout.addWidget(self.flip_lr_spin, 1, 3, alignment=Qt.AlignLeft)

        # 第三行
        augment_layout.addWidget(QLabel("HSV-H:"), 2, 0)
        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0, 0.1)
        self.hsv_h_spin.setValue(0.015)
        self.hsv_h_spin.setSingleStep(0.005)
        self.hsv_h_spin.setFixedWidth(100)
        augment_layout.addWidget(self.hsv_h_spin, 2, 1, alignment=Qt.AlignLeft)

        augment_layout.addWidget(QLabel("HSV-S:"), 2, 2)
        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0, 1)
        self.hsv_s_spin.setValue(0.7)
        self.hsv_s_spin.setSingleStep(0.1)
        self.hsv_s_spin.setFixedWidth(100)
        augment_layout.addWidget(self.hsv_s_spin, 2, 3, alignment=Qt.AlignLeft)

        # 第四行
        augment_layout.addWidget(QLabel("HSV-V:"), 3, 0)
        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0, 1)
        self.hsv_v_spin.setValue(0.4)
        self.hsv_v_spin.setSingleStep(0.1)
        self.hsv_v_spin.setFixedWidth(100)
        augment_layout.addWidget(self.hsv_v_spin, 3, 1, alignment=Qt.AlignLeft)

        augment_layout.addWidget(QLabel("Degrees:"), 3, 2)
        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(0, 180)
        self.degrees_spin.setValue(0.0)
        self.degrees_spin.setFixedWidth(100)
        augment_layout.addWidget(self.degrees_spin, 3, 3, alignment=Qt.AlignLeft)

        scroll_layout.addWidget(augment_group)

        # ========== 7. 输出配置 ==========
        output_group = QGroupBox("7. 输出配置")
        output_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        output_layout = QFormLayout(output_group)

        # 项目目录
        self.project_path = DropLineEdit(self, file_filter='folder')
        self.project_path.setPlaceholderText("拖拽输出目录到这里，或点击浏览...")
        self.project_path.setText(str(AppConfig.DEFAULT_OUTPUT_DIR))

        browse_project_btn = QPushButton("浏览...")
        browse_project_btn.clicked.connect(self.browse_project_folder)
        browse_project_btn.setFixedWidth(80)

        project_hbox = QHBoxLayout()
        project_hbox.addWidget(self.project_path, 1)
        project_hbox.addWidget(browse_project_btn)
        output_layout.addRow("项目目录:", project_hbox)

        # 实验名称
        self.experiment_name = QLineEdit()
        self.experiment_name.setPlaceholderText("例如: experiment_1")
        self.experiment_name.setText("train")
        output_layout.addRow("实验名称:", self.experiment_name)

        # 是否覆盖
        self.exist_ok_check = QCheckBox("允许覆盖已有实验")
        self.exist_ok_check.setChecked(True)
        output_layout.addRow("", self.exist_ok_check)

        scroll_layout.addWidget(output_group)

        # ========== 8. 训练控制按钮 ==========
        button_layout = QHBoxLayout()

        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        self.train_btn.setStyleSheet("""
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
        button_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.clicked.connect(self.stop_training)
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
                background-color: #cccccc;  /* 禁用时灰色 */
                color: #666666;
            }
        """)
        button_layout.addWidget(self.stop_btn)

        self.validate_btn = QPushButton("验证最佳模型")
        self.validate_btn.clicked.connect(self.validate_best_model)
        self.validate_btn.setEnabled(False)
        button_layout.addWidget(self.validate_btn)

        self.open_folder_btn = QPushButton("打开输出文件夹")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        self.open_folder_btn.setEnabled(False)
        button_layout.addWidget(self.open_folder_btn)

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

        # ========== 9. 进度显示 ==========
        progress_group = QGroupBox("训练进度")
        progress_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        progress_layout = QVBoxLayout(progress_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        # 训练信息
        info_layout = QHBoxLayout()

        self.epoch_label = QLabel("Epoch: 0/0")
        info_layout.addWidget(self.epoch_label)

        info_layout.addStretch()

        self.metric_label = QLabel("mAP50: --")
        info_layout.addWidget(self.metric_label)

        progress_layout.addLayout(info_layout)

        scroll_layout.addWidget(progress_group)

        # 添加伸缩项
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_widget)

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

        # 训练日志
        log_group = QGroupBox("训练日志")
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

        # 添加内容区域到分割器
        splitter.addWidget(log_group)

        # 设置初始大小比例（70% 上面，30% 日志）
        splitter.setSizes([700, 300])

        config_layout.addWidget(splitter)

        # 初始化UI状态
        self.update_ui_state()

    def on_task_changed(self, index):
        """任务类型改变回调"""
        self.task_type = self.task_combo.currentData()
        self.logger.info(f"任务类型已切换为: {self.task_type}")

        # 更新数据集输入框的占位提示
        if self.task_type == 'classify':
            self.data_path.setPlaceholderText("拖拽分类数据集文件夹（含 train/val 子目录）到这里，或点击浏览...")
        else:
            self.data_path.setPlaceholderText("拖拽 data.yaml 文件或数据集文件夹到这里，或点击浏览...")

        # 更新预训练模型列表
        self.update_pretrained_models()

        # 调整UI显示和参数 - 严格按照YOLO官方默认值
        if self.task_type == 'classify':
            self.logger.info("分类任务：使用分类模型和训练参数")
            self.metric_label.setText("Top-1 Acc: --")
            # 设置分类任务默认参数（YOLO官方标准）
            self.epochs_spin.setValue(300)  # YOLO官方默认
            self.imgsz_spin.setValue(224)   # 分类任务默认图像大小
            self.batch_spin.setValue(8)     # 分类任务常用batch size
            self.lr_spin.setValue(0.001)     # 分类任务学习率
            # 优化器参数
            self.optimizer_combo.setCurrentText("AdamW")  # 分类任务常用AdamW
            self.weight_decay_spin.setValue(0.01)  # 分类任务权重衰减
            self.momentum_spin.setValue(0.9)  # 分类任务动量
            # 训练策略参数
            self.lr_scheduler_combo.setCurrentText("cos")
            self.amp_check.setChecked(True)
            # 数据增强参数
            self.mosaic_spin.setValue(0.0)    # 分类任务不使用马赛克
            self.mixup_spin.setValue(0.1)     # 分类任务使用MixUp
            self.copypaste_spin.setValue(0.0) # 分类任务不使用Copy-Paste
            self.flip_lr_spin.setValue(0.5)   # 分类任务使用水平翻转
            self.hsv_h_spin.setValue(0.015)   # 分类任务色彩调整
            self.hsv_s_spin.setValue(0.4)     # 分类任务饱和度调整
            self.hsv_v_spin.setValue(0.4)     # 分类任务亮度调整
            self.degrees_spin.setValue(10.0)  # 分类任务旋转
        elif self.task_type == 'segment':
            self.logger.info("分割任务：使用分割模型")
            self.metric_label.setText("mAP50: --")
            # 设置分割任务默认参数（YOLO官方标准）
            self.epochs_spin.setValue(300)    # YOLO官方默认
            self.imgsz_spin.setValue(640)
            self.batch_spin.setValue(8)
            self.lr_spin.setValue(0.01)       # YOLO官方默认学习率
            # 优化器参数
            self.optimizer_combo.setCurrentText("auto")
            self.weight_decay_spin.setValue(0.0005)
            self.momentum_spin.setValue(0.937)
            # 训练策略参数
            self.lr_scheduler_combo.setCurrentText("cos")
            self.amp_check.setChecked(True)
            # 数据增强参数
            self.mosaic_spin.setValue(1.0)    # 分割任务使用马赛克
            self.mixup_spin.setValue(0.0)     # 分割任务不使用MixUp
            self.copypaste_spin.setValue(0.0) # 分割任务不使用Copy-Paste
            self.flip_lr_spin.setValue(0.5)   # 分割任务使用水平翻转
            self.hsv_h_spin.setValue(0.015)   # 分割任务色彩调整
            self.hsv_s_spin.setValue(0.7)     # 分割任务饱和度调整
            self.hsv_v_spin.setValue(0.4)     # 分割任务亮度调整
            self.degrees_spin.setValue(0.0)   # 分割任务不使用旋转
        elif self.task_type == 'pose':
            self.logger.info("姿态估计任务：使用姿态模型")
            self.metric_label.setText("mAP50: --")
            # 设置姿态估计任务默认参数（YOLO官方标准）
            self.epochs_spin.setValue(300)    # YOLO官方默认
            self.imgsz_spin.setValue(640)
            self.batch_spin.setValue(8)
            self.lr_spin.setValue(0.01)       # YOLO官方默认学习率
            # 优化器参数
            self.optimizer_combo.setCurrentText("auto")
            self.weight_decay_spin.setValue(0.0005)
            self.momentum_spin.setValue(0.937)
            # 训练策略参数
            self.lr_scheduler_combo.setCurrentText("cos")
            self.amp_check.setChecked(True)
            # 数据增强参数
            self.mosaic_spin.setValue(0.6)    # 姿态估计任务适度使用马赛克
            self.mixup_spin.setValue(0.0)     # 姿态估计任务不使用MixUp
            self.copypaste_spin.setValue(0.0) # 姿态估计任务不使用Copy-Paste
            self.flip_lr_spin.setValue(0.5)   # 姿态估计任务使用水平翻转
            self.hsv_h_spin.setValue(0.015)   # 姿态估计任务色彩调整
            self.hsv_s_spin.setValue(0.7)     # 姿态估计任务饱和度调整
            self.hsv_v_spin.setValue(0.4)     # 姿态估计任务亮度调整
            self.degrees_spin.setValue(0.0)   # 姿态估计任务不使用旋转
        elif self.task_type == 'obb':
            self.logger.info("旋转框检测任务：使用OBB模型")
            self.metric_label.setText("mAP50: --")
            # 设置旋转框检测任务默认参数（YOLO官方标准）
            self.epochs_spin.setValue(300)    # YOLO官方默认
            self.imgsz_spin.setValue(640)
            self.batch_spin.setValue(8)
            self.lr_spin.setValue(0.01)       # YOLO官方默认学习率
            # 优化器参数
            self.optimizer_combo.setCurrentText("auto")
            self.weight_decay_spin.setValue(0.0005)
            self.momentum_spin.setValue(0.937)
            # 训练策略参数
            self.lr_scheduler_combo.setCurrentText("cos")
            self.amp_check.setChecked(True)
            # 数据增强参数
            self.mosaic_spin.setValue(1.0)    # 旋转框检测任务使用马赛克
            self.mixup_spin.setValue(0.0)     # 旋转框检测任务不使用MixUp
            self.copypaste_spin.setValue(0.0) # 旋转框检测任务不使用Copy-Paste
            self.flip_lr_spin.setValue(0.5)   # 旋转框检测任务使用水平翻转
            self.hsv_h_spin.setValue(0.015)   # 旋转框检测任务色彩调整
            self.hsv_s_spin.setValue(0.7)     # 旋转框检测任务饱和度调整
            self.hsv_v_spin.setValue(0.4)     # 旋转框检测任务亮度调整
            self.degrees_spin.setValue(30.0)  # 旋转框检测任务使用旋转
        elif self.task_type == 'semantic':
            self.logger.info("语义分割任务：使用语义分割模型")
            self.metric_label.setText("mIoU: --")
            self.epochs_spin.setValue(300)
            self.imgsz_spin.setValue(640)
            self.batch_spin.setValue(8)
            self.lr_spin.setValue(0.01)
            self.optimizer_combo.setCurrentText("auto")
            self.weight_decay_spin.setValue(0.0005)
            self.momentum_spin.setValue(0.937)
            self.lr_scheduler_combo.setCurrentText("cos")
            self.amp_check.setChecked(True)
            self.mosaic_spin.setValue(1.0)
            self.mixup_spin.setValue(0.0)
            self.copypaste_spin.setValue(0.0)
            self.flip_lr_spin.setValue(0.5)
            self.hsv_h_spin.setValue(0.015)
            self.hsv_s_spin.setValue(0.7)
            self.hsv_v_spin.setValue(0.4)
            self.degrees_spin.setValue(0.0)
        elif self.task_type == 'depth':
            self.logger.info("深度估计任务：使用深度估计模型")
            self.metric_label.setText("AbsRel: --")
            self.epochs_spin.setValue(300)
            self.imgsz_spin.setValue(640)
            self.batch_spin.setValue(8)
            self.lr_spin.setValue(0.01)
            self.optimizer_combo.setCurrentText("auto")
            self.weight_decay_spin.setValue(0.0005)
            self.momentum_spin.setValue(0.937)
            self.lr_scheduler_combo.setCurrentText("cos")
            self.amp_check.setChecked(True)
            self.mosaic_spin.setValue(1.0)
            self.mixup_spin.setValue(0.0)
            self.copypaste_spin.setValue(0.0)
            self.flip_lr_spin.setValue(0.5)
            self.hsv_h_spin.setValue(0.015)
            self.hsv_s_spin.setValue(0.7)
            self.hsv_v_spin.setValue(0.4)
            self.degrees_spin.setValue(0.0)
        else:  # detect
            self.logger.info("检测任务：使用检测模型")
            self.metric_label.setText("mAP50: --")
            # 设置检测任务默认参数（YOLO官方标准）
            self.epochs_spin.setValue(300)    # YOLO官方默认
            self.imgsz_spin.setValue(640)
            self.batch_spin.setValue(8)
            self.lr_spin.setValue(0.01)       # YOLO官方默认学习率
            # 优化器参数
            self.optimizer_combo.setCurrentText("auto")
            self.weight_decay_spin.setValue(0.0005)
            self.momentum_spin.setValue(0.937)
            # 训练策略参数
            self.lr_scheduler_combo.setCurrentText("cos")
            self.amp_check.setChecked(True)
            # 数据增强参数
            self.mosaic_spin.setValue(1.0)    # 检测任务使用马赛克
            self.mixup_spin.setValue(0.0)     # 检测任务不使用MixUp
            self.copypaste_spin.setValue(0.0) # 检测任务不使用Copy-Paste
            self.flip_lr_spin.setValue(0.5)   # 检测任务使用水平翻转
            self.hsv_h_spin.setValue(0.015)   # 检测任务色彩调整
            self.hsv_s_spin.setValue(0.7)     # 检测任务饱和度调整
            self.hsv_v_spin.setValue(0.4)     # 检测任务亮度调整
            self.degrees_spin.setValue(0.0)   # 检测任务不使用旋转

        # 清空之前的结果
        self.clear_results()

    def clear_results(self):
        """清空训练结果"""
        # 清空进度条
        self.progress_bar.setValue(0)
        self.epoch_label.setText("Epoch: 0/0")
        self.metric_label.setText("mAP50: --" if self.task_type != 'classify' else "Top-1 Acc: --")
        
        # 清空日志
        self.log_display.clear()
        
        # 重置训练状态
        self.is_training = False
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("停止训练")
        self.validate_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    def update_pretrained_models(self):
        """更新预训练模型列表"""
        task_id = self.task_combo.currentData()
        models = AppConfig.get_pretrained_models(task_id)

        self.pretrained_combo.clear()
        self.pretrained_combo.addItems(models)

    def on_model_choice_changed(self, choice):
        """模型选择方式改变"""
        is_local = (choice == "加载本地模型文件")
        self.pretrained_combo.setEnabled(not is_local)
        self.local_model_path.setEnabled(is_local)

    def browse_local_model(self):
        """浏览本地模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "模型文件 (*.pt);;所有文件 (*.*)"
        )
        if file_path:
            self.local_model_path.setText(file_path)

    def browse_dataset(self):
        """浏览数据集（分类任务选文件夹，其他任务选 data.yaml 或含 data.yaml 的文件夹）"""
        task = getattr(self, 'task_type', 'detect')
        if task == 'classify':
            # 图像分类：选择数据集根文件夹（含 train/val 子目录）
            folder = QFileDialog.getExistingDirectory(self, "选择分类数据集文件夹", "")
            if folder:
                self.data_path.setText(folder)
                self.load_dataset_info(folder)
        else:
            # 检测/分割等：优先选择 data.yaml，也允许选择含 data.yaml 的文件夹
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择数据集配置文件", "",
                "YAML文件 (*.yaml *.yml);;所有文件 (*)"
            )
            if file_path:
                self.data_path.setText(file_path)
                self.load_dataset_info(file_path)

    def load_dataset_info(self, data_path: str):
        """加载数据集信息（带缓存，避免切换标签页时重复扫描目录）"""
        # 缓存：同一路径直接复用上次的显示文本
        if not hasattr(self, '_dataset_info_cache'):
            self._dataset_info_cache = {}
        if data_path in self._dataset_info_cache:
            self.data_info_label.setText(self._dataset_info_cache[data_path])
            self.update_ui_state()
            return

        try:
            if data_path.endswith('.yaml') or data_path.endswith('.yml'):
                import yaml
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                # 显示数据集信息
                info_text = f"类别数: {data.get('nc', 'N/A')} | "
                info_text += f"训练集: {data.get('train', 'N/A')} | "
                info_text += f"验证集: {data.get('val', 'N/A')}"

                if 'names' in data:
                    names = data['names']
                    if isinstance(names, list):
                        info_text += f"\n类别: {', '.join(names[:5])}"
                        if len(names) > 5:
                            info_text += f" 等{len(names)}个类别"

                self.data_info_label.setText(info_text)
                self.logger.info(f"数据集信息加载成功: {info_text}")
            else:
                task = getattr(self, 'task_type', 'detect')
                base_path = Path(data_path)

                if task == 'classify':
                    # 分类数据集：统计 train/val 子目录
                    train_path = base_path / 'train'
                    val_path = base_path / 'val'

                    train_classes = 0
                    val_classes = 0
                    train_images = 0
                    val_images = 0

                    if train_path.exists():
                        classes = [d for d in train_path.iterdir() if d.is_dir()]
                        train_classes = len(classes)
                        for class_dir in classes:
                            images = list(class_dir.glob('*.*'))
                            train_images += len(images)

                    if val_path.exists():
                        classes = [d for d in val_path.iterdir() if d.is_dir()]
                        val_classes = len(classes)
                        for class_dir in classes:
                            images = list(class_dir.glob('*.*'))
                            val_images += len(images)

                    info_text = f"训练集: {train_images} 张图片, {train_classes} 个类别 | "
                    info_text += f"验证集: {val_images} 张图片, {val_classes} 个类别"
                else:
                    # 检测/分割等任务：文件夹内需含 data.yaml
                    yaml_candidates = list(base_path.glob('*.yaml')) + list(base_path.glob('*.yml'))
                    if yaml_candidates:
                        import yaml
                        with open(yaml_candidates[0], 'r', encoding='utf-8') as f:
                            data = yaml.safe_load(f)
                        # 将路径修正为 yaml 文件路径，供训练使用
                        self.data_path.setText(str(yaml_candidates[0]))
                        info_text = f"已自动定位: {yaml_candidates[0].name} | "
                        info_text += f"类别数: {data.get('nc', 'N/A')} | "
                        info_text += f"训练集: {data.get('train', 'N/A')} | "
                        info_text += f"验证集: {data.get('val', 'N/A')}"
                    else:
                        info_text = "警告: 该文件夹下未找到 data.yaml，请选择含 data.yaml 的文件夹或直接选择 yaml 文件"

                self.data_info_label.setText(info_text)
                self.logger.info(f"数据集信息加载成功: {info_text}")

            # 写入缓存（同一路径下次切换标签页时直接复用）
            self._dataset_info_cache[data_path] = info_text

            # 更新UI状态
            self.update_ui_state()

        except Exception as e:
            self.logger.error(f"加载数据集信息失败: {e}")
            self.data_info_label.setText("加载数据集信息失败")
            # 更新UI状态
            self.update_ui_state()

    def browse_project_folder(self):
        """浏览项目目录"""
        folder = QFileDialog.getExistingDirectory(
            self, "选择项目目录"
        )
        if folder:
            self.project_path.setText(folder)

    def start_training(self):
        """开始训练"""
        # 验证输入
        data_path = self.data_path.text()
        if not data_path or not os.path.exists(data_path):
            QMessageBox.warning(self, "警告", "请选择有效的数据集")
            return

        # 获取模型路径
        model_choice = self.model_choice.currentText()
        if model_choice == "加载本地模型文件":
            model_path = self.local_model_path.text()
            if not model_path or not os.path.exists(model_path):
                QMessageBox.warning(self, "警告", "请选择有效的本地模型文件")
                return
        else:
            model_path = self.pretrained_combo.currentText()

        # 收集训练参数
        task = self.task_combo.currentData()
        epochs = self.epochs_spin.value()
        batch = self.batch_spin.value()
        imgsz = self.imgsz_spin.value()
        workers = self.workers_spin.value()
        lr = self.lr_spin.value()
        patience = self.patience_spin.value()
        optimizer = self.optimizer_combo.currentText().lower()
        lr_scheduler = self.lr_scheduler_combo.currentText()
        amp = self.amp_check.isChecked()

        # 输出路径
        project = self.project_path.text()
        name = self.experiment_name.text()
        exist_ok = self.exist_ok_check.isChecked()

        # 数据增强参数
        augment_params = {
            'mosaic': self.mosaic_spin.value(),
            'mixup': self.mixup_spin.value(),
            'copy_paste': self.copypaste_spin.value(),
            'fliplr': self.flip_lr_spin.value(),
            'hsv_h': self.hsv_h_spin.value(),
            'hsv_s': self.hsv_s_spin.value(),
            'hsv_v': self.hsv_v_spin.value(),
            'degrees': self.degrees_spin.value()
        }

        # 清空日志
        self.log_display.clear()
        self.logger.info("=" * 60)
        self.logger.info("开始模型训练")

        self.logger.info(f"任务类型: {task}")
        self.logger.info(f"模型: {model_path}")
        self.logger.info(f"数据集: {data_path}")
        self.logger.info(f"训练参数: epochs={epochs}, batch={batch}, imgsz={imgsz}")
        self.logger.info(f"优化器: {optimizer}, lr={lr}, patience={patience}")
        self.logger.info(f"训练策略: lr_scheduler={lr_scheduler}, amp={amp}")
        self.logger.info(f"输出路径: {os.path.join(project, name)}")
        self.logger.info("=" * 60)

        # 创建并启动工作线程
        self.worker = TrainingWorker(
            trainer=self.trainer,
            model=model_path,
            data=data_path,
            task=task,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            workers=workers,
            device=None,  # 自动选择
            project=project,
            name=name,
            exist_ok=exist_ok,
            pretrained=True,
            optimizer=optimizer,
            lr0=lr,
            patience=patience,
            cos_lr=(lr_scheduler == "cos"),
            warmup_epochs=self.warmup_spin.value(),
            warmup_bias_lr=self.warmup_lr_spin.value(),
            weight_decay=self.weight_decay_spin.value(),
            momentum=self.momentum_spin.value(),
            amp=amp,
            **augment_params
        )

        # 连接信号
        self.worker.finished.connect(self.on_training_finished)
        self.worker.error.connect(self.on_training_error)
        self.worker.log.connect(self.logger.info)
        self.worker.progress.connect(self.update_progress)
        self.worker.epoch_update.connect(self.update_epoch_info)
        self.worker.metric_update.connect(self.update_metric)

        # 更新UI状态
        self.is_training = True
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.validate_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 启动线程
        self.worker.start()

    def stop_training(self):
        """停止训练"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "确认停止",
                "确定要停止训练吗？\n训练进度将被保存。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.stop_btn.setEnabled(False)
                self.stop_btn.setText("正在停止...")

    def update_progress(self, value: int):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def update_epoch_info(self, current: int, total: int):
        """更新epoch信息"""
        self.epoch_label.setText(f"Epoch: {current}/{total}")

    def update_metric(self, name: str, value: float):
        """更新指标"""
        self.metric_label.setText(f"{name}: {value:.4f}")

    def on_training_finished(self, results):
        """训练完成回调（含正常完成和用户停止两种情况）"""
        worker_stopped = self.worker is not None and getattr(self.worker, 'is_stopped', False)
        was_stopped = worker_stopped or results is None

        self.logger.info("=" * 60)
        if was_stopped:
            self.logger.info("训练已停止")
        else:
            self.logger.info("训练完成！")

        # 获取结果信息
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {key}: {value:.4f}")

        # 获取最佳模型路径
        project = self.project_path.text()
        name = self.experiment_name.text()
        self.best_model_path = os.path.join(project, name, 'weights', 'best.pt')

        if os.path.exists(self.best_model_path):
            self.logger.info(f"最佳模型: {self.best_model_path}")

        self.logger.info("=" * 60)

        # 更新UI
        self.is_training = False
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("停止训练")
        self.validate_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # 保存配置
        self.save_config()

        # 仅在正常完成且存在最佳模型时询问是否验证
        if not was_stopped and self.best_model_path and os.path.exists(self.best_model_path):
            reply = QMessageBox.question(
                self, "训练完成",
                f"训练完成！最佳模型保存在:\n{self.best_model_path}\n\n是否立即验证模型？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                self.validate_best_model()

        # 清理线程
        self.worker = None

    def on_training_error(self, error_msg: str):
        """训练错误回调"""
        self.logger.error(f"训练失败: {error_msg}")

        QMessageBox.critical(self, "训练失败", f"错误信息:\n{error_msg}")

        self.is_training = False
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("停止训练")
        self.progress_bar.setVisible(False)
        self.worker = None

    def validate_best_model(self):
        """验证最佳模型"""
        if not self.best_model_path or not os.path.exists(self.best_model_path):
            QMessageBox.warning(self, "警告", "最佳模型文件不存在")
            return

        data_path = self.data_path.text()
        if not data_path:
            QMessageBox.warning(self, "警告", "请选择数据集")
            return

        self.logger.info(f"验证模型: {self.best_model_path}")

        try:
            # 执行验证
            metrics = self.trainer.validate(
                model_path=self.best_model_path,
                data=data_path,
                imgsz=self.imgsz_spin.value(),
                batch=self.batch_spin.value(),
                project=self.project_path.text(),
                name=f"{self.experiment_name.text()}_val"
            )

            # 显示结果
            if hasattr(metrics, 'results_dict'):
                result_text = "验证结果:\n"
                for key, value in metrics.results_dict.items():
                    if isinstance(value, (int, float)):
                        result_text += f"  {key}: {value:.4f}\n"

                QMessageBox.information(self, "验证结果", result_text)

        except Exception as e:
            self.logger.error(f"验证失败: {e}")
            QMessageBox.critical(self, "验证失败", str(e))

    def open_output_folder(self):
        """打开输出文件夹"""
        project = self.project_path.text()
        name = self.experiment_name.text()
        output_dir = os.path.join(project, name)

        if os.path.exists(output_dir):
            import subprocess
            import platform

            if platform.system() == "Windows":
                os.startfile(output_dir)
            elif platform.system() == "Darwin":
                subprocess.run(["open", output_dir])
            else:
                subprocess.run(["xdg-open", output_dir])
        else:
            QMessageBox.warning(self, "警告", "输出文件夹不存在")

    def update_ui_state(self):
        """更新UI状态"""
        has_data = bool(self.data_path.text())
        self.train_btn.setEnabled(has_data and not self.is_training)

    def save_config(self):
        """保存配置"""
        config = {
            'task': self.task_type,
            'model_choice': self.model_choice.currentText(),
            'pretrained_model': self.pretrained_combo.currentText(),
            'local_model': self.local_model_path.text(),
            'data_path': self.data_path.text(),
            'epochs': self.epochs_spin.value(),
            'batch': self.batch_spin.value(),
            'imgsz': self.imgsz_spin.value(),
            'workers': self.workers_spin.value(),
            'lr': self.lr_spin.value(),
            'patience': self.patience_spin.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'lr_schedule': self.lr_schedule_combo.currentText(),
            'warmup_epochs': self.warmup_spin.value(),
            'warmup_lr': self.warmup_lr_spin.value(),
            'weight_decay': self.weight_decay_spin.value(),
            'momentum': self.momentum_spin.value(),
            'project': self.project_path.text(),
            'experiment_name': self.experiment_name.text(),
            'exist_ok': self.exist_ok_check.isChecked(),
            # 数据增强参数
            'mosaic': self.mosaic_spin.value(),
            'mixup': self.mixup_spin.value(),
            'copypaste': self.copypaste_spin.value(),
            'fliplr': self.flip_lr_spin.value(),
            'hsv_h': self.hsv_h_spin.value(),
            'hsv_s': self.hsv_s_spin.value(),
            'hsv_v': self.hsv_v_spin.value(),
            'degrees': self.degrees_spin.value()
        }

        # 保存到文件
        config_dir = Path.home() / ".yolo_ui_workstation"
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "combined_train_config.json"
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")

    def load_config(self):
        """加载配置（现在不加载，每次启动使用默认值）"""
        # 不加载之前的配置，确保每次启动都使用默认值
        pass

    def reset_to_defaults(self):
        """恢复默认参数设置"""
        reply = QMessageBox.question(
            self, '确认恢复默认值',
            '确定要将所有训练参数恢复为默认值吗？\n\n这将重置：\n- 训练参数（轮数、批次大小等）\n- 优化器设置\n- 数据增强设置\n- 输出配置',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # 重置任务类型
                self.task_combo.setCurrentText("实例分割")

                # 重置模型选择
                self.model_choice.setCurrentText("使用预训练模型（自动下载）")
                self.on_model_choice_changed(0)  # 触发模型选择变化

                # 重置训练参数
                self.epochs_spin.setValue(AppConfig.DEFAULT_EPOCHS)
                self.batch_spin.setValue(AppConfig.DEFAULT_BATCH_SIZE)
                self.imgsz_spin.setValue(AppConfig.DEFAULT_IMAGE_SIZE)
                self.workers_spin.setValue(AppConfig.DEFAULT_WORKERS)
                self.lr_spin.setValue(AppConfig.DEFAULT_LR)
                self.patience_spin.setValue(AppConfig.DEFAULT_PATIENCE)
                self.lr_scheduler_combo.setCurrentText("cos")
                self.amp_check.setChecked(True)

                # 重置优化器设置
                self.optimizer_combo.setCurrentText("auto")
                self.lr_schedule_combo.setCurrentText("cos")
                self.warmup_spin.setValue(3)
                self.warmup_lr_spin.setValue(0.01)
                self.weight_decay_spin.setValue(0.0005)
                self.momentum_spin.setValue(0.937)

                # 重置数据增强（使用默认的适中设置）
                self.mosaic_spin.setValue(1.0)
                self.mixup_spin.setValue(0.0)
                self.copypaste_spin.setValue(0.0)
                self.flip_lr_spin.setValue(0.5)
                self.hsv_h_spin.setValue(0.015)
                self.hsv_s_spin.setValue(0.7)
                self.hsv_v_spin.setValue(0.4)
                self.degrees_spin.setValue(0.0)

                # 重置输出配置
                self.project_path.setText(str(AppConfig.DEFAULT_OUTPUT_DIR))
                self.experiment_name.setText("train")
                self.exist_ok_check.setChecked(True)

                # 清空模型和数据集路径（这些需要用户重新选择）
                self.local_model_path.clear()
                self.data_path.clear()
                self.data_info_label.setText("未选择数据集")
                self.data_info_label.setStyleSheet("color: #666666; font-size: 11px;")

                # 更新UI状态
                self.update_ui_state()

                self.logger.info("已恢复默认参数设置")
                QMessageBox.information(self, '恢复成功', '所有参数已恢复为默认值！')

            except Exception as e:
                self.logger.error(f"恢复默认值失败: {e}")
                QMessageBox.warning(self, '恢复失败', f'恢复默认值时发生错误：{e}')

    def on_tab_selected(self):
        """标签页被选中时的回调"""
        self.logger.info("模型训练标签页已激活")

        # 更新数据集信息
        if self.data_path.text():
            self.load_dataset_info(self.data_path.text())
            
        # 不自动保存配置，避免覆盖默认值
        # self.save_config()
