"""
模型验证标签页 - 验证训练好的模型性能
"""
import os
from pathlib import Path
from typing import Optional
import json
import yaml
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QTextEdit,
    QFileDialog, QMessageBox, QGroupBox, QLabel, QComboBox,
    QCheckBox, QProgressBar, QScrollArea, QGridLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from utils.logger import LogManager, QTextEditLogger
from core.trainer import ModelTrainer
from config.settings import AppConfig
from ui.widgets import DropLineEdit


class ValidationWorker(QThread):
    """验证工作线程"""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, model_path: str, data_yaml: str, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.kwargs = kwargs
        self.is_stopped = False

    def run(self):
        """执行验证"""
        import time
        t0 = time.time()
        try:
            self.log.emit(f"加载模型: {os.path.basename(self.model_path)}")
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            self.log.emit(f"模型加载完成 ({time.time()-t0:.1f}s)")

            self.log.emit("开始验证...")
            metrics = model.val(data=self.data_yaml, **self.kwargs)
            self.log.emit(f"验证完成 ({time.time()-t0:.1f}s)")

            if not self.is_stopped:
                self.finished.emit(metrics)

        except Exception as e:
            if not self.is_stopped:
                self.error.emit(str(e))

    def stop(self):
        self.is_stopped = True


class ValidateTab(QWidget):
    """模型验证标签页"""

    def __init__(
            self,
            logger: Optional[LogManager] = None,
            model_manager=None,
            trainer=None,
            image_display=None,
            image_label=None
    ):
        super().__init__()

        self.logger = logger or LogManager()
        self.model_manager = model_manager
        self.trainer = trainer or ModelTrainer(self.logger)
        self.image_display = image_display
        self.image_label = image_label

        # 工作线程
        self.worker = None

        # 验证结果
        self.last_metrics = None
        self.model_path = None
        self.data_yaml = None

        # 设置UI
        self.setup_ui()

        # 加载上次的配置
        self.load_config()

    def setup_ui(self):
        """设置UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)

        # ========== 1. 模型选择 ==========
        model_group = QGroupBox("1. 模型选择")
        model_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        model_layout = QFormLayout(model_group)

        # 模型路径
        self.model_path_edit = DropLineEdit(self, file_filter='model')
        self.model_path_edit.setPlaceholderText("拖拽模型文件(.pt/.onnx)到这里，或点击浏览...")
        self.model_path_edit.textChanged.connect(self.update_ui_state)

        browse_model_btn = QPushButton("浏览...")
        browse_model_btn.clicked.connect(self.browse_model)
        browse_model_btn.setFixedWidth(80)

        model_hbox = QHBoxLayout()
        model_hbox.addWidget(self.model_path_edit, 1)
        model_hbox.addWidget(browse_model_btn)
        model_layout.addRow("模型文件:", model_hbox)

        # 模型信息
        self.model_info_label = QLabel("未选择模型")
        self.model_info_label.setStyleSheet("color: #666666; font-size: 11px;")
        model_layout.addRow("", self.model_info_label)

        scroll_layout.addWidget(model_group)

        # ========== 2. 数据集配置 ==========
        data_group = QGroupBox("2. 数据集配置")
        data_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        data_layout = QFormLayout(data_group)

        # data.yaml路径
        self.data_yaml_edit = DropLineEdit(self, file_filter='yaml')
        self.data_yaml_edit.setPlaceholderText("拖拽data.yaml文件到这里，或点击浏览...")
        self.data_yaml_edit.textChanged.connect(self.update_ui_state)
        # 拖拽放置后加载数据集信息（与"浏览..."按钮行为保持一致）
        self.data_yaml_edit.pathDropped.connect(self.load_dataset_info)

        browse_data_btn = QPushButton("浏览...")
        browse_data_btn.clicked.connect(self.browse_data_yaml)
        browse_data_btn.setFixedWidth(80)

        data_hbox = QHBoxLayout()
        data_hbox.addWidget(self.data_yaml_edit, 1)
        data_hbox.addWidget(browse_data_btn)
        data_layout.addRow("数据集配置:", data_hbox)

        # 数据集信息
        self.data_info_label = QLabel("未加载数据集信息")
        self.data_info_label.setStyleSheet("color: #666666; font-size: 11px;")
        data_layout.addRow("", self.data_info_label)

        scroll_layout.addWidget(data_group)

        # ========== 3. 验证参数 ==========
        param_group = QGroupBox("3. 验证参数")
        param_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        param_layout = QGridLayout(param_group)
        # 设置列的拉伸比例，使标签和输入控件对齐
        param_layout.setColumnStretch(0, 0)
        param_layout.setColumnStretch(1, 1)
        param_layout.setColumnStretch(2, 0)
        param_layout.setColumnStretch(3, 1)

        # 图像尺寸
        param_layout.addWidget(QLabel("图像尺寸:"), 0, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 2048)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setFixedWidth(100)
        param_layout.addWidget(self.imgsz_spin, 0, 1, alignment=Qt.AlignLeft)

        # 批处理大小
        param_layout.addWidget(QLabel("批处理:"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)
        self.batch_spin.setFixedWidth(100)
        param_layout.addWidget(self.batch_spin, 0, 3, alignment=Qt.AlignLeft)

        # 置信度阈值
        param_layout.addWidget(QLabel("置信度:"), 1, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.001)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setToolTip("置信度阈值，用于计算指标")
        self.conf_spin.setFixedWidth(100)
        param_layout.addWidget(self.conf_spin, 1, 1, alignment=Qt.AlignLeft)

        # IoU阈值
        param_layout.addWidget(QLabel("IoU:"), 1, 2)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 1.0)
        self.iou_spin.setValue(0.6)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setToolTip("IoU阈值，用于计算mAP")
        self.iou_spin.setFixedWidth(100)
        param_layout.addWidget(self.iou_spin, 1, 3, alignment=Qt.AlignLeft)

        # 最大检测数
        param_layout.addWidget(QLabel("最大检测:"), 2, 0)
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 1000)
        self.max_det_spin.setValue(300)
        self.max_det_spin.setFixedWidth(100)
        param_layout.addWidget(self.max_det_spin, 2, 1, alignment=Qt.AlignLeft)

        # 工作线程数
        param_layout.addWidget(QLabel("工作线程:"), 2, 2)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(4)
        self.workers_spin.setFixedWidth(100)
        param_layout.addWidget(self.workers_spin, 2, 3, alignment=Qt.AlignLeft)

        # 增强选项
        self.augment_check = QCheckBox("使用测试时增强(TTA)")
        param_layout.addWidget(self.augment_check, 3, 0, 1, 2)

        self.half_check = QCheckBox("使用FP16精度")
        param_layout.addWidget(self.half_check, 3, 2, 1, 2)

        # 任务类型
        param_layout.addWidget(QLabel("任务类型:"), 4, 0)
        self.task_combo = QComboBox()
        self.task_combo.addItems(["自动检测", "目标检测", "实例分割", "语义分割", "深度估计", "图像分类", "姿态估计", "旋转框检测"])
        self.task_combo.setCurrentIndex(0)
        param_layout.addWidget(self.task_combo, 4, 1, 1, 3)

        scroll_layout.addWidget(param_group)

        # ========== 4. 输出配置 ==========
        output_group = QGroupBox("4. 输出配置")
        output_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        output_layout = QFormLayout(output_group)

        # 项目目录
        self.project_path = DropLineEdit(self, file_filter='folder')
        self.project_path.setPlaceholderText("拖拽输出目录到这里，或点击浏览...")
        self.project_path.setText(str(AppConfig.DEFAULT_OUTPUT_DIR / "val"))

        browse_project_btn = QPushButton("浏览...")
        browse_project_btn.clicked.connect(self.browse_project_folder)
        browse_project_btn.setFixedWidth(80)

        project_hbox = QHBoxLayout()
        project_hbox.addWidget(self.project_path, 1)
        project_hbox.addWidget(browse_project_btn)
        output_layout.addRow("项目目录:", project_hbox)

        # 实验名称
        self.experiment_name = QLineEdit()
        self.experiment_name.setPlaceholderText("例如: val_exp1")
        self.experiment_name.setText("val")
        output_layout.addRow("实验名称:", self.experiment_name)

        # 是否保存结果
        self.save_json_check = QCheckBox("保存JSON结果")
        self.save_json_check.setChecked(True)
        output_layout.addRow("", self.save_json_check)

        self.save_hybrid_check = QCheckBox("保存混合结果")
        self.save_hybrid_check.setChecked(False)
        output_layout.addRow("", self.save_hybrid_check)

        scroll_layout.addWidget(output_group)

        # ========== 5. 验证按钮 ==========
        button_layout = QHBoxLayout()

        self.validate_btn = QPushButton("开始验证")
        self.validate_btn.clicked.connect(self.start_validation)
        self.validate_btn.setEnabled(False)
        self.validate_btn.setStyleSheet("""
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
        button_layout.addWidget(self.validate_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_validation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #d83b01;
                color: white;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #a83200;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        button_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)

        self.open_folder_btn = QPushButton("打开文件夹")
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
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        button_layout.addWidget(self.reset_btn)

        scroll_layout.addLayout(button_layout)

        # ========== 6. 进度条 ==========
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        scroll_layout.addWidget(self.progress_bar)

        # ========== 7. 结果展示 ==========
        result_group = QGroupBox("验证结果")
        result_layout = QVBoxLayout(result_group)

        # 创建标签页
        result_tabs = QTabWidget()

        # 标签页1：指标表格
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        metrics_layout.addWidget(self.metrics_table)

        result_tabs.addTab(metrics_tab, "指标")

        # 标签页2：详细日志
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)

        self.detail_log = QTextEdit()
        self.detail_log.setReadOnly(True)
        self.detail_log.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.detail_log)

        result_tabs.addTab(log_tab, "详细日志")

        result_layout.addWidget(result_tabs)

        scroll_layout.addWidget(result_group)

        # 添加伸缩项
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_widget)

        # ========== 8. 日志显示 ==========
        log_group = QGroupBox("验证日志")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QVBoxLayout(log_group)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))

        # 设置日志处理器
        log_handler = QTextEditLogger(self.log_display)
        self.logger.add_handler(log_handler)

        log_layout.addWidget(self.log_display)

        # 创建垂直分割器，用于调节设置区域和日志栏的大小
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #e0e0e0;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QSplitter::handle:hover {
                background-color: #b0b0b0;
            }
        """)

        # 添加滚动区域到分割器
        splitter.addWidget(scroll_area)
        # 添加日志栏到分割器
        splitter.addWidget(log_group)
        # 设置初始大小比例（设置区域80%，日志栏20%）
        splitter.setSizes([800, 200])
        # 添加分割器到主布局
        main_layout.addWidget(splitter, 1)

        # 初始化UI状态
        self.update_ui_state()

    def browse_model(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "模型文件 (*.pt);;所有文件 (*.*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            self.model_path = file_path
            self.detect_model_info(file_path)

    def browse_data_yaml(self):
        """浏览data.yaml文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择data.yaml文件", "",
            "YAML文件 (*.yaml *.yml)"
        )
        if file_path:
            self.data_yaml_edit.setText(file_path)
            self.data_yaml = file_path
            self.load_dataset_info(file_path)

    def browse_project_folder(self):
        """浏览项目目录"""
        folder = QFileDialog.getExistingDirectory(
            self, "选择项目目录"
        )
        if folder:
            self.project_path.setText(folder)

    def detect_model_info(self, model_path: str):
        """检测模型信息"""
        try:
            # 使用模型管理器检测任务类型
            task = self.model_manager.detect_model_task(model_path)

            task_names = {
                'detect': '目标检测',
                'segment': '实例分割',
                'semantic': '语义分割',
                'depth': '深度估计',
                'classify': '图像分类',
                'pose': '姿态估计',
                'obb': '旋转框检测'
            }

            task_name = task_names.get(task, task)
            self.model_info_label.setText(f"模型类型: {task_name}")

            # 根据任务类型设置默认任务
            task_index = {
                'detect': 1,
                'segment': 2,
                'semantic': 3,
                'depth': 4,
                'classify': 5,
                'pose': 6,
                'obb': 7
            }.get(task, 0)

            if task_index > 0:
                self.task_combo.setCurrentIndex(task_index)

            self.logger.info(f"检测到模型类型: {task_name}")

        except Exception as e:
            self.logger.error(f"检测模型类型失败: {e}")
        finally:
            # 更新UI状态
            self.update_ui_state()

    def load_dataset_info(self, yaml_path: str):
        """加载数据集信息"""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            info_text = []

            if 'names' in data:
                names = data['names']
                if isinstance(names, list):
                    info_text.append(f"类别数: {len(names)}")
                    info_text.append(f"类别: {', '.join(names[:5])}")
                    if len(names) > 5:
                        info_text.append(f"... 等{len(names)}个类别")
                elif isinstance(names, dict):
                    info_text.append(f"类别数: {len(names)}")

            if 'nc' in data:
                info_text.append(f"nc: {data['nc']}")

            if 'path' in data:
                info_text.append(f"数据路径: {data['path']}")

            if info_text:
                self.data_info_label.setText('\n'.join(info_text))
            else:
                self.data_info_label.setText("数据集信息加载成功")

            self.logger.info("数据集信息加载成功")

        except Exception as e:
            self.logger.error(f"加载数据集信息失败: {e}")
            self.data_info_label.setText("加载数据集信息失败")
        finally:
            # 更新UI状态
            self.update_ui_state()

    def start_validation(self):
        """开始验证"""
        model_path = self.model_path_edit.text()
        data_yaml = self.data_yaml_edit.text()

        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "警告", "请选择有效的模型文件")
            return

        if not data_yaml or not os.path.exists(data_yaml):
            QMessageBox.warning(self, "警告", "请选择有效的data.yaml文件")
            return

        # 收集验证参数
        imgsz = self.imgsz_spin.value()
        batch = self.batch_spin.value()
        conf = self.conf_spin.value()
        iou = self.iou_spin.value()
        max_det = self.max_det_spin.value()
        workers = self.workers_spin.value()
        augment = self.augment_check.isChecked()
        half = self.half_check.isChecked()

        # 输出路径
        project = self.project_path.text()
        name = self.experiment_name.text()

        # 清空结果显示
        self.metrics_table.setRowCount(0)
        self.detail_log.clear()
        self.log_display.clear()

        self.logger.info("=" * 60)
        self.logger.info("开始模型验证")
        self.logger.info(f"模型: {model_path}")
        self.logger.info(f"数据集: {data_yaml}")
        self.logger.info(f"参数: imgsz={imgsz}, batch={batch}, conf={conf}, iou={iou}")
        self.logger.info(f"输出: {os.path.join(project, name)}")
        self.logger.info("=" * 60)

        # 创建并启动工作线程
        self.worker = ValidationWorker(
            model_path=model_path,
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            iou=iou,
            max_det=max_det,
            workers=workers,
            augment=augment,
            half=half,
            project=project,
            name=name,
            exist_ok=True,
            save_json=self.save_json_check.isChecked(),
            save_hybrid=self.save_hybrid_check.isChecked()
        )

        self.worker.finished.connect(self.on_validation_finished)
        self.worker.error.connect(self.on_validation_error)
        self.worker.log.connect(self.logger.info)

        self.validate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不定进度条

        self.worker.start()

    def stop_validation(self):
        """停止验证"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "确认停止",
                "确定要停止验证吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.worker.stop()
                # 先等待线程自行结束，超时才强制终止
                if not self.worker.wait(3000):
                    self.worker.terminate()
                    self.worker.wait()
                self.on_validation_stopped()

    def on_validation_finished(self, metrics):
        """验证完成回调"""
        self.last_metrics = metrics

        # 更新指标表格
        self.update_metrics_table(metrics)

        # 更新详细日志
        self.update_detail_log(metrics)

        self.logger.info("=" * 60)
        self.logger.info("验证完成！")

        # 提取关键指标
        if hasattr(metrics, 'box'):
            if hasattr(metrics.box, 'map50'):
                self.logger.info(f"mAP50: {metrics.box.map50:.4f}")
            if hasattr(metrics.box, 'map'):
                self.logger.info(f"mAP50-95: {metrics.box.map:.4f}")

        if hasattr(metrics, 'seg') and hasattr(metrics.seg, 'map50'):
            self.logger.info(f"Segmentation mAP50: {metrics.seg.map50:.4f}")

        if hasattr(metrics, 'obb') and hasattr(metrics.obb, 'map50'):
            self.logger.info(f"OBB mAP50: {metrics.obb.map50:.4f}")
            if hasattr(metrics.obb, 'map'):
                self.logger.info(f"OBB mAP50-95: {metrics.obb.map:.4f}")

        self.logger.info("=" * 60)

        # 更新UI
        self.validate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # 保存配置
        self.save_config()

        self.worker = None

    def on_validation_error(self, error_msg: str):
        """验证错误回调"""
        self.logger.error(f"验证失败: {error_msg}")

        QMessageBox.critical(self, "验证失败", f"错误信息:\n{error_msg}")

        self.validate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.worker = None

    def on_validation_stopped(self):
        """验证停止回调"""
        self.logger.warning("验证已停止")

        self.validate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.worker = None

    def update_metrics_table(self, metrics):
        """更新指标表格"""
        rows = []

        # 检测指标
        if hasattr(metrics, 'box'):
            box = metrics.box
            rows.append(("mAP50 (Box)", f"{box.map50:.4f}" if hasattr(box, 'map50') else "N/A"))
            rows.append(("mAP50-95 (Box)", f"{box.map:.4f}" if hasattr(box, 'map') else "N/A"))

            if hasattr(box, 'mp'):
                rows.append(("Precision", f"{box.mp:.4f}"))
            if hasattr(box, 'mr'):
                rows.append(("Recall", f"{box.mr:.4f}"))

        # 分割指标
        if hasattr(metrics, 'seg'):
            seg = metrics.seg
            rows.append(("mAP50 (Mask)", f"{seg.map50:.4f}" if hasattr(seg, 'map50') else "N/A"))
            rows.append(("mAP50-95 (Mask)", f"{seg.map:.4f}" if hasattr(seg, 'map') else "N/A"))

        # 分类指标
        if hasattr(metrics, 'top1'):
            rows.append(("Top-1 Accuracy", f"{metrics.top1:.4f}"))
            rows.append(("Top-5 Accuracy", f"{metrics.top5:.4f}"))

        # 姿态估计指标
        if hasattr(metrics, 'pose'):
            pose = metrics.pose
            rows.append(("mAP50 (Pose)", f"{pose.map50:.4f}" if hasattr(pose, 'map50') else "N/A"))
        
        # 旋转框检测指标
        if hasattr(metrics, 'obb'):
            obb = metrics.obb
            rows.append(("mAP50 (OBB)", f"{obb.map50:.4f}" if hasattr(obb, 'map50') else "N/A"))
            rows.append(("mAP50-95 (OBB)", f"{obb.map:.4f}" if hasattr(obb, 'map') else "N/A"))

        # 更新表格
        self.metrics_table.setRowCount(len(rows))
        for i, (name, value) in enumerate(rows):
            name_item = QTableWidgetItem(name)
            value_item = QTableWidgetItem(value)
            value_item.setTextAlignment(Qt.AlignRight)

            self.metrics_table.setItem(i, 0, name_item)
            self.metrics_table.setItem(i, 1, value_item)

    def update_detail_log(self, metrics):
        """更新详细日志"""
        log_text = []
        log_text.append("验证结果详细报告")
        log_text.append("=" * 60)

        # 尝试获取详细指标
        if hasattr(metrics, 'results_dict'):
            for key, value in metrics.results_dict.items():
                if isinstance(value, (int, float)):
                    log_text.append(f"{key}: {value:.4f}")
                else:
                    log_text.append(f"{key}: {value}")

        # 速度信息
        if hasattr(metrics, 'speed'):
            speed = metrics.speed
            log_text.append("\n速度指标:")
            if hasattr(speed, 'preprocess'):
                log_text.append(f"  预处理: {speed.preprocess:.1f}ms")
            if hasattr(speed, 'inference'):
                log_text.append(f"  推理: {speed.inference:.1f}ms")
            if hasattr(speed, 'postprocess'):
                log_text.append(f"  后处理: {speed.postprocess:.1f}ms")

        self.detail_log.setText('\n'.join(log_text))

    def save_results(self):
        """保存结果"""
        if not self.last_metrics:
            return

        # 选择保存路径
        default_name = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", default_name,
            "文本文件 (*.txt);;JSON文件 (*.json);;所有文件 (*.*)"
        )

        if save_path:
            try:
                if save_path.endswith('.json'):
                    # 保存为JSON
                    results_dict = {}
                    if hasattr(self.last_metrics, 'results_dict'):
                        results_dict = self.last_metrics.results_dict

                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(results_dict, f, indent=2, ensure_ascii=False)
                else:
                    # 保存为文本
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(self.detail_log.toPlainText())

                self.logger.info(f"结果已保存: {save_path}")

            except Exception as e:
                self.logger.error(f"保存结果失败: {e}")
                QMessageBox.critical(self, "错误", f"保存结果失败:\n{str(e)}")

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

    def update_ui_state(self):
        """更新UI状态"""
        has_model = bool(self.model_path_edit.text())
        has_data = bool(self.data_yaml_edit.text())
        self.validate_btn.setEnabled(has_model and has_data)

    def save_config(self):
        """保存配置"""
        config = {
            'model_path': self.model_path_edit.text(),
            'data_yaml': self.data_yaml_edit.text(),
            'imgsz': self.imgsz_spin.value(),
            'batch': self.batch_spin.value(),
            'conf': self.conf_spin.value(),
            'iou': self.iou_spin.value(),
            'max_det': self.max_det_spin.value(),
            'workers': self.workers_spin.value(),
            'augment': self.augment_check.isChecked(),
            'half': self.half_check.isChecked(),
            'task': self.task_combo.currentIndex(),
            'project': self.project_path.text(),
            'experiment_name': self.experiment_name.text(),
            'save_json': self.save_json_check.isChecked(),
            'save_hybrid': self.save_hybrid_check.isChecked()
        }

        config_dir = Path.home() / ".yolo_ui_workstation"
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "validate_config.json"
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
            '确定要将所有验证参数恢复为默认值吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # 重置验证参数
                self.imgsz_spin.setValue(640)
                self.batch_spin.setValue(16)
                self.conf_spin.setValue(0.001)
                self.iou_spin.setValue(0.6)
                self.max_det_spin.setValue(300)
                self.workers_spin.setValue(4)

                # 重置选项
                self.augment_check.setChecked(False)
                self.half_check.setChecked(False)
                self.save_json_check.setChecked(True)
                self.save_hybrid_check.setChecked(False)

                # 重置任务类型
                self.task_combo.setCurrentIndex(0)

                # 重置输出配置
                self.project_path.setText(str(AppConfig.DEFAULT_OUTPUT_DIR / "val"))
                self.experiment_name.setText("val")

                # 清空路径
                self.model_path_edit.clear()
                self.data_yaml_edit.clear()
                self.model_info_label.setText("模型类型: 未检测")

                # 更新UI状态
                self.update_ui_state()

                self.logger.info("已恢复验证默认参数设置")
                QMessageBox.information(self, '恢复成功', '所有参数已恢复为默认值！')

            except Exception as e:
                self.logger.error(f"恢复默认值失败: {e}")
                QMessageBox.warning(self, '恢复失败', f'恢复默认值时发生错误：{e}')

    def on_tab_selected(self):
        """标签页被选中时的回调"""
        self.logger.info("模型验证标签页已激活")