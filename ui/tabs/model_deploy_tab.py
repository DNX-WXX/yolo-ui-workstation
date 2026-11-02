"""
ONNX转换标签页 - 基于Ultralytics官方文档优化
支持：ONNX导出、INT8量化
"""
import os
import json
import shutil
from pathlib import Path
from typing import Optional
import numpy as np

# onnxruntime 采用惰性导入（在 validate_onnx 等方法内），减少启动耗时
# import onnxruntime as ort  ← 移到方法内

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QTextEdit,
    QFileDialog, QMessageBox, QGroupBox, QLabel, QComboBox,
    QCheckBox, QProgressBar, QScrollArea, QGridLayout, QSplitter,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

from utils.logger import LogManager, QTextEditLogger
from utils.image_utils import ImageProcessor, PreprocessPipeline
from core.inference_engine import InferenceEngine
from config.settings import AppConfig
from ui.widgets import DropLineEdit


# ============================================================================
# 工作线程
# ============================================================================

class ONNXConversionWorker(QThread):
    """ONNX转换工作线程 - 基于官方export API"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, pt_path: str, onnx_path: str, **kwargs):
        super().__init__()
        self.pt_path = pt_path
        self.onnx_path = onnx_path
        self.kwargs = kwargs
        self.is_stopped = False

    def run(self):
        import time
        t0 = time.time()
        try:
            self.log.emit(f"加载模型: {os.path.basename(self.pt_path)}")
            from ultralytics import YOLO
            model = YOLO(self.pt_path)
            self.log.emit(f"模型加载完成 ({time.time()-t0:.1f}s)")

            self.log.emit(f"导出ONNX... (参数: {self._format_kwargs()})")
            model.export(**self.kwargs)
            self.log.emit(f"ONNX导出完成 ({time.time()-t0:.1f}s)")

            # 查找生成的ONNX文件
            expected_path = os.path.splitext(self.pt_path)[0] + ".onnx"
            if self.kwargs.get('quantize') == 8:
                expected_path = os.path.splitext(self.pt_path)[0] + "_int8.onnx"

            if os.path.exists(expected_path):
                if expected_path != self.onnx_path:
                    if os.path.exists(self.onnx_path):
                        os.remove(self.onnx_path)
                    shutil.move(expected_path, self.onnx_path)
                    self.log.emit(f"文件已移动到: {self.onnx_path}")
            else:
                raise FileNotFoundError(f"ONNX文件未生成: {expected_path}")

            # 验证ONNX文件
            self.log.emit("验证ONNX文件...")
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
                self.log.emit(f"ONNX验证通过! 输入: {[i.name for i in session.get_inputs()]}")
            except Exception as e:
                self.log.emit(f"ONNX验证警告: {e}")

            self.log.emit(f"总耗时: {time.time()-t0:.1f}s")
            self.finished.emit(self.onnx_path)

        except Exception as e:
            self.error.emit(str(e))

    def _format_kwargs(self) -> str:
        """格式化参数显示"""
        items = []
        for k, v in self.kwargs.items():
            if k in ['format', 'data', 'device']:
                items.append(f"{k}={v}")
            elif k in ['imgsz', 'batch', 'opset']:
                items.append(f"{k}={v}")
            elif k in ['dynamic', 'simplify', 'nms']:
                items.append(f"{k}={v}")
            elif k == 'quantize':
                items.append(f"quantize={v}")
        return ", ".join(items)

    def stop(self):
        self.is_stopped = True


class ComparisonWorker(QThread):
    """模型对比工作线程"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, pt_path: str, onnx_path: str, image_path: str, imgsz: int):
        super().__init__()
        self.pt_path = pt_path
        self.onnx_path = onnx_path
        self.image_path = image_path
        self.imgsz = imgsz

    def run(self):
        try:
            results = {}

            # 1. PT模型推理
            self.log.emit("加载PT模型...")
            from ultralytics import YOLO
            pt_model = YOLO(self.pt_path)
            self.log.emit("执行PT模型推理...")
            pt_results = pt_model(self.image_path, imgsz=self.imgsz, verbose=False)
            results['pt'] = pt_results[0]

            # 2. ONNX包装器推理
            self.log.emit("加载ONNX包装器...")
            onnx_wrapper = YOLO(self.onnx_path)
            self.log.emit("执行ONNX包装器推理...")
            onnx_results = onnx_wrapper(self.image_path, imgsz=self.imgsz, verbose=False, device='cpu')
            results['onnx_wrapper'] = onnx_results[0]

            # 3. 直接ONNX推理
            self.log.emit("执行直接ONNX推理...")
            import onnxruntime as ort
            session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
            pipeline = PreprocessPipeline(input_size=self.imgsz)
            input_data = pipeline.preprocess_for_onnx(self.image_path)
            input_name = session.get_inputs()[0].name
            onnx_output = session.run(None, {input_name: input_data})
            results['onnx_direct'] = onnx_output[0]

            # 检测任务类型
            if hasattr(pt_results[0], 'boxes') and pt_results[0].boxes is not None:
                results['task'] = 'detect'
            elif hasattr(pt_results[0], 'probs') and pt_results[0].probs is not None:
                results['task'] = 'classify'
            elif hasattr(pt_results[0], 'masks') and pt_results[0].masks is not None:
                results['task'] = 'segment'
            elif hasattr(pt_results[0], 'keypoints') and pt_results[0].keypoints is not None:
                results['task'] = 'pose'
            else:
                results['task'] = 'unknown'

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class KnowledgeDistillationWorker(QThread):
    """知识蒸馏工作线程 - 基于官方 distill_model API
    支持 YOLOv8 / YOLO11 / YOLO26（教师与学生须同一家族）
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    # 蒸馏支持的训练任务（官方：detect/segment/pose/obb）
    SUPPORTED_TASKS = ('detect', 'segment', 'pose', 'obb')

    def __init__(self, student_path: str, teacher_path: str, data_yaml: str, **kwargs):
        super().__init__()
        self.student_path = student_path
        self.teacher_path = teacher_path
        self.data_yaml = data_yaml
        self.kwargs = kwargs
        self.is_stopped = False

    def run(self):
        import time
        t0 = time.time()
        try:
            self.log.emit(f"加载学生模型: {os.path.basename(self.student_path)}")
            from ultralytics import YOLO
            model = YOLO(self.student_path)
            self.log.emit(f"学生模型加载完成 ({time.time()-t0:.1f}s)")

            self.log.emit(f"加载教师模型: {os.path.basename(self.teacher_path)}")
            _ = YOLO(self.teacher_path)  # 预加载教师模型，提前暴露路径错误
            self.log.emit(f"教师模型加载完成 ({time.time()-t0:.1f}s)")

            self.log.emit(f"开始知识蒸馏训练... "
                          f"(教师={os.path.basename(self.teacher_path)}, "
                          f"dis={self.kwargs.get('dis', 6.0)}, "
                          f"epochs={self.kwargs.get('epochs', 100)})")
            self.log.emit("提示: 训练过程中教师模型冻结在eval模式，仅学生模型反传梯度")
            self.log.emit("提示: 蒸馏训练比普通训练慢约1.2-1.5倍，显存占用约多1.1倍")

            # 蒸馏训练（distill_model参数启用知识蒸馏）
            model.train(
                data=self.data_yaml,
                distill_model=self.teacher_path,
                **self.kwargs
            )
            self.log.emit(f"蒸馏训练完成 ({time.time()-t0:.1f}s)")

            # 输出模型路径
            project = self.kwargs.get('project', str(AppConfig.DEFAULT_OUTPUT_DIR))
            name = self.kwargs.get('name', 'distill')
            best_path = os.path.join(project, name, 'weights', 'best.pt')

            if not self.is_stopped:
                self.finished.emit(best_path)
        except Exception as e:
            if not self.is_stopped:
                self.error.emit(str(e))

    def stop(self):
        self.is_stopped = True


# ============================================================================
# 主界面
# ============================================================================

class ModelDeployTab(QWidget):
    """模型部署标签页 - 基于Ultralytics官方文档优化"""

    # YOLO版本支持矩阵
    YOLO_VERSION_COMPATIBILITY = {
        'yolo26': {
            'nms_supported': False,
            'nms_required': False,
            'nms_default': False,
            'dynamic_default': True,
            'fp16_default': True,
            'tasks': ['detect', 'segment', 'semantic', 'depth', 'classify', 'pose', 'obb'],
            'note': 'YOLO26是端到端模型，内置NMS，推荐FP16和动态批处理',
        },
        'yolo11': {
            'nms_supported': True,
            'nms_required': False,
            'nms_default': True,
            'dynamic_default': False,
            'fp16_default': False,
            'tasks': ['detect', 'segment', 'classify', 'pose', 'obb'],
            'note': 'YOLO11支持NMS，可选项',
        },
        'yolov8': {
            'nms_supported': True,
            'nms_required': False,
            'nms_default': True,
            'dynamic_default': False,
            'fp16_default': False,
            'tasks': ['detect', 'segment', 'classify', 'pose', 'obb'],
            'note': 'YOLOv8支持NMS，可选项',
        },
        'yolov5': {
            'nms_supported': True,
            'nms_required': False,
            'nms_default': True,
            'dynamic_default': False,
            'fp16_default': False,
            'tasks': ['detect', 'classify'],
            'note': 'YOLOv5支持NMS，可选项',
        }
    }

    def __init__(
        self,
        logger: Optional[LogManager] = None,
        model_manager=None,
        inference_engine=None,
        image_display=None,
        image_label=None
    ):
        super().__init__()

        self.logger = logger or LogManager()
        self.model_manager = model_manager
        self.engine = inference_engine or InferenceEngine(self.logger)
        self.image_display = image_display
        self.image_label = image_label

        # 工作线程
        self.conversion_worker = None
        self.comparison_worker = None
        self.distill_worker = None

        # 当前状态
        self.pt_model_path = None
        self.onnx_model_path = None
        self.test_image_path = None
        self.model_task = None
        self._current_model_version = '未知'

        # 设置UI
        self.setup_ui()

        # 加载配置
        self.load_config()

        # 初始化UI状态
        self.update_ui_state()

    # ========================================================================
    # UI设置
    # ========================================================================

    def setup_ui(self):
        """设置UI - 基于官方文档优化"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)

        # ====================================================================
        # Section 1: 模型文件选择
        # ====================================================================
        file_group = QGroupBox("1. 模型文件选择")
        file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_form = QFormLayout(file_group)
        file_form.setSpacing(10)
        file_form.setContentsMargins(12, 16, 12, 12)
        file_form.setLabelAlignment(Qt.AlignRight)
        file_form.setFormAlignment(Qt.AlignTop)
        file_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # PT模型文件
        pt_row_widget = QWidget()
        pt_row_layout = QHBoxLayout(pt_row_widget)
        pt_row_layout.setContentsMargins(0, 0, 0, 0)
        pt_row_layout.setSpacing(5)

        self.pt_path = DropLineEdit(self, file_filter='model')
        self.pt_path.setPlaceholderText("拖拽模型文件(.pt)到这里，或点击浏览...")
        self.pt_path.setToolTip("选择要导出的YOLO模型文件 (.pt)")
        self.pt_path.textChanged.connect(self.on_pt_path_changed)
        browse_pt_btn = QPushButton("浏览")
        browse_pt_btn.clicked.connect(self.browse_pt_model)
        browse_pt_btn.setFixedWidth(60)
        pt_row_layout.addWidget(self.pt_path, 1)
        pt_row_layout.addWidget(browse_pt_btn)
        file_form.addRow("PT模型文件:", pt_row_widget)

        # ONNX输出文件
        onnx_row_widget = QWidget()
        onnx_row_layout = QHBoxLayout(onnx_row_widget)
        onnx_row_layout.setContentsMargins(0, 0, 0, 0)
        onnx_row_layout.setSpacing(5)

        self.onnx_path = DropLineEdit(file_filter='folder')
        self.onnx_path.setPlaceholderText("拖拽输出目录到这里，或点击浏览...")
        self.onnx_path.setToolTip("ONNX模型的输出路径 (.onnx)")
        browse_onnx_btn = QPushButton("浏览")
        browse_onnx_btn.clicked.connect(self.browse_onnx_output)
        browse_onnx_btn.setFixedWidth(60)
        onnx_row_layout.addWidget(self.onnx_path, 1)
        onnx_row_layout.addWidget(browse_onnx_btn)
        file_form.addRow("ONNX输出:", onnx_row_widget)

        info_row = QHBoxLayout()
        self.model_info_label = QLabel("模型类型: 未检测")
        self.model_info_label.setStyleSheet("color: #666666; font-size: 11px;")
        info_row.addWidget(self.model_info_label)
        info_row.addStretch()

        self.compat_btn = QPushButton("版本兼容性")
        self.compat_btn.setFixedWidth(100)
        self.compat_btn.setToolTip("查看当前模型的版本兼容性信息")
        self.compat_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                padding: 2px 6px;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #1976d2; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.compat_btn.setEnabled(False)
        self.compat_btn.clicked.connect(lambda: self.show_version_compatibility_info(self._current_model_version))
        info_row.addWidget(self.compat_btn)
        file_form.addRow("", info_row)

        scroll_layout.addWidget(file_group)

        # ====================================================================
        # Section 2: 转换参数 (优化布局)
        # ====================================================================
        param_group = QGroupBox("2. 转换参数")
        param_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        param_form = QFormLayout(param_group)
        param_form.setSpacing(10)
        param_form.setContentsMargins(12, 16, 12, 12)
        param_form.setLabelAlignment(Qt.AlignRight)
        param_form.setFormAlignment(Qt.AlignTop)
        param_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        param_form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        # 输入尺寸
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 2048)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setToolTip("模型输入的图像尺寸，推荐640")
        self.imgsz_spin.setFixedWidth(120)
        param_form.addRow("输入尺寸:", self.imgsz_spin)

        # Opset版本
        self.opset_combo = QComboBox()
        self.opset_combo.addItems(AppConfig.ONNX_OPSET_VERSIONS)
        self.opset_combo.setCurrentText(AppConfig.DEFAULT_ONNX_OPSET)
        self.opset_combo.setToolTip("ONNX opset版本，推荐最新")
        self.opset_combo.setFixedWidth(120)
        param_form.addRow("Opset版本:", self.opset_combo)

        # 批处理
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(1)
        self.batch_spin.setToolTip("推理批处理大小")
        self.batch_spin.setFixedWidth(120)
        param_form.addRow("批处理:", self.batch_spin)

        # 量化精度
        self.quantize_combo = QComboBox()
        self.quantize_combo.addItem("FP32 (默认)", "32")
        self.quantize_combo.addItem("FP16 (推荐)", "16")
        self.quantize_combo.addItem("INT8 (PTQ量化)", "8")
        self.quantize_combo.setToolTip("量化精度: FP32为默认，FP16可减小模型，INT8需要校准数据集")
        self.quantize_combo.setFixedWidth(120)
        param_form.addRow("量化精度:", self.quantize_combo)

        # 校准数据集
        calib_row_widget = QWidget()
        calib_row_layout = QHBoxLayout(calib_row_widget)
        calib_row_layout.setContentsMargins(0, 0, 0, 0)
        calib_row_layout.setSpacing(5)

        self.calib_data_edit = DropLineEdit(self, file_filter='file')
        self.calib_data_edit.setPlaceholderText("选择校准数据集YAML文件...")
        self.calib_data_edit.setToolTip("INT8量化校准数据集，如 coco8.yaml")
        self.calib_data_edit.setEnabled(False)
        browse_calib_btn = QPushButton("浏览")
        browse_calib_btn.clicked.connect(self.browse_calib_data)
        browse_calib_btn.setFixedWidth(60)
        browse_calib_btn.setEnabled(False)
        calib_row_layout.addWidget(self.calib_data_edit, 1)
        calib_row_layout.addWidget(browse_calib_btn)
        param_form.addRow("校准数据集:", calib_row_widget)

        self.calib_browse_btn = browse_calib_btn

        self.calib_fraction_spin = QDoubleSpinBox()
        self.calib_fraction_spin.setRange(0.1, 1.0)
        self.calib_fraction_spin.setValue(1.0)
        self.calib_fraction_spin.setSingleStep(0.1)
        self.calib_fraction_spin.setToolTip("用于INT8量化的校准数据比例，默认1.0")
        self.calib_fraction_spin.setEnabled(False)
        self.calib_fraction_spin.setFixedWidth(120)
        param_form.addRow("校准比例:", self.calib_fraction_spin)

        # 选项勾选框
        checkbox_widget = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_widget)
        checkbox_layout.setSpacing(20)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)

        self.dynamic_check = QCheckBox("动态批处理")
        self.dynamic_check.setToolTip("允许动态输入尺寸，提高灵活性")
        checkbox_layout.addWidget(self.dynamic_check)

        self.nms_check = QCheckBox("包含NMS")
        self.nms_check.setChecked(True)
        self.nms_check.setToolTip("非极大值抑制，仅适用于检测模型")
        checkbox_layout.addWidget(self.nms_check)

        self.simplify_check = QCheckBox("ONNX简化 (onnxslim)")
        self.simplify_check.setChecked(True)
        self.simplify_check.setToolTip("使用onnxslim简化ONNX模型图，提升性能和兼容性")
        checkbox_layout.addWidget(self.simplify_check)

        checkbox_layout.addStretch()
        param_form.addRow("", checkbox_widget)

        # 连接量化精度选择
        self.quantize_combo.currentIndexChanged.connect(self._on_quantize_changed)

        scroll_layout.addWidget(param_group)

        # ====================================================================
        # Section 3: 转换按钮
        # ====================================================================
        convert_layout = QHBoxLayout()

        self.convert_btn = QPushButton("开始转换")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setEnabled(False)
        self.convert_btn.setToolTip("开始导出ONNX模型")
        self.convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #005a9e; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        convert_layout.addWidget(self.convert_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_conversion)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("停止正在进行的转换")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #d83b01;
                color: white;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #a83200; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        convert_layout.addWidget(self.stop_btn)

        convert_layout.addStretch()

        self.reset_btn = QPushButton("恢复默认值")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        self.reset_btn.setToolTip("恢复所有参数为默认值")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #5a6268; }
        """)
        convert_layout.addWidget(self.reset_btn)

        scroll_layout.addLayout(convert_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        scroll_layout.addWidget(self.progress_bar)

        # ====================================================================
        # Section 3: 知识蒸馏 (Knowledge Distillation)
        # 参考: https://docs.ultralytics.com/guides/knowledge-distillation
        # 支持 YOLOv8 / YOLO11 / YOLO26（教师与学生须同一家族）
        # ====================================================================
        distill_group = QGroupBox("3. 知识蒸馏 (Knowledge Distillation)")
        distill_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        distill_outer_layout = QVBoxLayout(distill_group)
        distill_outer_layout.setSpacing(10)
        distill_form = QFormLayout()
        distill_form.setSpacing(10)
        distill_form.setContentsMargins(12, 16, 12, 12)
        distill_form.setLabelAlignment(Qt.AlignRight)
        distill_form.setFormAlignment(Qt.AlignTop)
        distill_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        distill_outer_layout.addLayout(distill_form)

        # 学生模型（复用顶部PT模型，或在此直接选择）
        student_row_widget = QWidget()
        student_row_layout = QHBoxLayout(student_row_widget)
        student_row_layout.setContentsMargins(0, 0, 0, 0)
        student_row_layout.setSpacing(5)

        self.student_model_edit = DropLineEdit(self, file_filter='model')
        self.student_model_edit.setPlaceholderText("拖拽学生模型(.pt)到这里，或点击浏览...")
        self.student_model_edit.setToolTip("学生模型与顶部PT模型文件同步")
        self.student_model_edit.textChanged.connect(self.on_student_model_changed)
        browse_student_btn = QPushButton("浏览")
        browse_student_btn.clicked.connect(self.browse_student_model)
        browse_student_btn.setFixedWidth(60)
        student_row_layout.addWidget(self.student_model_edit, 1)
        student_row_layout.addWidget(browse_student_btn)
        distill_form.addRow("学生模型:", student_row_widget)

        # 教师模型
        teacher_row_widget = QWidget()
        teacher_row_layout = QHBoxLayout(teacher_row_widget)
        teacher_row_layout.setContentsMargins(0, 0, 0, 0)
        teacher_row_layout.setSpacing(5)

        self.teacher_path = DropLineEdit(self, file_filter='model')
        self.teacher_path.setPlaceholderText("拖拽教师模型(.pt)到这里，或点击浏览...")
        self.teacher_path.setToolTip("教师模型，须与学生模型同一家族（如同为 YOLO11）")
        self.teacher_path.textChanged.connect(self.update_ui_state)
        browse_teacher_btn = QPushButton("浏览")
        browse_teacher_btn.clicked.connect(self.browse_teacher_model)
        browse_teacher_btn.setFixedWidth(60)
        teacher_row_layout.addWidget(self.teacher_path, 1)
        teacher_row_layout.addWidget(browse_teacher_btn)
        distill_form.addRow("教师模型:", teacher_row_widget)

        # 蒸馏数据集
        distill_data_row_widget = QWidget()
        distill_data_row_layout = QHBoxLayout(distill_data_row_widget)
        distill_data_row_layout.setContentsMargins(0, 0, 0, 0)
        distill_data_row_layout.setSpacing(5)

        self.distill_data_edit = DropLineEdit(self, file_filter='yaml')
        self.distill_data_edit.setPlaceholderText("选择数据集YAML文件(data.yaml)...")
        self.distill_data_edit.setToolTip("蒸馏训练数据集，教师与学生须在同一数据集上训练")
        self.distill_data_edit.textChanged.connect(self.update_ui_state)
        browse_distill_data_btn = QPushButton("浏览")
        browse_distill_data_btn.clicked.connect(self.browse_distill_data)
        browse_distill_data_btn.setFixedWidth(60)
        distill_data_row_layout.addWidget(self.distill_data_edit, 1)
        distill_data_row_layout.addWidget(browse_distill_data_btn)
        distill_form.addRow("数据集:", distill_data_row_widget)

        # 蒸馏训练参数
        distill_param_widget = QWidget()
        distill_param_layout = QGridLayout(distill_param_widget)
        distill_param_layout.setContentsMargins(0, 0, 0, 0)
        distill_param_layout.setHorizontalSpacing(10)
        distill_param_layout.setVerticalSpacing(10)
        distill_param_layout.setColumnStretch(0, 0)
        distill_param_layout.setColumnStretch(1, 1)
        distill_param_layout.setColumnStretch(2, 0)
        distill_param_layout.setColumnStretch(3, 1)

        distill_param_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.distill_epochs_spin = QSpinBox()
        self.distill_epochs_spin.setRange(1, 1000)
        self.distill_epochs_spin.setValue(AppConfig.DEFAULT_EPOCHS)
        self.distill_epochs_spin.setFixedWidth(120)
        distill_param_layout.addWidget(self.distill_epochs_spin, 0, 1, alignment=Qt.AlignLeft)

        distill_param_layout.addWidget(QLabel("蒸馏权重 dis:"), 0, 2)
        self.dis_spin = QDoubleSpinBox()
        self.dis_spin.setRange(0.0, 20.0)
        self.dis_spin.setValue(6.0)
        self.dis_spin.setSingleStep(0.5)
        self.dis_spin.setDecimals(1)
        self.dis_spin.setToolTip("蒸馏损失权重，控制蒸馏损失对总损失的贡献，官方默认6.0")
        self.dis_spin.setFixedWidth(120)
        distill_param_layout.addWidget(self.dis_spin, 0, 3, alignment=Qt.AlignLeft)

        distill_param_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.distill_batch_spin = QSpinBox()
        self.distill_batch_spin.setRange(1, 256)
        self.distill_batch_spin.setValue(8)
        self.distill_batch_spin.setToolTip("蒸馏需同时加载教师和学生模型，建议比普通训练小")
        self.distill_batch_spin.setFixedWidth(120)
        distill_param_layout.addWidget(self.distill_batch_spin, 1, 1, alignment=Qt.AlignLeft)

        distill_param_layout.addWidget(QLabel("Image Size:"), 1, 2)
        self.distill_imgsz_spin = QSpinBox()
        self.distill_imgsz_spin.setRange(32, 2048)
        self.distill_imgsz_spin.setValue(640)
        self.distill_imgsz_spin.setSingleStep(32)
        self.distill_imgsz_spin.setFixedWidth(120)
        distill_param_layout.addWidget(self.distill_imgsz_spin, 1, 3, alignment=Qt.AlignLeft)

        distill_form.addRow("训练参数:", distill_param_widget)

        # 输出配置
        distill_out_row_widget = QWidget()
        distill_out_row_layout = QHBoxLayout(distill_out_row_widget)
        distill_out_row_layout.setContentsMargins(0, 0, 0, 0)
        distill_out_row_layout.setSpacing(5)

        self.distill_project_edit = DropLineEdit(self, file_filter='folder')
        self.distill_project_edit.setPlaceholderText("拖拽输出目录到这里，或点击浏览...")
        self.distill_project_edit.setText(str(AppConfig.DEFAULT_OUTPUT_DIR))
        browse_distill_project_btn = QPushButton("浏览")
        browse_distill_project_btn.clicked.connect(self.browse_distill_project)
        browse_distill_project_btn.setFixedWidth(60)
        distill_out_row_layout.addWidget(self.distill_project_edit, 1)
        distill_out_row_layout.addWidget(browse_distill_project_btn)
        distill_form.addRow("输出目录:", distill_out_row_widget)

        self.distill_name_edit = QLineEdit()
        self.distill_name_edit.setText("distill")
        self.distill_name_edit.setPlaceholderText("实验名称，如 distill_yolo26n")
        distill_form.addRow("实验名称:", self.distill_name_edit)

        # 蒸馏操作按钮（左对齐）
        distill_btn_widget = QWidget()
        distill_btn_layout = QHBoxLayout(distill_btn_widget)
        distill_btn_layout.setContentsMargins(12, 0, 12, 0)
        distill_btn_layout.setSpacing(8)
        self.distill_btn = QPushButton("开始知识蒸馏")
        self.distill_btn.clicked.connect(self.start_distillation)
        self.distill_btn.setEnabled(False)
        self.distill_btn.setToolTip("用教师模型指导学生模型进行蒸馏训练")
        self.distill_btn.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #59359d; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        distill_btn_layout.addWidget(self.distill_btn)

        self.distill_stop_btn = QPushButton("停止蒸馏")
        self.distill_stop_btn.clicked.connect(self.stop_distillation)
        self.distill_stop_btn.setEnabled(False)
        self.distill_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #d83b01;
                color: white;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #a83200; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        distill_btn_layout.addWidget(self.distill_stop_btn)
        distill_btn_layout.addStretch()
        distill_outer_layout.addWidget(distill_btn_widget)

        self.distill_progress_bar = QProgressBar()
        self.distill_progress_bar.setVisible(False)
        self.distill_progress_bar.setContentsMargins(12, 0, 12, 0)
        distill_outer_layout.addWidget(self.distill_progress_bar)

        scroll_layout.addWidget(distill_group)

        # ====================================================================
        # Section 4: 模型比较 (验证转换结果)
        # ====================================================================
        compare_group = QGroupBox("4. 模型比较 (验证转换结果)")
        compare_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        compare_layout = QVBoxLayout(compare_group)

        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("测试图片:"))
        test_layout.setSpacing(5)

        self.test_image_path_edit = DropLineEdit(self, file_filter='image')
        self.test_image_path_edit.setPlaceholderText("拖拽测试图片到这里，或点击浏览...")
        self.test_image_path_edit.setToolTip("选择用于比较的测试图片")

        browse_test_btn = QPushButton("浏览...")
        browse_test_btn.clicked.connect(self.browse_test_image)
        browse_test_btn.setFixedWidth(80)

        test_layout.addWidget(self.test_image_path_edit, 1)
        test_layout.addWidget(browse_test_btn)
        compare_layout.addLayout(test_layout)

        self.compare_btn = QPushButton("比较模型预测")
        self.compare_btn.clicked.connect(self.compare_models)
        self.compare_btn.setEnabled(False)
        self.compare_btn.setToolTip("比较PT模型和ONNX模型的预测结果")
        compare_layout.addWidget(self.compare_btn)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setFont(QFont("Consolas", 9))
        compare_layout.addWidget(self.result_text)

        # 对比图像显示
        image_compare_group = QGroupBox("预测结果对比")
        image_compare_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        image_compare_layout = QHBoxLayout(image_compare_group)

        pt_image_layout = QVBoxLayout()
        pt_image_layout.addWidget(QLabel("PT模型:"))
        self.pt_image_label = QLabel("暂无图像")
        self.pt_image_label.setAlignment(Qt.AlignCenter)
        self.pt_image_label.setMinimumHeight(280)
        self.pt_image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        pt_image_layout.addWidget(self.pt_image_label)
        image_compare_layout.addLayout(pt_image_layout)

        onnx_image_layout = QVBoxLayout()
        onnx_image_layout.addWidget(QLabel("ONNX模型:"))
        self.onnx_image_label = QLabel("暂无图像")
        self.onnx_image_label.setAlignment(Qt.AlignCenter)
        self.onnx_image_label.setMinimumHeight(280)
        self.onnx_image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        onnx_image_layout.addWidget(self.onnx_image_label)
        image_compare_layout.addLayout(onnx_image_layout)

        compare_layout.addWidget(image_compare_group)

        scroll_layout.addWidget(compare_group)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)

        # ====================================================================
        # 日志显示
        # ====================================================================
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

        splitter.addWidget(scroll_area)

        log_group = QGroupBox("转换日志")
        log_layout = QVBoxLayout(log_group)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(100)
        self.log_display.setFont(QFont("Consolas", 9))

        log_handler = QTextEditLogger(self.log_display)
        self.logger.add_handler(log_handler)

        log_layout.addWidget(self.log_display)

        splitter.addWidget(log_group)
        splitter.setSizes([800, 200])

        main_layout.addWidget(splitter, 1)

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _on_quantize_changed(self, index):
        """量化精度变化时更新UI"""
        quantize = self.quantize_combo.currentData()
        is_int8 = quantize == '8'

        try:
            if hasattr(self, 'calib_data_edit') and self.calib_data_edit is not None:
                self.calib_data_edit.setEnabled(is_int8)
            if hasattr(self, 'calib_browse_btn') and self.calib_browse_btn is not None:
                self.calib_browse_btn.setEnabled(is_int8)
            if hasattr(self, 'calib_fraction_spin') and self.calib_fraction_spin is not None:
                self.calib_fraction_spin.setEnabled(is_int8)

            if is_int8:
                self.calib_data_edit.setPlaceholderText("选择校准数据集YAML (INT8量化必需)")
            else:
                self.calib_data_edit.setPlaceholderText("选择校准数据集YAML文件...")
        except RuntimeError:
            pass

    # ========================================================================
    # 模型版本兼容性
    # ========================================================================

    def get_version_compatibility(self, model_version: str) -> dict:
        """获取模型版本的兼容性信息"""
        return self.YOLO_VERSION_COMPATIBILITY.get(model_version, {
            'nms_supported': True,
            'nms_required': False,
            'nms_default': True,
            'dynamic_default': False,
            'fp16_default': False,
            'tasks': ['detect'],
            'note': '未知版本，使用默认设置',
        })

    def show_version_compatibility_info(self, model_version: str):
        """显示版本兼容性信息"""
        info = self.get_version_compatibility(model_version)
        lines = [
            f"【{model_version.upper()} 版本兼容性说明】",
            "",
            f"✓ 支持的任务类型: {', '.join(info['tasks'])}",
            f"✓ NMS支持: {'是' if info['nms_supported'] else '否（内置）'}",
            f"  - 默认勾选NMS: {'是' if info['nms_default'] else '否'}",
            f"  - 必须包含NMS: {'是（端到端模型）' if info['nms_required'] else '否（可选）'}",
            f"✓ 推荐FP16: {'是' if info['fp16_default'] else '否'}",
            f"✓ 推荐动态批处理: {'是' if info['dynamic_default'] else '否'}",
            "",
            f"说明: {info['note']}",
        ]
        QMessageBox.information(self, f"{model_version} 版本兼容性", "\n".join(lines))

    # ========================================================================
    # 浏览方法
    # ========================================================================

    def browse_calib_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择校准数据集YAML文件", "",
            "YAML文件 (*.yaml *.yml);;所有文件 (*.*)"
        )
        if file_path:
            self.calib_data_edit.setText(file_path)
            self.logger.info(f"选择校准数据集: {file_path}")

    def browse_pt_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLO模型文件", "",
            "PyTorch模型 (*.pt);;所有文件 (*.*)"
        )
        if file_path:
            self.pt_path.setText(file_path)
            # 自动生成ONNX输出路径
            base_path = os.path.splitext(file_path)[0]
            self.onnx_path.setText(base_path + ".onnx")
            self.detect_model_type(file_path)
            self.update_ui_state()

    def browse_onnx_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存ONNX文件", "",
            "ONNX模型 (*.onnx);;所有文件 (*.*)"
        )
        if file_path:
            if not file_path.endswith('.onnx'):
                file_path += '.onnx'
            self.onnx_path.setText(file_path)

    def browse_test_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择测试图片", "",
            f"图片文件 (*{' *'.join(AppConfig.IMAGE_EXTENSIONS)})"
        )
        if file_path:
            self.test_image_path_edit.setText(file_path)
            self.test_image_path = file_path

    def browse_teacher_model(self):
        """浏览选择教师模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择教师模型", "",
            "PyTorch模型 (*.pt);;所有文件 (*.*)"
        )
        if file_path:
            self.teacher_path.setText(file_path)
            self.logger.info(f"选择教师模型: {file_path}")

    def browse_student_model(self):
        """浏览选择学生模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择学生模型", "",
            "PyTorch模型 (*.pt);;所有文件 (*.*)"
        )
        if file_path:
            self.student_model_edit.setText(file_path)
            self.logger.info(f"选择学生模型: {file_path}")

    def on_student_model_changed(self, text):
        """学生模型路径变化时同步顶部PT模型"""
        if text != self.pt_path.text():
            self.pt_path.setText(text)
            self.update_ui_state()

    def browse_distill_data(self):
        """浏览选择蒸馏数据集"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集YAML文件", "",
            "YAML文件 (*.yaml *.yml);;所有文件 (*.*)"
        )
        if file_path:
            self.distill_data_edit.setText(file_path)
            self.logger.info(f"选择蒸馏数据集: {file_path}")

    def browse_distill_project(self):
        """浏览选择蒸馏输出目录"""
        folder = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self.distill_project_edit.text() or str(AppConfig.DEFAULT_OUTPUT_DIR)
        )
        if folder:
            self.distill_project_edit.setText(folder)
            self.logger.info(f"选择蒸馏输出目录: {folder}")

    # ========================================================================
    # 知识蒸馏
    # 参考: https://docs.ultralytics.com/guides/knowledge-distillation
    # ========================================================================

    def start_distillation(self):
        """开始知识蒸馏训练"""
        is_valid, error_msg = self.validate_inputs('distillation')
        if not is_valid:
            QMessageBox.warning(self, "输入验证失败", error_msg)
            return

        student_path = self.pt_path.text()
        teacher_path = self.teacher_path.text()
        data_yaml = self.distill_data_edit.text()

        # 收集蒸馏训练参数
        distill_kwargs = {
            'epochs': self.distill_epochs_spin.value(),
            'batch': self.distill_batch_spin.value(),
            'imgsz': self.distill_imgsz_spin.value(),
            'dis': self.dis_spin.value(),
            'project': self.distill_project_edit.text(),
            'name': self.distill_name_edit.text().strip() or 'distill',
            'exist_ok': True,
            'device': None,  # 自动选择
            'verbose': True,
        }

        # 日志输出
        self.log_display.clear()
        self.logger.info("=" * 60)
        self.logger.info("开始知识蒸馏训练")
        self.logger.info(f"学生模型: {student_path}")
        self.logger.info(f"教师模型: {teacher_path}")
        self.logger.info(f"数据集: {data_yaml}")
        self.logger.info(f"参数: epochs={distill_kwargs['epochs']}, batch={distill_kwargs['batch']}, "
                        f"imgsz={distill_kwargs['imgsz']}, dis={distill_kwargs['dis']}")
        self.logger.info(f"输出: {os.path.join(distill_kwargs['project'], distill_kwargs['name'])}")
        self.logger.info("=" * 60)

        self.distill_worker = KnowledgeDistillationWorker(
            student_path=student_path,
            teacher_path=teacher_path,
            data_yaml=data_yaml,
            **distill_kwargs
        )

        self.distill_worker.finished.connect(self.on_distill_finished)
        self.distill_worker.error.connect(self.on_distill_error)
        self.distill_worker.log.connect(self.logger.info)

        self.distill_btn.setEnabled(False)
        self.distill_stop_btn.setEnabled(True)
        self.distill_progress_bar.setVisible(True)
        self.distill_progress_bar.setRange(0, 0)  # 不定进度条

        self.distill_worker.start()

    def stop_distillation(self):
        """停止蒸馏训练"""
        if self.distill_worker and self.distill_worker.isRunning():
            reply = QMessageBox.question(
                self, "确认停止",
                "确定要停止蒸馏训练吗？\n（训练可能无法立即中断，需等待当前epoch结束）",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.distill_worker.stop()
                self.logger.warning("正在请求停止蒸馏训练...")

    def on_distill_finished(self, best_path: str):
        """蒸馏训练完成"""
        self.logger.info("=" * 60)
        self.logger.info("知识蒸馏训练完成！")
        self.logger.info(f"最佳模型: {best_path}")
        self.logger.info("蒸馏导出的模型仅含学生权重，部署开销与学生模型一致")
        self.logger.info("=" * 60)

        self.distill_btn.setEnabled(True)
        self.distill_stop_btn.setEnabled(False)
        self.distill_progress_bar.setVisible(False)

        # 保存配置
        self.save_config()

        # 询问是否将蒸馏结果作为新模型用于导出
        if os.path.exists(best_path):
            reply = QMessageBox.question(
                self, "蒸馏完成",
                f"知识蒸馏训练完成！\n最佳模型: {best_path}\n\n"
                f"是否将该模型设置为当前PT模型，继续执行ONNX转换？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.pt_path.setText(best_path)

        self.distill_worker = None

    def on_distill_error(self, error_msg: str):
        """蒸馏训练失败"""
        self.logger.error(f"知识蒸馏失败: {error_msg}")
        QMessageBox.critical(self, "蒸馏失败", f"错误信息:\n{error_msg}")
        self.distill_btn.setEnabled(True)
        self.distill_stop_btn.setEnabled(False)
        self.distill_progress_bar.setVisible(False)
        self.distill_worker = None

    # ========================================================================
    # 模型检测
    # ========================================================================

    def on_pt_path_changed(self, text):
        """PT模型路径变化时自动检测并设置参数"""
        if text and os.path.exists(text) and text.endswith('.pt'):
            self.detect_model_type(text)
        self.update_ui_state()

    def detect_model_type(self, model_path: str):
        """检测模型类型和版本"""
        try:
            task = self.model_manager.detect_model_task(model_path)
            self.model_task = task

            model_filename = os.path.basename(model_path)
            model_base_name = os.path.splitext(model_filename)[0]
            model_version = self._identify_model_version(model_base_name)
            self._current_model_version = model_version

            self.compat_btn.setEnabled(model_version in self.YOLO_VERSION_COMPATIBILITY)

            task_names = {
                'detect': '目标检测', 'segment': '实例分割',
                'semantic': '语义分割', 'depth': '深度估计',
                'classify': '图像分类', 'pose': '姿态估计', 'obb': '旋转框检测'
            }
            task_name = task_names.get(task, task)
            self.model_info_label.setText(f"模型类型: {task_name} | 版本: {model_version}")
            self.logger.info(f"检测到模型: {model_version}, 任务类型: {task_name}")

            self._auto_set_options(model_version, task)

        except Exception as e:
            self.logger.error(f"检测模型类型失败: {e}")

    def _identify_model_version(self, model_name: str) -> str:
        """从模型名称识别版本"""
        version_order = ['yolo26', 'yolo11', 'yolov8', 'yolov5']
        for version in version_order:
            if model_name.startswith(version):
                return version
        return '未知'

    def _auto_set_options(self, model_version: str, task: str):
        """根据模型版本自动设置转换参数"""
        compat = self.get_version_compatibility(model_version)

        # ---- NMS设置 ----
        if task in ['detect', 'obb']:
            if compat.get('nms_supported', True):
                self.nms_check.setEnabled(True)
                if compat.get('nms_default', True):
                    self.nms_check.setChecked(True)
                    self.nms_check.setToolTip(f"{model_version}支持NMS，默认勾选")
                else:
                    self.nms_check.setChecked(False)
                    self.nms_check.setToolTip(f"{model_version}是端到端模型，内置NMS，不需要额外添加")

                if compat.get('nms_required', False):
                    self.nms_check.setToolTip(f"{model_version}是端到端模型，必须包含NMS")
                    self.nms_check.setChecked(True)
            else:
                self.nms_check.setEnabled(False)
                self.nms_check.setChecked(False)
                self.nms_check.setToolTip(f"{model_version}是端到端模型，内置NMS，不需要额外添加")
        else:
            self.nms_check.setEnabled(False)
            self.nms_check.setChecked(False)
            self.nms_check.setToolTip("非检测模型不需要NMS")

        # ---- FP16设置 ----
        if compat.get('fp16_default', False):
            self.quantize_combo.setCurrentText("FP16 (推荐)")
        else:
            self.quantize_combo.setCurrentText("FP32 (默认)")

        # ---- 动态批处理设置 ----
        self.dynamic_check.setChecked(compat.get('dynamic_default', False))

        # ---- 日志信息 ----
        nms_status = "✓支持" if compat.get('nms_supported', True) else "✗不支持(内置)"
        self.logger.info(f"兼容性: NMS={nms_status}")
        self.logger.info(f"自动设置完成: NMS={'勾选' if self.nms_check.isChecked() else '未勾选'}, "
                        f"量化={self.quantize_combo.currentText()}, "
                        f"动态批处理={'勾选' if self.dynamic_check.isChecked() else '未勾选'}")

    # ========================================================================
    # ONNX转换 (基于官方export API)
    # ========================================================================

    def start_conversion(self):
        is_valid, error_msg = self.validate_inputs('conversion')
        if not is_valid:
            QMessageBox.warning(self, "输入验证失败", error_msg)
            return

        pt_path = self.pt_path.text()
        onnx_path = self.onnx_path.text()

        if not onnx_path:
            onnx_path = os.path.splitext(pt_path)[0] + ".onnx"
            self.onnx_path.setText(onnx_path)

        # 收集参数 (基于官方export API)
        imgsz = self.imgsz_spin.value()
        batch = self.batch_spin.value()
        opset = int(self.opset_combo.currentText())
        dynamic = self.dynamic_check.isChecked()
        simplify = self.simplify_check.isChecked()
        nms = self.nms_check.isChecked() and self.nms_check.isEnabled()

        # 量化参数 (取代half/int8)
        quantize = self.quantize_combo.currentData()
        if quantize == '32':
            quantize = None  # FP32
        else:
            quantize = int(quantize)  # 8 或 16

        # 校准数据 (INT8需要)
        calib_data = self.calib_data_edit.text() if quantize == 8 else None
        calib_fraction = self.calib_fraction_spin.value() if quantize == 8 else 1.0

        # 构建导出参数
        export_kwargs = {
            'format': 'onnx',
            'imgsz': imgsz,
            'batch': batch,
            'opset': opset,
            'dynamic': dynamic,
            'simplify': simplify,
            'nms': nms,
        }

        # 添加量化参数
        if quantize is not None:
            export_kwargs['quantize'] = quantize

        if calib_data and quantize == 8:
            export_kwargs['data'] = calib_data
            export_kwargs['fraction'] = calib_fraction

        # 日志输出
        self.log_display.clear()
        self.logger.info("=" * 60)
        self.logger.info("开始ONNX转换 (基于Ultralytics官方API)")
        self.logger.info(f"PT模型: {pt_path}")
        self.logger.info(f"ONNX输出: {onnx_path}")
        self.logger.info(f"参数: imgsz={imgsz}, batch={batch}, opset={opset}")
        self.logger.info(f"量化精度: {quantize if quantize else 'FP32'}")
        self.logger.info(f"动态批处理={dynamic}, NMS={nms}, ONNX简化={simplify}")
        if quantize == 8:
            self.logger.info(f"校准数据集: {calib_data}")
            self.logger.info(f"校准数据比例: {calib_fraction}")
        self.logger.info("=" * 60)

        self.conversion_worker = ONNXConversionWorker(
            pt_path=pt_path,
            onnx_path=onnx_path,
            **export_kwargs
        )

        self.conversion_worker.finished.connect(self.on_conversion_finished)
        self.conversion_worker.error.connect(self.on_conversion_error)
        self.conversion_worker.log.connect(self.logger.info)

        self.convert_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不定进度条

        self.conversion_worker.start()

    def stop_conversion(self):
        if self.conversion_worker and self.conversion_worker.isRunning():
            reply = QMessageBox.question(
                self, "确认停止",
                "确定要停止转换吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.conversion_worker.stop()
                # 先等待线程自行结束，超时才强制终止
                if not self.conversion_worker.wait(3000):
                    self.conversion_worker.terminate()
                    self.conversion_worker.wait()
                self.on_conversion_stopped()

    def on_conversion_finished(self, onnx_path: str):
        self.logger.info("=" * 60)
        self.logger.info("ONNX转换完成！")
        self.logger.info(f"输出文件: {onnx_path}")

        # 检查文件大小
        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            self.logger.info(f"文件大小: {size_mb:.2f} MB")

        self.logger.info("=" * 60)

        self.convert_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        self.save_config()

        if self.test_image_path:
            reply = QMessageBox.question(
                self, "转换完成",
                f"ONNX转换完成！\n输出文件: {onnx_path}\n\n是否立即比较模型预测？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.compare_models()

        self.conversion_worker = None

    def on_conversion_error(self, error_msg: str):
        self.logger.error(f"转换失败: {error_msg}")
        QMessageBox.critical(self, "转换失败", f"错误信息:\n{error_msg}")
        self.convert_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.conversion_worker = None

    def on_conversion_stopped(self):
        self.logger.warning("转换已停止")
        self.convert_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.conversion_worker = None

    # ========================================================================
    # 模型比较
    # ========================================================================

    def compare_models(self):
        if not self.pt_path.text() or not self.onnx_path.text():
            QMessageBox.warning(self, "警告", "请先选择PT和ONNX模型文件")
            return
        if not self.test_image_path:
            QMessageBox.warning(self, "警告", "请选择测试图片")
            return
        if not os.path.exists(self.onnx_path.text()):
            QMessageBox.warning(self, "错误", f"ONNX文件不存在: {self.onnx_path.text()}")
            return

        self.result_text.clear()
        self.result_text.append("正在比较模型预测...")

        self.comparison_worker = ComparisonWorker(
            pt_path=self.pt_path.text(),
            onnx_path=self.onnx_path.text(),
            image_path=self.test_image_path,
            imgsz=self.imgsz_spin.value()
        )

        self.comparison_worker.finished.connect(self.on_comparison_finished)
        self.comparison_worker.error.connect(self.on_comparison_error)
        self.comparison_worker.log.connect(self.logger.info)

        self.compare_btn.setEnabled(False)
        self.comparison_worker.start()

    def on_comparison_finished(self, results: dict):
        task = results.get('task', 'unknown')

        if task == 'classify':
            result_text = self.format_classification_results(results)
        else:
            result_text = self.format_detection_results(results)

        self.result_text.clear()
        self.result_text.append(result_text)

        self.display_comparison_images(results)

        self.compare_btn.setEnabled(True)
        self.comparison_worker = None

    def on_comparison_error(self, error_msg: str):
        self.logger.error(f"比较失败: {error_msg}")
        self.result_text.append(f"\n错误: {error_msg}")
        self.compare_btn.setEnabled(True)
        self.comparison_worker = None

    def format_classification_results(self, results: dict) -> str:
        pt_result = results['pt']
        onnx_wrapper = results['onnx_wrapper']
        onnx_direct = results['onnx_direct']

        pt_probs = pt_result.probs
        pt_top1 = pt_probs.top1
        pt_conf = pt_probs.top1conf.item()
        pt_class = pt_result.names[pt_top1]

        onnx_probs = onnx_wrapper.probs
        onnx_top1 = onnx_probs.top1
        onnx_conf = onnx_probs.top1conf.item()
        onnx_class = onnx_wrapper.names[onnx_top1]

        _exp = np.exp(onnx_direct - np.max(onnx_direct, axis=-1, keepdims=True))
        onnx_direct_probs = (_exp / _exp.sum(axis=-1, keepdims=True)).squeeze()
        direct_top1 = np.argmax(onnx_direct_probs)
        direct_conf = onnx_direct_probs[direct_top1]
        direct_class = onnx_wrapper.names.get(direct_top1, f"Class_{direct_top1}")

        text = f"分类模型预测比较 (imgsz={self.imgsz_spin.value()}):\n\n"
        text += f"PT模型预测:\n  类别: {pt_class}\n  置信度: {pt_conf:.4f}\n\n"
        text += f"ONNX包装器预测:\n  类别: {onnx_class}\n  置信度: {onnx_conf:.4f}\n\n"
        text += f"直接ONNX推理:\n  类别: {direct_class}\n  置信度: {direct_conf:.4f}\n\n"
        text += "一致性检查:\n"
        if pt_class == onnx_class:
            text += "  ✓ PT vs ONNX包装器: 类别一致\n"
        else:
            text += f"  ✗ PT vs ONNX包装器: 类别不一致 (PT:{pt_class} vs ONNX:{onnx_class})\n"
        if abs(pt_conf - onnx_conf) < 0.01:
            text += "  ✓ PT vs ONNX包装器: 置信度接近\n"
        else:
            text += f"  ✗ PT vs ONNX包装器: 置信度差异较大 (差值: {abs(pt_conf - onnx_conf):.4f})\n"
        if onnx_class == direct_class:
            text += "  ✓ ONNX包装器 vs 直接ONNX: 类别一致"
        else:
            text += f"  ✗ ONNX包装器 vs 直接ONNX: 类别不一致 (Wrapper:{onnx_class} vs Direct:{direct_class})"
        return text

    def format_detection_results(self, results: dict) -> str:
        pt_result = results['pt']
        onnx_result = results['onnx_wrapper']

        pt_boxes = pt_result.boxes if pt_result.boxes is not None else []
        onnx_boxes = onnx_result.boxes if onnx_result.boxes is not None else []

        iou_scores = []
        if len(pt_boxes) > 0 and len(onnx_boxes) > 0:
            for pt_box in pt_boxes:
                for onnx_box in onnx_boxes:
                    iou = self.calculate_iou(
                        pt_box.xyxy[0].cpu().numpy(),
                        onnx_box.xyxy[0].cpu().numpy()
                    )
                    iou_scores.append(iou)

        avg_iou = np.mean(iou_scores) if iou_scores else 0

        text = f"检测模型预测比较 (imgsz={self.imgsz_spin.value()}):\n\n"
        text += f"PT模型检测到 {len(pt_boxes)} 个目标\n"
        text += f"ONNX模型检测到 {len(onnx_boxes)} 个目标\n\n"

        if len(pt_boxes) > 0 and len(onnx_boxes) > 0:
            text += f"平均IoU: {avg_iou:.4f}\n\n"
            if avg_iou > 0.5:
                text += "✓ 检测结果基本一致 (IoU > 0.5)\n"
            elif avg_iou > 0.3:
                text += "⚠ 检测结果部分一致 (0.3 < IoU ≤ 0.5)\n"
            else:
                text += "✗ 检测结果差异较大 (IoU ≤ 0.3)\n"

        if len(pt_boxes) > 0:
            pt_confs = [box.conf.item() for box in pt_boxes]
            text += f"\nPT模型置信度范围: {min(pt_confs):.3f} - {max(pt_confs):.3f}"

        if len(onnx_boxes) > 0:
            onnx_confs = [box.conf.item() for box in onnx_boxes]
            text += f"\nONNX模型置信度范围: {min(onnx_confs):.3f} - {max(onnx_confs):.3f}"

        return text

    def calculate_iou(self, box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def display_comparison_images(self, results: dict):
        try:
            pt_result = results['pt']
            onnx_result = results['onnx_wrapper']

            pt_plot = pt_result.plot()
            onnx_plot = onnx_result.plot()

            pt_pixmap = ImageProcessor.cv_to_qpixmap(pt_plot)
            onnx_pixmap = ImageProcessor.cv_to_qpixmap(onnx_plot)

            if pt_pixmap and not pt_pixmap.isNull():
                scaled_pt = pt_pixmap.scaled(
                    self.pt_image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.pt_image_label.setPixmap(scaled_pt)
                self.pt_image_label.setCursor(Qt.PointingHandCursor)
                self.pt_image_label.mousePressEvent = lambda event: self.show_enlarged_image(pt_pixmap, "PT模型预测结果")
            else:
                self.pt_image_label.setText("PT模型结果无法显示")

            if onnx_pixmap and not onnx_pixmap.isNull():
                scaled_onnx = onnx_pixmap.scaled(
                    self.onnx_image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.onnx_image_label.setPixmap(scaled_onnx)
                self.onnx_image_label.setCursor(Qt.PointingHandCursor)
                self.onnx_image_label.mousePressEvent = lambda event: self.show_enlarged_image(onnx_pixmap, "ONNX模型预测结果")
            else:
                self.onnx_image_label.setText("ONNX模型结果无法显示")

            if self.image_display and onnx_pixmap and not onnx_pixmap.isNull():
                self.image_display.set_pixmap(onnx_pixmap)

        except Exception as e:
            self.logger.error(f"显示图像失败: {e}")
            self.pt_image_label.setText(f"显示失败: {e}")
            self.onnx_image_label.setText(f"显示失败: {e}")

    def show_enlarged_image(self, pixmap: QPixmap, title: str):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        scaled = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        label.setFixedSize(scaled.size())

        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.exec_()

    # ========================================================================
    # UI更新
    # ========================================================================

    def update_ui_state(self):
        """更新UI状态"""
        try:
            has_pt = bool(self.pt_path.text())
            self.convert_btn.setEnabled(has_pt)

            has_test_image = bool(self.test_image_path_edit.text())
            has_onnx = bool(self.onnx_path.text())
            self.compare_btn.setEnabled(has_pt and has_onnx and has_test_image)

            # 蒸馏按钮：需要学生模型 + 教师模型 + 数据集，任务受支持且为同一家族
            has_teacher = bool(self.teacher_path.text())
            has_distill_data = bool(self.distill_data_edit.text())
            task_supported = self.model_task in KnowledgeDistillationWorker.SUPPORTED_TASKS
            # 学生与教师须同一家族（YOLOv8/YOLO11/YOLO26）
            student_version = self._current_model_version
            teacher_name = os.path.splitext(os.path.basename(self.teacher_path.text()))[0] if has_teacher else ''
            teacher_version = self._identify_model_version(teacher_name) if teacher_name else ''
            same_family = (student_version in ('yolov8', 'yolo11', 'yolo26')
                           and student_version == teacher_version)
            self.distill_btn.setEnabled(
                has_pt and has_teacher and has_distill_data and task_supported and same_family)
        except RuntimeError:
            # 控件已被删除，忽略错误
            pass

    # ========================================================================
    # 配置管理
    # ========================================================================

    def save_config(self):
        config = {
            'pt_path': self.pt_path.text(),
            'onnx_path': self.onnx_path.text(),
            'imgsz': self.imgsz_spin.value(),
            'batch': self.batch_spin.value(),
            'opset': self.opset_combo.currentText(),
            'quantize': self.quantize_combo.currentData(),
            'dynamic': self.dynamic_check.isChecked(),
            'nms': self.nms_check.isChecked(),
            'simplify': self.simplify_check.isChecked(),
            'test_image': self.test_image_path_edit.text(),
            'distill_teacher': self.teacher_path.text(),
            'distill_data': self.distill_data_edit.text(),
            'distill_epochs': self.distill_epochs_spin.value(),
            'distill_dis': self.dis_spin.value(),
            'distill_batch': self.distill_batch_spin.value(),
            'distill_imgsz': self.distill_imgsz_spin.value(),
            'distill_project': self.distill_project_edit.text(),
            'distill_name': self.distill_name_edit.text()
        }

        config_dir = Path.home() / ".yolo_ui_workstation"
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "onnx_config.json"
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")

    def load_config(self):
        config_dir = Path.home() / ".yolo_ui_workstation"
        config_path = config_dir / "onnx_config.json"
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 加载配置时阻塞信号，避免触发模型检测（导入 torch）影响启动速度
                for widget in (self.pt_path, self.onnx_path, self.test_image_path_edit,
                               self.teacher_path, self.distill_data_edit,
                               self.distill_project_edit, self.distill_name_edit):
                    widget.blockSignals(True)
                try:
                    self.pt_path.setText(config.get('pt_path', ''))
                    self.onnx_path.setText(config.get('onnx_path', ''))
                    self.test_image_path_edit.setText(config.get('test_image', ''))
                    if config.get('test_image'):
                        self.test_image_path = config['test_image']
                    self.teacher_path.setText(config.get('distill_teacher', ''))
                    self.distill_data_edit.setText(config.get('distill_data', ''))
                    self.distill_project_edit.setText(config.get('distill_project', str(AppConfig.DEFAULT_OUTPUT_DIR)))
                    self.distill_name_edit.setText(config.get('distill_name', 'distill'))
                finally:
                    for widget in (self.pt_path, self.onnx_path, self.test_image_path_edit,
                                   self.teacher_path, self.distill_data_edit,
                                   self.distill_project_edit, self.distill_name_edit):
                        widget.blockSignals(False)
        except Exception as e:
            self.logger.warning(f"加载配置失败: {e}")

    def reset_to_defaults(self):
        reply = QMessageBox.question(
            self, '确认恢复默认值',
            '确定要将所有参数恢复为默认值吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self.imgsz_spin.setValue(640)
                self.batch_spin.setValue(1)
                self.opset_combo.setCurrentText('14')
                self.quantize_combo.setCurrentText("FP32 (默认)")

                self.dynamic_check.setChecked(False)
                self.nms_check.setChecked(True)
                self.simplify_check.setChecked(True)

                self.pt_path.clear()
                self.onnx_path.clear()
                self.test_image_path_edit.clear()
                self.test_image_path = None

                # 蒸馏参数重置
                self.student_model_edit.clear()
                self.teacher_path.clear()
                self.distill_data_edit.clear()
                self.distill_epochs_spin.setValue(AppConfig.DEFAULT_EPOCHS)
                self.dis_spin.setValue(6.0)
                self.distill_batch_spin.setValue(8)
                self.distill_imgsz_spin.setValue(640)
                self.distill_project_edit.setText(str(AppConfig.DEFAULT_OUTPUT_DIR))
                self.distill_name_edit.setText("distill")

                self.model_info_label.setText("模型类型: 未检测")
                self.model_task = None

                self.update_ui_state()

                self.logger.info("已恢复默认参数设置")
                QMessageBox.information(self, '恢复成功', '所有参数已恢复为默认值！')

            except Exception as e:
                self.logger.error(f"恢复默认值失败: {e}")
                QMessageBox.warning(self, '恢复失败', f'恢复默认值时发生错误：{e}')

    def on_tab_selected(self):
        self.logger.info("模型部署标签页已激活")
        self.update_ui_state()


    # ========================================================================
    # 验证方法
    # ========================================================================

    def validate_inputs(self, validation_type: str = 'conversion') -> tuple:
        validation_map = {
            'conversion': self._validate_conversion,
            'comparison': self._validate_comparison,
            'distillation': self._validate_distillation,
        }
        validator = validation_map.get(validation_type)
        if validator:
            return validator()
        return False, f"未知的验证类型: {validation_type}"

    def _validate_conversion(self) -> tuple:
        pt_path = self.pt_path.text()
        if not pt_path:
            return False, "请选择PT模型文件"
        if not os.path.exists(pt_path):
            return False, f"PT模型文件不存在: {pt_path}"
        if not pt_path.endswith('.pt'):
            return False, "请选择有效的.pt模型文件"

        # INT8量化需要校准数据集
        if self.quantize_combo.currentData() == '8':
            calib_path = self.calib_data_edit.text()
            if not calib_path or not os.path.exists(calib_path):
                return False, "INT8量化需要选择有效的校准数据集YAML文件"
        return True, ""

    def _validate_comparison(self) -> tuple:
        test_image = self.test_image_path_edit.text()
        if not test_image:
            return False, "请选择测试图片"
        if not os.path.exists(test_image):
            return False, f"测试图片不存在: {test_image}"
        return True, ""

    def _validate_distillation(self) -> tuple:
        """验证知识蒸馏输入"""
        student_path = self.pt_path.text()
        if not student_path or not os.path.exists(student_path):
            return False, "请先在上方选择有效的学生模型(.pt)"

        if not student_path.endswith('.pt'):
            return False, "学生模型必须是.pt格式"

        # 支持 YOLOv8 / YOLO11 / YOLO26（教师与学生须同一家族）
        supported_families = ('yolov8', 'yolo11', 'yolo26')
        student_version = self._current_model_version
        if student_version not in supported_families:
            return False, (f"知识蒸馏支持 YOLOv8 / YOLO11 / YOLO26 系列模型，"
                           f"当前学生模型版本: {student_version}")

        teacher_path = self.teacher_path.text()
        if not teacher_path or not os.path.exists(teacher_path):
            return False, "请选择有效的教师模型(.pt)"

        if not teacher_path.endswith('.pt'):
            return False, "教师模型必须是.pt格式"

        teacher_name = os.path.splitext(os.path.basename(teacher_path))[0]
        teacher_version = self._identify_model_version(teacher_name)
        if teacher_version not in supported_families:
            return False, f"教师模型版本 {teacher_version} 不支持知识蒸馏"

        if teacher_version != student_version:
            return False, (f"教师与学生须同一家族：学生为 {student_version}，"
                           f"教师为 {teacher_version}，跨家族蒸馏不受支持")

        data_yaml = self.distill_data_edit.text()
        if not data_yaml or not os.path.exists(data_yaml):
            return False, "请选择有效的蒸馏数据集YAML文件"

        # 任务支持检查
        if self.model_task not in KnowledgeDistillationWorker.SUPPORTED_TASKS:
            supported = {'detect': '目标检测', 'segment': '实例分割', 'pose': '姿态估计', 'obb': '旋转框检测'}
            names = " / ".join(supported.values())
            return False, f"当前任务类型({self.model_task})不支持知识蒸馏，仅支持: {names}"
        return True, ""

    # ========================================================================
    # 资源清理
    # ========================================================================

    def cleanup_threads(self):
        workers = [self.conversion_worker, self.comparison_worker, self.distill_worker]
        for worker in workers:
            if worker and worker.isRunning():
                try:
                    worker.is_stopped = True
                    worker.quit()
                    worker.wait(3000)
                except Exception as e:
                    self.logger.warning(f"清理线程时出错: {e}")
        self.conversion_worker = None
        self.comparison_worker = None
        self.distill_worker = None

    def closeEvent(self, event):
        self.cleanup_threads()
        super().closeEvent(event)
