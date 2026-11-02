"""
主窗口 - 整合所有标签页和核心功能
"""
import os
import sys

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QStatusBar, QMessageBox,
    QPushButton, QCheckBox, QTabWidget, QShortcut, QSpinBox,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence

from config.settings import AppConfig
from utils.logger import LogManager
from utils.device_utils import DeviceManager, PowerManager
from utils.error_handler import setup_global_exception_handler
from core.model_manager import ModelManager
from core.inference_engine import InferenceEngine
from core.trainer import ModelTrainer

# 导入标签页
from ui.tabs.dataset_management_tab import DatasetManagementTab
from ui.tabs.validate_tab import ValidateTab
from ui.tabs.model_deploy_tab import ModelDeployTab
from ui.tabs.combined_inference_tab import CombinedInferenceTab
from ui.tabs.combined_train_tab import CombinedTrainTab
from ui.widgets.zoomable_scroll_area import ZoomableScrollArea


class DeviceInfoWorker(QThread):
    """后台探测设备信息，避免启动时同步导入 torch 造成界面卡顿"""

    finished_info = pyqtSignal(dict)

    def __init__(self, device_manager):
        super().__init__()
        self.device_manager = device_manager

    def run(self):
        try:
            self.finished_info.emit(self.device_manager.get_device_info())
        except Exception:
            pass


class MainWindow(QMainWindow):
    """主窗口"""

    # 自定义信号
    status_updated = pyqtSignal(str)
    model_loaded = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        # 初始化核心组件
        self.logger = LogManager(max_lines=AppConfig.MAX_LOG_LINES)
        self.device_manager = DeviceManager(verbose=True)  # 只在主窗口打印一次
        self.power_manager = PowerManager()
        self.model_manager = ModelManager(self.logger)
        self.inference_engine = InferenceEngine(self.logger)
        self.trainer = ModelTrainer(self.logger)

        # 共享的图像显示区域
        self.shared_image_display = ZoomableScrollArea()
        # 保留image_label属性以保持兼容性
        self.image_label = None
        # ZoomableScrollArea内部会创建自己的image_label，不需要外部设置
        # 移除手动设置widget的代码，让ZoomableScrollArea使用内部的image_label

        # 设置全局异常处理
        setup_global_exception_handler()

        # 设置UI
        self.setup_ui()
        self.setup_status_bar()
        self.setup_shortcuts()
        self.setup_connections()

        # 启动定时器
        self.start_timers()

        # 设备探测放到后台线程，避免窗口显示后因导入 torch 卡顿
        self.device_info_worker = DeviceInfoWorker(self.device_manager)
        self.device_info_worker.finished_info.connect(self.show_startup_info)
        self.device_info_worker.start()

    def setup_ui(self):
        """设置UI"""
        self.setWindowTitle(AppConfig.APP_NAME)
        self.setGeometry(100, 100, AppConfig.WINDOW_WIDTH, AppConfig.WINDOW_HEIGHT)

        # 创建菜单栏
        menu_bar = self.menuBar()
        
        # 文件菜单
        file_menu = menu_bar.addMenu("文件(&F)")
        file_menu.addAction("退出(&X)", self.close, "Ctrl+Q")

        # 视图菜单
        view_menu = menu_bar.addMenu("视图(&V)")
        view_menu.addAction("切换全屏(&F)", self.toggle_fullscreen, "F11")
        view_menu.addAction("适应图像到视图(&Z)", self.fit_image_to_view, "Ctrl+0")
        
        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助(&H)")
        help_menu.addAction("显示帮助(&H)", self.show_help, "F1")
        help_menu.addAction("关于(&A)", self.show_about)

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: white;
            }
            QPushButton {
                padding: 6px 12px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
            }
            QPushButton:pressed {
                background-color: #cccccc;
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #808080;
            }
            QLineEdit {
                padding: 4px 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QComboBox {
                padding: 4px 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 4px;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
        """)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 创建水平分割器
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #e0e0e0;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QSplitter::handle:hover {
                background-color: #b0b0b0;
            }
        """)

        # 设置分割器可以随着窗口大小变化而调整比例
        self.main_splitter.setChildrenCollapsible(False)

        # 左侧面板 - 图像显示区域
        left_widget = self.create_left_panel()
        self.main_splitter.addWidget(left_widget)

        # 右侧面板 - 标签页
        right_widget = self.create_right_panel()
        self.main_splitter.addWidget(right_widget)

        # 设置初始大小比例（窗口宽度的40%给左侧，60%给右侧）
        total_width = AppConfig.WINDOW_WIDTH
        left_width = int(total_width * 0.4)  # 40%
        right_width = total_width - left_width  # 60%
        self.main_splitter.setSizes([left_width, right_width])

        main_layout.addWidget(self.main_splitter)

        # 底部面板 - 电源管理和日志控制
        bottom_layout = self.create_bottom_panel()
        main_layout.addLayout(bottom_layout)

    def create_left_panel(self) -> QWidget:
        """创建左侧面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        title_label = QLabel("图像预览")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 图像显示区域
        layout.addWidget(self.shared_image_display, 1)

        # 缩放控制
        zoom_layout = QHBoxLayout()
        zoom_layout.addStretch()

        zoom_info = QLabel("使用 Ctrl+滚轮 缩放 | 鼠标拖拽移动")
        zoom_info.setStyleSheet("color: #666666; font-size: 11px;")
        zoom_layout.addWidget(zoom_info)

        reset_btn = QPushButton("适应窗口 (Ctrl+0)")
        reset_btn.clicked.connect(self.fit_image_to_view)
        reset_btn.setStyleSheet("padding: 4px 8px;")
        zoom_layout.addWidget(reset_btn)

        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)

        return widget

    def create_right_panel(self) -> QWidget:
        """创建右侧面板（标签页懒加载：仅创建首个标签页，其余切换时再创建）"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 已创建的标签页实例
        self.tabs = {}
        # 标签页工厂函数（名称 -> (标题, 创建函数)）
        self._tab_factories = {
            'dataset_management': (
                "0. 数据集管理",
                lambda: DatasetManagementTab(
                    logger=self.logger, model_manager=self.model_manager,
                    image_display=self.shared_image_display, image_label=self.image_label
                )
            ),
            'combined_train': (
                "1. 模型训练",
                lambda: CombinedTrainTab(
                    logger=self.logger, model_manager=self.model_manager,
                    trainer=self.trainer,
                    image_display=self.shared_image_display, image_label=self.image_label
                )
            ),
            'validate': (
                "2. 模型验证",
                lambda: ValidateTab(
                    logger=self.logger, model_manager=self.model_manager,
                    trainer=self.trainer,
                    image_display=self.shared_image_display, image_label=self.image_label
                )
            ),
            'combined_inference': (
                "3. 模型推理",
                lambda: CombinedInferenceTab(
                    logger=self.logger, model_manager=self.model_manager,
                    image_display=self.shared_image_display, image_label=self.image_label
                )
            ),
            'model_deploy': (
                "4. 模型部署",
                lambda: ModelDeployTab(
                    logger=self.logger, model_manager=self.model_manager,
                    inference_engine=self.inference_engine,
                    image_display=self.shared_image_display, image_label=self.image_label
                )
            ),
        }

        # 按顺序添加占位标签页（第一个标签页立即创建）
        self._tab_order = list(self._tab_factories.keys())
        for i, name in enumerate(self._tab_order):
            title, _ = self._tab_factories[name]
            if i == 0:
                self.tabs[name] = self._tab_factories[name][1]()
                self.tab_widget.addTab(self.tabs[name], title)
            else:
                # 占位空 widget，首次切换时替换
                placeholder = QWidget()
                self.tab_widget.addTab(placeholder, title)

        # 连接标签页切换信号
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        layout.addWidget(self.tab_widget)

        return widget

    def _ensure_tab_created(self, index: int):
        """确保指定索引的标签页已创建（懒加载）"""
        if index < 0 or index >= len(self._tab_order):
            return
        name = self._tab_order[index]
        if name not in self.tabs:
            title, factory = self._tab_factories[name]
            tab = factory()
            self.tabs[name] = tab
            # 替换占位 widget
            self.tab_widget.removeTab(index)
            self.tab_widget.insertTab(index, tab, title)
            self.tab_widget.setCurrentIndex(index)

    def create_bottom_panel(self) -> QHBoxLayout:
        """创建底部面板"""
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(5, 5, 5, 5)

        # 左侧 - 电源管理
        self.power_checkbox = QCheckBox("防止系统休眠/锁屏")
        self.power_checkbox.setChecked(True)
        self.power_checkbox.stateChanged.connect(
            lambda state: self.power_manager.set_sleep_prevention(state == Qt.Checked)
        )
        bottom_layout.addWidget(self.power_checkbox)

        bottom_layout.addStretch()

        # 右侧 - 日志控制
        log_layout = QHBoxLayout()
        log_layout.addWidget(QLabel("最大日志行数:"))

        self.max_log_spinbox = QSpinBox()
        self.max_log_spinbox.setRange(100, 10000)
        self.max_log_spinbox.setValue(AppConfig.MAX_LOG_LINES)
        self.max_log_spinbox.setSingleStep(100)
        self.max_log_spinbox.valueChanged.connect(self.update_max_log_lines)
        log_layout.addWidget(self.max_log_spinbox)

        bottom_layout.addLayout(log_layout)

        return bottom_layout

    def setup_status_bar(self):
        """设置状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 状态标签
        self.model_status = QLabel("模型: 未加载")
        self.model_status.setMinimumWidth(250)  # 增加最小宽度
        
        # 设备信息延迟更新，避免启动时同步导入 torch
        self.device_status = QLabel("设备: 检测中...")
        self.device_status.setMinimumWidth(200)  # 增加最小宽度

        self.memory_status = QLabel("内存: -- MB")
        self.memory_status.setMinimumWidth(120)

        self.time_status = QLabel()
        self.time_status.setMinimumWidth(100)

        # 添加到状态栏
        self.status_bar.addWidget(self.model_status)
        self.status_bar.addWidget(self.device_status)
        self.status_bar.addWidget(self.memory_status)
        self.status_bar.addPermanentWidget(self.time_status)

    def setup_shortcuts(self):
        """设置快捷键"""
        shortcuts = [
            # 通用快捷键
            (QKeySequence("Ctrl+0"), self.fit_image_to_view, "适应图像到视图"),
            (QKeySequence("Ctrl+Q"), self.close, "退出程序"),
            (QKeySequence("F1"), self.show_help, "显示帮助"),
            (QKeySequence("Ctrl+Shift+C"), self.clear_cache, "清空缓存"),
            (QKeySequence("Ctrl+R"), self.refresh_all, "刷新"),
            
            # 标签页切换快捷键
            (QKeySequence("Ctrl+1"), lambda: self.switch_tab(0), "切换到数据集管理"),
            (QKeySequence("Ctrl+2"), lambda: self.switch_tab(1), "切换到模型训练"),
            (QKeySequence("Ctrl+3"), lambda: self.switch_tab(2), "切换到模型验证"),
            (QKeySequence("Ctrl+4"), lambda: self.switch_tab(3), "切换到模型推理"),
            (QKeySequence("Ctrl+5"), lambda: self.switch_tab(4), "切换到模型部署"),
            (QKeySequence("Ctrl+Tab"), lambda: self.switch_tab(self.tab_widget.currentIndex() + 1), "下一个标签页"),
            (QKeySequence("Ctrl+Shift+Tab"), lambda: self.switch_tab(self.tab_widget.currentIndex() - 1), "上一个标签页"),
            
            # 图像导航快捷键
            (QKeySequence("Ctrl+N"), self.next_image, "下一张图片"),
            (QKeySequence("Ctrl+P"), self.prev_image, "上一张图片"),
            (QKeySequence("Right"), self.next_image, "下一张图片"),
            (QKeySequence("Left"), self.prev_image, "上一张图片"),
            (QKeySequence("Space"), self.next_image, "下一张图片"),
            
            # 图像缩放快捷键
            (QKeySequence("Ctrl+Plus"), lambda: self.zoom_image(1.25), "放大图像"),
            (QKeySequence("Ctrl+Minus"), lambda: self.zoom_image(0.8), "缩小图像"),
            (QKeySequence("Ctrl+="), lambda: self.zoom_image(1.25), "放大图像"),
            (QKeySequence("Ctrl+_"), lambda: self.zoom_image(0.8), "缩小图像"),
            
            # 窗口快捷键
            (QKeySequence("F11"), self.toggle_fullscreen, "切换全屏"),
            (QKeySequence("Escape"), self.exit_fullscreen, "退出全屏"),
            
            # 菜单快捷键
            (QKeySequence("Ctrl+H"), self.show_help, "显示帮助"),
        ]

        self.shortcut_list = shortcuts
        for key, callback, _ in shortcuts:
            shortcut = QShortcut(key, self)
            shortcut.activated.connect(callback)

    def switch_tab(self, index: int):
        """切换标签页"""
        count = self.tab_widget.count()
        if index < 0:
            index = count - 1
        elif index >= count:
            index = 0
        self.tab_widget.setCurrentIndex(index)

    def toggle_fullscreen(self):
        """切换全屏"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def exit_fullscreen(self):
        """退出全屏"""
        if self.isFullScreen():
            self.showNormal()

    def setup_connections(self):
        """设置信号连接"""
        # 模型加载信号
        self.model_loaded.connect(self.on_model_loaded)

        # 状态更新信号 - 不再使用showMessage，避免覆盖固定状态
        # self.status_updated.connect(self.status_bar.showMessage)

    def start_timers(self):
        """启动定时器"""
        # 内存监控定时器
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_status_bar)
        self.memory_timer.start(AppConfig.MEMORY_CLEANUP_INTERVAL)

        # 时间更新定时器
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)  # 每秒更新

    def show_startup_info(self, device_info: dict):
        """显示启动信息（由后台线程探测完成后回调）"""
        self.logger.info("=" * 60)
        self.logger.info(f"{AppConfig.APP_NAME} 启动")
        self.logger.info(f"Python: {sys.version.split()[0]}")
        self.logger.info("设备信息:")
        self.logger.info(f"  CUDA可用: {device_info['cuda_available']}")
        if device_info['cuda_devices'] > 0:
            for i, name in enumerate(device_info.get('cuda_device_names', [])):
                self.logger.info(f"    GPU {i}: {name}")
        self.logger.info(f"  ONNX提供程序: {device_info['onnx_providers']}")
        self.logger.info("=" * 60)

        # 设备信息已就绪，立即刷新状态栏（此时不再触发 torch 导入）
        self.update_status_bar()
        self.status_updated.emit("就绪")

    def update_status_bar(self):
        """更新状态栏"""
        # 更新内存使用
        memory = self.device_manager.get_memory_usage()
        self.memory_status.setText(f"内存: {memory:.1f} MB")

        # 更新模型状态
        model_info = self.model_manager.get_model_info()
        if model_info['status'] == '已加载':
            task_name = model_info.get('task_name', model_info['task'])
            model_path = os.path.basename(model_info['path'])
            self.model_status.setText(f"模型: {task_name} | {model_path}")
        else:
            self.model_status.setText("模型: 未加载")

        # 更新设备状态
        self.device_status.setText(f"设备: {self.device_manager.get_safe_device()}")

    def update_time(self):
        """更新时间显示"""
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_status.setText(current_time)

    def on_tab_changed(self, index: int):
        """标签页切换回调"""
        # 懒加载：首次切换到某标签页时才创建其实例
        self._ensure_tab_created(index)

        tab_name = self.tab_widget.tabText(index)
        self.logger.info(f"切换到: {tab_name}")

        # 清空图像预览，避免残留之前标签页的图像
        self.shared_image_display.clear()

        # 更新当前标签页的模型信息
        current_tab = self.tab_widget.widget(index)
        if hasattr(current_tab, 'on_tab_selected'):
            current_tab.on_tab_selected()

    def on_model_loaded(self, model_info: dict):
        """模型加载回调"""
        self.logger.info(f"模型已加载: {model_info}")

        # 更新已创建标签页的模型状态
        for tab_name, tab in self.tabs.items():
            if hasattr(tab, 'on_model_loaded'):
                tab.on_model_loaded(model_info)

    def fit_image_to_view(self):
        """适应图像到视图"""
        self.shared_image_display.fit_to_view()

    def zoom_image(self, factor: float):
        """缩放图像"""
        self.shared_image_display.zoom_by_factor(factor)

    def next_image(self):
        """下一张图片（转发到当前标签页）"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'next_image'):
            current_tab.next_image()

    def prev_image(self):
        """上一张图片（转发到当前标签页）"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'prev_image'):
            current_tab.prev_image()

    def clear_cache(self):
        """清空缓存"""
        self.model_manager.clear_cache()
        self.device_manager.clear_memory()
        self.logger.info("缓存已清空")
        self.status_updated.emit("缓存已清空")

    def refresh_all(self):
        """刷新所有"""
        self.update_status_bar()
        self.status_updated.emit("已刷新")

    def update_max_log_lines(self, value: int):
        """更新最大日志行数"""
        self.logger.max_lines = value

        # 更新所有标签页的日志显示
        for tab in self.tabs.values():
            if hasattr(tab, 'log_display'):
                # 这里需要实现日志行数限制
                pass

    def show_help(self):
        """显示帮助"""
        device_info = self.device_manager.get_device_info()

        help_text = f"""
        <h2>{AppConfig.APP_NAME}</h2>
        
        <h3>快捷键</h3>
        <h4>通用快捷键</h4>
        <table border="0" cellpadding="3">
        <tr><td><b>Ctrl+0</b></td><td>适应图像到视图</td></tr>
        <tr><td><b>Ctrl+Q</b></td><td>退出程序</td></tr>
        <tr><td><b>F1</b></td><td>显示帮助</td></tr>
        <tr><td><b>Ctrl+H</b></td><td>显示帮助</td></tr>
        <tr><td><b>Ctrl+Shift+C</b></td><td>清空缓存</td></tr>
        <tr><td><b>Ctrl+R</b></td><td>刷新</td></tr>
        <tr><td><b>F11</b></td><td>切换全屏</td></tr>
        <tr><td><b>Esc</b></td><td>退出全屏</td></tr>
        </table>
        
        <h4>标签页切换</h4>
        <table border="0" cellpadding="3">
        <tr><td><b>Ctrl+1</b></td><td>切换到数据集管理</td></tr>
        <tr><td><b>Ctrl+2</b></td><td>切换到模型训练</td></tr>
        <tr><td><b>Ctrl+3</b></td><td>切换到模型验证</td></tr>
        <tr><td><b>Ctrl+4</b></td><td>切换到模型推理</td></tr>
        <tr><td><b>Ctrl+5</b></td><td>切换到模型部署</td></tr>
        <tr><td><b>Ctrl+Tab</b></td><td>下一个标签页</td></tr>
        <tr><td><b>Ctrl+Shift+Tab</b></td><td>上一个标签页</td></tr>
        </table>
        
        <h4>图像导航</h4>
        <table border="0" cellpadding="3">
        <tr><td><b>Ctrl+N</b></td><td>下一张图片</td></tr>
        <tr><td><b>Ctrl+P</b></td><td>上一张图片</td></tr>
        <tr><td><b>→</b></td><td>下一张图片</td></tr>
        <tr><td><b>←</b></td><td>上一张图片</td></tr>
        <tr><td><b>Space</b></td><td>下一张图片</td></tr>
        </table>
        
        <h4>图像缩放</h4>
        <table border="0" cellpadding="3">
        <tr><td><b>Ctrl++</b></td><td>放大图像</td></tr>
        <tr><td><b>Ctrl+=</b></td><td>放大图像</td></tr>
        <tr><td><b>Ctrl+-</b></td><td>缩小图像</td></tr>
        <tr><td><b>Ctrl+_</b></td><td>缩小图像</td></tr>
        </table>
        
        <h3>图像操作</h3>
        <ul>
        <li><b>Ctrl+滚轮</b> - 缩放图像</li>
        <li><b>鼠标拖拽</b> - 移动图像</li>
        </ul>
        
        <h3>标签页功能</h3>
        <ol>
        <li><b>数据集管理</b> - 管理和处理数据集，包括数据增强和格式转换</li>
        <li><b>模型训练</b> - 训练检测、分割、姿态和分类模型</li>
        <li><b>模型验证</b> - 验证模型性能</li>
        <li><b>模型推理</b> - 使用检测、分割、姿态和分类模型进行推理</li>
        <li><b>模型部署</b> - 将模型转换为ONNX格式并部署到边缘设备</li>
        </ol>
        
        <h3>设备信息</h3>
        <ul>
        <li>CUDA可用: {device_info['cuda_available']}</li>
        """

        if device_info['cuda_available']:
            help_text += f"<li>CUDA设备数: {device_info['cuda_devices']}</li>"
            for i, name in enumerate(device_info.get('cuda_device_names', [])):
                help_text += f"<li>  GPU {i}: {name}</li>"

        help_text += f"<li>ONNX提供程序: {device_info['onnx_providers']}</li>"
        help_text += """
        </ul>
        
        <h3>性能优化</h3>
        <ul>
        <li>支持多GPU自动选择</li>
        <li>智能模型缓存管理</li>
        <li>批处理和并行推理</li>
        <li>智能内存管理和优化</li>
        <li>后台任务处理，保持UI响应</li>
        </ul>
        """

        QMessageBox.information(self, "帮助", help_text)

    def show_about(self):
        """显示关于信息"""
        from config.settings import AppConfig
        
        about_text = f"""
        <h2>{AppConfig.APP_NAME}</h2>
        <p>YOLO 多版本训练与部署工具</p>
        <p>支持目标检测、实例分割、语义分割、深度估计、图像分类、姿态估计、旋转框检测等任务</p>
        <p>© 2024 YOLO UI Tool</p>
        """
        
        QMessageBox.information(self, "关于", about_text)

    def closeEvent(self, event):
        """关闭事件"""
        # 确认退出
        reply = QMessageBox.question(
            self, "确认退出",
            "确定要退出程序吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 停止所有训练
            if hasattr(self.trainer, 'stop_training'):
                self.trainer.stop_training()

            # 卸载模型
            self.model_manager.unload_model()

            # 恢复电源设置
            self.power_manager.set_sleep_prevention(False)

            # 清理内存
            self.device_manager.clear_memory()

            # 等待设备探测线程结束，避免线程未退出导致崩溃
            if self.device_info_worker.isRunning():
                self.device_info_worker.wait(3000)

            self.logger.info("程序正常退出")
            event.accept()
        else:
            event.ignore()
