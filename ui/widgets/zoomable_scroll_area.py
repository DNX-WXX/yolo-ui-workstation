"""
可缩放滚动区域组件 - 支持缩放和平移的图像显示区域
"""
from PyQt5.QtWidgets import QScrollArea, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QWheelEvent, QMouseEvent


class ZoomableLabel(QLabel):
    """可缩放的标签"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(100, 100)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self._is_valid = True

    def mark_invalid(self):
        """标记为无效"""
        self._is_valid = False

    def is_valid(self) -> bool:
        """检查是否有效"""
        return self._is_valid and not (self is None)

    def __del__(self):
        """析构时标记为无效"""
        self._is_valid = False


class ZoomableScrollArea(QScrollArea):
    """支持缩放和拖拽的滚动区域"""

    # 信号定义
    zoom_request = pyqtSignal(float)  # 缩放请求
    zoom_changed = pyqtSignal(float)  # 缩放比例改变
    image_clicked = pyqtSignal(QPoint)  # 图像点击
    image_double_clicked = pyqtSignal(QPoint)  # 图像双击

    def __init__(self, parent=None):
        """
        初始化可缩放滚动区域

        Args:
            parent: 父窗口
        """
        super().__init__(parent)

        # 创建内部标签
        self.image_label = ZoomableLabel()
        self.setWidget(self.image_label)

        # 设置滚动区域属性
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.OpenHandCursor)

        # 设置样式
        self.setStyleSheet("""
            QScrollArea {
                border: 1px solid #cccccc;
                background-color: #f0f0f0;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #f0f0f0;
            }
        """)

        # 缩放和平移状态
        self._is_dragging = False
        self._last_drag_position = QPoint()
        self._zoom_scale = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        self._zoom_step = 1.2
        self._original_pixmap = None
        self._fit_to_view_enabled = True
        self._pending_update = False
        self._is_destroyed = False  # 标记是否已销毁

        # 网格线设置
        self._show_grid = False
        self._grid_size = 50
        self._grid_color = QColor(200, 200, 200)

        # 延迟更新定时器
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._delayed_update)

    def destroy(self, destroyWindow=True, destroySubWindows=True):
        """重写destroy方法"""
        self._is_destroyed = True
        if self.image_label:
            self.image_label.mark_invalid()
        super().destroy(destroyWindow, destroySubWindows)

    def _is_valid(self) -> bool:
        """检查对象是否有效"""
        return not self._is_destroyed and self.image_label is not None

    # ========== 图像操作 ==========

    def set_pixmap(self, pixmap: QPixmap):
        """
        设置显示的图片

        Args:
            pixmap: 要显示的QPixmap
        """
        # 检查对象是否有效
        if self._is_destroyed:
            return

        if pixmap is None or pixmap.isNull():
            self._original_pixmap = None
            if self._is_valid() and self.image_label.is_valid():
                self.image_label.clear()
                self.image_label.setText("无图像")
            return

        self._original_pixmap = pixmap

        # 检查标签是否有效
        if not self._is_valid() or not self.image_label.is_valid():
            return

        if self._fit_to_view_enabled:
            self.fit_to_view()
        else:
            self._update_display()


    def clear(self):
        """清空显示"""
        self._original_pixmap = None
        if self._is_valid() and self.image_label.is_valid():
            self.image_label.clear()
            self.image_label.setText("无图像")
        self._zoom_scale = 1.0

    # ========== 缩放控制 ==========

    def zoom_in(self):
        """放大"""
        self.zoom_by_factor(self._zoom_step)

    def zoom_out(self):
        """缩小"""
        self.zoom_by_factor(1.0 / self._zoom_step)

    def zoom_by_factor(self, factor: float):
        """
        按比例缩放

        Args:
            factor: 缩放因子 (>1 放大, <1 缩小)
        """
        if self._original_pixmap and not self._original_pixmap.isNull():
            new_scale = self._zoom_scale * factor
            self.set_zoom_level(new_scale)

    def set_zoom_level(self, level: float):
        """
        设置缩放级别

        Args:
            level: 缩放比例
        """
        if level < self._min_zoom:
            level = self._min_zoom
        if level > self._max_zoom:
            level = self._max_zoom

        if abs(level - self._zoom_scale) > 0.01:
            self._zoom_scale = level
            self._fit_to_view_enabled = False
            self._schedule_update()
            self.zoom_changed.emit(self._zoom_scale)


    def fit_to_view(self):
        """适应视图大小"""
        if not self._is_valid():
            return

        if not self._original_pixmap or self._original_pixmap.isNull():
            return

        view_size = self.viewport().size()
        pixmap_size = self._original_pixmap.size()

        if view_size.width() <= 0 or view_size.height() <= 0:
            return
        if pixmap_size.width() <= 0 or pixmap_size.height() <= 0:
            return

        # 计算适合视图的缩放比例
        w_ratio = view_size.width() / pixmap_size.width()
        h_ratio = view_size.height() / pixmap_size.height()
        self._zoom_scale = min(w_ratio, h_ratio)

        self._fit_to_view_enabled = True
        self._schedule_update()
        self.zoom_changed.emit(self._zoom_scale)


    def _schedule_update(self):
        """安排延迟更新"""
        if not self._is_valid():
            return

        if not self._pending_update:
            self._pending_update = True
            self._update_timer.start(50)  # 50ms延迟

    def _delayed_update(self):
        """延迟更新"""
        self._pending_update = False
        self._update_display()

    def _update_display(self):
        """更新显示"""
        if not self._is_valid():
            return

        if not self._original_pixmap or self._original_pixmap.isNull():
            return

        # 检查标签是否有效
        if not self.image_label or not self.image_label.is_valid():
            return

        try:
            # 计算新尺寸
            new_width = int(self._original_pixmap.width() * self._zoom_scale)
            new_height = int(self._original_pixmap.height() * self._zoom_scale)

            # 缩放图像
            scaled_pixmap = self._original_pixmap.scaled(
                new_width, new_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # 如果需要显示网格，在图像上绘制
            if self._show_grid:
                scaled_pixmap = self._draw_grid(scaled_pixmap)

            # 更新标签
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(new_width, new_height)

        except RuntimeError:
            # 对象已被删除，忽略错误
            pass

    # ========== 网格显示 ==========


    def _draw_grid(self, pixmap: QPixmap) -> QPixmap:
        """
        在图像上绘制网格

        Args:
            pixmap: 原始图像

        Returns:
            带网格的图像
        """
        if pixmap.isNull():
            return pixmap

        result = QPixmap(pixmap)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)

        # 设置画笔
        pen = QPen(self._grid_color, 1, Qt.DashLine)
        painter.setPen(pen)

        width = result.width()
        height = result.height()

        # 绘制垂直线
        for x in range(self._grid_size, width, self._grid_size):
            painter.drawLine(x, 0, x, height)

        # 绘制水平线
        for y in range(self._grid_size, height, self._grid_size):
            painter.drawLine(0, y, width, y)

        painter.end()
        return result

    # ========== 事件处理 ==========

    def wheelEvent(self, event: QWheelEvent):
        """
        滚轮事件 - 支持缩放

        Args:
            event: 滚轮事件
        """
        if not self._is_valid():
            return

        if event.modifiers() == Qt.ControlModifier:
            # Ctrl+滚轮缩放
            angle = event.angleDelta().y()
            if angle > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            # 普通滚轮滚动
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        """
        鼠标按下事件 - 开始拖拽

        Args:
            event: 鼠标事件
        """
        if not self._is_valid():
            return

        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._last_drag_position = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

            # 触发点击信号
            if self._original_pixmap:
                image_pos = self._viewport_to_image(event.pos())
                self.image_clicked.emit(image_pos)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        鼠标移动事件 - 拖拽图像

        Args:
            event: 鼠标事件
        """
        if not self._is_valid():
            return

        if self._is_dragging and event.buttons() & Qt.LeftButton:
            # 计算移动距离
            delta = event.pos() - self._last_drag_position

            # 移动滚动条
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )

            self._last_drag_position = event.pos()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        鼠标释放事件 - 结束拖拽

        Args:
            event: 鼠标事件
        """
        if not self._is_valid():
            return

        if event.button() == Qt.LeftButton:
            self._is_dragging = False
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """
        鼠标双击事件

        Args:
            event: 鼠标事件
        """
        if not self._is_valid():
            return

        if event.button() == Qt.LeftButton and self._original_pixmap:
            image_pos = self._viewport_to_image(event.pos())
            self.image_double_clicked.emit(image_pos)

            # 双击切换适应视图
            if self._fit_to_view_enabled:
                self._fit_to_view_enabled = False
                self.set_zoom_level(self._zoom_scale * 2)
            else:
                self.fit_to_view()

            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        """
        调整大小事件

        Args:
            event: 调整大小事件
        """
        if not self._is_valid():
            return

        super().resizeEvent(event)

        # 如果启用了适应视图，自动调整
        if self._fit_to_view_enabled and self._original_pixmap:
            self._schedule_update()

    def _viewport_to_image(self, viewport_pos: QPoint) -> QPoint:
        """
        将视口坐标转换为图像坐标

        Args:
            viewport_pos: 视口坐标

        Returns:
            图像坐标
        """
        if not self._original_pixmap:
            return QPoint(-1, -1)

        # 获取滚动条位置
        h_scroll = self.horizontalScrollBar().value()
        v_scroll = self.verticalScrollBar().value()

        # 计算图像在视口中的位置
        image_x = viewport_pos.x() + h_scroll
        image_y = viewport_pos.y() + v_scroll

        # 转换为原始图像坐标
        if self._zoom_scale > 0:
            orig_x = int(image_x / self._zoom_scale)
            orig_y = int(image_y / self._zoom_scale)
        else:
            orig_x = -1
            orig_y = -1

        return QPoint(orig_x, orig_y)

    # ========== 属性 ==========


    def set_fit_to_view_enabled(self, enabled: bool):
        """设置是否在 set_pixmap 时自动适应视图（视频模式应禁用）"""
        self._fit_to_view_enabled = enabled

    # ========== 截图功能 ==========


    # ========== 工具函数 ==========
