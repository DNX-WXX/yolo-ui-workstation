"""
支持拖拽的输入框组件
"""
import os
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import pyqtSignal


class DropLineEdit(QLineEdit):
    """支持拖拽的输入框组件"""

    # 文件拖拽放置完成后发射，参数为文件路径
    pathDropped = pyqtSignal(str)

    def __init__(self, parent=None, file_filter=None):
        super().__init__(parent)
        self.file_filter = file_filter  # 可选的文件过滤器: folder, file, image, model
        self.setAcceptDrops(True)
        self._setup_styles()

    def _setup_styles(self):
        """设置默认样式"""
        self.setStyleSheet("""
            QLineEdit {
                border: 1px solid #cccccc;
                padding: 3px 5px;
                border-radius: 3px;
                background-color: white;
            }
            QLineEdit:hover, QLineEdit:focus {
                border-color: #0078d4;
            }
        """)

    def _get_drop_highlight_style(self):
        """获取拖拽高亮样式"""
        return """
            QLineEdit {
                border: 2px dashed #0078d4;
                padding: 3px 5px;
                border-radius: 3px;
                background-color: #e6f4ff;
            }
        """

    def _is_valid_file(self, file_path):
        """检查文件是否符合过滤器要求"""
        if not self.file_filter:
            return True

        if self.file_filter == 'folder':
            return os.path.isdir(file_path)
        elif self.file_filter == 'file':
            return os.path.isfile(file_path)
        elif self.file_filter == 'image':
            ext = os.path.splitext(file_path)[1].lower()
            return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        elif self.file_filter == 'model':
            ext = os.path.splitext(file_path)[1].lower()
            return ext in ['.pt', '.onnx', '.pth']
        elif self.file_filter == 'yaml':
            ext = os.path.splitext(file_path)[1].lower()
            return ext in ['.yaml', '.yml']
        return True

    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._get_drop_highlight_style())

    def dragLeaveEvent(self, event):
        """拖拽离开事件"""
        self._setup_styles()

    def dropEvent(self, event):
        """放置事件"""
        self._setup_styles()

        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if self._is_valid_file(file_path):
                    self.setText(file_path)
                    self.pathDropped.emit(file_path)
                    break
            event.acceptProposedAction()
