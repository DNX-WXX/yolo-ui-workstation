"""
日志工具 - 分级日志管理
"""
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QTextEdit
from typing import List, Callable
from enum import Enum, auto
from datetime import datetime


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()

    @property
    def color(self) -> str:
        """获取级别对应的颜色"""
        colors = {
            LogLevel.DEBUG: "#808080",
            LogLevel.INFO: "#000000",
            LogLevel.WARNING: "#FFA500",
            LogLevel.ERROR: "#FF0000"
        }
        return colors.get(self, "#000000")


class LogEntry:
    """日志条目"""

    def __init__(self, level: LogLevel, message: str):
        self.level = level
        self.message = message
        self.timestamp = datetime.now()
        self.formatted = f"[{self.timestamp.strftime('%H:%M:%S')}] [{level.name}] {message}"

    def __str__(self):
        return self.formatted


class LogManager(QObject):
    """日志管理器"""

    log_signal = pyqtSignal(LogLevel, str)  # level, message

    def __init__(self, max_lines: int = 1000, level: LogLevel = LogLevel.INFO):
        """
        初始化日志管理器

        Args:
            max_lines: 最大保存行数
            level: 日志级别
        """
        super().__init__()
        self.max_lines = max_lines
        self.level = level
        self.handlers: List[Callable] = []
        self.entries: List[LogEntry] = []

    def add_handler(self, handler: Callable):
        """添加日志处理器"""
        self.handlers.append(handler)

    def log(self, level: LogLevel, message: str):
        """
        记录日志

        Args:
            level: 日志级别
            message: 日志消息
        """
        if level.value < self.level.value:
            return

        # 创建日志条目
        entry = LogEntry(level, message)

        # 存储日志
        self.entries.append(entry)
        if len(self.entries) > self.max_lines:
            self.entries.pop(0)

        # 发送信号
        self.log_signal.emit(level, message)

        # 调用处理器
        for handler in self.handlers:
            try:
                handler(level, entry.formatted)
            except Exception as e:
                print(f"日志处理器错误: {e}")


    def info(self, message: str):
        """记录信息日志"""
        self.log(LogLevel.INFO, message)

    def warning(self, message: str):
        """记录警告日志"""
        self.log(LogLevel.WARNING, message)

    def error(self, message: str):
        """记录错误日志"""
        self.log(LogLevel.ERROR, message)


    def clear(self):
        """清空日志"""
        self.entries.clear()


class QTextEditLogger:
    """QTextEdit日志处理器"""

    def __init__(self, text_edit: QTextEdit, max_lines: int = 1000):
        """
        初始化

        Args:
            text_edit: QTextEdit控件
            max_lines: 最大显示行数
        """
        self.text_edit = text_edit
        self.max_lines = max_lines

    def __call__(self, level: LogLevel, formatted: str):
        """
        处理日志

        Args:
            level: 日志级别
            formatted: 格式化后的日志消息
        """
        # 保存当前光标位置
        cursor = self.text_edit.textCursor()
        scrollbar = self.text_edit.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 5

        # 设置颜色
        self.text_edit.setTextColor(QColor(level.color))

        # 添加文本
        self.text_edit.append(formatted)
        
        # 限制行数
        doc = self.text_edit.document()
        if doc.blockCount() > self.max_lines:
            cursor = self.text_edit.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

        # 自动滚动到底部
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
