"""
错误处理工具 - 统一错误处理和异常捕获
"""
import traceback
import sys
from typing import Callable, Any
from functools import wraps
from PyQt5.QtWidgets import QMessageBox


class ErrorHandler:
    """统一错误处理器"""

    # 错误类型映射
    ERROR_MESSAGES = {
        "cuda out of memory": (
            "CUDA内存不足\n"
            "建议:\n"
            "• 减小batch size\n"
            "• 使用更小的模型\n"
            "• 切换到CPU模式"
        ),
        "connection": (
            "网络连接失败\n"
            "建议:\n"
            "• 检查网络连接\n"
            "• 手动下载模型文件"
        ),
        "download": (
            "模型下载失败\n"
            "建议:\n"
            "• 检查网络连接\n"
            "• 手动下载模型后使用本地加载"
        ),
        "file not found": (
            "文件不存在\n"
            "建议:\n"
            "• 检查文件路径\n"
            "• 确认文件是否被移动"
        ),
        "permission": (
            "权限不足\n"
            "建议:\n"
            "• 以管理员身份运行\n"
            "• 检查文件权限"
        ),
        "invalid model": (
            "无效的模型文件\n"
            "建议:\n"
            "• 确认文件是YOLO模型\n"
            "• 检查模型文件是否完整"
        ),
        "shape mismatch": (
            "张量形状不匹配\n"
            "建议:\n"
            "• 检查输入图像尺寸\n"
            "• 确认模型任务类型"
        )
    }

    @classmethod
    def get_user_message(cls, error: Exception) -> str:
        """
        获取用户友好的错误消息

        Args:
            error: 异常对象

        Returns:
            用户友好的错误消息
        """
        error_str = str(error).lower()

        # 检查已知错误类型
        for key, message in cls.ERROR_MESSAGES.items():
            if key in error_str:
                return message

        # 默认错误消息
        return f"发生错误: {str(error)}"

    @classmethod
    def get_detailed_message(cls, error: Exception) -> str:
        """
        获取详细错误信息（用于调试）

        Args:
            error: 异常对象

        Returns:
            详细错误信息
        """
        return ''.join(traceback.format_exception(
            type(error), error, error.__traceback__
        ))

    @classmethod
    def show_error_dialog(cls, parent, error: Exception,
                          title: str = "错误",
                          show_details: bool = True):
        """
        显示错误对话框

        Args:
            parent: 父窗口
            error: 异常对象
            title: 对话框标题
            show_details: 是否显示详细信息
        """
        msg_box = QMessageBox(parent)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(cls.get_user_message(error))

        if show_details:
            msg_box.setDetailedText(cls.get_detailed_message(error))

        msg_box.exec_()


def safe_execute(error_message: str = "操作失败",
                 show_dialog: bool = True,
                 default_return: Any = None):
    """
    安全执行函数的装饰器

    Args:
        error_message: 错误消息前缀
        show_dialog: 是否显示错误对话框
        default_return: 出错时的默认返回值

    Example:
        @safe_execute("加载模型失败")
        def load_model(path):
            return YOLO(path)
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取调用者（用于显示对话框）
                parent = None
                for arg in args:
                    if hasattr(arg, 'parent') and callable(getattr(arg, 'parent')):
                        parent = arg
                        break

                # 记录错误
                print(f"{error_message}: {e}")
                print(traceback.format_exc())

                # 显示错误对话框
                if show_dialog and parent and hasattr(parent, 'parent'):
                    ErrorHandler.show_error_dialog(parent, e, error_message)

                return default_return

        return wrapper

    return decorator


def setup_global_exception_handler():
    """设置全局异常处理"""

    def exception_hook(exctype, value, tb):
        """全局异常钩子"""
        error_msg = ''.join(traceback.format_exception(exctype, value, tb))

        # 打印到控制台
        print("未捕获的异常:")
        print(error_msg)

        # 显示错误对话框
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("程序发生未捕获的异常")
        msg_box.setText(str(value))
        msg_box.setDetailedText(error_msg)
        msg_box.exec_()

        # 调用默认的异常处理
        sys.__excepthook__(exctype, value, tb)

    sys.excepthook = exception_hook