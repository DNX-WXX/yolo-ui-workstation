#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO 多版本训练与部署工具 - 主程序入口
支持 YOLOv5、YOLOv8、YOLO11、YOLO26 等多版本模型，
涵盖目标检测、实例分割、图像分类、姿态估计等任务
"""

import sys
import os
import argparse
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# PyQt5 imports
from PyQt5.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont, QIcon

# 注意：主窗口采用延迟导入（在 show_splash_screen 之后），
# 避免 torch/cv2/onnxruntime 等重型库在启动画面出现前加载导致黑屏等待
# from ui.main_window import MainWindow  ← 移到 create_main_window 内

# 导入配置
from config.settings import AppConfig


class Application:
    """应用程序类 - 管理应用生命周期"""

    def __init__(self):
        self.app = None
        self.main_window = None
        self.splash = None

    def parse_arguments(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description='YOLO 多版本训练与部署工具')
        parser.add_argument('--no-splash', action='store_true', help='禁用启动画面')
        parser.add_argument('--model', type=str, help='启动时自动加载的模型路径')
        parser.add_argument('--image', type=str, help='启动时自动加载的图片路径')
        parser.add_argument('--version', action='version',
                            version=AppConfig.APP_NAME)

        return parser.parse_args()

    def setup_high_dpi(self):
        """设置高DPI支持"""
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    def setup_app_info(self):
        """设置应用程序信息"""
        self.app.setApplicationName(AppConfig.APP_NAME)
        self.app.setApplicationVersion(AppConfig.APP_VERSION)
        self.app.setOrganizationName(AppConfig.APP_AUTHOR)

        # 设置应用程序图标（如果有）
        icon_path = project_root / "resources" / "icon.png"
        if icon_path.exists():
            self.app.setWindowIcon(QIcon(str(icon_path)))

    def show_splash_screen(self):
        """显示启动画面"""
        splash_path = project_root / "resources" / "zyg.jpg"
        if splash_path.exists():
            splash_pixmap = QPixmap(str(splash_path))
            self.splash = QSplashScreen(splash_pixmap)
        else:
            # 创建默认启动画面
            splash_pixmap = QPixmap(600, 300)
            splash_pixmap.fill(Qt.white)
            self.splash = QSplashScreen(splash_pixmap)

            # 在启动画面上绘制文字
            from PyQt5.QtGui import QPainter, QPen
            painter = QPainter(splash_pixmap)
            painter.setPen(QPen(Qt.black))
            painter.setFont(QFont("Arial", 16, QFont.Bold))
            painter.drawText(splash_pixmap.rect(), Qt.AlignCenter,
                             f"{AppConfig.APP_NAME}")
            painter.end()
            self.splash.setPixmap(splash_pixmap)

        self.splash.show()
        self.app.processEvents()

        # 显示加载信息
        self.splash.showMessage("正在初始化...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
        self.app.processEvents()

    def setup_exception_handler(self):
        """设置全局异常处理"""

        def exception_hook(exctype, value, tb):
            """全局异常钩子"""
            error_msg = ''.join(traceback.format_exception(exctype, value, tb))

            # 记录到文件
            log_path = project_root / "error.log"
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"时间: {__import__('datetime').datetime.now()}\n")
                f.write(error_msg)
                f.write(f"\n{'=' * 60}\n")

            # 显示错误对话框
            if self.main_window and self.main_window.isVisible():
                msg_box = QMessageBox(self.main_window)
            else:
                msg_box = QMessageBox()

            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("程序发生未捕获的异常")
            msg_box.setText(str(value))
            msg_box.setDetailedText(error_msg)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()

            # 调用默认的异常处理
            sys.__excepthook__(exctype, value, tb)

        sys.excepthook = exception_hook

    def check_environment(self):
        """检查运行环境"""
        issues = []

        # 检查Python版本
        if sys.version_info < (3, 8):
            issues.append(f"Python版本过低: {sys.version}, 建议使用Python 3.8+")

        # 检查必要模块
        required_modules = [
            ('torch', 'PyTorch'),
            ('ultralytics', 'Ultralytics YOLO'),
            ('cv2', 'OpenCV'),
            ('PIL', 'Pillow'),
            ('numpy', 'NumPy'),
            ('onnxruntime', 'ONNX Runtime')
        ]

        for module_name, display_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                issues.append(f"缺少必要模块: {display_name}")

        return issues

    def create_output_directories(self):
        """创建必要的输出目录"""
        dirs = [
            project_root / "runs",
            Path.home() / ".yolo_ui_workstation"
        ]

        for dir_path in dirs:
            dir_path.mkdir(exist_ok=True, parents=True)

    def initialize(self):
        """初始化应用程序"""
        # 创建输出目录
        self.create_output_directories()

        # 检查环境
        issues = self.check_environment()
        if issues:
            print("环境检查发现以下问题:")
            for issue in issues:
                print(f"  - {issue}")

        # 更新启动画面
        if self.splash:
            self.splash.showMessage("正在加载主窗口...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
            self.app.processEvents()

    def create_main_window(self, args):
        """创建主窗口"""
        # 延迟导入主窗口及其依赖（cv2/onnxruntime 等重型库在此处加载）
        from ui.main_window import MainWindow

        self.main_window = MainWindow()

        # 设置窗口属性
        self.main_window.setWindowTitle(AppConfig.APP_NAME)

        # 自动加载模型
        if args.model:
            QTimer.singleShot(500, lambda: self.load_model(args.model))

        # 自动加载图片
        if args.image:
            QTimer.singleShot(1000, lambda: self.load_image(args.image))

        return self.main_window

    def _inference_tab(self):
        """获取模型推理标签页，并切换到该页"""
        if not self.main_window:
            return None
        self.main_window.tab_widget.setCurrentIndex(3)  # 推理标签页
        return self.main_window.tabs['combined_inference']

    def load_model(self, model_path: str):
        """自动加载模型"""
        if os.path.exists(model_path):
            tab = self._inference_tab()
            if tab:
                tab.model_path_edit.setText(model_path)
                tab.load_model(model_path)

    def load_image(self, image_path: str):
        """自动加载图片"""
        if os.path.exists(image_path):
            tab = self._inference_tab()
            if tab:
                tab.image_path.setText(image_path)

    def run(self):
        """运行应用程序"""
        # 解析命令行参数
        args = self.parse_arguments()

        # 创建QApplication
        self.setup_high_dpi()
        self.app = QApplication(sys.argv)
        self.setup_app_info()

        # 显示启动画面（除非禁用）
        if not args.no_splash:
            self.show_splash_screen()

        # 设置异常处理
        self.setup_exception_handler()

        # 初始化
        self.initialize()

        # 创建主窗口
        main_window = self.create_main_window(args)

        # 关闭启动画面并显示主窗口
        if self.splash:
            self.splash.finish(main_window)

        main_window.show()

        # 检查环境问题并显示警告
        issues = self.check_environment()
        if issues and not args.no_splash:
            QTimer.singleShot(1000, lambda: self.show_environment_warning(issues))

        # 进入事件循环
        return self.app.exec_()

    def show_environment_warning(self, issues):
        """显示环境警告"""
        if not self.main_window:
            return

        msg = "环境检查发现以下问题:\n\n"
        msg += "\n".join(f"• {issue}" for issue in issues)
        msg += "\n\n程序可能无法正常运行，建议安装必要的依赖。"

        QMessageBox.warning(self.main_window, "环境警告", msg)

    def cleanup(self):
        """清理资源"""
        print("程序退出，资源已清理")


def main():
    """主函数"""
    # 创建应用实例
    app = Application()

    # 运行应用
    exit_code = 0
    try:
        exit_code = app.run()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序运行出错: {e}")
        traceback.print_exc()
        exit_code = 1
    finally:
        # 清理资源
        app.cleanup()

    # 退出
    sys.exit(exit_code)


def setup_console():
    """设置控制台（Windows下隐藏控制台窗口）"""
    if sys.platform == 'win32':
        import ctypes
        # 获取控制台窗口句柄并隐藏
        try:
            ctypes.windll.user32.ShowWindow(
                ctypes.windll.kernel32.GetConsoleWindow(), 0)
        except:
            pass


if __name__ == '__main__':
    # 如果是打包后的可执行文件，隐藏控制台
    if getattr(sys, 'frozen', False):
        setup_console()

    # 运行主程序
    main()