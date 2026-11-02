"""
用户界面包 - 主窗口和标签页
"""
from .main_window import MainWindow

# 尝试导入标签页，如果失败则提供空导入
try:
    from .tabs.validate_tab import ValidateTab
except ImportError:
    ValidateTab = None

# 尝试导入控件
try:
    from .widgets.zoomable_scroll_area import ZoomableScrollArea
except ImportError:
    ZoomableScrollArea = None

__all__ = [
    # 主窗口
    'MainWindow',

    # 标签页
    'ValidateTab',

    # 控件
    'ZoomableScrollArea',
]

__version__ = '1.0.0'
