"""matplotlib 绘图样式

延迟导入并统一配置中文字体，避免在程序启动阶段加载 matplotlib 拖慢启动。
所有需要创建图表的模块都应先调用 setup_matplotlib()，再导入 Figure / FigureCanvas。
"""

_configured = False


def setup_matplotlib():
    """按需配置 matplotlib 中文字体，返回 matplotlib 模块"""
    global _configured
    import matplotlib

    if not _configured:
        matplotlib.use('Qt5Agg')
        matplotlib.rcParams['font.sans-serif'] = [
            'SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'DejaVu Sans'
        ]
        matplotlib.rcParams['axes.unicode_minus'] = False
        _configured = True

    return matplotlib
