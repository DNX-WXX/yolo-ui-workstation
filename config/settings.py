"""
配置文件 - 存储所有全局配置
"""
from PyQt5.QtGui import QColor
from pathlib import Path
from typing import Dict, List, Optional


class AppConfig:
    """应用全局配置"""

    # ==================== 应用信息 ====================
    APP_NAME = "YOLO 通用工具"
    APP_VERSION = "3.0.0"
    APP_AUTHOR = "JZBH"

    # ==================== 路径配置 ====================
    BASE_DIR = Path(__file__).parent.parent
    DEFAULT_OUTPUT_DIR = BASE_DIR / "runs"

    # ==================== 任务定义 ====================
    SUPPORTED_TASKS = {
        'detect': '目标检测',
        'segment': '实例分割',
        'semantic': '语义分割',
        'depth': '深度估计',
        'classify': '图像分类',
        'pose': '姿态估计',
        'obb': '旋转框检测'
    }

    # 数据集转换支持的任务类型
    # 目标检测: VOC + COCO
    # 实例分割: COCO (use_segments=True)
    # 姿态估计: COCO (use_keypoints=True)
    # 图像分类: 仅需要目录划分
    DATASET_SUPPORTED_TASKS = {
        'detect': '目标检测',
        'segment': '实例分割',
        'pose': '姿态估计',
        'classify': '图像分类'
    }

    # 数据集转换不支持的任务类型及原因
    DATASET_UNSUPPORTED_TASKS = {
        'obb': '旋转框检测需要旋转框标注格式，数据集转换暂不支持',
        'semantic': '语义分割需要像素级mask标注格式，数据集转换暂不支持',
        'depth': '深度估计需要深度图标注格式，数据集转换暂不支持'
    }

    # 任务类型与模型后缀的映射
    TASK_SUFFIX = {
        'detect': '',        # 无后缀
        'segment': '-seg',   # 实例分割
        'semantic': '-sem',  # 语义分割 (YOLO26新增独立后缀)
        'depth': '-depth',   # 深度估计 (官方后缀为-depth)
        'classify': '-cls',
        'pose': '-pose',
        'obb': '-obb'
    }

    # ==================== 模型版本支持矩阵 ====================
    # 各YOLO版本支持的任务列表
    VERSION_TASK_MATRIX = {
        'yolov5': ['detect', 'classify'],                             # YOLOv5
        'yolov8': ['detect', 'segment', 'classify', 'pose', 'obb'],  # YOLOv8
        'yolo11': ['detect', 'segment', 'classify', 'pose', 'obb'],  # YOLO11
        'yolo26': ['detect', 'segment', 'semantic', 'depth', 'classify', 'pose', 'obb'],  # YOLO26 全任务
    }

    # ==================== 预训练模型配置 ====================
    # 按版本和任务组织的模型列表
    MODELS_BY_VERSION = {
        'yolov5': {
            'sizes': ['n', 's', 'm', 'l', 'x'],
            'extra': ['n6', 's6', 'm6', 'l6', 'x6'],  # P6模型
        },
        'yolov8': {'sizes': ['n', 's', 'm', 'l', 'x']},
        'yolo11': {'sizes': ['n', 's', 'm', 'l', 'x']},
        'yolo26': {'sizes': ['n', 's', 'm', 'l', 'x']},
    }

    # 数据集特定模型（如ADE20K预训练）
    DATASET_SPECIFIC_MODELS = {
        'semantic': {
            'ade20k': ['yolo26n-sem-ade20k', 'yolo26s-sem-ade20k',
                       'yolo26m-sem-ade20k', 'yolo26l-sem-ade20k',
                       'yolo26x-sem-ade20k'],
            'cityscapes': ['yolo26n-sem', 'yolo26s-sem', 'yolo26m-sem',
                           'yolo26l-sem', 'yolo26x-sem']
        }
    }

    # 特殊模型（预留扩展位）
    SPECIAL_MODELS: Dict[str, List[str]] = {}

    @classmethod
    def _build_model_list(cls, version: str, task: str) -> List[str]:
        """
        根据版本和任务生成模型列表

        Args:
            version: 版本名称 (yolo5, yolov8, yolo11, yolo26等)
            task: 任务类型 (detect, segment, classify等)

        Returns:
            模型名称列表
        """
        # 检查任务是否被该版本支持
        if task not in cls.VERSION_TASK_MATRIX.get(version, []):
            return []

        # 获取该版本的尺寸列表
        version_config = cls.MODELS_BY_VERSION.get(version, {})
        sizes = version_config.get('sizes', [])

        # 构建基础模型名
        suffix = cls.TASK_SUFFIX.get(task, '')
        base_models = [f'{version}{size}{suffix}' for size in sizes]

        # 特殊处理：YOLOv5额外添加P6模型（仅检测和分类）
        if version == 'yolov5' and task in ['detect', 'classify']:
            extra_sizes = version_config.get('extra', [])
            extra_models = [f'{version}{size}{suffix}' for size in extra_sizes]
            base_models.extend(extra_models)

        return base_models

    # 预训练模型完整映射（由构建函数生成）
    PRETRAINED_MODELS: Dict[str, List[str]] = {}

    @classmethod
    def _init_pretrained_models(cls):
        """初始化预训练模型映射"""
        versions = list(cls.MODELS_BY_VERSION.keys())
        tasks = list(cls.SUPPORTED_TASKS.keys())

        for task in tasks:
            models = []
            for version in versions:
                models.extend(cls._build_model_list(version, task))

            # 添加特殊模型（仅检测任务）
            if task == 'detect':
                models.extend(cls.SPECIAL_MODELS.get('detect', []))

            # 添加数据集特定模型（语义分割）
            if task == 'semantic':
                for dataset_models in cls.DATASET_SPECIFIC_MODELS.get('semantic', {}).values():
                    models.extend(dataset_models)

            # 去重并保持顺序
            cls.PRETRAINED_MODELS[task] = list(dict.fromkeys(models))

    @classmethod
    def get_pretrained_models(cls, task: str, dataset: Optional[str] = None) -> List[str]:
        """
        获取指定任务的预训练模型列表

        Args:
            task: 任务类型
            dataset: 可选，数据集名称 (如 'cityscapes', 'ade20k')

        Returns:
            模型列表
        """
        if task == 'semantic' and dataset:
            return cls.DATASET_SPECIFIC_MODELS.get('semantic', {}).get(dataset, [])
        return cls.PRETRAINED_MODELS.get(task, [])

    @classmethod
    def get_task_name(cls, task: str) -> str:
        """获取任务的中文名称"""
        return cls.SUPPORTED_TASKS.get(task, task)


    @classmethod
    def get_model_info(cls, model_name: str) -> Dict:
        """
        获取模型的详细信息（性能指标等）

        Args:
            model_name: 模型名称

        Returns:
            模型信息字典
        """
        # 语义分割模型性能数据
        SEMANTIC_PERFORMANCE = {
            # Cityscapes数据集
            'yolo26n-sem': {'dataset': 'Cityscapes', 'mIoU': 78.3, 'speed': 4.4, 'params': 1.6, 'flops': 22.7},
            'yolo26s-sem': {'dataset': 'Cityscapes', 'mIoU': 80.8, 'speed': 8.4, 'params': 6.5, 'flops': 88.8},
            'yolo26m-sem': {'dataset': 'Cityscapes', 'mIoU': 82.0, 'speed': 19.9, 'params': 14.3, 'flops': 304.5},
            'yolo26l-sem': {'dataset': 'Cityscapes', 'mIoU': 82.9, 'speed': 26.5, 'params': 17.9, 'flops': 384.7},
            'yolo26x-sem': {'dataset': 'Cityscapes', 'mIoU': 83.6, 'speed': 48.9, 'params': 40.2, 'flops': 861.7},
            # ADE20K数据集
            'yolo26n-sem-ade20k': {'dataset': 'ADE20K', 'mIoU': 38.8, 'speed': 3.9, 'params': 1.6, 'flops': 4.4},
            'yolo26s-sem-ade20k': {'dataset': 'ADE20K', 'mIoU': 45.6, 'speed': 4.2, 'params': 6.5, 'flops': 17.4},
            'yolo26m-sem-ade20k': {'dataset': 'ADE20K', 'mIoU': 47.4, 'speed': 4.7, 'params': 14.3, 'flops': 59.5},
            'yolo26l-sem-ade20k': {'dataset': 'ADE20K', 'mIoU': 49.7, 'speed': 8.3, 'params': 17.9, 'flops': 75.0},
            'yolo26x-sem-ade20k': {'dataset': 'ADE20K', 'mIoU': 51.5, 'speed': 9.9, 'params': 40.2, 'flops': 168.1},
        }
        return SEMANTIC_PERFORMANCE.get(model_name, {})

    # ==================== 图像配置 ====================
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')
    DEFAULT_IMAGE_SIZE = 640
    MIN_IMAGE_SIZE = 32
    MAX_IMAGE_SIZE = 4096  # 语义分割可能需要更大尺寸

    # ==================== 颜色配置 ====================
    DEFAULT_COLORS = {
        'bbox': QColor("red"),
        'label_text': QColor("white"),
        'mask': QColor("red"),
        'cls_label': QColor("yellow"),
        'keypoint': QColor("lime"),
    }

    # ==================== 可视化配置 ====================
    DEFAULT_MASK_OPACITY = 60
    DEFAULT_LINE_WIDTH = 3
    DEFAULT_FONT_SIZE = 8
    DEFAULT_CONF_THRESHOLD = 0.25

    # ==================== 训练配置 ====================
    DEFAULT_EPOCHS = 300
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_WORKERS = 8
    DEFAULT_PATIENCE = 50
    DEFAULT_LR = 0.01

    # ==================== 日志配置 ====================
    MAX_LOG_LINES = 1000

    # ==================== 性能配置 ====================
    MEMORY_CLEANUP_INTERVAL = 5000  # 毫秒

    # ==================== 导出配置 ====================
    ONNX_OPSET_VERSIONS = ['12', '13', '14', '15', '16', '17']
    DEFAULT_ONNX_OPSET = '12'

    # ==================== UI配置 ====================
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900

    # ==================== 辅助方法 ====================


# 模块级别便捷函数
def get_models(task: str, dataset: Optional[str] = None) -> List[str]:
    """获取指定任务的预训练模型列表"""
    return AppConfig.get_pretrained_models(task, dataset)


def get_task_name(task: str) -> str:
    """获取任务的中文名称"""
    return AppConfig.get_task_name(task)


# 初始化预训练模型列表
AppConfig._init_pretrained_models()

# 导出配置实例
config = AppConfig
