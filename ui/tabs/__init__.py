"""
标签页包 - 所有功能标签页
"""
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入各个标签页，如果失败则提供空对象并记录警告
try:
    from .validate_tab import ValidateTab
except ImportError as e:
    logger.warning(f"导入 ValidateTab 失败: {e}")
    ValidateTab = None

# 模型部署标签页导入
try:
    from .model_deploy_tab import ModelDeployTab
except ImportError as e:
    logger.warning(f"导入 ModelDeployTab 失败: {e}")
    ModelDeployTab = None
except Exception as e:
    logger.warning(f"导入 ModelDeployTab 时发生未知错误: {e}")
    ModelDeployTab = None

# 导入合并标签页
try:
    from .combined_inference_tab import CombinedInferenceTab
except ImportError as e:
    logger.warning(f"导入 CombinedInferenceTab 失败: {e}")
    CombinedInferenceTab = None

try:
    from .combined_train_tab import CombinedTrainTab
except ImportError as e:
    logger.warning(f"导入 CombinedTrainTab 失败: {e}")
    CombinedTrainTab = None

try:
    from .dataset_management_tab import DatasetManagementTab
except ImportError as e:
    logger.warning(f"导入 DatasetManagementTab 失败: {e}")
    DatasetManagementTab = None

# 定义 __all__ 列表，只包含成功导入的类
__all__ = []

# 动态添加成功导入的类到 __all__
if ValidateTab is not None:
    __all__.append('ValidateTab')
if ModelDeployTab is not None:
    __all__.append('ModelDeployTab')
if CombinedInferenceTab is not None:
    __all__.append('CombinedInferenceTab')
if CombinedTrainTab is not None:
    __all__.append('CombinedTrainTab')
if DatasetManagementTab is not None:
    __all__.append('DatasetManagementTab')