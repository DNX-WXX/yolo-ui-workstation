"""
核心功能包 - 模型管理、推理引擎和训练器
"""
from .model_manager import ModelManager, ModelCache
from .inference_engine import (
    InferenceEngine,
    InferenceResult,
    GPUResourceManager,
    InferenceTask
)
from .trainer import (
    ModelTrainer,
    TrainingCallback,
)

__all__ = [
    # 模型管理
    'ModelManager',
    'ModelCache',

    # 推理引擎
    'InferenceEngine',
    'InferenceResult',
    'GPUResourceManager',
    'InferenceTask',

    # 训练器
    'ModelTrainer',
    'TrainingCallback',
]

__version__ = '1.0.2'
__author__ = 'DNX'
