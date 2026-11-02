"""
配置文件包 - 应用配置和常量
"""
from .settings import AppConfig

__all__ = ['AppConfig']

__version__ = '1.0.0'

# 便捷访问
config = AppConfig