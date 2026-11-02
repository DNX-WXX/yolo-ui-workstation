"""
设备管理工具 - 负责CUDA、ONNX运行时和内存管理
"""
import psutil
import platform
import ctypes
from typing import Dict


class DeviceManager:
    """设备管理器 - 单例模式"""

    _instance = None
    _detection_done = False  # 标记是否已经完成检测并打印
    _initialized = False  # 标记是否已经初始化

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, verbose=True):
        """
        初始化设备管理器

        Args:
            verbose: 是否打印详细信息（只在第一次初始化时有效）
        """
        if DeviceManager._initialized:
            return
        DeviceManager._initialized = True
        self._verbose = verbose
        self._cuda_available = None  # None 表示尚未检测
        self._cuda_device_count = 0
        self._cuda_device_names = []
        self._onnx_providers = []
        # 不在 __init__ 中同步检测设备，避免在主线程导入 torch 导致启动卡顿
        # 设备检测会在首次访问设备信息时惰性触发

    def _ensure_detected(self):
        """惰性触发设备检测（首次访问设备信息时执行）"""
        if self._cuda_available is None:
            self._detect_devices()

    def _detect_devices(self):
        """检测可用设备 - 只打印一次"""
        import torch
        import onnxruntime as ort
        try:
            # 检测CUDA
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                self._cuda_device_count = torch.cuda.device_count()
                self._cuda_device_names = [
                    torch.cuda.get_device_name(i)
                    for i in range(self._cuda_device_count)
                ]

            # 检测ONNX提供程序
            self._onnx_providers = ort.get_available_providers()

            # 只在第一次检测且verbose为True时打印
            if self._verbose and not DeviceManager._detection_done:
                print(f"CUDA可用: {self._cuda_device_count}个设备")
                if self._cuda_available and self._cuda_device_count > 0:
                    for i, name in enumerate(self._cuda_device_names):
                        print(f"  GPU {i}: {name}")
                print(f"ONNX提供程序: {self._onnx_providers}")
                DeviceManager._detection_done = True

        except Exception as e:
            if self._verbose and not DeviceManager._detection_done:
                print(f"设备检测失败: {e}")
                DeviceManager._detection_done = True

    def get_safe_device(self, index: int = 0) -> str:
        """
        安全获取可用设备

        Args:
            index: GPU设备索引

        Returns:
            'cuda:{index}' 或 'cpu'
        """
        import torch
        self._ensure_detected()
        if not self._cuda_available or index >= self._cuda_device_count:
            return 'cpu'

        try:
            # 测试CUDA设备
            device = f'cuda:{index}'
            torch.tensor([1.0]).to(device)
            return device
        except Exception as e:
            print(f"CUDA设备 {index} 不可用: {e}")
            return 'cpu'


    def get_best_device(self) -> str:
        """
        获取最佳设备（基于内存使用情况）

        Returns:
            最佳设备
        """
        import torch
        self._ensure_detected()
        if not self._cuda_available:
            return 'cpu'

        best_device = 'cpu'
        max_free_memory = 0

        for i in range(self._cuda_device_count):
            try:
                device = f'cuda:{i}'
                free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = device
            except:
                pass

        return best_device


    def get_device_info(self) -> Dict:
        """获取设备信息"""
        self._ensure_detected()
        return {
            'cuda_available': self._cuda_available,
            'cuda_devices': self._cuda_device_count,
            'cuda_device_names': self._cuda_device_names,
            'onnx_providers': self._onnx_providers,
            'current_device': self.get_safe_device()
        }

    @staticmethod
    def get_memory_usage() -> float:
        """获取当前进程内存使用情况 (MB)"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # MB
        except:
            return 0.0

    @staticmethod
    def get_cuda_memory_info() -> Dict:
        """获取CUDA内存使用情况"""
        import torch
        if not torch.cuda.is_available():
            return {}

        try:
            memory_info = {}
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024  # MB
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024  # MB
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024  # MB
                free = total - allocated
                
                memory_info[f'cuda:{i}'] = {
                    'total': total,
                    'allocated': allocated,
                    'reserved': reserved,
                    'free': free,
                    'usage_percent': (allocated / total) * 100
                }
            return memory_info
        except:
            return {}

    @staticmethod
    def clear_memory(force: bool = False):
        """清理内存"""
        import gc
        import torch

        # 收集垃圾
        gc.collect()

        # 清理CUDA缓存
        if torch.cuda.is_available():
            # 只有当内存使用超过阈值或强制清理时才清理
            memory_info = DeviceManager.get_cuda_memory_info()
            should_clean = force
            
            for device, info in memory_info.items():
                if info['usage_percent'] > 70:  # 当内存使用超过70%时清理
                    should_clean = True
                    break
            
            if should_clean:
                torch.cuda.empty_cache()
                print("CUDA内存已清理")


class PowerManager:
    """电源管理器 - Windows系统防休眠"""

    # Windows API常量
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    def __init__(self):
        self.prevent_sleep = False
        self._is_windows = platform.system() == "Windows"

    def set_sleep_prevention(self, enabled: bool):
        """
        设置防休眠

        Args:
            enabled: True启用防休眠，False禁用
        """
        self.prevent_sleep = enabled

        if not self._is_windows:
            return

        try:
            if enabled:
                # 阻止系统休眠和显示器关闭
                ctypes.windll.kernel32.SetThreadExecutionState(
                    self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED | self.ES_DISPLAY_REQUIRED
                )
                print("已启用防休眠模式")
            else:
                # 恢复默认电源设置
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
                print("已禁用防休眠模式")
        except Exception as e:
            print(f"设置电源管理失败: {e}")

    def __del__(self):
        """析构时恢复电源设置"""
        if self._is_windows and self.prevent_sleep:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            except:
                pass
