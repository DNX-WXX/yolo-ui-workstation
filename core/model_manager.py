"""
模型管理器 - 负责模型的加载、缓存和任务识别
支持 YOLOv5、YOLOv8、YOLO11、YOLO26
"""
import os
from typing import Optional, Dict, Any
import time
from utils.device_utils import DeviceManager
from utils.logger import LogManager

# 支持的YOLO模型系列
YOLO_MODELS = {
    'yolov5': ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
               'yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6'],
    'yolov8': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
    'yolo11': ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
    'yolo26': ['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x'],
}

# 任务类型映射
TASK_MAPPING = {
    'detect': '目标检测',
    'segment': '实例分割',
    'semantic': '语义分割',
    'depth': '深度估计',
    'classify': '图像分类',
    'pose': '姿态估计',
    'obb': '旋转框检测'
}

# 模型版本兼容性信息
MODEL_COMPATIBILITY = {
    'yolov5': {'ultralytics_min': '8.0.0', 'torch_min': '1.8.0'},
    'yolov8': {'ultralytics_min': '8.0.0', 'torch_min': '1.8.0'},
    'yolo11': {'ultralytics_min': '8.10.0', 'torch_min': '2.0.0'},
    'yolo26': {'ultralytics_min': '8.20.0', 'torch_min': '2.0.0'},
}


class ModelCache:
    """模型缓存 - LRU策略"""

    def __init__(self, max_models=5):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_models = max_models
        self.access_count = {}  # 访问计数

    def get(self, key: str):
        """获取缓存模型"""
        if key in self.cache:
            # 更新访问时间和计数
            self.cache[key]['last_access'] = time.time()
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]['model']
        return None

    def put(self, key: str, model, metadata: dict = None):
        """缓存模型"""
        import torch
        # 检查缓存大小
        if len(self.cache) >= self.max_models:
            # 基于访问频率和时间的智能淘汰
            def get_eviction_score(k):
                # 结合访问计数和最后访问时间
                access_count = self.access_count.get(k, 0)
                last_access = self.cache[k]['last_access']
                # 分数越低越优先淘汰
                return (1 / (access_count + 1)) * (time.time() - last_access)

            # 选择分数最低的淘汰
            evict_key = min(self.cache.keys(), key=get_eviction_score)
            del self.cache[evict_key]
            if evict_key in self.access_count:
                del self.access_count[evict_key]

        # 缓存新模型
        self.cache[key] = {
            'model': model,
            'metadata': metadata or {},
            'last_access': time.time(),
            'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
        self.access_count[key] = 1

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()

    def get_cache_info(self):
        """获取缓存信息"""
        info = []
        for key, value in self.cache.items():
            info.append({
                'key': key,
                'access_count': self.access_count.get(key, 0),
                'last_access': value['last_access'],
                'memory_usage': value.get('memory_usage', 0)
            })
        return info


def get_yolo_model_name(base_name: str, task: str) -> str:
    """获取YOLO模型名称（统一函数）"""
    task_suffix = {
        'detect': '',
        'segment': '-seg',
        'semantic': '-sem',
        'depth': '-depth',
        'classify': '-cls',
        'pose': '-pose',
        'obb': '-obb'
    }
    return f"{base_name}{task_suffix.get(task, '')}"

# 保留兼容函数
get_yolov9_model_name = get_yolo_model_name
get_yolov10_model_name = get_yolo_model_name


class ModelManager:
    """模型管理器 - 支持多种YOLO变体和自定义模型"""

    def __init__(self, logger: Optional[LogManager] = None):
        self.device_manager = DeviceManager()
        self.logger = logger or LogManager()
        self.cache = ModelCache()

        # 当前活动模型
        self.current_model = None
        self.current_model_path = None
        self.current_model_task = None
        self.current_model_type = None  # 'pt', 'onnx', 'wrapper'
        self.current_model_info = None  # 模型详细信息
        
        # 模型预加载队列
        self.preload_queue = []
        self.is_preloading = False
        
        # 已加载的自定义配置
        self.custom_configs = {}


    def detect_model_task(self, model_path: str) -> str:
        """
        检测模型任务类型
        返回: 'detect', 'segment', 'semantic', 'depth', 'classify', 'pose', 'obb'
        """
        try:
            # 缓存：同一文件（路径+mtime+size未变）直接返回上次结果，
            # 避免每次选择模型都在UI线程反序列化整个.pt文件
            if not hasattr(self, '_task_detect_cache'):
                self._task_detect_cache = {}
            try:
                stat = os.stat(model_path)
                cache_key = (model_path, stat.st_mtime, stat.st_size)
            except OSError:
                cache_key = None
            if cache_key and cache_key in self._task_detect_cache:
                return self._task_detect_cache[cache_key]

            # 通过文件名判断（优先检查语义分割和深度估计后缀）
            filename = os.path.basename(model_path).lower()

            # 配置字典驱动：后缀到任务类型的映射
            TASK_SUFFIX_MAP = {
                '-sem': 'semantic',
                '-depth': 'depth',
                '-seg': 'segment',
                '-cls': 'classify',
                '-pose': 'pose',
                '-obb': 'obb',
            }

            # 按优先级检查后缀
            for suffix, task in TASK_SUFFIX_MAP.items():
                if suffix in filename:
                    if cache_key:
                        self._task_detect_cache[cache_key] = task
                    return task

            # 文件不存在时直接返回默认类型，避免触发 YOLO 自动下载/安装
            if not os.path.exists(model_path):
                self.logger.warning(f"模型文件不存在，默认检测为 detect: {model_path}")
                return 'detect'

            # 尝试加载模型
            try:
                import torch
                from ultralytics import YOLO
                # mmap=True（torch>=2.1）惰性映射权重，仅读取元信息时显著加速
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', mmap=True)
                except TypeError:
                    checkpoint = torch.load(model_path, map_location='cpu')

                detected = None
                if 'model' in checkpoint:
                    model_str = str(checkpoint['model'])
                    if 'Classify' in model_str:
                        detected = 'classify'
                    elif 'Segment' in model_str:
                        detected = 'segment'
                    elif 'Pose' in model_str:
                        detected = 'pose'

                # 检查配置
                if detected is None and 'train_args' in checkpoint:
                    detected = checkpoint['train_args'].get('task')

            except:
                # torch.load 失败时不再回退到 YOLO() 构造，
                # 因为 YOLO() 对非官方模型名会触发 AutoInstall 挂起。
                # 直接返回默认类型 detect。
                detected = None

            result = detected or 'detect'  # 默认
            if cache_key:
                self._task_detect_cache[cache_key] = result
            return result

        except Exception as e:
            self.logger.error(f"检测模型类型失败: {e}")
            return 'detect'

    def load_pt_model(self, model_path: str, task: Optional[str] = None, device: Optional[str] = None):
        """加载PyTorch模型"""
        cache_key = f"pt_{model_path}_{device or 'auto'}"
        cached = self.cache.get(cache_key)

        if cached:
            self.logger.info(f"从缓存加载模型: {os.path.basename(model_path)}")
            self.current_model = cached
            self.current_model_path = model_path
            self.current_model_task = task or self.detect_model_task(model_path)
            self.current_model_type = 'pt'
            return cached

        from ultralytics import YOLO
        self.logger.info(f"加载PT模型: {os.path.basename(model_path)}")

        try:
            # 选择最佳设备
            target_device = device or self.device_manager.get_best_device()
            self.logger.info(f"使用设备: {target_device}")

            model = YOLO(model_path, task=task)

            # 移动到目标设备（仅对 PyTorch 模型有效；ONNX 等格式 model.model 为字符串路径）
            import torch.nn as nn
            if hasattr(model, 'model') and isinstance(model.model, nn.Module):
                model.model.to(target_device)

            # 缓存
            self.cache.put(cache_key, model, {'task': task, 'device': target_device})

            self.current_model = model
            self.current_model_path = model_path
            self.current_model_task = task or self.detect_model_task(model_path)
            self.current_model_type = 'pt'

            self.logger.info(f"模型加载成功: {self.current_model_task} (设备: {target_device})")
            return model

        except Exception as e:
            self.logger.error(f"加载PT模型失败: {e}")
            raise

    def preload_model(self, model_path: str, task: Optional[str] = None):
        """预加载模型到缓存"""
        import threading
        
        def _preload():
            self.is_preloading = True
            try:
                self.logger.info(f"预加载模型: {os.path.basename(model_path)}")
                # 使用CPU预加载，后续使用时会迁移到合适的GPU
                self.load_pt_model(model_path, task, device='cpu')
                self.logger.info(f"模型预加载完成: {os.path.basename(model_path)}")
            except Exception as e:
                self.logger.error(f"预加载模型失败: {e}")
            finally:
                self.is_preloading = False
                if self.preload_queue:
                    # 预加载下一个模型
                    next_model = self.preload_queue.pop(0)
                    self.preload_model(next_model['path'], next_model['task'])

        # 如果已经在预加载，加入队列
        if self.is_preloading:
            self.preload_queue.append({'path': model_path, 'task': task})
            self.logger.info(f"模型已加入预加载队列: {os.path.basename(model_path)}")
            return

        # 启动预加载线程
        thread = threading.Thread(target=_preload)
        thread.daemon = True
        thread.start()


    def unload_model(self, clear_cache: bool = False):
        """卸载当前模型"""
        import torch
        self.current_model = None
        self.current_model_path = None
        self.current_model_task = None
        self.current_model_type = None

        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 清理未使用的变量
            import gc
            gc.collect()

        # 可选清理缓存
        if clear_cache:
            self.cache.clear()
            self.logger.info("模型已卸载并清空缓存")
        else:
            self.logger.info("模型已卸载")

    def get_cache_info(self):
        """获取缓存信息"""
        return self.cache.get_cache_info()

    def clear_cache(self):
        """清空缓存"""
        import torch
        self.cache.clear()
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        self.logger.info("缓存已清空")

    def get_model_info(self) -> dict:
        """获取当前模型信息"""
        if not self.current_model:
            return {'status': '未加载'}

        import torch
        info = {
            'status': '已加载',
            'type': self.current_model_type,
            'path': self.current_model_path,
            'task': self.current_model_task,
            'device': torch.cuda.current_device() if torch.cuda.is_available() else 'cpu',
            'cache_info': self.get_cache_info()
        }

        # 添加额外信息
        if hasattr(self.current_model, 'names') and self.current_model.names:
            info['num_classes'] = len(self.current_model.names)
        
        # 添加模型大小信息
        if hasattr(self.current_model, 'model'):
            try:
                model_params = sum(p.numel() for p in self.current_model.model.parameters())
                info['params'] = model_params
                info['params_mb'] = model_params * 4 / 1024 / 1024  # 假设每个参数4字节
            except:
                pass

        return info
