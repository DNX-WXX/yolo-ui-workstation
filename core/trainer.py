"""
训练器 - 负责YOLO模型训练的统一接口
支持检测、分割、分类、姿态估计、语义分割、深度估计、旋转框检测等任务
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO

from utils.logger import LogManager
from utils.device_utils import DeviceManager
from utils.error_handler import safe_execute
from config.settings import AppConfig


class TrainingCallback:
    """训练回调基类（抽象接口）

    定义训练生命周期钩子，子类按需覆写（如 LoggingCallback 记录日志），
    未覆写的方法默认空操作，无需全部实现。
    """

    def on_train_start(self, trainer):
        """训练开始时触发

        Args:
            trainer: Trainer 实例，可读取 task/model_name/device/train_params 等属性
        """
        pass

    def on_train_end(self, trainer, results):
        """训练结束时触发

        Args:
            trainer: Trainer 实例
            results: 训练结果（最终指标与产物路径）
        """
        pass

    def on_epoch_end(self, trainer):
        """每个 epoch 结束时触发

        Args:
            trainer: Trainer 实例
        """
        pass


    def on_validation_end(self, trainer, metrics):
        """验证结束时触发

        Args:
            trainer: Trainer 实例
            metrics: 验证指标字典
        """
        pass


class LoggingCallback(TrainingCallback):
    """日志记录回调"""

    def __init__(self, logger: LogManager):
        self.logger = logger
        self.start_time = None
        self.best_map = 0

    def on_train_start(self, trainer):
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("训练开始")
        self.logger.info(f"任务类型: {trainer.task}")
        self.logger.info(f"模型: {trainer.model_name}")
        self.logger.info(f"设备: {trainer.device}")
        self.logger.info(f"训练参数: {trainer.train_params}")
        self.logger.info("=" * 60)

    def on_train_end(self, trainer, results):
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        self.logger.info("=" * 60)
        if not self._stop_training:
            self.logger.info("训练完成！")
        self.logger.info(f"总耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")

        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {key}: {value:.4f}")

        self.logger.info(f"最佳模型保存: {trainer.best_model_path}")
        self.logger.info(f"最后模型保存: {trainer.last_model_path}")
        self.logger.info("=" * 60)

    def on_epoch_end(self, trainer):
        if trainer.epoch % 10 == 0:  # 每10个epoch打印一次
            self.logger.info(f"Epoch {trainer.epoch}/{trainer.total_epochs} 完成")

    def on_validation_end(self, trainer, metrics):
        if 'metrics/mAP50(B)' in metrics:
            current_map = metrics['metrics/mAP50(B)']
            if current_map > self.best_map:
                self.best_map = current_map
                self.logger.info(f"新的最佳mAP50: {self.best_map:.4f}")


class ModelTrainer:
    """模型训练器"""

    # 任务类型到默认模型的映射
    DEFAULT_MODELS = {
        'detect': 'yolo11n.pt',
        'segment': 'yolo11n-seg.pt',
        'semantic': 'yolo26n-sem.pt',
        'depth': 'yolo26n-depth.pt',
        'classify': 'yolo11n-cls.pt',
        'pose': 'yolo11n-pose.pt',
        'obb': 'yolo11n-obb.pt'
    }

    def __init__(self, logger: Optional[LogManager] = None):
        """
        初始化训练器

        Args:
            logger: 日志管理器
        """
        self.logger = logger or LogManager()
        self.device_manager = DeviceManager()

        # 训练状态
        self.model = None
        self.task = None
        self.model_name = None
        self.device = 'cpu'
        self.epoch = 0
        self.total_epochs = 0
        self.train_params = {}

        # 路径
        self.project_dir = None
        self.experiment_name = None
        self.best_model_path = None
        self.last_model_path = None

        # 回调
        self.callbacks = []
        self.add_callback(LoggingCallback(self.logger))

        # 训练是否停止
        self._stop_training = False

    def add_callback(self, callback: TrainingCallback):
        """添加训练回调"""
        self.callbacks.append(callback)


    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """触发回调事件"""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                try:
                    method(self, *args, **kwargs)
                except Exception as e:
                    self.logger.error(f"回调执行失败 {event}: {e}")

    @safe_execute("训练模型失败")
    def train(
            self,
            model: Union[str, YOLO],
            data: str,
            task: Optional[str] = None,
            epochs: int = AppConfig.DEFAULT_EPOCHS,
            batch: int = AppConfig.DEFAULT_BATCH_SIZE,
            imgsz: int = AppConfig.DEFAULT_IMAGE_SIZE,
            workers: int = AppConfig.DEFAULT_WORKERS,
            device: Optional[str] = None,
            project: str = './runs',
            name: str = 'train',
            exist_ok: bool = True,
            pretrained: bool = True,
            optimizer: str = 'auto',
            lr0: float = AppConfig.DEFAULT_LR,
            resume: bool = False,
            patience: int = AppConfig.DEFAULT_PATIENCE,
            seed: int = 42,
            deterministic: bool = True,
            single_cls: bool = False,
            rect: bool = False,
            cos_lr: bool = False,
            warmup_epochs: int = 3,
            warmup_momentum: float = 0.8,
            warmup_bias_lr: float = 0.1,
            dropout: float = 0.0,
            amp: bool = True,  # 混合精度训练
            verbose: bool = True,
            **kwargs
    ) -> Any:
        """
        训练YOLO模型

        Args:
            model: 模型路径或YOLO实例
            data: 数据集配置文件路径
            task: 任务类型（自动检测）
            epochs: 训练轮数
            batch: 批次大小
            imgsz: 输入图像尺寸
            workers: 数据加载线程数
            device: 设备（cuda/cpu）
            project: 项目目录
            name: 实验名称
            exist_ok: 是否覆盖现有实验
            pretrained: 是否使用预训练权重
            optimizer: 优化器
            lr0: 初始学习率
            resume: 是否恢复训练
            patience: 早停耐心值
            seed: 随机种子
            deterministic: 是否确定性训练
            single_cls: 是否单类别训练
            rect: 是否矩形训练
            cos_lr: 是否使用余弦学习率（新版YOLO使用cos_lr控制学习率调度）
            warmup_epochs: 预热轮数
            warmup_momentum: 预热动量
            warmup_bias_lr: 预热偏置学习率
            dropout: Dropout率
            amp: 是否使用混合精度训练
            verbose: 是否详细输出
            **kwargs: 其他训练参数

        Returns:
            训练结果
        """
        # 重置停止标志
        self._stop_training = False

        # 1. 准备训练参数
        self._prepare_training(
            model, task, epochs, batch, imgsz, workers, device,
            project, name, exist_ok, pretrained, optimizer, lr0,
            seed, deterministic, **kwargs
        )

        # 2. 触发训练开始回调
        self._trigger_callbacks('on_train_start')

        # 3. 执行训练
        results = self._run_training(
            data, resume, patience, single_cls, rect, cos_lr,
            warmup_epochs, warmup_momentum, warmup_bias_lr,
            dropout, amp, verbose, **kwargs
        )

        # 4. 触发训练结束回调
        self._trigger_callbacks('on_train_end', results)

        return results

    def _prepare_training(self, model, task, epochs, batch, imgsz, workers,
                          device, project, name, exist_ok, pretrained,
                          optimizer, lr0, seed, deterministic, **kwargs):
        """准备训练参数和模型"""
        from ultralytics import YOLO

        # 设置设备
        self.device = device or self.device_manager.get_safe_device()

        # 保存训练参数
        self.train_params = {
            'epochs': epochs,
            'batch': batch,
            'imgsz': imgsz,
            'workers': workers,
            'device': self.device,
            'optimizer': optimizer,
            'lr0': lr0,
            'seed': seed,
            'deterministic': deterministic,
            **kwargs
        }

        self.total_epochs = epochs

        # 设置项目目录
        self.project_dir = Path(project)
        self.experiment_name = name

        # 加载模型
        if isinstance(model, YOLO):
            self.model = model
            self.model_name = "custom_model"
        else:
            self.model_name = os.path.basename(model)

            # 检测任务类型
            if task is None:
                from core.model_manager import ModelManager
                model_manager = ModelManager(self.logger)
                self.task = model_manager.detect_model_task(model)
            else:
                self.task = task

            self.logger.info(f"加载模型: {self.model_name}")
            self.logger.info(f"任务类型: {self.task}")

            # 加载模型
            try:
                from ultralytics import YOLO
                self.model = YOLO(model, task=self.task)
            except Exception as e:
                self.logger.error(f"模型加载失败: {e}")
                raise

    def _run_training(self, data, resume, patience, single_cls, rect, cos_lr,
                      warmup_epochs, warmup_momentum, warmup_bias_lr,
                      dropout, amp, verbose, **kwargs) -> Any:
        """执行训练"""
        self.logger.info("开始训练循环...")

        # 构建训练参数
        train_args = {
            'data': data,
            'epochs': self.total_epochs,
            'batch': self.train_params['batch'],
            'imgsz': self.train_params['imgsz'],
            'workers': self.train_params['workers'],
            'device': self.device,
            'project': str(self.project_dir),
            'name': self.experiment_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': self.train_params['optimizer'],
            'lr0': self.train_params['lr0'],
            'resume': resume,
            'patience': patience,
            'single_cls': single_cls,
            'rect': rect,
            'cos_lr': cos_lr,
            'warmup_epochs': warmup_epochs,
            'warmup_momentum': warmup_momentum,
            'warmup_bias_lr': warmup_bias_lr,
            'dropout': dropout,
            'amp': amp,
            'seed': self.train_params['seed'],
            'deterministic': self.train_params['deterministic'],
            'verbose': verbose,
            **kwargs
        }

        # 添加回调来监控训练进度
        if hasattr(self.model, 'add_callback'):
            def progress_callback(trainer):
                self.epoch = trainer.epoch
                self._trigger_callbacks('on_epoch_end')

                # 检查是否应该停止训练
                if self._stop_training:
                    trainer.stop = True

            self.model.add_callback('on_train_epoch_end', progress_callback)

            # 每个 batch 结束时检查停止标志，实现更快响应（不必等整个 epoch 跑完）
            def batch_stop_callback(trainer):
                if self._stop_training:
                    trainer.stop = True

            self.model.add_callback('on_train_batch_end', batch_stop_callback)

            def val_callback(trainer):
                # ultralytics 的 trainer.metrics 可能是 DetMetrics 对象，转成字典再回调
                metrics = getattr(trainer, 'metrics', None) or {}
                results_dict = getattr(metrics, 'results_dict', None)
                if results_dict is not None:
                    metrics = results_dict
                self._trigger_callbacks('on_validation_end', metrics)

            self.model.add_callback('on_val_end', val_callback)

        try:
            # 执行训练
            results = self.model.train(**train_args)

            # 获取最佳模型路径
            self.best_model_path = str(self.project_dir / self.experiment_name / 'weights' / 'best.pt')
            self.last_model_path = str(self.project_dir / self.experiment_name / 'weights' / 'last.pt')

            return results

        except KeyboardInterrupt:
            # 用户主动停止训练（trainer.stop=True 时 Ultralytics 抛出 KeyboardInterrupt）
            self.logger.info("训练已被用户停止")
            # 停止时模型可能仍保存了 last.pt
            self.last_model_path = str(self.project_dir / self.experiment_name / 'weights' / 'last.pt')
            best_path = self.project_dir / self.experiment_name / 'weights' / 'best.pt'
            if best_path.exists():
                self.best_model_path = str(best_path)
            return None
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}")
            raise
        finally:
            # 清理回调
            if hasattr(self.model, 'reset_callbacks'):
                self.model.reset_callbacks()


    def validate(
            self,
            model_path: str,
            data: str,
            imgsz: int = AppConfig.DEFAULT_IMAGE_SIZE,
            batch: int = AppConfig.DEFAULT_BATCH_SIZE,
            device: Optional[str] = None,
            project: Optional[str] = None,
            name: Optional[str] = None,
            **kwargs
    ) -> Any:
        """
        验证模型

        Args:
            model_path: 模型路径
            data: 数据集配置文件
            imgsz: 输入图像尺寸
            batch: 批次大小
            device: 设备
            project: 项目目录
            name: 实验名称
            **kwargs: 其他验证参数

        Returns:
            验证结果
        """
        from ultralytics import YOLO
        self.logger.info(f"验证模型: {os.path.basename(model_path)}")

        device = device or self.device_manager.get_safe_device()

        # 加载模型
        model = YOLO(model_path)

        # 构建验证参数
        val_args = {
            'data': data,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'verbose': True,
            **kwargs
        }

        if project:
            val_args['project'] = project
        if name:
            val_args['name'] = name

        # 执行验证
        metrics = model.val(**val_args)

        self.logger.info("验证完成")

        return metrics

    def stop_training(self):
        """停止训练"""
        self._stop_training = True
        self.logger.info("正在停止训练...")
