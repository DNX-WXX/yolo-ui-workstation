"""
core/trainer.py 单元测试
"""
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trainer import (
    TrainingCallback, LoggingCallback, ModelTrainer
)


class TestTrainingCallback:
    """TrainingCallback 测试类"""

    def test_methods_exist(self):
        cb = TrainingCallback()
        assert hasattr(cb, 'on_train_start')
        assert hasattr(cb, 'on_train_end')
        assert hasattr(cb, 'on_epoch_end')
        assert hasattr(cb, 'on_validation_end')

    def test_methods_callable(self):
        cb = TrainingCallback()
        cb.on_train_start(None)
        cb.on_train_end(None, None)
        cb.on_epoch_end(None)
        cb.on_validation_end(None, {})


class TestLoggingCallback:
    """LoggingCallback 测试类"""

    def test_on_train_start(self):
        from utils.logger import LogManager
        logger = LogManager()
        cb = LoggingCallback(logger)
        trainer = MagicMock()
        trainer.task = 'detect'
        trainer.model_name = 'yolov8n'
        trainer.device = 'cpu'
        trainer.train_params = {}
        cb.on_train_start(trainer)
        assert cb.start_time is not None

    def test_on_validation_end(self):
        from utils.logger import LogManager
        logger = LogManager()
        cb = LoggingCallback(logger)
        trainer = MagicMock()
        cb.on_validation_end(trainer, {'metrics/mAP50(B)': 0.85})
        assert cb.best_map == 0.85


class TestModelTrainer:
    """ModelTrainer 测试类"""

    def setup_method(self):
        self.trainer = ModelTrainer()

    def test_default_models(self):
        assert ModelTrainer.DEFAULT_MODELS['detect'] == 'yolo11n.pt'
        assert ModelTrainer.DEFAULT_MODELS['segment'] == 'yolo11n-seg.pt'
        assert ModelTrainer.DEFAULT_MODELS['semantic'] == 'yolo26n-sem.pt'
        assert ModelTrainer.DEFAULT_MODELS['depth'] == 'yolo26n-depth.pt'


    def test_stop_training(self):
        self.trainer.stop_training()
        assert self.trainer._stop_training is True


    def test_train(self):
        mock_results = MagicMock()
        mock_results.results_dict = {'metrics/mAP50(B)': 0.8}

        with patch.object(self.trainer, '_prepare_training') as mock_prep, \
             patch.object(self.trainer, '_run_training') as mock_run:
            mock_run.return_value = mock_results
            result = self.trainer.train(
                model='yolov8n.pt',
                data='dummy.yaml',
                epochs=1,
                batch=2,
                imgsz=320
            )
        assert result is mock_results
        mock_prep.assert_called_once()
        mock_run.assert_called_once()

    @patch('ultralytics.YOLO')
    def test_validate(self, mock_yolo):
        mock_model = MagicMock()
        mock_model.val.return_value = {'metrics/mAP50(B)': 0.8}
        mock_yolo.return_value = mock_model

        with patch.object(self.trainer.device_manager, 'get_safe_device', return_value='cpu'):
            metrics = self.trainer.validate(
                model_path='yolov8n.pt',
                data='dummy.yaml',
                imgsz=320,
                batch=2
            )
        assert 'metrics/mAP50(B)' in metrics
