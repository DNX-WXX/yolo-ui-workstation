"""
config/settings.py 单元测试
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import AppConfig, get_models, get_task_name


class TestAppConfig:
    """AppConfig 测试类"""

    def test_app_version(self):
        assert AppConfig.APP_VERSION

    def test_task_suffix(self):
        assert AppConfig.TASK_SUFFIX['detect'] == ''
        assert AppConfig.TASK_SUFFIX['segment'] == '-seg'
        assert AppConfig.TASK_SUFFIX['semantic'] == '-sem'
        assert AppConfig.TASK_SUFFIX['depth'] == '-depth'
        assert AppConfig.TASK_SUFFIX['classify'] == '-cls'
        assert AppConfig.TASK_SUFFIX['pose'] == '-pose'
        assert AppConfig.TASK_SUFFIX['obb'] == '-obb'

    def test_version_task_matrix(self):
        assert 'detect' in AppConfig.VERSION_TASK_MATRIX['yolov8']
        assert 'segment' in AppConfig.VERSION_TASK_MATRIX['yolov8']
        assert 'semantic' not in AppConfig.VERSION_TASK_MATRIX['yolov8']
        assert 'semantic' in AppConfig.VERSION_TASK_MATRIX['yolo26']
        assert AppConfig.VERSION_TASK_MATRIX['yolov5'] == ['detect', 'classify']

    def test_build_model_list_detect(self):
        models = AppConfig._build_model_list('yolov8', 'detect')
        assert 'yolov8n' in models
        assert 'yolov8x' in models
        assert len(models) == 5

    def test_build_model_list_segment(self):
        models = AppConfig._build_model_list('yolov8', 'segment')
        assert 'yolov8n-seg' in models
        assert len(models) == 5

    def test_build_model_list_unsupported_task(self):
        models = AppConfig._build_model_list('yolov5', 'segment')
        assert models == []

    def test_build_model_list_yolov5_extra(self):
        models = AppConfig._build_model_list('yolov5', 'detect')
        assert 'yolov5n' in models
        assert 'yolov5n6' in models
        assert len(models) == 10

    def test_get_pretrained_models_detect(self):
        models = AppConfig.get_pretrained_models('detect')
        assert len(models) > 0
        assert 'yolov8n' in models
        assert 'yolo11n' in models

    def test_get_pretrained_models_semantic_with_dataset(self):
        ade_models = AppConfig.get_pretrained_models('semantic', dataset='ade20k')
        assert 'yolo26n-sem-ade20k' in ade_models

    def test_get_task_name(self):
        assert AppConfig.get_task_name('detect') == '目标检测'
        assert AppConfig.get_task_name('semantic') == '语义分割'
        assert AppConfig.get_task_name('unknown') == 'unknown'


    def test_get_model_info(self):
        info = AppConfig.get_model_info('yolo26n-sem')
        assert 'mIoU' in info
        info_empty = AppConfig.get_model_info('unknown')
        assert info_empty == {}


class TestModuleHelpers:
    """模块级便捷函数测试"""

    def test_get_models(self):
        models = get_models('detect')
        assert len(models) > 0

    def test_get_task_name(self):
        assert get_task_name('detect') == '目标检测'
