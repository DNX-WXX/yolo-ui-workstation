"""
core/model_manager.py 单元测试
"""
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_manager import ModelCache, ModelManager, get_yolo_model_name


class TestModelCache:
    """ModelCache 测试类"""

    def test_put_and_get(self):
        cache = ModelCache(max_models=2)
        cache.put('m1', 'model1')
        assert cache.get('m1') == 'model1'

    def test_lru_eviction(self):
        cache = ModelCache(max_models=2)
        cache.put('m1', 'model1')
        cache.put('m2', 'model2')
        cache.put('m3', 'model3')
        assert cache.get('m1') is None or cache.get('m2') is None

    def test_clear(self):
        cache = ModelCache()
        cache.put('m1', 'model1')
        cache.clear()
        assert cache.get('m1') is None
        assert cache.get_cache_info() == []

    def test_get_cache_info(self):
        cache = ModelCache()
        cache.put('m1', 'model1')
        info = cache.get_cache_info()
        assert len(info) == 1
        assert info[0]['key'] == 'm1'
        assert info[0]['access_count'] == 1


class TestGetYOLOModelName:
    """get_yolo_model_name 测试类"""

    def test_detect(self):
        assert get_yolo_model_name('yolov8n', 'detect') == 'yolov8n'

    def test_segment(self):
        assert get_yolo_model_name('yolov8n', 'segment') == 'yolov8n-seg'

    def test_semantic(self):
        assert get_yolo_model_name('yolo26n', 'semantic') == 'yolo26n-sem'

    def test_depth(self):
        assert get_yolo_model_name('yolo26n', 'depth') == 'yolo26n-depth'

    def test_classify(self):
        assert get_yolo_model_name('yolo11n', 'classify') == 'yolo11n-cls'

    def test_pose(self):
        assert get_yolo_model_name('yolo11n', 'pose') == 'yolo11n-pose'

    def test_obb(self):
        assert get_yolo_model_name('yolo11n', 'obb') == 'yolo11n-obb'


class TestModelManager:
    """ModelManager 测试类"""

    def setup_method(self):
        self.manager = ModelManager()


    def test_detect_model_task_by_suffix(self):
        assert self.manager.detect_model_task('model-seg.pt') == 'segment'
        assert self.manager.detect_model_task('model-cls.pt') == 'classify'
        assert self.manager.detect_model_task('model-depth.pt') == 'depth'
        assert self.manager.detect_model_task('model-pose.pt') == 'pose'
        assert self.manager.detect_model_task('model-obb.pt') == 'obb'
        assert self.manager.detect_model_task('model-sem.pt') == 'semantic'

    def test_detect_model_task_nas_world(self):
        assert self.manager.detect_model_task('yolo_nas_s.pt') == 'detect'
        assert self.manager.detect_model_task('yolo_world_v2.pt') == 'detect'

    @patch('ultralytics.YOLO')
    def test_load_pt_model(self, mock_yolo):
        mock_model = MagicMock()
        mock_model.model.to.return_value = None
        mock_yolo.return_value = mock_model
        with patch.object(self.manager.device_manager, 'get_best_device', return_value='cpu'):
            model = self.manager.load_pt_model('dummy.pt', task='detect')
        assert model is not None
        mock_yolo.assert_called_with('dummy.pt', task='detect')
        assert self.manager.current_model_type == 'pt'

    @patch('ultralytics.YOLO')
    def test_load_pt_model_cached(self, mock_yolo):
        mock_model = MagicMock()
        mock_model.model.to.return_value = None
        mock_yolo.return_value = mock_model
        with patch.object(self.manager.device_manager, 'get_best_device', return_value='cpu'):
            self.manager.load_pt_model('dummy.pt', task='detect')
            # 第二次调用应命中缓存
            model2 = self.manager.load_pt_model('dummy.pt', task='detect')
        assert model2 is mock_model

    def test_unload_model(self):
        self.manager.current_model = 'dummy'
        self.manager.unload_model()
        assert self.manager.current_model is None
        assert self.manager.current_model_path is None

    def test_get_model_info_no_model(self):
        info = self.manager.get_model_info()
        assert info['status'] == '未加载'

    @patch('ultralytics.YOLO')
    def test_get_model_info_loaded(self, mock_yolo):
        mock_model = MagicMock()
        mock_model.names = {0: 'a'}
        mock_model.model.to.return_value = None
        mock_yolo.return_value = mock_model
        with patch.object(self.manager.device_manager, 'get_best_device', return_value='cpu'):
            self.manager.load_pt_model('dummy.pt', task='detect')
        info = self.manager.get_model_info()
        assert info['status'] == '已加载'
        assert info['task'] == 'detect'

    def test_clear_cache(self):
        self.manager.cache.put('k', 'v')
        self.manager.clear_cache()
        assert self.manager.get_cache_info() == []
