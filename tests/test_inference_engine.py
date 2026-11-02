"""
推理引擎单元测试
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.inference_engine import InferenceEngine, GPUResourceManager, InferenceResult


class TestGPUResourceManager(unittest.TestCase):
    """GPU资源管理器测试类"""

    def setUp(self):
        """设置测试环境"""
        self.manager = GPUResourceManager()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.manager)
        self.assertTrue(hasattr(self.manager, 'device'))

    def test_get_available_memory(self):
        """测试获取可用内存"""
        memory = self.manager.get_available_memory()
        self.assertIsInstance(memory, int)
        self.assertGreaterEqual(memory, 0)

    def test_calculate_optimal_batch_size(self):
        """测试计算最优批大小"""
        batch_size = self.manager.calculate_optimal_batch_size()
        self.assertIsInstance(batch_size, int)
        self.assertGreaterEqual(batch_size, 1)

    def test_estimate_batch_memory(self):
        """测试估算批处理内存"""
        memory = self.manager.estimate_batch_memory(8)
        self.assertIsInstance(memory, int)
        self.assertGreater(memory, 0)


class TestInferenceEngine(unittest.TestCase):
    """推理引擎测试类"""

    def setUp(self):
        """设置测试环境"""
        self.engine = InferenceEngine()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.engine)
        self.assertTrue(hasattr(self.engine, 'gpu_manager'))

    def test_get_device_info(self):
        """测试获取设备信息"""
        info = self.engine.get_device_info()
        self.assertIsInstance(info, dict)
        self.assertIn('device_type', info)
        self.assertIn('is_cuda', info)


class TestInferenceResult:
    """InferenceResult 测试类"""

    def test_init_detect(self, mock_raw_result_detect):
        result = InferenceResult([mock_raw_result_detect], 'test.jpg', 'detect')
        assert result.task == 'detect'
        assert result.boxes is not None
        assert len(result.boxes) == 2
        assert result.names[0] == 'person'

    def test_get_detection_summary_detect(self, mock_raw_result_detect):
        result = InferenceResult([mock_raw_result_detect], 'test.jpg', 'detect')
        summary = result.get_detection_summary()
        assert '检测到 2 个目标' in summary

    def test_get_detection_summary_classify(self, mock_raw_result_classify):
        result = InferenceResult([mock_raw_result_classify], 'test.jpg', 'classify')
        summary = result.get_detection_summary()
        assert 'Top-5 分类结果' in summary

    def test_get_detection_summary_semantic(self, mock_raw_result_semantic):
        result = InferenceResult([mock_raw_result_semantic], 'test.jpg', 'semantic')
        summary = result.get_detection_summary()
        assert '语义分割完成' in summary
        assert '3 个类别' in summary

    def test_get_detection_summary_depth(self, mock_raw_result_depth):
        result = InferenceResult([mock_raw_result_depth], 'test.jpg', 'depth')
        summary = result.get_detection_summary()
        assert '深度估计完成' in summary
        assert '平均' in summary

    def test_get_detection_summary_pose(self, mock_raw_result_pose):
        result = InferenceResult([mock_raw_result_pose], 'test.jpg', 'pose')
        summary = result.get_detection_summary()
        assert '人体姿态' in summary

    def test_get_detection_summary_obb(self, mock_raw_result_obb):
        result = InferenceResult([mock_raw_result_obb], 'test.jpg', 'obb')
        summary = result.get_detection_summary()
        assert '旋转目标' in summary

    def test_get_detection_summary_empty(self):
        mock_result = MagicMock()
        mock_result.names = {}
        mock_result.boxes = None
        mock_result.probs = None
        mock_result.keypoints = None
        mock_result.obb = None
        mock_result.semantic_mask = None
        mock_result.depth = None
        mock_result.orig_shape = (64, 64)
        result = InferenceResult([mock_result], 'test.jpg', 'detect')
        summary = result.get_detection_summary()
        assert '未检测到' in summary


if __name__ == '__main__':
    unittest.main()
