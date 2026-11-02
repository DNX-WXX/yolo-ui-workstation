"""
pytest 共享 fixtures
"""
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def qapp():
    """确保 PyQt5 QApplication 存在（绘制跟踪结果需要）"""
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def tmp_dir(tmp_path):
    """临时目录（字符串路径）"""
    return str(tmp_path)


@pytest.fixture
def test_image_path(tmp_path):
    """生成测试用图像文件 (64x64 RGB)"""
    from PIL import Image
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img_path = tmp_path / "test_image.jpg"
    img.save(str(img_path))
    return str(img_path)


@pytest.fixture
def mock_yolo_model():
    """Mock 的 YOLO 模型对象"""
    model = MagicMock()
    model.names = {0: 'person', 1: 'car', 2: 'dog'}
    return model


def _make_mock_box(cls_id=0, conf=0.9, track_id=None):
    box = MagicMock()
    box.cls = cls_id
    conf_mock = MagicMock()
    conf_mock.item.return_value = conf
    box.conf = conf_mock
    box.xyxy = [torch.tensor([10.0, 10.0, 50.0, 50.0])]
    box.id = track_id
    return box


@pytest.fixture
def mock_raw_result_detect():
    """Mock detect 任务的原始结果"""
    result = MagicMock()
    result.names = {0: 'person', 1: 'car'}
    result.orig_shape = (64, 64)
    result.boxes = [_make_mock_box(0, 0.9), _make_mock_box(1, 0.8)]
    result.masks = None
    result.probs = None
    result.keypoints = None
    result.obb = None
    result.semantic_mask = None
    result.depth = None
    return result


@pytest.fixture
def mock_raw_result_semantic():
    """Mock semantic 任务的原始结果"""
    result = MagicMock()
    result.names = {0: 'sky', 1: 'road', 2: 'person'}
    result.orig_shape = (64, 64)
    result.boxes = None
    result.masks = None
    result.probs = None
    result.keypoints = None
    result.obb = None
    sem = MagicMock()
    sem.data = torch.tensor([[
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ]])
    result.semantic_mask = sem
    result.depth = None
    return result


@pytest.fixture
def mock_raw_result_depth():
    """Mock depth 任务的原始结果"""
    result = MagicMock()
    result.names = {0: 'depth'}
    result.orig_shape = (64, 64)
    result.boxes = None
    result.masks = None
    result.probs = None
    result.keypoints = None
    result.obb = None
    result.semantic_mask = None
    depth = MagicMock()
    depth.data = torch.tensor([[
        [0.5, 1.2, 3.0],
        [0.1, 2.5, 4.0],
        [0.0, 0.8, 1.5],
    ]])
    result.depth = depth
    return result


@pytest.fixture
def mock_raw_result_classify():
    """Mock classify 任务的原始结果"""
    result = MagicMock()
    result.names = {0: 'cat', 1: 'dog', 2: 'bird'}
    result.orig_shape = (64, 64)
    result.boxes = None
    result.masks = None
    probs = MagicMock()
    probs.data = torch.tensor([0.1, 0.7, 0.2])
    result.probs = probs
    result.keypoints = None
    result.obb = None
    result.semantic_mask = None
    result.depth = None
    return result


@pytest.fixture
def mock_raw_result_pose():
    """Mock pose 任务的原始结果"""
    result = MagicMock()
    result.names = {0: 'person'}
    result.orig_shape = (64, 64)
    result.boxes = None
    result.masks = None
    result.probs = None
    kp = MagicMock()
    kp.data = torch.zeros((1, 17, 3))
    result.keypoints = [kp]
    result.obb = None
    result.semantic_mask = None
    result.depth = None
    return result


@pytest.fixture
def mock_raw_result_obb():
    """Mock obb 任务的原始结果"""
    result = MagicMock()
    result.names = {0: 'ship'}
    result.orig_shape = (64, 64)
    result.boxes = None
    result.masks = None
    result.probs = None
    result.keypoints = None
    result.obb = [MagicMock()]
    result.semantic_mask = None
    result.depth = None
    return result
