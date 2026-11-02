"""
core/tracking_engine.py 单元测试
"""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tracking_engine import TrackingEngine, TrackingResult


class TestTrackingResult:
    """TrackingResult 测试类"""

    def test_get_track_summary(self):
        tr = TrackingResult(1, [{'track_id': 1}, {'track_id': 2}], 0.1)
        assert '2 个目标' in tr.get_track_summary()


class TestTrackingEngine:
    """TrackingEngine 测试类"""

    def setup_method(self):
        self.engine = TrackingEngine()

    def test_initialization(self):
        assert self.engine.tracking_history == {}
        assert self.engine.track_stats['total_frames'] == 0

    def test_set_tracker_config(self):
        self.engine.set_tracker_config(confidence=0.5)
        assert self.engine.track_config['confidence'] == 0.5

    def test_track_frame(self):
        import numpy as np
        mock_model = MagicMock()
        mock_box = MagicMock()
        mock_box.id = 1
        mock_box.cls = 0
        mock_box.conf.item.return_value = 0.9
        mock_box.xyxy = [MagicMock()]
        arr = np.array([10.0, 10.0, 50.0, 50.0])
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = arr
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: 'person'}
        mock_result.orig_img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_model.track.return_value = [mock_result]

        result = self.engine.track_frame(mock_model, 'dummy.jpg')
        assert result is not None
        assert len(result.boxes_data) == 1
        assert self.engine.track_stats['total_frames'] == 1

    def test_track_frame_no_boxes(self):
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_result.names = {}
        mock_result.orig_img = MagicMock()
        mock_model.track.return_value = [mock_result]

        result = self.engine.track_frame(mock_model, 'dummy.jpg')
        assert result is not None
        assert len(result.boxes_data) == 0

    def test_draw_tracking_results_no_result(self):
        result = self.engine.draw_tracking_results(None)
        assert result is None

    def test_draw_tracking_results_with_frame_image(self, qapp):
        import numpy as np
        from unittest.mock import patch
        self.engine.tracking_history[1] = __import__('collections').deque(
            [(30, 30, 0), (35, 35, 1)], maxlen=50
        )
        tr = TrackingResult(1, [{
            'track_id': 1,
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.9,
            'bbox': [10, 10, 50, 50]
        }], 0.1)
        tr.frame_image = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch('core.tracking_engine.QPixmap') as mock_qpixmap, \
             patch('core.tracking_engine.QPainter') as mock_qpainter, \
             patch('PyQt5.QtGui.QImage'):
            mock_instance = MagicMock()
            mock_qpixmap.fromImage.return_value = mock_instance
            mock_fm = MagicMock()
            mock_fm.height.return_value = 12
            mock_fm.ascent.return_value = 9
            mock_fm.horizontalAdvance.return_value = 50
            mock_painter = MagicMock()
            mock_painter.fontMetrics.return_value = mock_fm
            mock_qpainter.return_value = mock_painter
            pixmap = self.engine.draw_tracking_results(tr)
            assert pixmap is mock_instance

    def test_get_track_stats(self):
        self.engine.track_stats['total_frames'] = 10
        stats = self.engine.get_track_stats()
        assert stats['total_frames'] == 10


    def test_clear_tracking_history(self):
        self.engine.tracking_history[1] = __import__('collections').deque([(1, 2, 3)])
        self.engine.clear_tracking_history()
        assert self.engine.tracking_history == {}

    def test_get_tracked_objects(self):
        tr = TrackingResult(1, [{
            'track_id': 1,
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.9,
            'bbox': [10, 10, 50, 50]
        }], 0.1)
        self.engine.current_result = tr
        objs = self.engine.get_tracked_objects()
        assert 1 in objs
        assert objs[1]['class_name'] == 'person'
