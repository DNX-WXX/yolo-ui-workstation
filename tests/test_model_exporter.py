"""
core/model_exporter.py 单元测试
"""
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_exporter import ModelExporter


class TestModelExporter:
    """ModelExporter 测试类"""

    def setup_method(self):
        self.exporter = ModelExporter()

    def test_supported_formats(self):
        assert 'onnx' in ModelExporter.SUPPORTED_FORMATS
        assert 'tensorrt' in ModelExporter.SUPPORTED_FORMATS
        assert 'tflite' in ModelExporter.SUPPORTED_FORMATS
        assert len(ModelExporter.SUPPORTED_FORMATS) == 13

    def test_format_descriptions(self):
        desc = ModelExporter.FORMAT_DESCRIPTIONS
        assert 'onnx' in desc
        assert '跨平台' in desc['onnx']

    @patch('ultralytics.YOLO')
    def test_export_model(self, mock_yolo):
        mock_model = MagicMock()
        mock_model.export.return_value = '/tmp/model.onnx'
        mock_yolo.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmp:
            out = self.exporter.export_model('model.pt', tmp, 'onnx', imgsz=320)
            assert out.endswith('.onnx')
            mock_model.export.assert_called_once()

    @patch('ultralytics.YOLO')
    def test_export_model_unsupported_format(self, mock_yolo):
        try:
            self.exporter.export_model('model.pt', '/tmp', 'unknown')
            assert False, "应抛出 ValueError"
        except ValueError:
            pass


    def test_validate_export_file_not_exist(self):
        result = self.exporter.validate_export('/nonexistent/model.onnx', 'onnx')
        assert result['valid'] is False
        assert '导出文件不存在' in result['errors']

    def test_validate_export_generic(self, tmp_path):
        f = tmp_path / 'model.bin'
        f.write_bytes(b'12345')
        result = self.exporter.validate_export(str(f), 'pb')
        assert result['valid'] is True
        assert result['file_size_mb'] > 0

    def test_validate_export_empty_file(self, tmp_path):
        f = tmp_path / 'model.empty'
        f.write_bytes(b'')
        result = self.exporter.validate_export(str(f), 'pb')
        assert result['valid'] is False

    def test_get_supported_formats(self):
        fmts = self.exporter.get_supported_formats()
        assert len(fmts) == 13

    def test_get_format_description(self):
        desc = self.exporter.get_format_description('onnx')
        assert 'ONNX' in desc
        assert self.exporter.get_format_description('unknown') == ''

    def test_get_platform_requirements(self):
        req = self.exporter.get_platform_requirements('onnx')
        assert 'onnxruntime' in req.get('pip_packages', [])
        req_empty = self.exporter.get_platform_requirements('unknown')
        assert req_empty == {}

    def test_get_format_extension(self):
        assert self.exporter._get_format_extension('onnx') == '.onnx'
        assert self.exporter._get_format_extension('openvino') == ''
        assert self.exporter._get_format_extension('unknown') == ''
