"""
模型导出器 - 支持多种格式导出
基于Ultralytics文档指南
支持ONNX、TensorRT、CoreML、TFLite、OpenVINO等多种格式
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ModelExporter:
    """模型导出器"""

    def __init__(self):
        self.device = None

    # 支持的导出格式
    SUPPORTED_FORMATS = [
        'onnx', 'tensorrt', 'coreml', 'tflite', 'openvino',
        'torchscript', 'saved_model', 'pb', 'paddle', 'ncnn',
        'edgetpu', 'tfjs', 'keras'
    ]

    # 格式描述
    FORMAT_DESCRIPTIONS = {
        'onnx': 'ONNX格式，跨平台通用',
        'tensorrt': 'NVIDIA TensorRT，高性能推理',
        'coreml': 'Apple CoreML，iOS/macOS部署',
        'tflite': 'TensorFlow Lite，移动端部署',
        'openvino': 'Intel OpenVINO，Intel硬件优化',
        'torchscript': 'PyTorch TorchScript格式',
        'saved_model': 'TensorFlow SavedModel格式',
        'pb': 'TensorFlow Frozen Graph格式',
        'paddle': 'PaddlePaddle格式',
        'ncnn': 'ncnn，移动端高性能推理框架',
        'edgetpu': 'Google Edge TPU，嵌入式部署',
        'tfjs': 'TensorFlow.js，Web浏览器部署',
        'keras': 'Keras格式，TensorFlow高级API'
    }

    def export_model(self,
                    model_path: str,
                    output_path: str,
                    export_format: str,
                    data_path: Optional[str] = None,
                    imgsz: int = 640,
                    batch: int = 1,
                    half: bool = False,
                    int8: bool = False,
                    dynamic: bool = False,
                    simplify: bool = True,
                    opset: int = 12,
                    **kwargs) -> str:
        """
        导出模型到指定格式

        Args:
            model_path: 输入模型路径
            output_path: 输出目录
            export_format: 导出格式
            data_path: 数据集路径（用于量化校准）
            imgsz: 图像尺寸
            batch: 批大小
            half: 是否使用FP16
            int8: 是否使用INT8量化
            dynamic: 是否使用动态形状
            simplify: 是否简化模型
            opset: ONNX opset版本
            
        Returns:
            导出文件路径
        """
        logger.info(f"开始导出模型: {model_path}")
        logger.info(f"导出格式: {export_format}, 图像尺寸: {imgsz}")

        # 验证格式
        if export_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的导出格式: {export_format}")

        from ultralytics import YOLO
        # 加载模型
        model = YOLO(model_path)

        # 构建导出参数
        # 注意：ultralytics 8.x 已弃用 half/int8，统一使用 quantize 参数
        # quantize=None -> FP32, quantize=16 -> FP16, quantize=8 -> INT8
        quantize = None
        if int8:
            quantize = 8
        elif half:
            quantize = 16

        export_args = {
            'format': export_format,
            'imgsz': imgsz,
            'batch': batch,
            'dynamic': dynamic,
            'simplify': simplify,
            'opset': opset,
            **kwargs
        }
        if quantize is not None:
            export_args['quantize'] = quantize

        # 添加数据集路径（用于量化）
        if data_path and quantize == 8:
            export_args['data'] = data_path

        # 执行导出
        results = model.export(**export_args)

        # 获取导出文件路径
        if isinstance(results, str):
            export_file_path = results
        elif hasattr(results, 'save_dir'):
            # 从结果对象获取保存目录
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            ext = self._get_format_extension(export_format)
            export_file_path = os.path.join(str(results.save_dir), f"{model_name}{ext}")
        else:
            # 默认路径
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            ext = self._get_format_extension(export_format)
            export_file_path = os.path.join(output_path, f"{model_name}{ext}")

        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)

        # 如果导出文件不在指定输出目录，复制过去
        if os.path.exists(export_file_path) and os.path.dirname(export_file_path) != output_path:
            import shutil
            dest_path = os.path.join(output_path, os.path.basename(export_file_path))
            shutil.copy2(export_file_path, dest_path)
            export_file_path = dest_path

        logger.info(f"模型导出完成，保存到: {export_file_path}")
        return export_file_path


    def validate_export(self, export_path: str, export_format: str) -> Dict[str, Any]:
        """
        验证导出的模型

        Args:
            export_path: 导出文件路径
            export_format: 导出格式
            
        Returns:
            验证结果
        """
        logger.info(f"开始验证导出模型: {export_path}")

        results = {
            'valid': False,
            'format': export_format,
            'file_size_mb': 0,
            'errors': [],
            'warnings': []
        }

        # 检查文件是否存在
        if not os.path.exists(export_path):
            results['errors'].append("导出文件不存在")
            return results

        # 检查文件大小
        file_size = os.path.getsize(export_path)
        results['file_size_mb'] = file_size / (1024 * 1024)

        # 根据格式进行验证
        if export_format == 'onnx':
            results.update(self._validate_onnx(export_path))
        elif export_format == 'tflite':
            results.update(self._validate_tflite(export_path))
        elif export_format == 'openvino':
            results.update(self._validate_openvino(export_path))
        else:
            # 通用验证：文件或目录存在且非空即视为有效
            if os.path.isdir(export_path):
                entries = os.listdir(export_path)
                results['valid'] = len(entries) > 0
                if not results['valid']:
                    results['errors'].append("导出目录为空")
            else:
                results['valid'] = file_size > 0
                if not results['valid']:
                    results['errors'].append("导出文件为空")

        logger.info(f"模型验证完成: {'有效' if results['valid'] else '无效'}")
        return results

    def _validate_onnx(self, onnx_path: str) -> Dict[str, Any]:
        """验证ONNX模型"""
        try:
            import onnx
            model = onnx.load(onnx_path)
            
            # 检查模型格式
            onnx.checker.check_model(model)
            
            # 获取模型信息
            graph = model.graph
            input_names = [node.name for node in graph.input]
            output_names = [node.name for node in graph.output]
            
            return {
                'valid': True,
                'input_names': input_names,
                'output_names': output_names,
                'num_nodes': len(graph.node)
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)]
            }

    def _validate_tflite(self, tflite_path: str) -> Dict[str, Any]:
        """验证TFLite模型"""
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            # 获取输入输出信息
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            return {
                'valid': True,
                'input_shape': input_details[0]['shape'].tolist() if input_details else [],
                'output_shape': output_details[0]['shape'].tolist() if output_details else []
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)]
            }

    def _validate_openvino(self, openvino_dir: str) -> Dict[str, Any]:
        """验证OpenVINO模型"""
        try:
            # 检查是否存在XML和BIN文件
            xml_path = os.path.join(openvino_dir, 'model.xml')
            bin_path = os.path.join(openvino_dir, 'model.bin')
            
            if os.path.exists(xml_path) and os.path.exists(bin_path):
                return {
                    'valid': True,
                    'xml_size_mb': os.path.getsize(xml_path) / (1024 * 1024),
                    'bin_size_mb': os.path.getsize(bin_path) / (1024 * 1024)
                }
            else:
                return {
                    'valid': False,
                    'errors': ['缺少XML或BIN文件']
                }
        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)]
            }

    def _get_format_extension(self, export_format: str) -> str:
        """获取格式对应的文件扩展名"""
        extensions = {
            'onnx': '.onnx',
            'tensorrt': '.engine',
            'coreml': '.mlpackage',
            'tflite': '.tflite',
            'openvino': '',  # 目录
            'torchscript': '.pt',
            'saved_model': '',  # 目录
            'pb': '.pb',
            'paddle': '.pdmodel',
            'ncnn': '',  # 目录
            'edgetpu': '_edgetpu.tflite',
            'tfjs': '',  # 目录
            'keras': '.keras'
        }
        return extensions.get(export_format, '')

    def get_supported_formats(self) -> List[str]:
        """获取所有支持的导出格式"""
        return self.SUPPORTED_FORMATS

    def get_format_description(self, export_format: str) -> str:
        """获取格式描述"""
        return self.FORMAT_DESCRIPTIONS.get(export_format, '')

    def get_platform_requirements(self, target_platform: str) -> Dict[str, Any]:
        """
        获取目标平台的依赖要求

        Args:
            target_platform: 目标平台
            
        Returns:
            依赖要求字典
        """
        requirements = {
            'onnx': {
                'description': 'ONNX Runtime',
                'pip_packages': ['onnxruntime'],
                'notes': '跨平台通用格式'
            },
            'tensorrt': {
                'description': 'NVIDIA TensorRT',
                'pip_packages': ['tensorrt'],
                'notes': '需要NVIDIA GPU和CUDA环境'
            },
            'coreml': {
                'description': 'Apple CoreML',
                'pip_packages': ['coremltools'],
                'notes': '仅支持macOS系统'
            },
            'tflite': {
                'description': 'TensorFlow Lite',
                'pip_packages': ['tensorflow'],
                'notes': '适用于移动端部署'
            },
            'openvino': {
                'description': 'Intel OpenVINO',
                'pip_packages': ['openvino'],
                'notes': '适用于Intel硬件优化'
            },
            'ncnn': {
                'description': 'ncnn',
                'pip_packages': [],
                'notes': '需要编译安装，适用于移动端'
            },
            'edgetpu': {
                'description': 'Google Edge TPU',
                'pip_packages': ['tflite-runtime'],
                'notes': '需要Edge TPU硬件'
            }
        }
        return requirements.get(target_platform, {})
