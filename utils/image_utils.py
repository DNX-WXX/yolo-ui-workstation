"""
图像处理工具 - 图像加载、转换和预处理
"""
from __future__ import annotations

# 重型库（cv2/numpy/PIL）采用惰性导入，减少模块加载时间，加快应用启动
from PyQt5.QtGui import QPixmap, QImage
from typing import Optional


class ImageProcessor:
    """图像处理器 - 负责图像格式转换和加载"""


    @staticmethod
    def cv_to_qpixmap(cv_image) -> QPixmap:
        """
        OpenCV图像转QPixmap

        Args:
            cv_image: OpenCV图像 (BGR格式)

        Returns:
            QPixmap对象
        """
        import cv2
        # BGR转RGB
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv_image

        # 转换为QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimage = QImage(
            rgb_image.data,
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        return QPixmap.fromImage(qimage)

    @staticmethod
    def load_image_safe(image_path: str) -> Optional[QPixmap]:
        """
        安全加载图像，支持多种格式

        Args:
            image_path: 图像路径

        Returns:
            QPixmap对象，失败返回None
        """
        try:
            # 先尝试使用QPixmap直接加载
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                return pixmap

            # 失败则使用OpenCV
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                return ImageProcessor.cv_to_qpixmap(img)

            return None
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return None


class PreprocessPipeline:
    """预处理流水线 - 用于模型推理前的图像预处理"""

    def __init__(self, input_size: int = 640, normalize: bool = False):
        """
        初始化预处理流水线

        Args:
            input_size: 模型输入尺寸
            normalize: 是否进行ImageNet标准化
        """
        self.input_size = input_size
        self.normalize = normalize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def preprocess_for_onnx(self, image_path: str):
        """
        为ONNX推理预处理图像

        Args:
            image_path: 图像路径

        Returns:
            预处理后的numpy数组 (1, C, H, W)
        """
        import numpy as np
        import torchvision.transforms as transforms
        from PIL import Image
        pil_img = Image.open(image_path).convert('RGB')

        # 调整大小（保持宽高比）
        img_w, img_h = pil_img.size
        if img_h < img_w:
            target_h = self.input_size
            target_w = int(round(img_w * (float(self.input_size) / img_h)))
        else:
            target_w = self.input_size
            target_h = int(round(img_h * (float(self.input_size) / img_w)))

        resize = transforms.Resize(
            (target_h, target_w),
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        img_resized = resize(pil_img)

        # 中心裁剪
        crop = transforms.CenterCrop(self.input_size)
        img_cropped = crop(img_resized)

        # 转换为张量
        transforms_list = [transforms.ToTensor()]
        if self.normalize:
            transforms_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        transform = transforms.Compose(transforms_list)
        img_tensor = transform(img_cropped)

        # 添加batch维度并转为numpy
        img_np = img_tensor.unsqueeze(0).cpu().numpy()

        return img_np
