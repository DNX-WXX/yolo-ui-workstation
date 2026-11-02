"""
数据集管理标签页 - 整合数据集转换、划分和格式转换功能
"""
import os
import hashlib
import shutil
from pathlib import Path
from typing import Optional, List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QTextEdit, QFileDialog,
    QMessageBox, QGroupBox, QLabel, QComboBox,
    QRadioButton, QSpinBox,
    QScrollArea, QGridLayout, QSplitter, QTabWidget,
    QProgressBar, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap

# PIL.Image 采用惰性导入（在使用图像操作的方法内），减少启动耗时
# from PIL import Image  ← 移到方法内

from utils.logger import LogManager, QTextEditLogger
from ui.widgets import DropLineEdit


class PreprocessWorker(QThread):
    """图片预处理工作线程"""

    finished = pyqtSignal(int, int, int)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, input_dir: str, output_dir: str, target_width: int, target_height: int,
                 keep_ratio: bool, resize_mode: str, remove_duplicates: bool,
                 remove_corrupted: bool, skip_exist: bool):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_width = target_width
        self.target_height = target_height
        self.keep_ratio = keep_ratio
        self.resize_mode = resize_mode
        self.remove_duplicates = remove_duplicates
        self.remove_corrupted = remove_corrupted
        self.skip_exist = skip_exist
        self.is_stopped = False

    def run(self):
        try:
            from pathlib import Path
            from PIL import Image

            supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')

            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if self.remove_duplicates:
                self.remove_duplicate_images(self.input_dir, supported_formats)

            if self.remove_corrupted:
                self.remove_corrupted_images(self.input_dir, supported_formats)

            image_files = []
            input_path = Path(self.input_dir)
            for file_path in input_path.iterdir():
                if self.is_stopped:
                    return
                if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                    image_files.append(file_path)

            if not image_files:
                self.log.emit("未找到支持的有效图片文件")
                self.finished.emit(0, 0, 0)
                return

            self.log.emit(f"找到 {len(image_files)} 个有效图片文件，开始调整尺寸...")

            success_count = 0
            error_count = 0
            skip_count = 0

            total = len(image_files)
            for i, file_path in enumerate(image_files):
                if self.is_stopped:
                    return

                try:
                    name_without_ext = file_path.stem
                    ext = file_path.suffix
                    output_file_name = f"{name_without_ext}_resized{ext}"
                    output_file_path = output_path / output_file_name

                    if self.skip_exist and output_file_path.exists():
                        skip_count += 1
                        self.log.emit(f"跳过已存在: {file_path.name}")
                        continue

                    with Image.open(file_path) as img:
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                        elif img.mode == 'L':
                            img = img.convert('RGB')

                        if self.keep_ratio:
                            resized_img = self.resize_with_keep_ratio(
                                img, self.target_width, self.target_height, self.resize_mode
                            )
                        else:
                            resized_img = img.resize(
                                (self.target_width, self.target_height),
                                Image.Resampling.LANCZOS
                            )

                        if ext.lower() in ('.jpg', '.jpeg'):
                            resized_img.save(output_file_path, 'JPEG', quality=95, optimize=True, progressive=True)
                        elif ext.lower() == '.png':
                            resized_img.save(output_file_path, 'PNG', compress_level=3)
                        elif ext.lower() == '.webp':
                            resized_img.save(output_file_path, 'WEBP', quality=90)
                        else:
                            resized_img.save(output_file_path)

                        success_count += 1
                        self.log.emit(f"成功: {file_path.name} ({img.size[0]}x{img.size[1]} → {self.target_width}x{self.target_height})")

                except Exception as e:
                    error_count += 1
                    self.log.emit(f"失败: {file_path.name} - {str(e)}")

                progress = int((i + 1) / total * 100)
                self.progress.emit(progress)

            self.finished.emit(success_count, error_count, skip_count)

        except Exception as e:
            if not self.is_stopped:
                self.error.emit(str(e))

    def stop(self):
        self.is_stopped = True

    def get_file_md5(self, file_path):
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                buf = f.read(65536)
                while buf and not self.is_stopped:
                    hasher.update(buf)
                    buf = f.read(65536)
            return hasher.hexdigest()
        except Exception as e:
            self.log.emit(f"计算文件MD5失败: {file_path.name} - {str(e)}")
            return None

    def backup_file(self, file_path):
        backup_root_dir = Path(self.input_dir).parent / "image_backup"
        backup_dir = backup_root_dir / Path(file_path).relative_to(Path(self.input_dir).parent)
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(file_path, backup_dir)
            self.log.emit(f"已备份: {file_path.name} -> {backup_dir}")
        except Exception as e:
            self.log.emit(f"备份失败: {file_path.name} - {str(e)}")

    def remove_duplicate_images(self, input_folder, supported_formats):
        self.log.emit("开始检测并删除重复文件...")
        md5_map = {}
        duplicate_count = 0

        input_path = Path(input_folder)
        for file_path in input_path.iterdir():
            if self.is_stopped:
                return
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                file_md5 = self.get_file_md5(file_path)
                if not file_md5:
                    continue

                if file_md5 in md5_map:
                    self.backup_file(file_path)
                    try:
                        file_path.unlink()
                        duplicate_count += 1
                        self.log.emit(f"已删除重复文件: {file_path.name}")
                    except Exception as e:
                        self.log.emit(f"删除重复文件失败: {file_path.name} - {str(e)}")
                else:
                    md5_map[file_md5] = file_path

        self.log.emit(f"重复文件检测完成：删除了 {duplicate_count} 个重复文件")

    def remove_corrupted_images(self, input_folder, supported_formats):
        from PIL import Image
        self.log.emit("开始检测并删除损坏文件...")
        corrupted_count = 0

        input_path = Path(input_folder)
        for file_path in input_path.iterdir():
            if self.is_stopped:
                return
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                is_corrupted = False

                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        crop_w = min(100, width)
                        crop_h = min(100, height)
                        img_crop = img.crop((0, 0, crop_w, crop_h))
                        img_crop.load()
                except (IOError, SyntaxError, ValueError):
                    is_corrupted = True
                except Exception as e:
                    self.log.emit(f"轻微异常，保留文件: {file_path.name} - {str(e)}")

                if is_corrupted:
                    self.backup_file(file_path)
                    try:
                        file_path.unlink()
                        corrupted_count += 1
                        self.log.emit(f"已删除损坏文件: {file_path.name}")
                    except Exception as e:
                        self.log.emit(f"删除损坏文件失败: {file_path.name} - {str(e)}")

        self.log.emit(f"损坏文件检测完成：删除了 {corrupted_count} 个损坏文件")

    def resize_with_keep_ratio(self, img, target_width, target_height, mode='pad'):
        from PIL import Image
        original_w, original_h = img.size

        if mode == 'pad':
            scale = min(target_width / original_w, target_height / original_h)
        else:
            scale = max(target_width / original_w, target_height / original_h)

        new_w = int(round(original_w * scale))
        new_h = int(round(original_h * scale))

        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if mode == 'pad':
            padded_img = Image.new(img.mode, (target_width, target_height), (0, 0, 0))
            x_offset = (target_width - new_w) // 2
            y_offset = (target_height - new_h) // 2
            padded_img.paste(resized_img, (x_offset, y_offset))
            return padded_img
        else:
            x_start = (new_w - target_width) // 2
            y_start = (new_h - target_height) // 2
            x_end = x_start + target_width
            y_end = y_start + target_height
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(new_w, x_end)
            y_end = min(new_h, y_end)
            cropped_img = resized_img.crop((x_start, y_start, x_end, y_end))
            return cropped_img


class DatasetConvertWorker(QThread):
    """基于ultralytics官方转换器的数据集转换工作线程"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, task_type, source_format,
                 train_ratio, val_ratio, test_ratio, **kwargs):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.task_type = task_type
        self.source_format = source_format
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.kwargs = kwargs
        self.is_stopped = False

    def run(self):
        try:
            if self.task_type == 'classify':
                self._split_classify_dataset()
            elif self.source_format == 'YOLO':
                self._split_yolo_dataset()
            elif self.source_format == 'COCO':
                self._convert_coco_format()
            elif self.source_format == 'VOC':
                self._convert_voc_format()
        except Exception as e:
            if not self.is_stopped:
                self.error.emit(str(e))

    def _split_classify_dataset(self):
        import shutil
        import random
        from PIL import Image

        input_path = Path(self.input_dir)
        output_path = Path(self.output_dir)

        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        def is_image_file(file_path):
            # 优先按扩展名判断；仅扩展名不在常见列表时才用 Image.open 探测，
            # 避免对全量文件做打开操作导致大目录下扫描极慢
            if file_path.suffix.lower() in image_exts:
                return True
            if file_path.suffix:
                return False
            try:
                with Image.open(file_path):
                    return True
            except Exception:
                return False

        classes = []
        # 已按 train/val/test 划分好的数据集：类别从 train/（或首个 split）下读取，
        # 直接把各 split 复制到输出目录，不再按用户比例重新打乱
        split_names = ('train', 'val', 'test')
        existing_splits = [d for d in input_path.iterdir() if d.is_dir() and d.name in split_names]
        if existing_splits:
            scan_dir = input_path / 'train' if (input_path / 'train').is_dir() else existing_splits[0]
            for d in scan_dir.iterdir():
                if d.is_dir():
                    has_images = False
                    for f in d.rglob("*"):
                        if f.is_file() and is_image_file(f):
                            has_images = True
                            break
                    if has_images:
                        classes.append(d.name)

            if classes:
                self.log.emit(f"检测到已划分的数据集 (splits: {', '.join(d.name for d in existing_splits)})，直接复制不重新划分")
                self.log.emit(f"检测到 {len(classes)} 个类别: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")

                total_copied = 0
                for split_dir in existing_splits:
                    if self.is_stopped:
                        self.log.emit("转换已停止")
                        return
                    for cls in classes:
                        src_cls = split_dir / cls
                        if not src_cls.is_dir():
                            continue
                        dst_cls = output_path / split_dir.name / cls
                        os.makedirs(dst_cls, exist_ok=True)
                        n = 0
                        for f in src_cls.rglob("*"):
                            if f.is_file() and is_image_file(f):
                                shutil.copy2(f, dst_cls / f.name)
                                n += 1
                        if n:
                            total_copied += n
                            self.log.emit(f"{split_dir.name}/{cls}: {n}张")

                self.log.emit(f"分类数据集复制完成: 共 {total_copied} 张")
                self.finished.emit(str(output_path))
                return

        for d in input_path.iterdir():
            if d.is_dir():
                has_images = False
                for f in d.rglob("*"):
                    if f.is_file() and is_image_file(f):
                        has_images = True
                        break
                if has_images:
                    classes.append(d.name)

        if not classes:
            raise Exception("未在数据集中找到包含图片的类别文件夹，请确保每个类别在独立子文件夹中")

        self.log.emit(f"检测到 {len(classes)} 个类别: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")

        for split_name in ['train', 'val', 'test']:
            for cls in classes:
                os.makedirs(output_path / split_name / cls, exist_ok=True)

        total_train = 0
        total_val = 0
        total_test = 0

        for cls in classes:
            if self.is_stopped:
                self.log.emit("转换已停止")
                return
            cls_path = input_path / cls
            image_files = []
            for f in cls_path.rglob("*"):
                if f.is_file() and is_image_file(f):
                    image_files.append(f)

            if not image_files:
                self.log.emit(f"警告: 类别 {cls} 中未找到图像文件")
                continue

            random.shuffle(image_files)
            total = len(image_files)
            train_num = int(total * self.train_ratio)
            val_num = int(total * self.val_ratio)
            test_num = total - train_num - val_num

            self.log.emit(f"类别 {cls}: 共 {total} 张，划分训练集 {train_num}，验证集 {val_num}，测试集 {test_num}")

            train_files = image_files[:train_num]
            val_files = image_files[train_num:train_num + val_num]
            test_files = image_files[train_num + val_num:]

            for f in train_files:
                shutil.copy2(f, output_path / 'train' / cls / f.name)
            for f in val_files:
                shutil.copy2(f, output_path / 'val' / cls / f.name)
            for f in test_files:
                shutil.copy2(f, output_path / 'test' / cls / f.name)

            total_train += train_num
            total_val += val_num
            total_test += test_num

        self.log.emit(f"分类数据集划分完成: 训练集 {total_train}张, 验证集 {total_val}张, 测试集 {total_test}张")
        self.finished.emit(str(output_path))

    def _split_yolo_dataset(self):
        import shutil
        import random

        input_path = Path(self.input_dir)
        output_path = Path(self.output_dir)

        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        def is_image_file(file_path):
            if file_path.suffix.lower() in image_exts:
                return True
            return False

        image_label_pairs = []

        images_dir = input_path / 'images'
        labels_dir = input_path / 'labels'

        if images_dir.is_dir() and labels_dir.is_dir():
            img_files = []
            for f in images_dir.rglob("*"):
                if f.is_file() and is_image_file(f):
                    img_files.append(f)

            for img_path in img_files:
                rel_path = img_path.relative_to(images_dir)
                label_path = labels_dir / rel_path.with_suffix('.txt')
                if label_path.exists():
                    image_label_pairs.append((img_path, label_path, str(rel_path)))

        if not image_label_pairs:
            img_files = []
            for ext in image_exts:
                img_files.extend(list(input_path.glob(f'*{ext}')))
                img_files.extend(list(input_path.glob(f'*{ext.upper()}')))

            for img_path in img_files:
                label_path = input_path / f"{img_path.stem}.txt"
                if label_path.exists():
                    image_label_pairs.append((img_path, label_path, img_path.name))

        if not image_label_pairs:
            all_images = []
            for ext in image_exts:
                all_images.extend(list(input_path.rglob(f'*{ext}')))

            for img_path in all_images:
                label_path = input_path / f"{img_path.stem}.txt"
                if label_path.exists():
                    image_label_pairs.append((img_path, label_path, img_path.name))

        if not image_label_pairs:
            raise Exception("未找到有效的图像-标签对，请确保数据集包含对应的.txt标签文件")

        self.log.emit(f"检测到 {len(image_label_pairs)} 对图像-标签")

        random.shuffle(image_label_pairs)
        total = len(image_label_pairs)
        train_num = int(total * self.train_ratio)
        val_num = int(total * self.val_ratio)
        test_num = total - train_num - val_num

        train_pairs = image_label_pairs[:train_num]
        val_pairs = image_label_pairs[train_num:train_num + val_num]
        test_pairs = image_label_pairs[train_num + val_num:]

        self.log.emit(f"划分: 训练集 {train_num}, 验证集 {val_num}, 测试集 {test_num}")

        for split_name, pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
            for img_src, lbl_src, rel_path in pairs:
                if self.is_stopped:
                    self.log.emit("转换已停止")
                    return
                img_dst_dir = output_path / 'images' / split_name
                img_dst_dir.mkdir(parents=True, exist_ok=True)
                img_dst = img_dst_dir / Path(rel_path).name
                shutil.copy2(img_src, img_dst)

                lbl_rel = Path(rel_path).with_suffix('.txt')
                lbl_dst_dir = output_path / 'labels' / split_name
                lbl_dst_dir.mkdir(parents=True, exist_ok=True)
                lbl_dst = lbl_dst_dir / lbl_rel.name
                shutil.copy2(lbl_src, lbl_dst)

        yaml_path = self._create_dataset_yaml(output_path, [])
        self.log.emit(f"YOLO数据集划分完成: 训练集 {train_num}张, 验证集 {val_num}张, 测试集 {test_num}张")
        self.log.emit("⚠ 注意：data.yaml中的nc(类别数)和names(类别列表)需自行补充")
        self.finished.emit(yaml_path)

    def _convert_coco_format(self):
        import shutil
        from ultralytics.data.converter import convert_coco

        use_segments = (self.task_type == 'segment')
        use_keypoints = (self.task_type == 'pose')

        self.log.emit(f"使用ultralytics官方转换器: convert_coco(cls91to80=False, use_segments={use_segments}, use_keypoints={use_keypoints})")

        ann_dir = os.path.join(self.input_dir, 'annotations')
        if not os.path.exists(ann_dir):
            json_files = list(Path(self.input_dir).rglob('*.json'))
            if json_files:
                ann_dir = str(Path(json_files[0]).parent)
            else:
                raise FileNotFoundError(f"在 {self.input_dir} 中未找到COCO JSON标注文件")

        # ultralytics convert_coco 只输出 labels/<json_stem>/，不拷贝图像，
        # 这里单独建立图像索引并按划分比例把图像和标签一起归整到 train/val/test
        save_dir = os.path.join(self.output_dir, 'converted')

        convert_coco(
            labels_dir=ann_dir,
            save_dir=save_dir,
            cls91to80=False,
            use_segments=use_segments,
            use_keypoints=use_keypoints
        )

        # convert_coco 内部会 increment_path，已存在 converted 时实际写到 converted-2 等
        actual_save_dir = Path(save_dir)
        if not (actual_save_dir / 'labels').is_dir():
            candidates = sorted(
                (p.parent for p in Path(self.output_dir).glob('converted*/labels')),
                key=lambda p: os.path.getmtime(str(p)),
            )
            if candidates:
                actual_save_dir = candidates[-1]

        yaml_path = self._organize_coco_output(actual_save_dir)
        # 清掉中间产物，避免输出目录里多套一层 converted*/
        try:
            shutil.rmtree(actual_save_dir)
        except OSError:
            pass
        self.log.emit(f"COCO转换完成，输出目录: {self.output_dir}")
        self.finished.emit(yaml_path)

    def _convert_voc_format(self):
        import shutil
        import random
        import xml.etree.ElementTree as ET
        from PIL import Image

        input_path = Path(self.input_dir)
        output_path = Path(self.output_dir)

        img_dir = input_path / 'JPEGImages'
        ann_dir = input_path / 'Annotations'

        if not img_dir.exists() or not ann_dir.exists():
            raise Exception("VOC格式数据集应包含JPEGImages和Annotations目录")

        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        # 仅按扩展名判断图片；对每个文件执行 Image.open 会导致全量磁盘IO，大目录下极慢
        image_files = [
            f for f in img_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in image_exts
        ]

        if not image_files:
            raise Exception("未找到图像文件")

        self.log.emit(f"找到 {len(image_files)} 个图像文件")

        for split_name in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                os.makedirs(output_path / split_name / subdir, exist_ok=True)

        classes_set = set()
        all_labels = {}

        for ann_file in ann_dir.glob('*.xml'):
            try:
                root = ET.parse(ann_file).getroot()
                labels = []
                for obj in root.iter('object'):
                    name_el = obj.find('name')
                    cls_name = name_el.text.strip() if name_el is not None and name_el.text else None
                    if not cls_name:
                        continue
                    classes_set.add(cls_name)
                    bbox = obj.find('bndbox')
                    if bbox is None:
                        continue
                    try:
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)
                    except (AttributeError, ValueError):
                        continue
                    labels.append((cls_name, xmin, ymin, xmax, ymax))
                all_labels[ann_file.stem] = labels
            except Exception as e:
                self.log.emit(f"解析 {ann_file.name} 失败: {e}")

        class_list = sorted(classes_set)
        class_to_id = {name: idx for idx, name in enumerate(class_list)}
        self.log.emit(f"检测到 {len(class_list)} 个类别: {class_list}")

        random.shuffle(image_files)
        total = len(image_files)
        train_num = int(total * self.train_ratio)
        val_num = int(total * self.val_ratio)

        train_files = image_files[:train_num]
        val_files = image_files[train_num:train_num + val_num]
        test_files = image_files[train_num + val_num:]

        for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for img_path in files:
                if self.is_stopped:
                    self.log.emit("转换已停止")
                    return
                shutil.copy2(img_path, output_path / split_name / 'images' / img_path.name)

                name_stem = img_path.stem
                if name_stem in all_labels:
                    try:
                        with Image.open(img_path) as img:
                            img_w, img_h = img.size
                    except:
                        img_w, img_h = 640, 640

                    label_lines = []
                    for cls_name, xmin, ymin, xmax, ymax in all_labels[name_stem]:
                        cls_id = class_to_id[cls_name]
                        x_center = (xmin + xmax) / 2 / img_w
                        y_center = (ymin + ymax) / 2 / img_h
                        width = (xmax - xmin) / img_w
                        height = (ymax - ymin) / img_h
                        label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                    txt_path = output_path / split_name / 'labels' / (name_stem + '.txt')
                    with open(txt_path, 'w') as f:
                        f.write('\n'.join(label_lines))

        yaml_path = self._create_dataset_yaml(output_path, class_list)
        self.log.emit(f"VOC转换完成: 训练集 {len(train_files)}张, 验证集 {len(val_files)}张, 测试集 {len(test_files)}张")
        self.finished.emit(yaml_path)

    def _organize_coco_output(self, save_dir):
        """把 convert_coco 生成的 labels/<json_stem>/* 归整为 train/val/test 结构，
        并从输入目录把对应图像一起拷过来（convert_coco 本身不拷贝图像）。"""
        import shutil
        output_path = Path(self.output_dir)
        input_path = Path(self.input_dir)
        save_dir = Path(save_dir)
        labels_root = save_dir / 'labels'

        if not labels_root.is_dir():
            raise Exception("convert_coco 未产生 labels 输出")

        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        # 建图像索引：stem -> 完整路径（输入目录下所有图片，允许 train2014/val2014/images 等任意层级）
        image_index = {}
        for f in input_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in image_exts:
                image_index[f.stem] = f

        # 按 json 子目录推断 split（instances_train2014 -> train，instances_val2014 -> val）
        def infer_split(stem: str) -> str:
            s = stem.lower()
            if 'train' in s:
                return 'train'
            if 'val' in s:
                return 'val'
            if 'test' in s:
                return 'test'
            return ''

        split_counts = {'train': 0, 'val': 0, 'test': 0}
        unmatched = []

        for label_subdir in sorted(labels_root.iterdir()):
            if not label_subdir.is_dir():
                continue
            split = infer_split(label_subdir.name) or 'train'
            label_files = list(label_subdir.glob('*.txt'))

            for lbl in label_files:
                if self.is_stopped:
                    self.log.emit("转换已停止")
                    return
                stem = lbl.stem
                img_src = image_index.get(stem)
                if img_src is None:
                    unmatched.append(stem)
                    continue

                dst_img_dir = output_path / split / 'images'
                dst_lbl_dir = output_path / split / 'labels'
                dst_img_dir.mkdir(parents=True, exist_ok=True)
                dst_lbl_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_src, dst_img_dir / img_src.name)
                shutil.copy2(lbl, dst_lbl_dir / lbl.name)
                split_counts[split] += 1

        if unmatched:
            self.log.emit(f"警告: {len(unmatched)} 个标签找不到对应图像，已跳过（如 {unmatched[:3]}）")

        self.log.emit(
            f"归整完成: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}"
        )

        class_list = self._get_class_list_from_coco()
        yaml_path = self._create_dataset_yaml(output_path, class_list)
        return yaml_path

    def _get_class_list_from_coco(self):
        import json
        ann_dir = os.path.join(self.input_dir, 'annotations')
        if not os.path.exists(ann_dir):
            json_files = list(Path(self.input_dir).rglob('*.json'))
            if json_files:
                ann_dir = self.input_dir
            else:
                return []

        for ann_file in os.listdir(ann_dir):
            if ann_file.endswith('.json'):
                try:
                    with open(os.path.join(ann_dir, ann_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    categories = data.get('categories', [])
                    if categories:
                        return [cat['name'] for cat in sorted(categories, key=lambda c: c['id'])]
                except Exception:
                    continue
        return []

    def _create_dataset_yaml(self, output_path, class_list):
        import yaml

        output_path = Path(output_path)
        abs_path = str(output_path.resolve())

        yaml_data = {'path': abs_path}
        # 只写入实际存在且非空的 split，避免训练时读到空目录报错
        for split in ('train', 'val', 'test'):
            split_img_dir = output_path / 'images' / split
            if split_img_dir.is_dir() and any(split_img_dir.iterdir()):
                yaml_data[split] = f'images/{split}'
        if 'val' not in yaml_data and 'train' in yaml_data:
            # ultralytics 要求必须有 val，缺失时回退到 train
            yaml_data['val'] = yaml_data['train']

        yaml_data['nc'] = len(class_list)
        yaml_data['names'] = class_list

        yaml_path = output_path / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        self.log.emit(f"配置文件已创建: {yaml_path}")
        return str(yaml_path)

    def stop(self):
        self.is_stopped = True


class DatasetManagementTab(QWidget):
    """数据集管理标签页 - 整合数据集转换、划分和格式转换功能"""

    def __init__(
            self,
            logger: Optional[LogManager] = None,
            model_manager=None,
            image_display=None,
            image_label=None
    ):
        super().__init__()

        self.logger = logger or LogManager()
        self.model_manager = model_manager
        self.image_display = image_display
        self.image_label = image_label

        self.worker = None
        self.dataset_path = ""
        self.output_path = ""

        # 存储当前检测到的格式
        self.current_format = 'UNKNOWN'
        self.format_desc = ''

        # 格式检测缓存: {(folder, task): (fmt, desc)}，避免重复递归扫描大目录
        self._format_cache = {}
        # 输入路径防抖定时器：手动输入时避免逐字符触发全盘扫描
        self._format_detect_timer = QTimer(self)
        self._format_detect_timer.setSingleShot(True)
        self._format_detect_timer.setInterval(400)
        self._format_detect_timer.timeout.connect(self.detect_and_update_format)

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        self.tab_widget = QTabWidget()

        self.convert_tab = QWidget()
        self.setup_convert_tab()
        self.tab_widget.addTab(self.convert_tab, "数据集转换")

        self.preprocess_tab = QWidget()
        self.setup_preprocess_tab()
        self.tab_widget.addTab(self.preprocess_tab, "图像预处理")


        main_layout.addWidget(self.tab_widget)


    def is_image_file(self, file_path: Path) -> bool:
        """按扩展名判断是否为图片文件（纯后缀判断，避免逐文件打开的开销）"""
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        return file_path.suffix.lower() in image_exts

    def check_dataset_folder(self, folder_path: str):
        """检查数据集文件夹 - 单次递归扫描同时收集图片和JSON"""
        folder = Path(folder_path)

        # 单次递归扫描，同时收集图片和JSON文件（避免两次rglob大目录）
        image_files = []
        json_files = []
        for file in folder.rglob("*"):
            if not file.is_file():
                continue
            if self.is_image_file(file):
                image_files.append(file)
            elif file.suffix.lower() == '.json':
                json_files.append(file)

        # 自动检测格式（有缓存，路径未变时不会重复扫描）
        self.detect_and_update_format()

        self.logger.info(f"数据集检查: {folder_path}")
        self.logger.info(f"  图像文件: {len(image_files)} 个")
        self.logger.info(f"  JSON文件: {len(json_files)} 个")

        # 检查配对
        json_stems = {f.stem for f in json_files}
        paired = sum(1 for f in image_files if f.stem in json_stems)

        self.logger.info(f"  已配对: {paired} 个")

        if image_files:
            sample_names = [f.name for f in image_files[:5]]
            self.logger.info(f"  图片示例: {', '.join(sample_names)}")

        if paired == 0:
            self.logger.warning("警告：没有找到配对的图像和JSON文件")

    def _detect_dataset_format(self, folder_path: str, task_type: str) -> tuple:
        """检测数据集格式（带缓存，同一路径+任务不重复递归扫描）"""
        folder = Path(folder_path)
        cache_key = (str(folder), task_type)
        if cache_key in self._format_cache:
            return self._format_cache[cache_key]

        result = self._detect_dataset_format_impl(folder, task_type)
        self._format_cache[cache_key] = result
        return result

    def _detect_dataset_format_impl(self, folder: Path, task_type: str) -> tuple:
        """检测数据集格式 - 实际扫描逻辑

        检测顺序很关键：必须先识别有明确标志的格式（YOLO/VOC/COCO），
        再尝试分类文件夹，否则 COCO 的 train2014/ 等图片子目录会被误判为类别文件夹。
        """
        if not folder.exists():
            return 'UNKNOWN', '目录不存在'

        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        def is_image_file(file_path):
            return file_path.suffix.lower() in image_exts

        def count_images(d: Path) -> int:
            n = 0
            for f in d.rglob("*"):
                if f.is_file() and is_image_file(f):
                    n += 1
            return n

        def count_flat_pairs(d: Path) -> int:
            """统计目录内「图像 + 同名.txt」的对数"""
            n = 0
            for f in d.iterdir():
                if f.is_file() and is_image_file(f) and (d / f'{f.stem}.txt').exists():
                    n += 1
            return n

        # 1. YOLO 格式: images/ + labels/(masks/depth)
        images_dir = folder / 'images'
        if images_dir.is_dir():
            for label_name in ('labels', 'masks', 'depth'):
                labels_dir = folder / label_name
                if labels_dir.is_dir():
                    img_count = count_images(images_dir)
                    if label_name == 'labels':
                        label_count = len(list(labels_dir.rglob('*.txt')))
                    else:
                        label_count = count_images(labels_dir)
                    if img_count > 0:
                        return 'YOLO', f'YOLO格式: {img_count}张图像, {label_count}个{label_name}文件'

        # 2. VOC 格式
        jpeg_dir = folder / 'JPEGImages'
        ann_dir = folder / 'Annotations'
        if jpeg_dir.is_dir() and ann_dir.is_dir():
            img_count = count_images(jpeg_dir)
            xml_count = len(list(ann_dir.glob('*.xml')))
            if img_count > 0:
                return 'VOC', f'VOC格式: {img_count}张图像, {xml_count}个XML标注'

        # 3. COCO 格式: annotations/*.json 或任意位置的 COCO JSON
        json_candidates = []
        coco_ann_dir = folder / 'annotations'
        if coco_ann_dir.is_dir():
            json_candidates.extend(sorted(coco_ann_dir.glob('*.json')))
        if not json_candidates:
            json_candidates.extend(sorted(folder.rglob('*.json'))[:5])
        if json_candidates:
            try:
                import json as _json
                for jf in json_candidates[:3]:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = _json.load(f)
                    if 'images' in data and 'annotations' in data and 'categories' in data:
                        return 'COCO', f'COCO格式: {jf.name}'
            except Exception:
                pass
            if coco_ann_dir.is_dir() and json_candidates:
                return 'COCO', f'COCO格式: {len(json_candidates)}个JSON标注文件'

        # 4. 扁平 YOLO: 图像与同名.txt标签混放在同一目录（根目录或某个子目录）
        flat_pairs = count_flat_pairs(folder)
        flat_dir_desc = '根目录'
        if not flat_pairs:
            for sd in sorted(folder.iterdir()):
                if sd.is_dir():
                    n = count_flat_pairs(sd)
                    if n:
                        flat_pairs = n
                        flat_dir_desc = sd.name
                        break
        if flat_pairs:
            return 'YOLO', f'YOLO扁平结构（图像与标签混放在{flat_dir_desc}）: {flat_pairs}对图像-标签'

        # 5. 图像分类文件夹结构
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        if subdirs:
            # 已按 train/val/test 划分好的分类数据集：类别取 train 下的子目录
            split_names = {'train', 'val', 'test'}
            split_dirs = [d for d in subdirs if d.name in split_names]
            scan_root = folder
            if split_dirs:
                train_dir = folder / 'train'
                if train_dir.is_dir():
                    scan_root = train_dir

            valid_classes = []
            for sd in scan_root.iterdir():
                if sd.is_dir():
                    has_images = False
                    for file in sd.rglob("*"):
                        if file.is_file() and is_image_file(file):
                            has_images = True
                            break
                    if has_images:
                        valid_classes.append(sd.name)

            if valid_classes:
                total_images = count_images(folder)
                desc = f"分类文件夹结构: {len(valid_classes)}个类别, 共{total_images}张图片"
                if split_dirs:
                    desc += f" (已划分: {', '.join(d.name for d in split_dirs)})"
                if len(valid_classes) <= 5:
                    desc += f" (类别: {', '.join(valid_classes)})"
                else:
                    desc += f" (包括: {', '.join(valid_classes[:3])}...)"
                return 'CLASS_FOLDER', desc

        # 6. 最后尝试：检测是否有任何图片
        img_files = []
        for file in folder.rglob("*"):
            if file.is_file() and is_image_file(file):
                img_files.append(file)

        if img_files:
            if subdirs:
                return 'UNKNOWN', f'检测到{len(img_files)}张图片，但不在类别子文件夹中。请将图片按类别放入子文件夹'
            else:
                return 'UNKNOWN', f'检测到{len(img_files)}张图片，但格式无法识别。支持: YOLO / VOC / COCO / 分类文件夹'

        return 'UNKNOWN', '未检测到有效数据'

    def detect_and_update_format(self):
        """检测数据集格式并更新UI显示"""
        if not hasattr(self, 'dataset_path_edit') or not self.dataset_path_edit.text():
            return

        folder = self.dataset_path_edit.text()
        if not os.path.exists(folder):
            return

        task = self.task_combo.currentData() if hasattr(self, 'task_combo') else 'detect'
        fmt, desc = self._detect_dataset_format(folder, task)

        # 存储当前格式
        self.current_format = fmt
        self.format_desc = desc

        # 更新格式显示
        if hasattr(self, 'detected_format_label'):
            if fmt == 'UNKNOWN':
                self.detected_format_label.setText(f"⚠ {desc}")
                self.detected_format_label.setStyleSheet("color: #d83b01; padding: 8px; background-color: #fff0f0; border-radius: 4px;")
            else:
                self.detected_format_label.setText(f"✓ 检测到格式: {fmt}\n{desc}")
                self.detected_format_label.setStyleSheet("color: #107c10; padding: 8px; background-color: #f0fff0; border-radius: 4px;")

        # 智能锁定任务类型
        self._auto_lock_task_type(fmt)

        self.logger.info(f"数据集格式检测: {fmt} - {desc}")

    def _auto_lock_task_type(self, fmt: str):
        """根据检测到的格式自动锁定任务类型"""
        if not hasattr(self, 'task_combo'):
            return

        # 保存当前选中的值
        current_data = self.task_combo.currentData()

        if fmt == 'CLASS_FOLDER':
            # 分类文件夹 → 自动切换到图像分类并锁定
            index = self.task_combo.findData('classify')
            if index >= 0:
                self.task_combo.setCurrentIndex(index)
                self.task_combo.setEnabled(False)
                self._update_task_hint(fmt)
                self.logger.info("🔒 检测到分类文件夹，已自动锁定为「图像分类」任务（仅划分）")
                return

        elif fmt == 'YOLO':
            # YOLO格式 → 自动切换到目标检测并锁定
            index = self.task_combo.findData('detect')
            if index >= 0:
                self.task_combo.setCurrentIndex(index)
                self.task_combo.setEnabled(False)
                self._update_task_hint(fmt)
                self.logger.info("🔒 检测到YOLO格式数据集，已自动锁定为「目标检测」任务（仅划分）")
                return

        elif fmt == 'VOC':
            # VOC 只有目标检测标注，直接锁定为检测任务
            index = self.task_combo.findData('detect')
            if index >= 0:
                self.task_combo.setCurrentIndex(index)
                self.task_combo.setEnabled(False)
                self._update_task_hint(fmt)
                self.logger.info("🔒 检测到VOC格式，已自动锁定为「目标检测」任务")
                return

        elif fmt == 'COCO':
            # COCO → 启用任务类型选择，只显示三个选项
            self.task_combo.setEnabled(True)
            self._update_task_combo_options(['detect', 'segment', 'pose'])
            self._update_task_hint(fmt)
            self.logger.info("🔓 检测到COCO格式，请选择转换任务类型")

            # 尝试恢复之前的选择
            if current_data in ['detect', 'segment', 'pose']:
                index = self.task_combo.findData(current_data)
                if index >= 0:
                    self.task_combo.setCurrentIndex(index)
            return

        else:
            # 未知格式 → 启用所有选项
            self.task_combo.setEnabled(True)
            self._update_task_combo_options(['detect', 'segment', 'pose', 'classify'])
            self._update_task_hint(fmt)
            self.logger.info("⚠️ 格式未识别，请手动选择任务类型")

            # 尝试恢复之前的选择
            if current_data:
                index = self.task_combo.findData(current_data)
                if index >= 0:
                    self.task_combo.setCurrentIndex(index)
            return

    def _update_task_combo_options(self, allowed_tasks: List[str]):
        """更新任务类型下拉框的选项"""
        if not hasattr(self, 'task_combo'):
            return

        # 保存当前选中的值
        current_data = self.task_combo.currentData()

        # 清空并重新添加
        self.task_combo.clear()

        task_names = {
            'detect': '目标检测 (detect)',
            'segment': '实例分割 (segment)',
            'pose': '姿态估计 (pose)',
            'classify': '图像分类 (classify)'
        }

        for task in allowed_tasks:
            if task in task_names:
                self.task_combo.addItem(task_names[task], task)

        # 尝试恢复之前的选择
        if current_data in allowed_tasks:
            index = self.task_combo.findData(current_data)
            if index >= 0:
                self.task_combo.setCurrentIndex(index)

    def _update_task_hint(self, fmt: str):
        """根据格式更新任务提示"""
        hints = {
            'CLASS_FOLDER': '📁 图像分类数据集：将按类别自动划分 train/val/test',
            'YOLO': '📦 YOLO格式数据集：保持原格式，自动划分 train/val/test',
            'VOC': '📋 VOC格式数据集：可选择转换为 YOLO 或 COCO 格式',
            'COCO': '📋 COCO格式数据集：可选择转换为 YOLO 或 VOC 格式',
            'UNKNOWN': '❓ 未知格式，请确认数据集结构'
        }

        if hasattr(self, 'detected_format_label'):
            hint = hints.get(fmt, '请选择数据集目录后自动检测格式...')
            # 如果已经有格式信息，附加到提示中
            if self.format_desc:
                hint = f"💡 {hint}\n{self.format_desc}"
            self.detected_format_label.setText(hint)
            self.detected_format_label.setStyleSheet("color: #0066cc; padding: 8px; background-color: #e6f2ff; border-radius: 4px;")

    def on_input_changed(self, text):
        """输入路径改变回调"""
        if hasattr(self, 'same_dir_btn'):
            self.same_dir_btn.setEnabled(bool(text))

        if text and hasattr(self, 'output_path_edit'):
            output_dir = os.path.join(text, "yolo_output")
            self.output_path_edit.setText(output_dir)

        # 自动检测格式并更新任务类型（防抖：手动输入时等停顿后再扫描）
        if text and os.path.exists(text):
            self._format_detect_timer.start()
        else:
            self._format_detect_timer.stop()

        if hasattr(self, 'update_ui_state'):
            self.update_ui_state()

    def update_ui_state(self):
        """更新UI状态"""
        if hasattr(self, 'dataset_path_edit') and hasattr(self, 'output_path_edit'):
            has_input = bool(self.dataset_path_edit.text())
            has_output = bool(self.output_path_edit.text())
            if hasattr(self, 'convert_btn'):
                self.convert_btn.setEnabled(has_input and has_output)
            if hasattr(self, 'preview_btn'):
                # 预览按钮仅在输出目录存在 data.yaml 时启用
                yaml_exists = False
                if has_output:
                    out = self.output_path_edit.text()
                    yaml_exists = os.path.exists(os.path.join(out, "data.yaml"))
                self.preview_btn.setEnabled(yaml_exists)

        if hasattr(self, 'preprocess_input_edit') and hasattr(self, 'preprocess_output_edit'):
            has_pre_input = bool(self.preprocess_input_edit.text())
            has_pre_output = bool(self.preprocess_output_edit.text())
            if hasattr(self, 'start_preprocess_btn'):
                self.start_preprocess_btn.setEnabled(has_pre_input and has_pre_output)


    def browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "选择原始数据集文件夹")
        if folder:
            self.dataset_path_edit.setText(folder)
            self.check_dataset_folder(folder)

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "选择YOLO格式输出文件夹")
        if folder:
            self.output_path_edit.setText(folder)

    def on_output_changed(self, text):
        if hasattr(self, 'update_ui_state'):
            self.update_ui_state()

    def set_output_same_dir(self):
        if hasattr(self, 'dataset_path_edit') and hasattr(self, 'output_path_edit'):
            input_dir = self.dataset_path_edit.text()
            if input_dir:
                output_dir = os.path.join(input_dir, "yolo_output")
                self.output_path_edit.setText(output_dir)

    def on_task_changed(self, index):
        if hasattr(self, 'task_combo'):
            task = self.task_combo.currentData()
            self.logger.info(f"任务类型已切换为: {task}")


    def update_test_ratio(self):
        if hasattr(self, 'train_ratio') and hasattr(self, 'val_ratio') and hasattr(self, 'test_ratio'):
            train = self.train_ratio.value()
            val = self.val_ratio.value()
            test = 100 - train - val
            self.test_ratio.setValue(max(0, test))

    def start_conversion(self):
        if hasattr(self, 'dataset_path_edit') and hasattr(self, 'output_path_edit'):
            input_dir = self.dataset_path_edit.text()
            output_dir = self.output_path_edit.text()

            if not input_dir or not os.path.exists(input_dir):
                QMessageBox.warning(self, "警告", "请选择有效的输入目录")
                return

            if not output_dir:
                QMessageBox.warning(self, "警告", "请选择输出目录")
                return

            # 使用检测到的格式，而不是用户选择的
            task = self.task_combo.currentData()
            source_format, fmt_desc = self._detect_dataset_format(input_dir, task)

            # 根据格式自动调整任务类型
            if source_format == 'CLASS_FOLDER':
                task = 'classify'
                self.logger.info("🔒 检测到分类文件夹，自动使用「图像分类」模式")
            elif source_format == 'YOLO':
                task = 'detect'
                self.logger.info("🔒 检测到YOLO格式，自动使用「目标检测」模式（仅划分）")
            elif source_format == 'VOC':
                task = 'detect'
                self.logger.info("🔒 VOC格式自动使用「目标检测」模式")
            elif source_format == 'COCO':
                # 需要用户选择转换类型
                if task not in ['detect', 'segment', 'pose']:
                    QMessageBox.warning(
                        self, "请选择任务类型",
                        "检测到COCO格式数据集，请选择转换目标类型：\n\n"
                        "• 目标检测 (detect) → 转换为YOLO格式\n"
                        "• 实例分割 (segment) → 转换为YOLO分割格式\n"
                        "• 姿态估计 (pose) → 转换为YOLO姿态格式"
                    )
                    return

            if source_format == 'UNKNOWN':
                QMessageBox.warning(
                    self, "格式检测失败",
                    f"无法识别数据集格式：\n{fmt_desc}\n\n"
                    f"请确保数据集符合以下格式之一：\n"
                    f"• YOLO: images/ + labels/ 目录结构 (含.txt标签)\n"
                    f"• VOC: JPEGImages/ + Annotations/ 目录结构\n"
                    f"• COCO: annotations/*.json 标注文件\n"
                    f"• 分类: 按类别分文件夹存放图像"
                )
                return

            if task == 'segment' and source_format not in ('COCO', 'YOLO'):
                QMessageBox.warning(self, "格式不匹配",
                    f"实例分割任务需要COCO或YOLO格式数据集，当前检测到: {source_format}")
                return
            if task == 'pose' and source_format != 'COCO':
                QMessageBox.warning(self, "格式不匹配",
                    f"姿态估计任务需要COCO格式数据集，当前检测到: {source_format}")
                return
            if task == 'classify' and source_format not in ('CLASS_FOLDER', 'UNKNOWN'):
                QMessageBox.warning(self, "格式不匹配",
                    f"图像分类任务需要文件夹结构数据集，当前检测到: {source_format}")
                return
            if task == 'detect' and source_format not in ('YOLO', 'VOC', 'COCO'):
                QMessageBox.warning(self, "格式不匹配",
                    f"目标检测任务需要YOLO/VOC/COCO格式数据集，当前检测到: {source_format}")
                return

            train_ratio = self.train_ratio.value() / 100.0
            val_ratio = self.val_ratio.value() / 100.0
            test_ratio = self.test_ratio.value() / 100.0

            import random
            random.seed(42)

            if hasattr(self, 'log_display'):
                self.log_display.clear()
            self.logger.info("=" * 60)
            self.logger.info("开始数据集转换")
            self.logger.info(f"输入目录: {input_dir}")
            self.logger.info(f"输出目录: {output_dir}")
            self.logger.info(f"任务类型: {task}")
            self.logger.info(f"自动检测格式: {source_format} ({fmt_desc})")
            self.logger.info(f"划分比例: 训练:{train_ratio:.0%} 验证:{val_ratio:.0%} 测试:{test_ratio:.0%}")
            self.logger.info("=" * 60)

            self.worker = DatasetConvertWorker(
                input_dir=input_dir,
                output_dir=output_dir,
                task_type=task,
                source_format=source_format,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )

            self.worker.finished.connect(self.on_conversion_finished)
            self.worker.error.connect(self.on_conversion_error)
            self.worker.log.connect(self.logger.info)
            if hasattr(self, 'progress_bar'):
                self.worker.progress.connect(self.progress_bar.setValue)

            if hasattr(self, 'convert_btn') and hasattr(self, 'stop_btn') and hasattr(self, 'progress_bar'):
                self.convert_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)

            self.worker.start()

    def stop_conversion(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "确认停止",
                "确定要停止转换吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.stop()
                if hasattr(self, 'stop_btn'):
                    self.stop_btn.setEnabled(False)
                self.logger.info("正在停止转换...")
                # 不阻塞界面：用定时器轮询线程是否退出，超时才强制终止
                self._stop_wait_ms = 0
                timer = QTimer(self)
                timer.setInterval(100)
                timer.timeout.connect(lambda: self._check_conversion_stopped(timer))
                timer.start()

    def _check_conversion_stopped(self, timer):
        """轮询转换线程是否已退出；3秒未退出则强制终止（convert_coco 等阻塞调用无法响应 stop 标记）"""
        self._stop_wait_ms += 100
        if not self.worker.isRunning():
            timer.stop()
            timer.deleteLater()
            self.on_conversion_stopped()
        elif self._stop_wait_ms >= 3000:
            timer.stop()
            timer.deleteLater()
            self.worker.terminate()
            self.worker.wait()
            self.on_conversion_stopped()

    def on_conversion_finished(self, yaml_path: str):
        self.logger.info("=" * 60)
        self.logger.info("数据集转换完成！")
        self.logger.info(f"配置文件: {yaml_path}")
        self.logger.info("=" * 60)

        if hasattr(self, 'convert_btn') and hasattr(self, 'stop_btn') and hasattr(self, 'progress_bar') and hasattr(self, 'preview_btn'):
            self.convert_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.preview_btn.setEnabled(True)

        reply = QMessageBox.question(
            self, "转换完成",
            f"数据集转换完成！\n配置文件: {yaml_path}\n\n是否打开输出目录？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            self.open_output_folder()

        self.worker = None

    def on_conversion_error(self, error_msg: str):
        self.logger.error(f"转换失败: {error_msg}")

        QMessageBox.critical(self, "转换失败", f"错误信息:\n{error_msg}")

        if hasattr(self, 'convert_btn') and hasattr(self, 'stop_btn') and hasattr(self, 'progress_bar'):
            self.convert_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
        self.worker = None

    def on_conversion_stopped(self):
        self.logger.warning("转换已停止")

        if hasattr(self, 'convert_btn') and hasattr(self, 'stop_btn') and hasattr(self, 'progress_bar'):
            self.convert_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
        self.worker = None

    def preview_dataset(self):
        if hasattr(self, 'output_path_edit'):
            output_dir = self.output_path_edit.text()
            if not output_dir or not os.path.exists(output_dir):
                QMessageBox.warning(self, "警告", "输出目录不存在")
                return

            yaml_path = os.path.join(output_dir, "data.yaml")
            if os.path.exists(yaml_path):
                try:
                    import yaml
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    if data is None:
                        data = {}

                    preview_text = "数据集配置:\n"
                    preview_text += f"  路径: {data.get('path', 'N/A')}\n"
                    preview_text += f"  训练集: {data.get('train', 'N/A')}\n"
                    preview_text += f"  验证集: {data.get('val', 'N/A')}\n"
                    preview_text += f"  测试集: {data.get('test', 'N/A')}\n"
                    preview_text += f"  类别数: {data.get('nc', 0)}\n"
                    preview_text += f"  类别列表: {data.get('names', [])}\n"

                    if 'stats' in data:
                        preview_text += "\n统计信息:\n"
                        for key, value in data['stats'].items():
                            preview_text += f"  {key}: {value}\n"

                    QMessageBox.information(self, "数据集预览", preview_text)

                except Exception as e:
                    self.logger.error(f"读取配置文件失败: {e}")
                    QMessageBox.critical(self, "预览失败", f"读取配置文件失败:\n{e}")
            else:
                QMessageBox.warning(
                    self, "无法预览",
                    f"未在输出目录找到 data.yaml。\n请先执行数据集转换。\n输出目录: {output_dir}"
                )

    def open_output_folder(self):
        if hasattr(self, 'output_path_edit'):
            output_dir = self.output_path_edit.text()
            if output_dir and os.path.exists(output_dir):
                import subprocess
                import platform

                if platform.system() == "Windows":
                    os.startfile(output_dir)
                elif platform.system() == "Darwin":
                    subprocess.run(["open", output_dir])
                else:
                    subprocess.run(["xdg-open", output_dir])

    def reset_to_defaults(self):
        reply = QMessageBox.question(
            self, '确认恢复默认值',
            '确定要将所有数据集转换参数恢复为默认值吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                if hasattr(self, 'task_combo'):
                    self.task_combo.setCurrentIndex(0)
                    self.task_combo.setEnabled(True)  # 恢复启用

                if hasattr(self, 'train_ratio') and hasattr(self, 'val_ratio') and hasattr(self, 'test_ratio'):
                    self.train_ratio.setValue(80)
                    self.val_ratio.setValue(10)
                    self.test_ratio.setValue(10)

                if hasattr(self, 'dataset_path_edit') and hasattr(self, 'output_path_edit'):
                    self.dataset_path_edit.clear()
                    self.output_path_edit.clear()

                if hasattr(self, 'detected_format_label'):
                    self.detected_format_label.setText("请选择数据集目录后自动检测格式...")
                    self.detected_format_label.setStyleSheet("color: #666666; padding: 8px;")

                if hasattr(self, 'update_ui_state'):
                    self.update_ui_state()

                self.current_format = 'UNKNOWN'
                self.format_desc = ''

                self.logger.info("已恢复数据集转换默认参数设置")
                QMessageBox.information(self, '恢复成功', '所有参数已恢复为默认值！')

            except Exception as e:
                self.logger.error(f"恢复默认值失败: {e}")
                QMessageBox.warning(self, '恢复失败', f'恢复默认值时发生错误：{e}')

    # ========== 图像预处理相关方法 ==========
    def setup_preprocess_tab(self):
        """设置图像预处理标签页"""
        layout = QVBoxLayout(self.preprocess_tab)
        layout.setSpacing(10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)

        # 输入输出配置
        io_group = QGroupBox("1. 输入输出配置")
        io_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        io_layout = QFormLayout(io_group)

        self.preprocess_input_edit = DropLineEdit(file_filter='folder')
        self.preprocess_input_edit.setPlaceholderText("拖拽图片文件夹到这里，或点击浏览...")
        self.preprocess_input_edit.textChanged.connect(self.update_ui_state)

        browse_input_btn = QPushButton("浏览...")
        browse_input_btn.clicked.connect(self.browse_preprocess_input)
        browse_input_btn.setFixedWidth(80)

        input_hbox = QHBoxLayout()
        input_hbox.addWidget(self.preprocess_input_edit, 1)
        input_hbox.addWidget(browse_input_btn)
        io_layout.addRow("输入文件夹:", input_hbox)

        self.preprocess_output_edit = DropLineEdit(file_filter='folder')
        self.preprocess_output_edit.setPlaceholderText("拖拽输出文件夹到这里，或点击浏览...")
        self.preprocess_output_edit.textChanged.connect(self.update_ui_state)

        browse_output_btn = QPushButton("浏览...")
        browse_output_btn.clicked.connect(self.browse_preprocess_output)
        browse_output_btn.setFixedWidth(80)

        output_hbox = QHBoxLayout()
        output_hbox.addWidget(self.preprocess_output_edit, 1)
        output_hbox.addWidget(browse_output_btn)
        io_layout.addRow("输出文件夹:", output_hbox)

        scroll_layout.addWidget(io_group)

        # 尺寸设置
        size_group = QGroupBox("2. 尺寸设置")
        size_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        size_layout = QFormLayout(size_group)

        size_grid = QGridLayout()
        size_grid.setColumnStretch(0, 0)
        size_grid.setColumnStretch(1, 1)
        size_grid.setColumnStretch(2, 0)
        size_grid.setColumnStretch(3, 1)

        self.target_width_spin = QSpinBox()
        self.target_width_spin.setRange(1, 4096)
        self.target_width_spin.setValue(800)
        self.target_width_spin.setFixedWidth(100)
        size_grid.addWidget(QLabel("目标宽度:"), 0, 0)
        size_grid.addWidget(self.target_width_spin, 0, 1, alignment=Qt.AlignLeft)

        self.target_height_spin = QSpinBox()
        self.target_height_spin.setRange(1, 4096)
        self.target_height_spin.setValue(600)
        self.target_height_spin.setFixedWidth(100)
        size_grid.addWidget(QLabel("目标高度:"), 0, 2)
        size_grid.addWidget(self.target_height_spin, 0, 3, alignment=Qt.AlignLeft)

        size_layout.addRow("", size_grid)

        self.keep_ratio_check = QCheckBox("保持比例")
        self.keep_ratio_check.setChecked(True)
        self.keep_ratio_check.setToolTip("保持图片原始比例")
        size_layout.addRow("", self.keep_ratio_check)

        mode_layout = QHBoxLayout()
        self.pad_rb = QRadioButton("填充")
        self.pad_rb.setChecked(True)
        self.pad_rb.setToolTip("保持比例，使用黑色边框填充到目标尺寸")
        mode_layout.addWidget(self.pad_rb)

        self.crop_rb = QRadioButton("裁剪")
        self.crop_rb.setToolTip("保持比例，裁剪多余部分到目标尺寸")
        mode_layout.addWidget(self.crop_rb)
        size_layout.addRow("处理模式:", mode_layout)

        scroll_layout.addWidget(size_group)

        # 预处理选项
        option_group = QGroupBox("3. 预处理选项")
        option_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        option_layout = QFormLayout(option_group)

        self.remove_duplicates_check = QCheckBox("去除重复图片")
        self.remove_duplicates_check.setChecked(True)
        self.remove_duplicates_check.setToolTip("基于MD5哈希值检测并删除重复图片")
        option_layout.addRow("", self.remove_duplicates_check)

        self.remove_corrupted_check = QCheckBox("删除损坏图片")
        self.remove_corrupted_check.setChecked(True)
        self.remove_corrupted_check.setToolTip("检测并删除完全损坏的图片")
        option_layout.addRow("", self.remove_corrupted_check)

        self.skip_exist_check = QCheckBox("跳过已存在文件")
        self.skip_exist_check.setChecked(True)
        self.skip_exist_check.setToolTip("跳过输出目录中已存在的文件")
        option_layout.addRow("", self.skip_exist_check)

        scroll_layout.addWidget(option_group)

        # 操作按钮
        button_layout = QHBoxLayout()

        self.start_preprocess_btn = QPushButton("开始处理")
        self.start_preprocess_btn.clicked.connect(self.start_preprocess)
        self.start_preprocess_btn.setEnabled(False)
        self.start_preprocess_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        button_layout.addWidget(self.start_preprocess_btn)

        self.stop_preprocess_btn = QPushButton("停止")
        self.stop_preprocess_btn.clicked.connect(self.stop_preprocess)
        self.stop_preprocess_btn.setEnabled(False)
        self.stop_preprocess_btn.setStyleSheet("""
            QPushButton {
                background-color: #d83b01;
                color: white;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #a83200;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        button_layout.addWidget(self.stop_preprocess_btn)

        button_layout.addStretch()
        scroll_layout.addLayout(button_layout)

        # 进度条
        self.preprocess_progress = QProgressBar()
        self.preprocess_progress.setVisible(False)
        scroll_layout.addWidget(self.preprocess_progress)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)

        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #d0d0d0;
                height: 8px;
                border: 1px solid #a0a0a0;
                border-radius: 4px;
            }
            QSplitter::handle:hover {
                background-color: #b0b0b0;
            }
        """)

        splitter.addWidget(scroll_area)

        log_group = QGroupBox("处理日志")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QVBoxLayout(log_group)

        self.preprocess_log_display = QTextEdit()
        self.preprocess_log_display.setReadOnly(True)
        self.preprocess_log_display.setMinimumHeight(100)
        self.preprocess_log_display.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.preprocess_log_display)

        splitter.addWidget(log_group)
        splitter.setSizes([700, 300])

        layout.addWidget(splitter)
        self.update_ui_state()

    def browse_preprocess_input(self):
        folder = QFileDialog.getExistingDirectory(self, "选择要处理的图片文件夹")
        if folder:
            self.preprocess_input_edit.setText(folder)

    def browse_preprocess_output(self):
        folder = QFileDialog.getExistingDirectory(self, "选择处理后的输出文件夹")
        if folder:
            self.preprocess_output_edit.setText(folder)

    def start_preprocess(self):
        input_dir = self.preprocess_input_edit.text()
        output_dir = self.preprocess_output_edit.text()

        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.warning(self, "警告", "请选择有效的输入文件夹")
            return

        if not output_dir:
            QMessageBox.warning(self, "警告", "请选择输出文件夹")
            return

        target_width = self.target_width_spin.value()
        target_height = self.target_height_spin.value()
        keep_ratio = self.keep_ratio_check.isChecked()
        resize_mode = "pad" if self.pad_rb.isChecked() else "crop"
        remove_duplicates = self.remove_duplicates_check.isChecked()
        remove_corrupted = self.remove_corrupted_check.isChecked()
        skip_exist = self.skip_exist_check.isChecked()

        self.preprocess_log_display.clear()
        self.log_preprocess("=" * 60)
        self.log_preprocess("开始图片预处理")
        self.log_preprocess(f"输入文件夹: {input_dir}")
        self.log_preprocess(f"输出文件夹: {output_dir}")
        self.log_preprocess(f"目标尺寸: {target_width}x{target_height}")
        self.log_preprocess(f"保持比例: {'是' if keep_ratio else '否'}")
        self.log_preprocess(f"处理模式: {'填充' if resize_mode == 'pad' else '裁剪'}")
        self.log_preprocess("=" * 60)

        self.worker = PreprocessWorker(
            input_dir=input_dir,
            output_dir=output_dir,
            target_width=target_width,
            target_height=target_height,
            keep_ratio=keep_ratio,
            resize_mode=resize_mode,
            remove_duplicates=remove_duplicates,
            remove_corrupted=remove_corrupted,
            skip_exist=skip_exist
        )

        self.worker.finished.connect(self.on_preprocess_finished)
        self.worker.error.connect(self.on_preprocess_error)
        self.worker.log.connect(self.log_preprocess)
        self.worker.progress.connect(self.preprocess_progress.setValue)

        self.start_preprocess_btn.setEnabled(False)
        self.stop_preprocess_btn.setEnabled(True)
        self.preprocess_progress.setVisible(True)
        self.preprocess_progress.setValue(0)

        self.worker.start()

    def stop_preprocess(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "确认停止",
                "确定要停止处理吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.terminate()
                self.worker.wait()
                self.on_preprocess_stopped()

    def on_preprocess_finished(self, success_count, error_count, skip_count):
        self.log_preprocess("=" * 60)
        self.log_preprocess("图片预处理完成！")
        self.log_preprocess(f"成功处理: {success_count} 个")
        self.log_preprocess(f"处理失败: {error_count} 个")
        self.log_preprocess(f"跳过已存在: {skip_count} 个")
        self.log_preprocess("=" * 60)

        self.start_preprocess_btn.setEnabled(True)
        self.stop_preprocess_btn.setEnabled(False)
        self.preprocess_progress.setVisible(False)

        reply = QMessageBox.question(
            self, "处理完成",
            f"图片预处理完成！\n成功处理: {success_count} 个\n\n是否打开输出目录？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            output_dir = self.preprocess_output_edit.text()
            if output_dir and os.path.exists(output_dir):
                import subprocess
                import platform

                if platform.system() == "Windows":
                    os.startfile(output_dir)
                elif platform.system() == "Darwin":
                    subprocess.run(["open", output_dir])
                else:
                    subprocess.run(["xdg-open", output_dir])

        self.worker = None

    def on_preprocess_error(self, error_msg: str):
        self.log_preprocess(f"错误: {error_msg}")
        QMessageBox.critical(self, "处理失败", f"错误信息:\n{error_msg}")

        self.start_preprocess_btn.setEnabled(True)
        self.stop_preprocess_btn.setEnabled(False)
        self.preprocess_progress.setVisible(False)
        self.worker = None

    def on_preprocess_stopped(self):
        self.log_preprocess("处理已停止")

        self.start_preprocess_btn.setEnabled(True)
        self.stop_preprocess_btn.setEnabled(False)
        self.preprocess_progress.setVisible(False)
        self.worker = None

    def log_preprocess(self, message: str):
        self.preprocess_log_display.append(message)
        self.preprocess_log_display.verticalScrollBar().setValue(
            self.preprocess_log_display.verticalScrollBar().maximum()
        )

    def get_task_name(self, task_type: str) -> str:
        names = {
            "classify": "图像分类",
            "detect": "目标检测",
            "segment": "实例分割",
            "pose": "姿态估计"
        }
        return names.get(task_type, task_type)

    def on_tab_selected(self):
        self.logger.info("数据集管理标签页已激活")

        if hasattr(self, 'image_label') and self.image_label is not None:
            self.image_label.setText("请在这里预览数据集图像")
            self.image_label.setPixmap(QPixmap())

    def setup_convert_tab(self):
        """设置数据集转换标签页"""
        layout = QVBoxLayout(self.convert_tab)
        layout.setSpacing(10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)

        # 输入输出配置
        io_group = QGroupBox("1. 输入输出配置")
        io_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        io_layout = QFormLayout(io_group)

        self.dataset_path_edit = DropLineEdit(self, file_filter='folder')
        self.dataset_path_edit.setPlaceholderText("拖拽数据集文件夹到这里，或点击浏览...")
        self.dataset_path_edit.textChanged.connect(self.on_input_changed)

        browse_dataset_btn = QPushButton("浏览...")
        browse_dataset_btn.clicked.connect(self.browse_dataset)
        browse_dataset_btn.setFixedWidth(80)

        dataset_hbox = QHBoxLayout()
        dataset_hbox.addWidget(self.dataset_path_edit, 1)
        dataset_hbox.addWidget(browse_dataset_btn)
        io_layout.addRow("原始数据集:", dataset_hbox)

        self.output_path_edit = DropLineEdit(self, file_filter='folder')
        self.output_path_edit.setPlaceholderText("拖拽输出目录到这里，或点击浏览...")
        self.output_path_edit.textChanged.connect(self.on_output_changed)

        browse_output_btn = QPushButton("浏览...")
        browse_output_btn.clicked.connect(self.browse_output)
        browse_output_btn.setFixedWidth(80)

        output_hbox = QHBoxLayout()
        output_hbox.addWidget(self.output_path_edit, 1)
        output_hbox.addWidget(browse_output_btn)
        io_layout.addRow("输出路径:", output_hbox)

        quick_hbox = QHBoxLayout()
        quick_hbox.addStretch()

        self.same_dir_btn = QPushButton("使用输入目录下的yolo_output")
        self.same_dir_btn.clicked.connect(self.set_output_same_dir)
        self.same_dir_btn.setEnabled(False)
        quick_hbox.addWidget(self.same_dir_btn)

        io_layout.addRow("", quick_hbox)

        scroll_layout.addWidget(io_group)

        # 任务类型
        task_group = QGroupBox("2. 任务类型")
        task_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        task_layout = QFormLayout(task_group)

        self.task_combo = QComboBox()
        # 初始显示所有选项
        self.task_combo.addItem("目标检测 (detect)", "detect")
        self.task_combo.addItem("实例分割 (segment)", "segment")
        self.task_combo.addItem("姿态估计 (pose)", "pose")
        self.task_combo.addItem("图像分类 (classify)", "classify")
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        task_layout.addRow("任务类型:", self.task_combo)

        scroll_layout.addWidget(task_group)

        # 格式自动检测
        format_hint_group = QGroupBox("3. 格式自动检测")
        format_hint_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        format_hint_layout = QVBoxLayout(format_hint_group)

        self.detected_format_label = QLabel("请选择数据集目录后自动检测格式...")
        self.detected_format_label.setStyleSheet("color: #666666; padding: 8px;")
        self.detected_format_label.setWordWrap(True)
        format_hint_layout.addWidget(self.detected_format_label)

        scroll_layout.addWidget(format_hint_group)

        # 数据集划分
        split_group = QGroupBox("4. 数据集划分")
        split_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        split_layout = QFormLayout(split_group)

        self.train_ratio = QSpinBox()
        self.train_ratio.setRange(0, 100)
        self.train_ratio.setValue(80)
        self.train_ratio.setSuffix("%")
        self.train_ratio.valueChanged.connect(self.update_test_ratio)
        split_layout.addRow("训练集:", self.train_ratio)

        self.val_ratio = QSpinBox()
        self.val_ratio.setRange(0, 100)
        self.val_ratio.setValue(10)
        self.val_ratio.setSuffix("%")
        self.val_ratio.valueChanged.connect(self.update_test_ratio)
        split_layout.addRow("验证集:", self.val_ratio)

        self.test_ratio = QSpinBox()
        self.test_ratio.setRange(0, 100)
        self.test_ratio.setValue(10)
        self.test_ratio.setSuffix("%")
        self.test_ratio.setReadOnly(True)
        split_layout.addRow("测试集:", self.test_ratio)

        ratio_hint = QLabel("提示：测试集比例会自动计算")
        ratio_hint.setStyleSheet("color: #666666; font-size: 11px;")
        split_layout.addRow("", ratio_hint)

        scroll_layout.addWidget(split_group)

        # 操作按钮
        button_layout = QHBoxLayout()

        self.convert_btn = QPushButton("开始转换")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setEnabled(False)
        self.convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        button_layout.addWidget(self.convert_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_conversion)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #d83b01;
                color: white;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #a83200;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        button_layout.addWidget(self.stop_btn)

        self.preview_btn = QPushButton("预览数据集")
        self.preview_btn.clicked.connect(self.preview_dataset)
        self.preview_btn.setEnabled(False)
        self.preview_btn.setStyleSheet("""
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        button_layout.addWidget(self.preview_btn)

        button_layout.addStretch()

        self.reset_btn = QPushButton("恢复默认值")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        button_layout.addWidget(self.reset_btn)

        scroll_layout.addLayout(button_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        scroll_layout.addWidget(self.progress_bar)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)

        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #d0d0d0;
                height: 8px;
                border: 1px solid #a0a0a0;
                border-radius: 4px;
            }
            QSplitter::handle:hover {
                background-color: #b0b0b0;
            }
        """)

        splitter.addWidget(scroll_area)

        log_group = QGroupBox("转换日志")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QVBoxLayout(log_group)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(100)
        self.log_display.setFont(QFont("Consolas", 9))

        log_handler = QTextEditLogger(self.log_display)
        self.logger.add_handler(log_handler)

        log_layout.addWidget(self.log_display)

        splitter.addWidget(log_group)
        splitter.setSizes([700, 300])

        layout.addWidget(splitter)

        self.update_ui_state()
