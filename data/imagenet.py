from torchvision.datasets.folder import DatasetFolder, default_loader
import torchvision
import os
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
from typing import Callable, cast, Tuple
from packaging import version

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    return filename.lower().endswith(extensions)

def make_subsampled_dataset(
        directory, class_to_idx, extensions=None,is_valid_file=None, sampling_ratio=1., nb_classes=None):

    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for i, target_class in enumerate(sorted(class_to_idx.keys())):
        if nb_classes is not None and i>=nb_classes:
            break
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        num_imgs = int(len(os.listdir(target_dir))*sampling_ratio)
        imgs=0
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if imgs==num_imgs :
                    break
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
                    imgs+=1
    return instances


class SubsampledDatasetFolder(DatasetFolder):

    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None, sampling_ratio=1., nb_classes=None):

        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        classes, class_to_idx = self._find_classes(self.root) if version.parse(torchvision.__version__) < version.parse("0.10.0") else self.find_classes(self.root)
        samples = make_subsampled_dataset(self.root, class_to_idx, extensions, is_valid_file, sampling_ratio=sampling_ratio, nb_classes=nb_classes)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]



class ImageNetDataset(SubsampledDatasetFolder):
    def __init__(self, root, loader=default_loader, is_valid_file=None,  **kwargs):
        super(ImageNetDataset, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                              is_valid_file=is_valid_file, **kwargs)
        self.imgs = self.samples
