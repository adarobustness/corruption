# -*- coding: utf-8 -*-

"""Util functions."""

import logging
import os
from PIL import Image
# import accimage
import torch
from tqdm import tqdm
import torch.utils.data as data
import collections
from corruption_methods import *

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    #class_to_idx = {classes[i]: i for i in range(len(classes))}
    class_to_idx = {classes[i]: int(classes[i]) for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            # logger.debug("fname: {}".format(fname))
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, "NO_CLASS_IDX")
                images.append(item)

    # logger.debug("dir: {}".format(dir))
    # for target in sorted(os.listdir(dir)):
    #     d = os.path.join(dir, target)
    #     #if not os.path.isdir(d):
    #     if os.path.isdir(d):
    #         continue
    #     # logger.debug("d: {}".format(d))
    #     for root, _, fnames in sorted(os.walk(d)):
    #         for fname in sorted(fnames):
    #             logger.debug("fname: {}".format(fname))
    #             if is_image_file(fname):
    #                 path = os.path.join(root, fname)
    #                 if class_to_idx is not None:
    #                     item = (path, class_to_idx[target])
    #                 else:
    #                     item = (path, "NO_CLASS_IDX")
    #                 images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# def accimage_loader(path):
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def default_loader(path):
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    return pil_loader(path)


class DistortImageFolder(data.Dataset):
    def __init__(
            self,
            root,
            method,
            severity,
            corrupted_root_path,
            with_class_info=False,
            transform=None,
            target_transform=None,
            loader=default_loader
    ):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx if with_class_info else None)
        if len(imgs) == 0:
            raise (
                RuntimeError(f"Found 0 images in subfolders of: {root} \n"
                             f"Supported image extensions are: {IMG_EXTENSIONS}")
            )

        self.root = root
        self.method = method
        self.severity = severity
        self.imgs = imgs
        if with_class_info:
            self.classes = classes
            self.class_to_idx = class_to_idx
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # create a folder to save corrupted data
        self.corrupted_root_path = corrupted_root_path
        os.makedirs(self.corrupted_root_path, exist_ok=True)

    def generate_corrupted_image_save_path(self, original_image_path):
        image_file_name = original_image_path[original_image_path.rindex('/') + 1:]
        # save_path = os.path.join(self.corrupted_root_path,
        #                          f"{image_file_name}_{self.method.__name__}_{str(self.severity)}.png")
        save_path = os.path.join(self.corrupted_root_path,
                                 f"{image_file_name}")
        return save_path

    def __getitem__(self, index):
        path, target = self.imgs[index]
        ori_size = os.stat(path).st_size
        save_path = self.generate_corrupted_image_save_path(path)
        if os.path.exists(save_path):
            try:
                existing_corr_img = self.loader(save_path)
                logger.info(f"Corrupted image {save_path} exists, skip.")
                return 0
            except Exception as e:
                logger.info(f"Corrupted image {save_path} exists but non-valid, regenerate.")

        img = self.loader(path)
        original_img_shape = img.size
        if self.transform is not None:
            img = self.transform(img)
        try:
            # corruption here
            img = self.method(img, self.severity)
        except Exception as e:
            logger.error(f"Error in during processing file {path} "
                         f"with method {self.method.__name__} and severity {self.severity}")
            logger.error(f"Original Image shape: {original_img_shape}")
            logger.error(f"Error: {e}")
            return 0
        if self.target_transform is not None:
            target = self.target_transform(target)

        Image.fromarray(np.uint8(img)).save(save_path, quality=85, optimize=True)
        return 0  # we do not care about returning the data

    def __len__(self):
        return len(self.imgs)


def get_method_dict():
    d = collections.OrderedDict()
    d["blank_image"] = blank_image
    d['gaussian_noise'] = gaussian_noise
    d['shot_noise'] = shot_noise
    d['impulse_noise'] = impulse_noise
    d['defocus_blur'] = defocus_blur
    d['glass_blur'] = glass_blur

    # this does not work on SLURM cuz the missing package of ImageMagick
    d['motion_blur'] = motion_blur

    # the following 4 methods cannot be fully applied to all images
    d['zoom_blur'] = zoom_blur
    d["gaussian_blur"] = gaussian_blur
    d['snow'] = snow
    d['frost'] = frost
    d['fog'] = fog

    d['brightness'] = brightness
    d['contrast'] = contrast
    d['elastic_transform'] = elastic_transform
    d['pixelate'] = pixelate
    d['jpeg_compression'] = jpeg_compression

    d['speckle_noise'] = speckle_noise
    d['gaussian_blur'] = gaussian_blur
    d['spatter'] = spatter
    d['saturate'] = saturate
    return d


def save_distorted(
        method,
        image_root_folder="",
        corrupted_path_prefix="",
        corrupted_root_path="",
        severity_begin=1,
        severity_end=5,
        transform=None,
        data_loader_batch_size=100,
        data_loader_num_workers=5
):
    """
    A wrapper function to save distorted images to disk.
    Args:
        method (): the corruption method function
        image_root_folder (): folder containing the original images
        corrupted_path_prefix (): a prefix to the corrupted images folder name
        corrupted_root_path (): folder path to store the corrupted images, if not specified, it will be
            generated automatically located in the same parent folder of the original images.
            For example, if `image_root_folder` is `/home/user/images`, and corrupted_path_prefix is `corrupted_images`,
            method is `gaussian_blur`, and severity is 1, then the corrupted images will be saved to
            `/home/user/corrupted_images_gaussian_blur_1`
        severity_begin (int): lower bound of severity
        severity_end (int): upper bound of severity which is inclusive
        transform (): the transform function to apply to the original images if needed
        data_loader_batch_size (int): batch size for the data loader
        data_loader_num_workers (int): number of workers for the data loader

    Returns:
        None. The corrupted images will be saved to disk.

    """
    for severity in tqdm(range(severity_begin, severity_end+1), desc="severity bar"):
        if corrupted_root_path == "" or corrupted_root_path is None:
            if corrupted_path_prefix == "" or corrupted_path_prefix is None:
                corrupted_path_prefix = image_root_folder[image_root_folder.rindex('/') + 1:]
                logger.debug(f"corrupted_path_prefix is not specified, using {corrupted_path_prefix} as the prefix")
            logger.debug(f" severity is {severity}")
            saving_path = os.path.join(os.path.dirname(image_root_folder),
                                               f"{corrupted_path_prefix}_{method.__name__}_{severity}")
        else:
            saving_path = corrupted_root_path
        logger.info(f"Corrupted images will be saved to {saving_path}")
        distorted_dataset = DistortImageFolder(
            root=image_root_folder,
            method=method,
            severity=severity,
            corrupted_root_path=saving_path,
            transform=transform
            )
        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset,
            batch_size=data_loader_batch_size,
            shuffle=False,
            num_workers=data_loader_num_workers
        )
        for _ in tqdm(distorted_dataset_loader, desc="corrupted images"):
            continue


def corrupt_images(image_paths, method, severity: int, save_path: str = ""):
    """
    A wrapper function to corrupt images directly.
    Mainly used for debugging.
    Args:
        image_paths (): a list of image paths or a single image path
        method (): corruption method function or the name of the function
        severity ():
        save_path (): path to save the corrupted images, if not specified, the corrupted images will be saved to the
            same folder as the original images with the name of `original_image_name_method_severity`

    Returns:

    """
    if type(image_paths) is not list:
        image_paths = [image_paths]
    if type(method) is str:
        method = get_method_dict()[method]

    for image_path in image_paths:
        img_full_name = image_path[image_path.rindex('/') + 1:].split('.')
        img_file_name = img_full_name[0]
        img_extension = img_full_name[1]
        img = default_loader(image_path)
        original_img_shape = img.size
        logger.info(f"Original Image shape: {original_img_shape}")
        try:
            # corruption here
            img = method(img, severity)
            # numpy.ndarray
            # logger.debug(f"img type after method {type(img)}")
        except Exception as e:
            logger.error(f"Error in during processing file {image_path} "
                         f"with method {method.__name__} and severity {severity}")
            logger.error(f"Original Image shape: {original_img_shape}")
            logger.error(f"Error: {e}")
            return 0
        if save_path == "":
            corrupted_img_path = os.path.join(os.path.dirname(image_path), f"{img_file_name}_{method.__name__}_{severity}.{img_extension}")
        else:
            corrupted_img_path = os.path.join(save_path, f"{img_file_name}_{method.__name__}_{severity}.{img_extension}")
        logger.info(f"Corrupted image ({image_path}) will be saved to {corrupted_img_path}")
        Image.fromarray(np.uint8(img)).save(corrupted_img_path, quality=85, optimize=True)


