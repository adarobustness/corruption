# -*- coding: utf-8 -*-

"""Entrance to generate and save corrupted images."""

import logging
from utils import save_distorted, get_method_dict, corrupt_images
import argparse
import os

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="gaussian_noise", help="corruption method name")
    parser.add_argument("--image_root_folder", type=str, required=True, help="path to the folder storing images")
    parser.add_argument("--corrupted_path_prefix", type=str, default="")
    parser.add_argument("--corrupted_root_path", type=str, default="",
                        help="path to the folder storing corrupted images, if not specified, it will be "
                             "generated automatically using the corrupted_path_prefix "
                             "check `save_distorted` for details")
    parser.add_argument("--severity_begin", type=int, default=1,)
    parser.add_argument("--severity_end", type=int, default=5, help="inclusive")
    parser.add_argument("--batch_size", type=int, default=100,)
    parser.add_argument("--num_workers", type=int, default=5,)
    args = parser.parse_args()
    return args

def corrupt_a_folder():
    args = parse_args()
    method_dict = get_method_dict()
    logger.info(f"args: {args}")
    assert args.method in method_dict.keys(), (
        f"Please specify a method name from {method_dict.keys()}",
        f"Got {args.method}"
    )
    save_distorted(
        method_dict[args.method],
        image_root_folder=args.image_root_folder,
        corrupted_path_prefix=args.corrupted_path_prefix,
        corrupted_root_path=args.corrupted_root_path,
        severity_begin=args.severity_begin,
        severity_end=args.severity_end,
        data_loader_batch_size=args.batch_size,
        data_loader_num_workers=args.num_workers,
    )



if __name__ == "__main__":
    corrupt_a_folder()
