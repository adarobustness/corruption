# nohup python -u corruption.py > corruption_mlm_suggestion.log 2>&1 &

from tqdm import tqdm
from utils import (
    corrupt_nlvr,
    corrupt_vqa,
    corrupt_gqa,
    CHANGE_CHAR_METHOD_MAP,
    TEXT_STYLE_METHOD_MAP,
    ADD_TEXT_METHOD_MAP,
    DROP_TEXT_ON_POS_METHOD_MAP,
    SWAP_TEXT_METHOD_MAP,
    POSITIONAL_METHOD_MAP,
    ALL_METHOD_MAP,
    METHODS_WITHOUT_SEVERITY
)
import argparse
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)


CORRUPTION_CATEGORY_TO_METHOD_MAP = {
    "change_char": CHANGE_CHAR_METHOD_MAP,
    "text_style": TEXT_STYLE_METHOD_MAP,
    "add_text": ADD_TEXT_METHOD_MAP,
    "drop_text_on_pos": DROP_TEXT_ON_POS_METHOD_MAP,
    "swap_text": SWAP_TEXT_METHOD_MAP,
    "positional": POSITIONAL_METHOD_MAP
}

TASK_TO_FUNC = {
    "nlvr": corrupt_nlvr,
    "vqa": corrupt_vqa,
    "gqa": corrupt_gqa
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="nlvr", choices=["nlvr", "vqa", "gqa"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--severity_begin", type=int, default=-1)
    parser.add_argument("--severity_end", type=int, default=-1)
    parser.add_argument(
        "--corruption_category",
        type=str,
        default="none",
        choices=["change_char",
                 "text_style",
                 "add_text",
                 "drop_text_on_pos",
                 "swap_text",
                 "positional",
                "none"
                 ]
    )
    parser.add_argument("--corruption_method", type=str, default="none", choices=list(ALL_METHOD_MAP.keys()) + ["none"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.task == "nlvr":
        assert args.split in ["test"]
    if args.task == "vqa":
        assert args.split in ["karpathy_test"]
    if args.task == "gqa":
        assert args.split in ["testdev"]

    if args.corruption_category != "none":
        logger.info(f"Corrupting using {args.corruption_category} methods")
        for method in tqdm(CORRUPTION_CATEGORY_TO_METHOD_MAP[args.corruption_category].keys(),
                           desc=f"Corrupting using different {args.corruption_category} methods"):
            if args.severity_begin != -1 and args.severity_end != -1:
                for severity in tqdm(range(args.severity_begin, args.severity_end + 1), desc=f"Severity Progress"):
                    TASK_TO_FUNC[args.task](split=args.split, json_path=None, method_name=method, severity=severity)
            else:
                TASK_TO_FUNC[args.task](split=args.split, json_path=None, method_name=method, severity=args.severity)

    elif args.corruption_method != "none":
        logger.info(f"Corrupting using {args.corruption_method} method")
        if args.severity_begin != -1 and args.severity_end != -1:
            for severity in tqdm(range(args.severity_begin, args.severity_end + 1), desc=f"Severity Progress"):
                TASK_TO_FUNC[args.task](split=args.split, json_path=None, method_name=args.corruption_method, severity=severity)
        else:
            TASK_TO_FUNC[args.task](
                split=args.split,
                json_path=None,
                method_name=args.corruption_method,
                severity=args.severity
            )
    else:
        raise ValueError("Please specify a corruption method or category")

