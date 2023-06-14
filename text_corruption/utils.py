# -*- coding: utf-8 -*-

"""Utility functions for text corruptions."""

import logging
import json
import socket
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from ensure_fidelity import obtain_similarity
from corruption_methods import (
    ocr_corruption,
    swap_prefix_corruption,
    punctuation_corruption,
    typos_corruption,
    keyboard_corruption,
    spell_error_corruption,
    random_char_delete_corruption,
    random_char_insert_corruption,
    random_char_replace_corruption,
    random_char_swap_corruption,

    to_passive_corruption,
    to_active_corruption,
    to_casual_corruption,
    to_formal_corruption,
    tense_corruption,
    double_denial_corruption,

    insert_adv_corruption,
    append_irr_corruption,
    random_word_insert_corruption,

    drop_nn_corruption,
    drop_vb_corruption,
    drop_vb_nn_corruption,
    drop_rand_one_nn_corruption,
    drop_rand_one_vb_corruption,
    only_nn_corruption,
    only_vb_corruption,
    only_vb_nn_corruption,
    drop_first_corruption,
    drop_last_corruption,
    drop_first_and_last_corruption,
    shuffle_order_corruption,
    random_word_delete_corruption,
    nonsense_corruption,

    mlm_suggestion_corruption,
    swap_syn_word_emb_corruption,
    swap_syn_word_net_corruption,
    swap_multi_pos_jj_corruption,
    swap_multi_pos_nn_corruption,
    back_trans_corruption,
    random_word_swap_corruption,
)

# Modify this to point to the location of the datasets on your machine.
DATASET_BASE = "/home/.data/datasets"

logger = logging.getLogger(__name__)


ALL_METHOD_MAP = {
    "ocr": ocr_corruption,
    "swap_prefix": swap_prefix_corruption,
    "punctuation": punctuation_corruption,
    "typos": typos_corruption,
    "keyboard": keyboard_corruption,
    "spell_error": spell_error_corruption,
    "random_char_delete": random_char_delete_corruption,
    "random_char_insert": random_char_insert_corruption,
    "random_char_replace": random_char_replace_corruption,
    "random_char_swap": random_char_swap_corruption,
    "to_passive": to_passive_corruption,
    "to_active": to_active_corruption,
    "to_casual": to_casual_corruption,
    "to_formal": to_formal_corruption,
    "tense": tense_corruption,
    "double_denial": double_denial_corruption,
    "insert_adv": insert_adv_corruption,
    "append_irr": append_irr_corruption,
    "random_word_insert": random_word_insert_corruption,
    "drop_nn": drop_nn_corruption,
    "drop_vb": drop_vb_corruption,
    "drop_vb_nn": drop_vb_nn_corruption,
    "drop_rand_one_nn": drop_rand_one_nn_corruption,
    "drop_rand_one_vb": drop_rand_one_vb_corruption,
    "only_nn": only_nn_corruption,
    "only_vb": only_vb_corruption,
    "only_vb_nn": only_vb_nn_corruption,
    "drop_first": drop_first_corruption,
    "drop_last": drop_last_corruption,
    "drop_first_and_last": drop_first_and_last_corruption,
    "shuffle_order": shuffle_order_corruption,
    "random_word_delete": random_word_delete_corruption,
    "nonsense": nonsense_corruption,

    "mlm_suggestion": mlm_suggestion_corruption,
    "swap_syn_word_emb": swap_syn_word_emb_corruption,
    "swap_syn_word_net": swap_syn_word_net_corruption,
    "swap_multi_pos_jj": swap_multi_pos_jj_corruption,
    "swap_multi_pos_nn": swap_multi_pos_nn_corruption,
    "back_trans": back_trans_corruption,
    "random_word_swap": random_word_swap_corruption,
}

CHANGE_CHAR_METHOD_MAP = {
    "ocr": ocr_corruption,
    # "swap_prefix": swap_prefix_corruption,
    "punctuation": punctuation_corruption,
    "typos": typos_corruption,
    "keyboard": keyboard_corruption,
    "spell_error": spell_error_corruption,
    "random_char_delete": random_char_delete_corruption,
    "random_char_insert": random_char_insert_corruption,
    "random_char_replace": random_char_replace_corruption,
    "random_char_swap": random_char_swap_corruption,
}

TEXT_STYLE_METHOD_MAP = {
    "to_passive": to_passive_corruption,
    "to_active": to_active_corruption,
    "to_casual": to_casual_corruption,
    "to_formal": to_formal_corruption,
    "tense": tense_corruption,
    "double_denial": double_denial_corruption,
}

ADD_TEXT_METHOD_MAP = {
    "insert_adv": insert_adv_corruption,
    "append_irr": append_irr_corruption,
    "random_word_insert": random_word_insert_corruption,
}

DROP_TEXT_ON_POS_METHOD_MAP = {
    "drop_nn": drop_nn_corruption,
    "drop_vb": drop_vb_corruption,
    "drop_vb_nn": drop_vb_nn_corruption,
    "drop_rand_one_nn": drop_rand_one_nn_corruption,
    "drop_rand_one_vb": drop_rand_one_vb_corruption,
    "only_nn": only_nn_corruption,
    "only_vb": only_vb_corruption,
    "only_vb_nn": only_vb_nn_corruption,
}

POSITIONAL_METHOD_MAP = {
    "drop_first": drop_first_corruption,
    "drop_last": drop_last_corruption,
    "drop_first_and_last": drop_first_and_last_corruption,
    "shuffle_order": shuffle_order_corruption,
    "random_word_delete": random_word_delete_corruption,
}

SWAP_TEXT_METHOD_MAP = {
    # "mlm_suggestion": mlm_suggestion_corruption,
    "swap_syn_word_emb": swap_syn_word_emb_corruption,
    "swap_syn_word_net": swap_syn_word_net_corruption,
    "swap_multi_pos_jj": swap_multi_pos_jj_corruption,
    "swap_multi_pos_nn": swap_multi_pos_nn_corruption,
    "back_trans": back_trans_corruption,
    "random_word_swap": random_word_swap_corruption,
}

METHODS_WITHOUT_SEVERITY = [
    "punctuation",

    "to_active",
    "to_passive",
    "to_formal",
    "to_casual",
    "tense",
    "double_denial",

    "append_irr",

    "insert_adv",

    "drop_first",
    "drop_last",
    "drop_first_and_last",
    "shuffle_order",
    "drop_nn",
    "drop_rand_one_nn",
    "drop_rand_one_vb",
    "drop_vb",
    "drop_vb_nn",
    "only_nn",
    "only_vb",
    "only_vb_nn",

    "back_trans",

]

ALPHA_SIMILARITY = 0.2
RETRY_LIMIT = 100


def ensure_fidelity_of_corruption(
        original_sentence,
        method, severity,
        alpha_similarity=ALPHA_SIMILARITY,
        retry_limit=RETRY_LIMIT
):
    similarity = 0
    times = 0
    corrupted_sent = original_sentence
    while similarity < alpha_similarity and times < retry_limit:
        # logger.info(f"Try to find a sentence with similarity > {alpha_similarity} for {original_sentence} ")
        corrupted_sent = method(original_sentence, severity=severity)
        # logger.critical(f"original sentence: {original_sentence}")
        # logger.critical(f"corrupted sentence: {corrupted_sent}")
        similarity = obtain_similarity(original_sentence, corrupted_sent)
        times += 1
    if times == retry_limit:
        # TODO: remove this example or not?
        logger.warning(f"Cannot find a sentence with similarity > {alpha_similarity} for {original_sentence} "
                       f"after 100 times of trying. Set the corrupted sentence to be the original sentence.")
        logger.warning(f"original sentence: {original_sentence}")
        logger.warning(f"corrupted sentence: {corrupted_sent}")
        logger.warning(f"similarity: {similarity}")
        corrupted_sent = original_sentence
    return corrupted_sent


def corrupt_nlvr(split: str, json_path=None, method_name="ocr", severity=1, output_path=None):
    """
    Corrupt NLVR2 dataset.
    :param split:
    :type split:
    :param json_path:
    :type json_path:
    :param method_name:
    :type method_name:
    :param severity:
    :type severity:
    :param output_path:
    :type output_path:
    :return:
    :rtype:
    """
    if split is None:
        assert json_path is not None, "split and json_path are both None!"
    else:
        assert split in ["test"], f"split = {split} is not valid!"
    if output_path is None:
        output_path = os.path.join(DATASET_BASE, "nlvr", f"{split}_corrupted_{method_name}_{severity}.json")
    if json_path is None:
        json_path = os.path.join(DATASET_BASE, "nlvr", f"{split}.json")

    assert isinstance(method_name, str)
    assert method_name in ALL_METHOD_MAP, f"method_name = {method_name} is not valid!"
    method = ALL_METHOD_MAP[method_name]
    if method_name in METHODS_WITHOUT_SEVERITY:
        severity = 0

    data_records = json.load(open(json_path, "r"))
    for data_record in tqdm(data_records, desc=f"Corrupting NLVR2 {split} data using {method_name} with severity {severity}..."):
        original_sentence = data_record["sent"]
        if method_name != "nonsense":
            corrupted_sent = ensure_fidelity_of_corruption(original_sentence, method, severity)
        else:
            corrupted_sent = method(original_sentence, severity)
        data_record["sent"] = corrupted_sent
    json.dump(data_records, open(output_path, "w"))
    return output_path


def corrupt_vqa(split: str, json_path=None, method_name="ocr", severity=1, output_path=None):
    if split is None:
        assert json_path is not None, "split and json_path are both None!"
    else:
        assert split in ["karpathy_test"], f"split = {split} is not valid!"
    if output_path is None:
        output_path = os.path.join(DATASET_BASE, "vqa", f"{split}_corrupted_{method_name}_{severity}.json")
    if json_path is None:
        json_path = os.path.join(DATASET_BASE, "vqa", f"{split}.json")

    assert isinstance(method_name, str)
    assert method_name in ALL_METHOD_MAP, f"method_name = { method_name} is not valid!"
    method = ALL_METHOD_MAP[method_name]
    if method_name in METHODS_WITHOUT_SEVERITY:
        severity = 0

    data_records = json.load(open(json_path, "r"))
    for data_record in tqdm(data_records, desc=f"Corrupting VQA {split} data using {method_name} with severity {severity}..."):
        original_sentence = data_record["sent"]
        if method_name != "nonsense":
            corrupted_sent = ensure_fidelity_of_corruption(original_sentence, method, severity)
        else:
            corrupted_sent = method(original_sentence, severity)
        data_record["sent"] = corrupted_sent
    json.dump(data_records, open(output_path, "w"))
    return output_path


def corrupt_gqa(split: str, json_path=None, method_name="ocr", severity=1, output_path=None):
    if split is None:
        assert json_path is not None, "split and json_path are both None!"
    else:
        assert split in ["testdev"], f"split = {split} is not valid!"
    if output_path is None:
        output_path = os.path.join(DATASET_BASE, "GQA", f"{split}_corrupted_{method_name}_{severity}.json")
    if json_path is None:
        json_path = os.path.join(DATASET_BASE, "GQA", f"{split}.json")

    assert isinstance(method_name, str)
    assert method_name in ALL_METHOD_MAP, f"method_name = {method_name} is not valid!"
    method = ALL_METHOD_MAP[method_name]
    if method_name in METHODS_WITHOUT_SEVERITY:
        severity = 0

    data_records = json.load(open(json_path, "r"))
    for data_record in tqdm(data_records, desc=f"Corrupting GQA {split} data using {method_name} with severity {severity}..."):
        original_sentence = data_record["sent"]
        if method_name != "nonsense":
            corrupted_sent = ensure_fidelity_of_corruption(original_sentence, method, severity)
        else:
            corrupted_sent = method(original_sentence, severity)
        data_record["sent"] = corrupted_sent
    json.dump(data_records, open(output_path, "w"))
    return output_path


# def corrupt_gqa_dataset_version():
#     distort_gqa_text = DistortText(
#         task="gqa",
#         split="testdev",
#         method_name="ocr",
#         severity=1,
#     )
#     dataloader = DataLoader(distort_gqa_text,
#                             batch_size=10,
#                             shuffle=False,
#                             num_workers=10)
#     for batch in dataloader:
#         print(batch)
#         break
#
#
# class DistortText(Dataset):
#     def __init__(
#             self,
#             task,
#             split,
#             json_path=None,
#             output_path=None,
#             method_name="ocr",
#             severity=1,
#     ):
#         self.task = task.lower()
#         self.split = split
#         self.method_name = method_name
#         self.severity = severity
#
#         if task == "vqa":
#             valid_split = ["karpathy_test"]
#             if split is None:
#                 assert json_path is not None, "split and json_path are both None!"
#             else:
#                 assert split in valid_split, f"split = {split} is not valid!"
#             if output_path is None:
#                 self.output_path = os.path.join(DATASET_BASE, "vqa", f"{split}_corrupted_{method_name}_{severity}.json")
#             else:
#                 self.output_path = output_path
#             if json_path is None:
#                 json_path = os.path.join(DATASET_BASE, "vqa", f"{split}.json")
#         elif task == "nlvr2" or task == "nlvr":
#             valid_split = ["test"]
#             if split is None:
#                 assert json_path is not None, "split and json_path are both None!"
#             else:
#                 assert split in valid_split, f"split = {split} is not valid!"
#             if output_path is None:
#                 self.output_path = os.path.join(DATASET_BASE, "nlvr", f"{split}_corrupted_{method_name}_{severity}.json")
#             else:
#                 self.output_path = output_path
#             if json_path is None:
#                 json_path = os.path.join(DATASET_BASE, "nlvr", f"{split}.json")
#         elif task == "gqa":
#             valid_split = ["testdev"]
#             if split is None:
#                 assert json_path is not None, "split and json_path are both None!"
#             else:
#                 assert split in valid_split, f"split = {split} is not valid!"
#             if output_path is None:
#                 self.output_path = os.path.join(DATASET_BASE, "GQA", f"{split}_corrupted_{method_name}_{severity}.json")
#             else:
#                 self.output_path = output_path
#             if json_path is None:
#                 json_path = os.path.join(DATASET_BASE, "GQA", f"{split}.json")
#
#         assert isinstance(method_name, str)
#         assert method_name in ALL_METHOD_MAP, f"method_name = {method_name} is not valid!"
#         self.method = ALL_METHOD_MAP[method_name]
#         if method_name in METHODS_WITHOUT_SEVERITY:
#             self.severity = 0
#
#         self.data_records = json.load(open(json_path, "r"))
#
#     def __getitem__(self, index):
#         data_record = self.data_records[index]
#         corrupted_sent = ensure_fidelity_of_corruption(data_record["sent"], self.method, self.severity)
#         data_record["sent"] = corrupted_sent
#         return data_record
#
#     def __len__(self):
#         return len(self.data_records)


if __name__ == "__main__":
    # corrupt_gqa("testdev", method_name="to_passive", severity=5)
    to_passive_corruption(
        [
            "this is a test sentence",
            "this is a test sentence",
            "this is a test sentence",
        ]
    )