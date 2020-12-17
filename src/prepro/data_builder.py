# -*- coding:utf-8 -*-

import gc
import glob
import json
import os
import torch
from os.path import join as pjoin

from others.logging import logger
from others.tokenization import BertTokenizer


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_temp_dir)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.tgt_bos = '[unused1]'
        self.tgt_eos = '[unused2]'
        self.tgt_sent_split = '[unused3]'
        self.role_1 = '[unused4]'
        self.role_2 = '[unused5]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.unk_vid = self.tokenizer.vocab[self.unk_token]

    def preprocess_train(self, content, info=None):
        if_exceed_length = False

        if not (info == "c2b" or info == 'b2c'):
            return None
        if len(content) < self.args.min_src_ntokens_per_sent:
            return None
        if len(content) > self.args.max_src_ntokens_per_sent:
            if_exceed_length = True
        original_txt = ' '.join(content)

        content_subtokens = self.tokenizer.tokenize(original_txt)

        # [CLS] + T0 + T1 + ... + Tn
        if info == 'c2b':
            src_subtokens = [self.cls_token, self.role_1] + content_subtokens
        else:
            src_subtokens = [self.cls_token, self.role_2] + content_subtokens
        # src_subtokens = [self.cls_token] + content_subtokens
        tgt_subtokens = [self.tgt_bos] + content_subtokens + [self.tgt_eos]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtokens)
        segments_ids = len(src_subtoken_idxs) * [0]

        return src_subtoken_idxs, tgt_subtoken_idxs, segments_ids, original_txt, \
            src_subtokens, tgt_subtokens, if_exceed_length

    def preprocess_test(self, content):

        original_txt = ' '.join(content)

        content_subtokens = self.tokenizer.tokenize(original_txt)

        subtoken_idxs = self.tokenizer.convert_tokens_to_ids(content_subtokens)

        return subtoken_idxs, original_txt, content_subtokens


def prepro(args):

    a_lst = []
    for json_f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = json_f.split('/')[-1]
        a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))

    total_statistic = {
        "instances": 0,
        "sent_num": 0.,
        "max_sent_num": -1,
        "truncated_sent_num": 0,
        "truncated_dial_num": 0,
        "total_length": 0.,
        "sent_count": [0] * 11,
        "sent_length_count": [0] * 11,
        "dial_length_count": [0] * 11,
        "total_summ_length": 0.
    }
    for d in a_lst:
        statistic = _prepro(d)
        if statistic is None:
            continue
        total_statistic["instances"] += statistic["instances"]
        total_statistic["sent_num"] += statistic["sent_num"]
        total_statistic["max_sent_num"] = max(total_statistic["max_sent_num"], statistic["max_sent_num"])
        total_statistic["truncated_sent_num"] += statistic["truncated_sent_num"]
        total_statistic["truncated_dial_num"] += statistic["truncated_dial_num"]
        total_statistic["total_length"] += statistic["total_length"]
        total_statistic["total_summ_length"] += statistic["total_summ_length"]
        for idx in range(len(total_statistic["sent_count"])):
            total_statistic["sent_count"][idx] += statistic["sent_count"][idx]
        for idx in range(len(total_statistic["sent_length_count"])):
            total_statistic["sent_length_count"][idx] += statistic["sent_length_count"][idx]
        for idx in range(len(total_statistic["dial_length_count"])):
            total_statistic["dial_length_count"][idx] += statistic["dial_length_count"][idx]

    if total_statistic["instances"] > 0:
        logger.info("Total dials: %d" % total_statistic["instances"])
        logger.info("Averaged sent number: %f" % (total_statistic["sent_num"] / total_statistic["instances"]))
        logger.info("Total sent number: %d" % total_statistic["sent_num"])
        for idx, num in enumerate(total_statistic["sent_count"]):
            logger.info(" Sent num %d ~ %d: %d, %.2f%%" % (idx * 4, (idx+1) * 4, num, (num / total_statistic["instances"])))
        logger.info("Averaged sent length: %f" % (total_statistic["total_length"] / total_statistic["sent_num"]))
        for idx, num in enumerate(total_statistic["sent_length_count"]):
            logger.info(" Sent length %d ~ %d: %d, %.2f%%" % (idx * 5, (idx+1) * 5, num, (num / total_statistic["sent_num"])))
        logger.info("Averaged dial length: %f" % (total_statistic["total_length"] / total_statistic["instances"]))
        for idx, num in enumerate(total_statistic["dial_length_count"]):
            logger.info(" Dial length %d ~ %d: %d, %.2f%%" % (idx * 30, (idx+1) * 30, num, (num / total_statistic["instances"])))
        logger.info("Averaged summ length:: %f" % (total_statistic["total_summ_length"] / total_statistic["instances"]))
        logger.info("Truncated sent number: %d" % total_statistic["truncated_sent_num"])
        logger.info("Truncated dial number: %d" % total_statistic["truncated_dial_num"])


def _prepro(params):
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    datasets = []
    truncated_sent_num = 0
    truncated_dial_num = 0
    total_length = 0.
    total_summ_length = 0.
    sent_length_count = [0] * 11
    dial_length_count = [0] * 11
    max_sent_num = 0
    sent_count = [0] * 11
    sent_num = 0.

    for dialogue in jobs:
        dialogue_b_data = []
        dial_length = 0
        for index, sent in enumerate(dialogue['session']):
            content, info = sent['content'], sent['type']

            b_data = bert.preprocess_train(content, info)
            if (b_data is None):
                continue
            src_subtoken_idxs, tgt_subtoken_idxs, segments_ids, original_txt, \
                src_subtokens, tgt_subtokens, exceed_length = b_data
            b_data_dict = {"index": index, "type": info, "src_id": src_subtoken_idxs,
                           "tgt_id": tgt_subtoken_idxs, "segs": segments_ids,
                           "original_txt": original_txt, "src_tokens": src_subtokens,
                           "tgt_tokens": tgt_subtokens}
            # remove the start token and end token.
            sent_length = len(tgt_subtoken_idxs) - 2
            sent_length_count[min(sent_length // 5, 10)] += 1
            dial_length += sent_length
            total_length += sent_length
            dialogue_b_data.append(b_data_dict)
            if exceed_length:
                truncated_sent_num += 1
        if len(dialogue_b_data) >= args.max_turns:
            truncated_dial_num += 1
        if len(dialogue_b_data) <= 0:
            continue

        dialogue_example = {"session": dialogue_b_data}
        # test & dev data process
        if "summary" in dialogue.keys():
            content = dialogue["summary"]
            b_data = bert.preprocess_test(content)
            subtoken_idxs, original_txt, content_subtokens = b_data
            b_data_dict = {"id": subtoken_idxs,
                           "original_txt": original_txt,
                           "content_tokens": content_subtokens}
            dialogue_example["summary"] = b_data_dict
            total_summ_length += len(subtoken_idxs)
        datasets.append(dialogue_example)
        sent_count[min(len(dialogue_b_data) // 4, 10)] += 1
        dial_length_count[min(dial_length // 30, 10)] += 1
        max_sent_num = max(max_sent_num, len(dialogue_b_data))
        sent_num += len(dialogue_b_data)

    statistic = {
        "instances": len(datasets),
        "sent_num": sent_num,
        "max_sent_num": max_sent_num,
        "truncated_sent_num": truncated_sent_num,
        "truncated_dial_num": truncated_dial_num,
        "total_length": total_length,
        "sent_count": sent_count,
        "sent_length_count": sent_length_count,
        "dial_length_count": dial_length_count,
        "total_summ_length": total_summ_length
    }

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
    return statistic
