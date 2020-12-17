#!/usr/bin/env python
# -*-coding:utf8-*-
""" Translator Class and builder """
from __future__ import print_function
import codecs
import torch

from tensorboardX import SummaryWriter
from others.utils import rouge_results_to_str, test_bleu, test_length
from translate.beam import GNMTGlobalScorer
from rouge import Rouge, FilesRouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def build_predictor(args, tokenizer, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha, length_penalty='wu')

    translator = Translator(args, model, tokenizer,
                            global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 tokenizer,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.start_token = self.vocab['[unused1]']
        self.end_token = self.vocab['[unused2]']
        self.seg_token = self.vocab['[unused3]']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(
            tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch_dev(self, batch, doc_batch):

        translations = []

        preds = doc_batch

        batch_size = len(batch)

        special_tokens = ['[PAD]', '[unused1]', '[unused2]', '[unused3]']

        doc_extract = batch.doc_tgt

        for b in range(batch_size):

            # doc extracted text
            doc_original_sent = self.tokenizer.convert_ids_to_tokens([int(n) for n in doc_extract[b]])
            doc_original_sent = ' '.join([ch for ch in doc_original_sent if ch not in special_tokens])

            # short doc context text
            segment_preds = self.tokenizer.convert_ids_to_tokens([int(n) for n in preds[b]])
            segment_preds = ' '.join([ch for ch in segment_preds if ch not in special_tokens])

            translation = (doc_original_sent, segment_preds)
            translations.append(translation)

        return translations

    def from_batch_test(self, batch, doc_batch):

        special_tokens = ['[PAD]', '[unused1]']
        translations = []

        preds = doc_batch

        batch_size = len(batch)

        doc_extract, context_doc_extract, origin_txt, ex_segs, doc_segs = \
            batch.doc_tgt, batch.context_doc_tgt, batch.original_str, batch.ex_segs, batch.doc_segs

        ex_segs = [sum(ex_segs[:i]) for i in range(len(ex_segs)+1)]

        for b in range(batch_size):
            # original text
            original_sent = ' [unused3] '.join(origin_txt[ex_segs[b]:ex_segs[b+1]])

            # lead text
            lead_sent = " ".join(origin_txt[ex_segs[b]:ex_segs[b+1]][:doc_segs[b+1]-doc_segs[b]])

            # doc extracted text
            doc_original_sent = self.tokenizer.convert_ids_to_tokens(
                [int(n) for n in doc_extract[b]])
            doc_original_sent = [ch for ch in doc_original_sent if ch not in special_tokens]
            doc_original_sent = ' '.join(doc_original_sent)

            # context doc extracted text
            context_doc_extract_sent = context_doc_extract[doc_segs[b]:doc_segs[b+1]]
            context_doc_original_sents = ''
            for sent in context_doc_extract_sent:
                context_doc_original_sent = self.tokenizer.convert_ids_to_tokens([int(n) for n in sent])
                context_doc_original_sent = [ch for ch in context_doc_original_sent if ch not in special_tokens]
                context_doc_original_sent = ' '.join(context_doc_original_sent).replace('##', '')
                context_doc_original_sents = context_doc_original_sents + ' ' + context_doc_original_sent

            # long doc context text
            doc_pred_sents = self.tokenizer.convert_ids_to_tokens(
                [int(n) for n in preds[b]])
            doc_pred_sents = [ch for ch in doc_pred_sents if ch not in special_tokens]
            doc_pred_sents = ' '.join(doc_pred_sents).replace('##', '')

            translation = (original_sent, doc_original_sent, context_doc_original_sents,
                           doc_pred_sents, lead_sent)
            translations.append(translation)

        return translations

    def validate(self, data_iter, step, attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.step.%d.gold_temp' % step
        pred_path = self.args.result_path + '.step.%d.pred_temp' % step
        gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        pred_out_file = codecs.open(pred_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                doc_data, summ_data = self.translate_batch(batch)
                translations = self.from_batch_dev(batch, doc_data)

                for idx in range(len(translations)):
                    if ct % 100 == 0:
                        print("Processing %d" % ct)
                    doc_short_context = translations[idx][1]
                    gold_data = summ_data[idx]
                    pred_out_file.write(doc_short_context + '\n')
                    gold_out_file.write(gold_data + '\n')
                    ct += 1
                pred_out_file.flush()
                gold_out_file.flush()

        pred_out_file.close()
        gold_out_file.close()

        if (step != -1):
            pred_bleu = test_bleu(pred_path, gold_path)
            file_rouge = FilesRouge(hyp_path=pred_path, ref_path=gold_path)
            pred_rouges = file_rouge.get_scores(avg=True)
            self.logger.info('Gold Length at step %d: %.2f' %
                             (step, test_length(gold_path, gold_path, ratio=False)))
            self.logger.info('Prediction Length ratio at step %d: %.2f' %
                             (step, test_length(pred_path, gold_path)))
            self.logger.info('Prediction Bleu at step %d: %.2f' %
                             (step, pred_bleu*100))
            self.logger.info('Prediction Rouges at step %d: \n%s\n' %
                             (step, rouge_results_to_str(pred_rouges)))
            rouge_results = (pred_rouges["rouge-1"]['f'],
                             pred_rouges["rouge-l"]['f'])
        return rouge_results

    def translate(self, data_iter, step, attn_debug=False):

        self.model.eval()
        output_path = self.args.result_path + '.%d.output' % step
        output_file = codecs.open(output_path, 'w', 'utf-8')
        gold_path = self.args.result_path + '.%d.gold_test' % step
        pred_path = self.args.result_path + '.%d.pred_test' % step
        ex_single_path = self.args.result_path + '.%d.ex_test' % step + ".short"
        ex_context_path = self.args.result_path + '.%d.ex_test' % step + ".long"
        gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        pred_out_file = codecs.open(pred_path, 'w', 'utf-8')
        short_ex_out_file = codecs.open(ex_single_path, 'w', 'utf-8')
        long_ex_out_file = codecs.open(ex_context_path, 'w', 'utf-8')
        # pred_results, gold_results = [], []

        ct = 0
        with torch.no_grad():
            rouge = Rouge()
            for batch in data_iter:
                doc_data, summ_data = self.translate_batch(batch)
                translations = self.from_batch_test(batch, doc_data)

                for idx in range(len(translations)):
                    origin_sent, doc_extract, context_doc_extract, \
                        doc_pred, lead = translations[idx]
                    if ct % 100 == 0:
                        print("Processing %d" % ct)
                    output_file.write("ID      : %d\n" % ct)
                    output_file.write("ORIGIN  : " + origin_sent.replace('<S>', '\n          ') + "\n")
                    gold_data = summ_data[idx]
                    output_file.write("GOLD    : " + gold_data + "\n")
                    output_file.write("LEAD    : " + lead + "\n")
                    output_file.write("DOC_EX  : " + doc_extract.strip() + "\n")
                    output_file.write("DOC_CONT: " + context_doc_extract.strip() + "\n")
                    output_file.write("DOC_GEN : " + doc_pred.strip() + "\n")

                    gold_list = gold_data.strip().split()
                    lead_list = lead.strip().replace("[unused2]", "").replace("[unused3]", "").split()
                    rouge_score = rouge.get_scores(lead, gold_data)
                    bleu_score = sentence_bleu([gold_list], lead_list, smoothing_function=SmoothingFunction().method1)
                    output_file.write("LEAD     bleu & rouge-f 1/2/l:    %.4f & %.4f/%.4f/%.4f\n" %
                                      (bleu_score, rouge_score[0]["rouge-1"]["f"], rouge_score[0]["rouge-2"]["f"], rouge_score[0]["rouge-l"]["f"]))

                    doc_extract_list = doc_extract.strip().replace("[unused2]", "").replace("[unused3]", "").split()
                    rouge_score = rouge.get_scores(doc_extract, gold_data)
                    bleu_score = sentence_bleu([gold_list], doc_extract_list, smoothing_function=SmoothingFunction().method1)
                    output_file.write("DOC_EX   bleu & rouge-f 1/2/l:    %.4f & %.4f/%.4f/%.4f\n" %
                                      (bleu_score, rouge_score[0]["rouge-1"]["f"], rouge_score[0]["rouge-2"]["f"], rouge_score[0]["rouge-l"]["f"]))

                    doc_context_list = context_doc_extract.strip().replace("[unused2]", "").replace("[unused3]", "").split()
                    rouge_score = rouge.get_scores(context_doc_extract, gold_data)
                    bleu_score = sentence_bleu([gold_list], doc_context_list, smoothing_function=SmoothingFunction().method1)
                    output_file.write("DOC_CONT bleu & rouge-f 1/2/l:    %.4f & %.4f/%.4f/%.4f\n" %
                                      (bleu_score, rouge_score[0]["rouge-1"]["f"], rouge_score[0]["rouge-2"]["f"], rouge_score[0]["rouge-l"]["f"]))

                    doc_long_list = doc_pred.strip().replace("[unused2]", "").replace("[unused3]", "").split()
                    rouge_score = rouge.get_scores(doc_pred, gold_data)
                    bleu_score = sentence_bleu([gold_list], doc_long_list, smoothing_function=SmoothingFunction().method1)
                    output_file.write("DOC_GEN  bleu & rouge-f 1/2/l:    %.4f & %.4f/%.4f/%.4f\n\n" %
                                      (bleu_score, rouge_score[0]["rouge-1"]["f"], rouge_score[0]["rouge-2"]["f"], rouge_score[0]["rouge-l"]["f"]))

                    short_ex_out_file.write(doc_extract.strip().replace("[unused2]", "").replace("[unused3]", "") + '\n')
                    long_ex_out_file.write(context_doc_extract.strip().replace("[unused2]", "").replace("[unused3]", "") + '\n')
                    pred_out_file.write(doc_pred.strip().replace("[unused2]", "").replace("[unused3]", "") + '\n')
                    gold_out_file.write(gold_data.strip() + '\n')
                    ct += 1
                pred_out_file.flush()
                short_ex_out_file.flush()
                long_ex_out_file.flush()
                gold_out_file.flush()
                output_file.flush()

        pred_out_file.close()
        short_ex_out_file.close()
        long_ex_out_file.close()
        gold_out_file.close()
        output_file.close()

        if (step != -1):
            ex_short_bleu = test_bleu(gold_path, ex_single_path)
            ex_long_bleu = test_bleu(gold_path, ex_context_path)
            pred_bleu = test_bleu(gold_path, pred_path)

            file_rouge = FilesRouge(hyp_path=ex_single_path, ref_path=gold_path)
            ex_short_rouges = file_rouge.get_scores(avg=True)

            file_rouge = FilesRouge(hyp_path=ex_context_path, ref_path=gold_path)
            ex_long_rouges = file_rouge.get_scores(avg=True)

            file_rouge = FilesRouge(hyp_path=pred_path, ref_path=gold_path)
            pred_rouges = file_rouge.get_scores(avg=True)

            self.logger.info('Gold Length at step %d: %.2f\n' %
                             (step, test_length(gold_path, gold_path, ratio=False)))
            self.logger.info('Short Extraction Length ratio at step %d: %.2f' %
                             (step, test_length(ex_single_path, gold_path)))
            self.logger.info('Short Extraction Bleu at step %d: %.2f' %
                             (step, ex_short_bleu*100))
            self.logger.info('Short Extraction Rouges at step %d \n%s' %
                             (step, rouge_results_to_str(ex_short_rouges)))
            self.logger.info('Long Extraction Length ratio at step %d: %.2f' %
                             (step, test_length(ex_context_path, gold_path)))
            self.logger.info('Long Extraction Bleu at step %d: %.2f' %
                             (step, ex_long_bleu*100))
            self.logger.info('Long Extraction Rouges at step %d \n%s' %
                             (step, rouge_results_to_str(ex_long_rouges)))
            self.logger.info('Prediction Length ratio at step %d: %.2f' %
                             (step, test_length(pred_path, gold_path)))
            self.logger.info('Prediction Bleu at step %d: %.2f' %
                             (step, pred_bleu*100))
            self.logger.info('Prediction Rouges at step %d \n%s' %
                             (step, rouge_results_to_str(pred_rouges)))

    def translate_batch(self, batch):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """

        summ_txt = batch.summ_txt

        _, _, doc_data = self.model(batch)
        return doc_data, summ_txt


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
