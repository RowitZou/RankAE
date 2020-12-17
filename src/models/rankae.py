import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence

from models.decoder import TransformerDecoder
from models.encoder import Bert, TransformerEncoder, PositionalEncoding
from models.generator import Generator
from others.utils import tile


class RankAE(nn.Module):
    def __init__(self, args, device, vocab, checkpoint=None):
        super(RankAE, self).__init__()
        self.args = args
        self.device = device
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.beam_size = args.beam_size
        self.max_length = args.max_length
        self.min_length = args.min_length

        self.start_token = vocab['[unused1]']
        self.end_token = vocab['[unused2]']
        self.pad_token = vocab['[PAD]']
        self.mask_token = vocab['[MASK]']
        self.seg_token = vocab['[unused3]']
        self.cls_token = vocab['[CLS]']

        self.hidden_size = args.enc_hidden_size
        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)

        if args.encoder == 'bert':
            self.encoder = Bert(args.bert_dir, args.finetune_bert)
            if(args.max_pos > 512):
                my_pos_embeddings = nn.Embedding(args.max_pos, self.encoder.model.config.hidden_size)
                my_pos_embeddings.weight.data[:512] = self.encoder.model.embeddings.position_embeddings.weight.data
                my_pos_embeddings.weight.data[512:] = self.encoder.model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(args.max_pos-512, 1)
                self.encoder.model.embeddings.position_embeddings = my_pos_embeddings
            tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
        else:
            self.encoder = TransformerEncoder(self.hidden_size, args.enc_ff_size, args.enc_heads,
                                              args.enc_dropout, args.enc_layers)
            tgt_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)

        self.hier_encoder = TransformerEncoder(self.hidden_size, args.hier_ff_size, args.hier_heads,
                                               args.hier_dropout, args.hier_layers)
        self.cup_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        self.pos_emb = PositionalEncoding(0., self.hidden_size)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout,
            embeddings=tgt_embeddings)

        self.generator = Generator(self.vocab_size, self.args.dec_hidden_size, self.pad_token)

        self.generator.linear.weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.encoder == "transformer":
                for module in self.encoder.modules():
                    self._set_parameter_tf(module)
                xavier_uniform_(self.embeddings.weight)
            for module in self.decoder.modules():
                self._set_parameter_tf(module)
            for module in self.hier_encoder.modules():
                self._set_parameter_tf(module)
            for p in self.generator.parameters():
                self._set_parameter_linear(p)
            for p in self.cup_bilinear.parameters():
                self._set_parameter_linear(p)
            if args.share_emb:
                if args.encoder == 'bert':
                    self.embeddings = self.encoder.model.embeddings.word_embeddings
                    tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
                    tgt_embeddings.weight = copy.deepcopy(self.encoder.model.embeddings.word_embeddings.weight)
                else:
                    tgt_embeddings = self.embeddings
                self.decoder.embeddings = tgt_embeddings
                self.generator.linear.weight = self.decoder.embeddings.weight

        self.to(device)

    def _set_parameter_tf(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_parameter_linear(self, p):
        if p.dim() > 1:
            xavier_uniform_(p)
        else:
            p.data.zero_()

    def _rebuild_tgt(self, origin, index, sep_token=None):

        tgt_list = [torch.tensor([self.start_token], device=self.device)]
        selected = origin.index_select(0, index)
        for sent in selected:
            filted_sent = sent[sent != self.pad_token][1:]
            if sep_token is not None:
                filted_sent[-1] = sep_token
            else:
                filted_sent = filted_sent[:-1]
            tgt_list.append(filted_sent)
        new_tgt = torch.cat(tgt_list, 0)
        if sep_token is not None:
            new_tgt[-1] = self.end_token
        else:
            new_tgt = torch.cat([new_tgt, torch.tensor([self.end_token], device=self.device)], 0)
        return new_tgt

    def _build_memory_window(self, ex_segs, keep_clss, replace_clss=None, mask=None, samples=None):
        keep_cls_list = torch.split(keep_clss, ex_segs)
        window_list = []
        for ex in keep_cls_list:
            ex_pad = F.pad(ex, (0, 0, self.args.win_size, self.args.win_size)).unsqueeze(1)
            ex_context = torch.cat([ex_pad[:ex.size(0)], ex.unsqueeze(1),
                                    ex_pad[self.args.win_size*2:]], 1)
            window_list.append(ex_context)
        memory = torch.cat(window_list, 0)
        if replace_clss is not None:
            replace_cls_list = torch.split(replace_clss, ex_segs)
            window_list = []
            for ex in replace_cls_list:
                ex_pad = F.pad(ex, (0, 0, self.args.win_size, self.args.win_size)).unsqueeze(1)
                ex_context = torch.cat([ex_pad[:ex.size(0)], ex.unsqueeze(1),
                                        ex_pad[self.args.win_size*2:]], 1)
                window_list.append(ex_context)
            origin_memory = torch.cat(window_list, 0)
            sample_list = torch.split(samples, ex_segs)
            sample_tensor_list = []
            for i in range(len(ex_segs)):
                sample_index_ = torch.randint(0, samples.size(-1), [mask.size(-1)], device=self.device)
                sample_index = torch.index_select(sample_list[i], 1, sample_index_)
                sample_tensor = replace_cls_list[i][sample_index]
                sample_tensor_list.append(sample_tensor)
            sample_memory = torch.cat(sample_tensor_list, 0)
            memory = memory * (mask == 2).unsqueeze(-1).float() + \
                sample_memory * (mask == 0).unsqueeze(-1).float() + \
                origin_memory * (mask == 1).unsqueeze(-1).float()
        return memory

    def _src_add_noise(self, sent, sampled_sent, expand_ratio=0.):
        role_emb = sent[1:2]
        filted_sent = sent[sent != self.pad_token][2:]
        # filted_sent = sent[sent != self.pad_token][1:]
        rand_size = sampled_sent.size(0)
        length = max(int(filted_sent.size(0)*(1+expand_ratio)), filted_sent.size(0)+1)
        while filted_sent.size(0) < length:
            target_length = length - filted_sent.size(0)
            rand_sent = sampled_sent[random.randint(0, rand_size-1)]
            rand_sent = rand_sent[rand_sent != self.pad_token][2:]  # remove cls and role embedding
            # rand_sent = rand_sent[rand_sent != self.pad_token][1:] # no role embedding
            start_point = random.randint(0, rand_sent.size(0)-1)
            end_point = random.randint(start_point, rand_sent.size(0))
            rand_segment = rand_sent[start_point:min(end_point, start_point+10, start_point+target_length)]
            insert_point = random.randint(0, filted_sent.size(0)-1)
            filted_sent = torch.cat([filted_sent[:insert_point],
                                    rand_segment,
                                    filted_sent[insert_point:]], 0)
        # return filted_sent
        return torch.cat([role_emb, filted_sent], 0)

    def _build_noised_src(self, src, ex_segs, samples, expand_ratio=0.):
        src_list = torch.split(src, ex_segs)
        new_src_list = []
        sample_list = torch.split(samples, ex_segs)

        for i, ex in enumerate(src_list):
            for j, sent in enumerate(ex):
                sampled_sent = ex.index_select(0, sample_list[i][j])
                expanded_sent = self._src_add_noise(sent, sampled_sent, expand_ratio)
                new_src = torch.cat([torch.tensor([self.cls_token], device=self.device), expanded_sent], 0)
                new_src_list.append(new_src)

        new_src = pad_sequence(new_src_list, batch_first=True, padding_value=self.pad_token)
        new_mask = new_src.data.ne(self.pad_token)
        new_segs = torch.zeros_like(new_src)
        return new_src, new_mask, new_segs

    def _build_context_tgt(self, tgt, ex_segs, win_size=1, modify=False, mask=None):

        tgt_list = torch.split(tgt, ex_segs)
        new_tgt_list = []
        if modify and mask is not None:
            # 1 means keeping the sentence
            mask_list = torch.split(mask, ex_segs)
        for i in range(len(tgt_list)):
            sent_num = tgt_list[i].size(0)
            for j in range(sent_num):
                if modify:
                    low = j-win_size
                    up = j+win_size+1
                    index = torch.arange(low, up, device=self.device)
                    index = index[mask_list[i][j] > 0]
                else:
                    low = max(0, j-win_size)
                    up = min(sent_num, j+win_size+1)
                    index = torch.arange(low, up, device=self.device)
                new_tgt_list.append(self._rebuild_tgt(tgt_list[i], index, self.seg_token))

        new_tgt = pad_sequence(new_tgt_list, batch_first=True, padding_value=self.pad_token)

        return new_tgt

    def _build_doc_tgt(self, tgt, vec, ex_segs, win_size=1, max_k=6, sigma=1.0):

        vec_list = torch.split(vec, ex_segs)
        tgt_list = torch.split(tgt, ex_segs)

        new_tgt_list = []
        index_list = []
        shift_list = []
        accum_index = 0
        for idx in range(len(ex_segs)):
            ex_vec = vec_list[idx]
            sent_num = ex_segs[idx]
            ex_tgt = tgt_list[idx]
            tgt_length = ex_tgt[:, 1:].ne(self.pad_token).sum(dim=1).float()
            topk_ids = self._centrality_rank(ex_vec, sent_num, tgt_length, win_size, max_k, sigma)
            new_tgt_list.append(self._rebuild_tgt(ex_tgt, topk_ids, self.seg_token))
            shift_list.append(topk_ids)
            index_list.append(topk_ids + accum_index)
            accum_index += sent_num
        new_tgt = pad_sequence(new_tgt_list, batch_first=True, padding_value=self.pad_token)
        return new_tgt, index_list, shift_list

    def _centrality_rank(self, vec, sent_num, tgt_length, win_size, max_k, sigma, eta=0.5, min_length=5):

        assert vec.size(0) == sent_num
        sim = torch.sigmoid(self.cup_bilinear(vec.unsqueeze(1).expand(sent_num, sent_num, -1).contiguous(),
                                              vec.unsqueeze(0).expand(sent_num, sent_num, -1).contiguous())
                            ).squeeze().detach()
        # sim = torch.sigmoid(torch.mm(vec, vec.transpose(0, 1)))
        # sim = torch.cosine_similarity(
        #    vec.unsqueeze(1).expand(sent_num, sent_num, -1).contiguous().view(sent_num * sent_num, -1),
        #    vec.unsqueeze(0).expand(sent_num, sent_num, -1).contiguous().view(sent_num * sent_num, -1)
        # ).view(sent_num, sent_num).detach()

        # calculate sim weight
        k = min(max(sent_num // (win_size*2+1), 1), max_k)
        var = sent_num / k * 1.
        x = torch.arange(sent_num, device=self.device, dtype=torch.float).unsqueeze(0).expand_as(sim)
        u = torch.arange(sent_num, device=self.device, dtype=torch.float).unsqueeze(1)
        weight = torch.exp(-(x-u)**2 / (2. * var**2)) * (1. - torch.eye(sent_num, device=self.device))
        # weight = 1. - torch.eye(sent_num, device=self.device)
        sim[tgt_length < min_length, :] = -1e20

        # Calculate centrality and select top k sentence.
        topk_ids = torch.empty(0, dtype=torch.long, device=self.device)
        mask = torch.zeros([sent_num, sent_num], dtype=torch.float, device=self.device)
        for _ in range(k):
            mean_score = torch.sum(sim * weight, dim=1) / max(sent_num-1, 1)
            max_v, _ = torch.max(sim * weight * mask, dim=1)
            centrality = eta*mean_score - (1-eta)*max_v
            _, top_id = torch.topk(centrality, 1, dim=0, sorted=False)
            topk_ids = torch.cat([topk_ids, top_id], 0)
            sim[topk_ids, :] = -1e20
            mask[:, topk_ids] = 1.
        topk_ids, _ = torch.sort(topk_ids)
        """
        centrality = torch.sum(sim * weight, dim=1)
        _, topk_ids = torch.topk(centrality, k, dim=0, sorted=False)
        topk_ids, _ = torch.sort(topk_ids)
        """
        return topk_ids

    def _add_mask(self, src, mask_src):
        pm_index = torch.empty_like(mask_src).float().uniform_().le(self.args.mask_token_prob)
        ps_index = torch.empty_like(mask_src[:, 0]).float().uniform_().gt(self.args.select_sent_prob)
        pm_index[ps_index] = 0
        # Avoid mask [PAD]
        pm_index[(1-mask_src).byte()] = 0
        # Avoid mask [CLS]
        pm_index[:, 0] = 0
        # Avoid mask [SEG]
        pm_index[src == self.seg_token] = 0
        src[pm_index] = self.mask_token
        return src

    def _build_cup(self, bsz, ex_segs, win_size=1, negative_num=2):

        cup = torch.split(torch.arange(0, bsz, dtype=torch.long, device=self.device), ex_segs)
        tgt = torch.split(torch.ones(bsz), ex_segs)
        cup_list = []
        cup_origin_list = []
        tgt_list = []
        negative_list = []
        for i in range(len(ex_segs)):
            sent_num = ex_segs[i]
            cup_low = cup[i][0].item()
            cup_up = cup[i][sent_num-1].item()
            cup_index = cup[i].repeat(win_size*2*(negative_num+1))
            tgt_index = tgt[i].repeat(win_size*2*(negative_num+1))
            cup_origin_list.append(cup[i].repeat(win_size*2*(negative_num+1)))
            tgt_index[sent_num*win_size*2:] = 0
            for j in range(cup_index.size(0)):
                if tgt_index[j] == 1:
                    cup_temp = cup_index[j]
                    window_list = [t for t in range(max(cup_index[j]-win_size, cup_low),
                                                    min(cup_index[j]+win_size, cup_up)+1)
                                   if t != cup_index[j]]
                    cup_temp = window_list[(j // sent_num) % len(window_list)]
                else:
                    cand_list = [t for t in range(cup_low, max(cup_index[j]-win_size, cup_low))] + \
                                [t for t in range(min(cup_index[j]+win_size, cup_up), cup_up)]
                    cup_temp = cand_list[random.randint(0, len(cand_list)-1)]
                cup_index[j] = cup_temp
            negative_list.append((cup_index[sent_num*win_size*2:]-cup_low).
                                 view(negative_num*win_size*2, -1).transpose(0, 1))
            cup_list.append(cup_index)
            tgt_list.append(tgt_index)

        tgt = torch.cat(tgt_list, dim=0).float().to(self.device)
        cup_origin = torch.cat(cup_origin_list, dim=0)
        cup = torch.cat(cup_list, dim=0)
        negative_sample = torch.cat(negative_list, dim=0)

        return cup, cup_origin, tgt[cup != -1], negative_sample

    def _build_option_window(self, bsz, ex_segs, win_size=1, keep_ratio=0.1, replace_ratio=0.2):

        assert keep_ratio + replace_ratio <= 1.
        noise_ratio = 1 - keep_ratio - replace_ratio

        window_size = 2*win_size+1
        index = torch.split(torch.arange(1, bsz+1, dtype=torch.long, device=self.device), ex_segs)
        # 2 means noise addition, 1 means keep the memory, 0 means replacement
        tgt = torch.zeros([bsz, window_size], device=self.device, dtype=torch.int)
        prob = torch.empty([bsz, window_size], device=self.device).uniform_()
        tgt.masked_fill_(prob.lt(noise_ratio), 2)
        tgt.masked_fill_(prob.ge(1-keep_ratio), 1)
        tgt = torch.split(tgt, ex_segs)

        for i in range(len(ex_segs)):
            sent_num = ex_segs[i]
            index_pad = F.pad(index[i], (self.args.win_size, self.args.win_size))
            for j in range(sent_num):
                window = index_pad[j:j+window_size]
                # Avoiding that all elements are 0
                if torch.sum(tgt[i][j].byte()*(window > 0)) == 0:
                    tgt[i][j][win_size] = 2
                tgt[i][j][window == 0] = -1
        tgt = torch.cat(tgt, 0)
        return tgt

    def _fast_translate_batch(self, batch, memory_bank, max_length, init_tokens=None, memory_mask=None,
                              min_length=2, beam_size=3, ignore_mem_attn=False):

        batch_size = memory_bank.size(0)

        dec_states = self.decoder.init_decoder_state(batch.src, memory_bank, with_cache=True)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        memory_bank = tile(memory_bank, beam_size, dim=0)
        init_tokens = tile(init_tokens, beam_size, dim=0)
        memory_mask = tile(memory_mask, beam_size, dim=0)

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=self.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=self.device)

        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=self.device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=self.device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = [[] for _ in range(batch_size)]  # noqa: F812

        for step in range(max_length):
            if step > 0:
                init_tokens = None
            # Decoder forward.
            decoder_input = alive_seq[:, -1].view(1, -1)
            decoder_input = decoder_input.transpose(0, 1)

            dec_out, dec_states, _ = self.decoder(decoder_input, memory_bank, dec_states, init_tokens, step=step,
                                                  memory_masks=memory_mask, ignore_memory_attn=ignore_mem_attn)

            # Generator forward.
            log_probs = self.generator(dec_out.transpose(0, 1).squeeze(0))

            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if(cur_len > 3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        if(len(words) <= 3):
                            continue
                        trigrams = [(words[i-1], words[i], words[i+1]) for i in range(1, len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            log_probs[i] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.args.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        _, pred = best_hyp[0]
                        results[b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            if memory_bank is not None:
                memory_bank = memory_bank.index_select(0, select_indices)
            if memory_mask is not None:
                memory_mask = memory_mask.index_select(0, select_indices)
            if init_tokens is not None:
                init_tokens = init_tokens.index_select(0, select_indices)

            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        results = [t[0] for t in results]
        return results

    def forward(self, batch):

        src = batch.src
        tgt = batch.tgt
        segs = batch.segs
        mask_src = batch.mask_src
        ex_segs = batch.ex_segs

        if self.training:
            # Sample some dialogue utterances to do auto-encoder
            ex_size = batch.src.size(0)
            ex_index = [i for i in range(ex_size)]
            random.shuffle(ex_index)
            ex_indexs = torch.tensor(ex_index, dtype=torch.long, device=self.device)
            ex_sample_indexs = ex_indexs[:max(int(ex_size * self.args.sample_ratio), 1)]

            # Get Context utterance training samples and targets
            cup_index, cup_original_index, cup_tgt, negative_samples = \
                self._build_cup(src.size(0), ex_segs, self.args.win_size, self.args.negative_sample_num)
            setattr(batch, 'cup_tgt', cup_tgt)

        option_mask = self._build_option_window(src.size(0), ex_segs, win_size=self.args.win_size,
                                                keep_ratio=self.args.ps if self.training else 1.,
                                                replace_ratio=self.args.pr if self.training else 0.)

        if self.training:
            # Build noised src
            noised_src, noised_src_mask, noised_src_segs = \
                self._build_noised_src(src, ex_segs, samples=negative_samples,
                                       expand_ratio=self.args.expand_ratio)
        # build context tgt
        context_tgt = self._build_context_tgt(tgt, ex_segs, self.args.win_size,
                                              modify=self.training, mask=option_mask)
        setattr(batch, 'context_tgt', context_tgt)

        # DAE: Randomly mask tokens
        if self.training:
            src = self._add_mask(src.clone(), mask_src)
            noised_src = self._add_mask(noised_src, noised_src_mask)

        if self.args.encoder == "bert":
            top_vec = self.encoder(src, segs, mask_src)
        else:
            src_emb = self.embeddings(src)
            top_vec = self.encoder(src_emb, 1-mask_src)
        clss = top_vec[:, 0, :]

        # Hierarchical encoder
        cls_list = torch.split(clss, ex_segs)
        cls_input = nn.utils.rnn.pad_sequence(cls_list, batch_first=True, padding_value=0.)
        cls_mask_list = [mask_src.new_zeros([length]) for length in ex_segs]
        cls_mask = nn.utils.rnn.pad_sequence(cls_mask_list, batch_first=True, padding_value=1)

        hier = self.hier_encoder(cls_input, cls_mask)
        hier = hier.view(-1, hier.size(-1))[(1-cls_mask.view(-1)).byte()]

        if self.training:

            # calculate cup score
            cup_tensor = torch.index_select(clss, 0, cup_index)
            origin_tensor = torch.index_select(clss, 0, cup_original_index)
            cup_score = torch.sigmoid(self.cup_bilinear(origin_tensor, cup_tensor)).squeeze()
            # cup_score = torch.sigmoid(origin_tensor.unsqueeze(1).bmm(cup_tensor.unsqueeze(-1)).squeeze())

            # noised src encode
            if self.args.encoder == "bert":
                noised_top_vec = self.encoder(noised_src, noised_src_segs, noised_src_mask)
            else:
                noised_src_emb = self.embeddings(noised_src)
                noised_top_vec = self.encoder(noised_src_emb, 1-noised_src_mask)
            noised_clss = noised_top_vec[:, 0, :]
            noised_cls_mem = self._build_memory_window(ex_segs, noised_clss, clss, option_mask, negative_samples)
            noised_cls_mem = self.pos_emb(noised_cls_mem)

            # sample training examples
            context_tgt_sample = torch.index_select(context_tgt, 0, ex_sample_indexs)
            noised_cls_mem_sample = torch.index_select(noised_cls_mem, 0, ex_sample_indexs)
            hier_sample = torch.index_select(hier, 0, ex_sample_indexs)
        else:
            cup_score = None

        if self.training:

            dec_state = self.decoder.init_decoder_state(noised_src, noised_cls_mem_sample)

            decode_context, _, _ = self.decoder(context_tgt_sample[:, :-1], noised_cls_mem_sample, dec_state,
                                                init_tokens=hier_sample)
            doc_data = None

            # For loss computation.
            if ex_sample_indexs is not None:
                batch.context_tgt = context_tgt_sample

        else:
            decode_context = None
            # Build paragraph tgt based on centrality rank.
            doc_tgt, doc_index, _ = self._build_doc_tgt(tgt, clss, ex_segs, self.args.win_size, self.args.ranking_max_k)
            centrality_segs = [len(iex) for iex in doc_index]
            centrality_index = [sum(centrality_segs[:i]) for i in range(len(centrality_segs)+1)]
            doc_index = torch.cat(doc_index, 0)
            setattr(batch, 'doc_tgt', doc_tgt)

            doc_hier_sample = torch.index_select(hier, 0, doc_index)

            # original cls mem
            cls_mem = self._build_memory_window(ex_segs, clss)
            cls_mem = self.pos_emb(cls_mem)
            doc_cls_mem = torch.index_select(cls_mem, 0, doc_index)

            # Context aware doc target
            context_doc_tgt = torch.index_select(context_tgt, 0, doc_index)
            setattr(batch, 'context_doc_tgt', context_doc_tgt)
            setattr(batch, 'doc_segs', centrality_index)

            doc_context_long = self._fast_translate_batch(batch, doc_cls_mem, self.max_length, init_tokens=doc_hier_sample,
                                                          min_length=2, beam_size=self.beam_size)
            doc_context_long = [torch.cat(doc_context_long[centrality_index[i]:centrality_index[i+1]], 0) for i in range(len(centrality_segs))]

            doc_data = doc_context_long

        return cup_score, decode_context, doc_data
