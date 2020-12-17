import torch
import torch.nn as nn
from models.neural import aeq
from models.neural import gumbel_softmax


class Generator(nn.Module):
    def __init__(self, vocab_size, dec_hidden_size, pad_idx):
        super(Generator, self).__init__()
        self.linear = nn.Linear(dec_hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.pad_idx = pad_idx

    def forward(self, x, use_gumbel_softmax=False):
        output = self.linear(x)
        output[:, self.pad_idx] = -float('inf')
        if use_gumbel_softmax:
            output = gumbel_softmax(output, log_mode=True, dim=-1)
        else:
            output = self.softmax(output)
        return output


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.
    These networks consider copying words
    directly from the source sequence.
    The copy generator is an extended version of the standard
    generator that computes three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary,
    computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    .. mermaid::
       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O
    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, output_size, input_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map, use_gumbel_softmax=False):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.
        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(batch, src_len, extra_words)``
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        batch, slen_, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        if use_gumbel_softmax:
            prob = gumbel_softmax(logits, log_mode=False, dim=1)
        else:
            prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(batch, -1, slen),
            src_map
        )
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs=None,
                         batch_dim=0, batch_offset=None, beam_size=1, segs_index=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = len(tgt_vocab)
    if segs_index is None:
        segs_index = torch.repeat_interleave(torch.arange(len(batch.ex_segs), dtype=torch.long),
                                             torch.tensor(batch.ex_segs) * beam_size, dim=0)

    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []

        if src_vocabs is None:
            src_vocab = batch.src_ex_vocab[segs_index[b]]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]

        for i in range(1, len(src_vocab)):
            sw = src_vocab.itos[i]
            ti = tgt_vocab[sw]
            if ti != 0:
                blank.append(offset + i)
                fill.append(ti)
        if blank:
            blank = torch.tensor(blank, device=scores.device)
            fill = torch.tensor(fill, device=scores.device)
            score = scores[:, b] if batch_dim == 1 else scores[b]
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 0.)
    return scores
