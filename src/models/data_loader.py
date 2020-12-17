import gc
import glob
import random
import torch

from others.logging import logger


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, args, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = (x[0] for x in data)

            ex_segs = [len(s) for s in pre_src]

            src = torch.tensor(self._pad(sum((x[0] for x in data), []), 0))
            tgt = torch.tensor(self._pad(sum((x[1] for x in data), []), 0))
            segs = torch.tensor(self._pad(sum((x[2] for x in data), []), 0))

            mask_src = 1 - (src == 0)
            mask_tgt = 1 - (tgt == 0)

            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))
            setattr(self, 'ex_segs', ex_segs)

            original_str = sum((x[3] for x in data), [])
            setattr(self, 'original_str', original_str)

            if is_test:
                summ_id = [x[-2] for x in data]
                summ_txt = [x[-1] for x in data]
                setattr(self, 'summ_id', summ_id)
                setattr(self, 'summ_txt', summ_txt)

    def __len__(self):
        return self.batch_size


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "dev", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        pt = args.data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def batch_size_fn(new, count):
    tgt = new[1]
    global sent_num, max_n_tokens
    if count == 1:
        sent_num = 0
        max_n_tokens = 0
    max_n_tokens = max(max_n_tokens, max([len(s) for s in tgt]))
    sent_num += len(tgt)
    src_elements = sent_num * max_n_tokens
    if (count > 6):
        return src_elements + 1e3
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, batch_size, batch_ex_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.batch_ex_size = batch_ex_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args, dataset=self.cur_dataset,
                            batch_size=self.batch_size, batch_ex_size=self.batch_ex_size,
                            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, batch_ex_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.batch_ex_size, self.is_test, self.dataset = \
            batch_size, batch_ex_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        self.batch_size_fn = batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex):

        src_ex = []
        tgt_ex = []
        segs_ex = []
        original_txt_ex = []

        session = ex["session"]
        if len(session) <= 2*self.args.win_size+1:
            return None

        if "summary" in ex.keys():
            summ_id = ex["summary"]["id"]
            summ_txt = ex["summary"]["original_txt"]
        else:
            summ_id, summ_txt = None, None

        for turn in session:
            src = turn['src_id'][:self.args.max_src_len+1]
            tgt = turn['tgt_id'][:self.args.max_src_len+2][:-1]+[2]
            segs = turn['segs'][:self.args.max_src_len+1]
            original_txt = turn['original_txt']

            end_id = [src[-1]]
            src = src[:-1][:self.args.max_pos - 1] + end_id
            segs = segs[:self.args.max_pos]

            src_ex.append(src)
            tgt_ex.append(tgt)
            segs_ex.append(segs)
            original_txt_ex.append(original_txt)
            if len(src_ex) >= self.args.max_src_num:
                break

        return src_ex, tgt_ex, segs_ex, original_txt_ex, summ_id, summ_txt

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if len(ex) == 0:
                continue
            ex = self.preprocess(ex)
            if ex is None:
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size, batch_ex_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            if len(minibatch) >= batch_ex_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            p_batch = self.batch(buffer, self.batch_size, self.batch_ex_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(self.args, minibatch, self.device, self.is_test)

                yield batch
            return
