# Tokenizer

A tokenizer is in charge of preparing the inputs for a model. 

The `PretrainedTokenizer` is doing the below things:

1. Tokenizing (splitting strings in sub-word token strings), converting tokens strings to ids and back, and encoding/decoding. `This is actually what Vocab does in previous Recurrent Neural network`. See `class Vocab` below;

2. Adding new tokens to the vocabulary in a way that is independent of the underlying structure.

3. Managing special tokens (like mask, beginning-of-sentence.) 

The docunment for `bertTokenizer` is [Link](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/tokenization_bert.py#L117)

```Python
import collections

def count_corpus(X):
    tokens = [token for data_dample in X for token in data_sample]
    cnt = collections.Counter(tokens)
    return cnt


class Vocab:
    def __init__(self, X, min_freq, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        cnt = count_corpus(X)

        self._token_freqs = sorted(cnt.items(), key=lambda x: x[1], reverse=True)

        # idx_to_token -- a List
        self.idx_to_token = ['<unk>'] + reserved_tokens 
        # token_to_idx -- a Dict
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

        # adding token to the index
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

        
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, data_sample):
        if not isinstance(data_sample, (list, tuple)):
            return self.token_to_idx.get(data_sample, self.unk) 
        return [self.__getitem__(token) for token in data_sample]

    @property
    def unk(self):
        return 0

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token(index) for index in indices]
```

## Important

**BertTokenizer** is a subclass inheritated from class **PretrainedTokenizer**.

The BertTokenizer can use method from PretrainedTokenizer --> **encode_plus** method to perform tokenization and generate the necessary outputs, namely: `ids`, `attention_mask`, `token_type_ids`.