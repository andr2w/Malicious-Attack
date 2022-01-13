from transformers import BertTokenizer


class Config:
    def __init__(self):
        self.data_path = 'Fake'
        self.max_len = 1200
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      