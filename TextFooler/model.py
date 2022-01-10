class Config:
    def __init__(self):
        self.data_path = 'Pheme.csv'
        self.batch_size = 64
        self.min_freq = 3
        self.pad_length = 35