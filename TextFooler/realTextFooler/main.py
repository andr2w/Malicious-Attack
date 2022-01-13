import IPython
from model import Config
from data import build_vocab


def main():
    config = Config()
    dataset = build_vocab(config.data_path, config.tokenizer, config.max_len)
    import IPython; IPython.embed(); exit(1)

if __name__ == '__main__':
    main()