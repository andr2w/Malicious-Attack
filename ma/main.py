from model import Config, Model
from data import build_vocab, build_iter
from attack import attack


def main():
    config = Config()
    data_set = build_vocab(config.data_path, config.tokenizer)
    data_iter = build_iter(data_set, config.batch_size)
    net = Model(config).to(device=config.device)
    attack(data_iter, net, config)

if __name__ == '__main__':
    main()