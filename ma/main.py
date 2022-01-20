from re import I
from model import Config, Model
from data import build_vocab, build_iter
from attack import attack
from wrapper.model_wrapper import HuggingFaceModelWrapper
import torch


def main():
    config = Config()
    data_set = build_vocab(config.data_path, config.tokenizer)
    data_iter = build_iter(data_set, config.batch_size)
    net = Model(config).to(device=config.device)
    net.to(config.device)
    net.load_state_dict(torch.load(config.saved_path, map_location=torch.device('cpu')))
    net = HuggingFaceModelWrapper(net, config.tokenizer) 
    attack(data_iter, net, config)

if __name__ == '__main__':
    main()