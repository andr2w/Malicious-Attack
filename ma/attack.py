from re import I
import torch


def attack(data_iter, net, config):
    net.to(config.device)
    net.load_state_dict(torch.load(config.saved_path, map_location=torch.device('cpu')))
    net.eval()

    for _, data in enumerate(data_iter, 0):
        ids = torch.LongTensor(data['ids']).to(device=config.device)
        mask = torch.LongTensor(data['mask']).to(device=config.device)
        y = torch.LongTensor(data['y']).squeeze(1).to(device=config.device)

        break 

    import IPython; IPython.embed(); exit(1)
    