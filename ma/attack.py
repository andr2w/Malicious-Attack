from re import I
import torch

def attack(data_iter, net, config):
    

    for _, data in enumerate(data_iter, 0):
        ids = torch.LongTensor(data['ids']).to(device=config.device)
        mask = torch.LongTensor(data['mask']).to(device=config.device)
        y = torch.LongTensor(data['y']).squeeze(1).to(device=config.device)

    text =  [config.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for id in ids]

    
   
    

    import IPython; IPython.embed(); exit(1)
    