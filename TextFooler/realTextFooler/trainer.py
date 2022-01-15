import torch
from tqdm import tqdm
import util as ut


def train(train_iter, net, optimizer, loss_function, config, debug=True):
    if debug == True:
        
        for _, data in enumerate(train_iter, 0):
            ids = torch.LongTensor(data['ids']).to(device=config.device)
            mask = torch.LongTensor(data['mask']).to(device=config.device)
            y = torch.LongTensor(data['y']).squeeze(1).to(device=config.device)
            
            break

        loop = tqdm(range(config.num_epochs), total= config.num_epochs, leave=False)  
        for epoch in loop:    
            y_hat = net(ids, mask)
            loss = loss_function(y_hat, y)
            


            # clean the gradient
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update the parameters
            optimizer.step()

            acc = ut.compute_acc(y_hat, y)
            
            loop.set_description('[{}/ {}]'.format(epoch + 1, config.num_epochs))
            loop.set_postfix(loss = loss.item(), acc = acc)
        
    if debug == False:

        loss_list = [] 
        acc_list = []
        net.train()
        for epoch in range(config.num_epochs):
            loop = tqdm(enumerate(train_iter, 0), total=len(train_iter), leave=False)
            for batch_idx, data in loop:
                ids = torch.LongTensor(data['ids']).to(device=config.device)
                mask = torch.LongTensor(data['mask']).to(device=config.device)
                y = torch.LongTensor(data['y']).squeeze(1).to(device=config.device)
                # [batch_size, 1]
                # [batch_size]


                y_hat = net(ids, mask)
                loss = loss_function(y_hat, y)

                # clean the gradient
                optimizer.zero_grad()
                # back prob
                loss.backward()
                # update the para      
                optimizer.step()

                acc = ut.compute_acc(y_hat, y)
                loop.set_description('[{}/ {}]'.format(epoch + 1, config.num_epochs))
                loop.set_postfix(iter = batch_idx, loss = loss.item(), acc = acc)
                loss_list.append(loss.item())
                acc_list.append(acc.item())

        ut.plot_loss_acc(loss_list, acc_list)
        torch.save(net.state_dict(), config.saved_path)

def test(test_iter, net, config):
    net.to(config.device)
    net.load_state_dict(torch.load(config.saved_path), strict=False)
    net.eval()

    with torch.no_grad():
        true = 0
        total = 0
        for _, data in enumerate(test_iter, 0):
            ids = torch.LongTensor(data['ids']).to(device=config.device)
            mask = torch.LongTensor(data['mask']).to(device=config.device)
            y = torch.LongTensor(data['y']).squeeze(1).to(device=config.device)

            y_hat = net(ids, mask)
            _, predict = torch.max(y_hat.data, 1)
            total += y.size(0)
            true += (predict == y).sum().item()

        print('Test Acc: {}%'.format((true/total) * 100))

                