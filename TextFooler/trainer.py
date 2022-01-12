from re import I
import util as ut
import torch
import torch.nn as nn
from tqdm import tqdm



def train(train_iter, net, optimizer, loss_function, config, devices=ut.try_all_gpus()):
    net = nn.DataParallel(net, device_ids =devices).to(devices[0])
    print('Total batchs: {}'.format(len(train_iter)))
    net.train()
    for epoch in range(config.num_epochs):
        loop = tqdm(enumerate(train_iter), total=len(train_iter), leave=False)
        for batch_index, (X, y) in loop:
            X = X.to(devices[0]) 
            y = y.to(devices[0])

            # forward
            y_hat = net(X)

            # compute the loss
            loss = loss_function(y_hat, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # Update the model's parameters
            optimizer.step()

            # update progress bar 
            loop.set_description('[{}/{}]'.format(epoch + 1, config.num_epochs))
            train_acc = ut.compute_accuracy(y_hat, y)
            loop.set_postfix(iter = batch_index, loss = loss.item(), acc = train_acc)            

    torch.save(net.state_dict(), config.saved_path)

    # ut.plot_acc_loss(loss_list, acc_list)


def test(test_iter, net, config, devices=ut.try_all_gpus()):
    net = nn.DataParallel(net, device_ids = devices).to(devices[0])
    net.load_state_dict(torch.load(config.saved_path), strict=False)
    net.eval()
    with torch.no_grad:
        true = 0
        total = 0 
        loop = tqdm(test_iter, total=len(test_iter), leave=False)
        for X, y in loop:
            X = X.to(devices[0])
            y = y.to(devices[0])
            y_hat = net(X)
            _, predict = torch.max(y_hat.data, 1)
            total += y.size(0)
            true += (predict == y).sum().item()



    print('Test Acc: {}%'.format((true / total) * 100))


