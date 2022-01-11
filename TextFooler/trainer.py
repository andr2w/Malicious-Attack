import util as ut
import torch
import torch.nn as nn

def train(train_iter, net, optimizer, loss_function, config, devices=ut.try_all_gpus()):
    # train
    num_epochs = config.num_epochs
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    loss_list = []
    acc_list = []
    net.train()
    for epoch in range(num_epochs):
        print('Epoch: [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (X, y) in enumerate(train_iter):
            X = X.to(devices[0])
            y = y.to(devices[0])
            
            optimizer.zero_grad()
            # forward
            y_hat = net(X)
            loss = loss_function(y_hat, y)

            # backward
            loss.backward()
            optimizer.step()

            # if (i + 1) % 10 == 0:
            #     total_steps = len(train_iter)
            #     train_acc = ut.compute_accuracy(y_hat, y)
            #     msg = 'Iter: [{}/{}], Loss: {}, Train_acc: {}%'
            #     print(msg.format(i+1, total_steps, loss.item(), train_acc * 100))
            #     loss_list.append(loss.item())
            #     acc_list.append(train_acc)
            train_acc = ut.compute_accuracy(y_hat, y)
            print('Loss: {}, Train_acc: {}%'.format(loss.item(), train_acc * 100))

    torch.save(net.state_dict(), config.saved_path)
    # ut.plot_acc_loss(loss_list, acc_list)
    print(loss_list, acc_list)



# def test(test_iter, net, config):
#     net.load_state_dict(torch.load(config.saved_path), strict=False)
#     with torch.no_grad():
#         true = 0
#         total = 0
#         for X, y in test_iter:
#             y_hat = net(X)

#             _, predict = torch.max(y_hat.data, 1)
#             total += y.size(0)
#             true += (predict == y).sum().item()

#     print('Test Accuracy: {}%'.format((true/total) * 100))

