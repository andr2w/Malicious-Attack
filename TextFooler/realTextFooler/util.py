import torch
import matplotlib.pyplot as plt

def compute_acc(y_hat, y):
    _, predict = torch.max(y_hat.data, 1)
    total = y.size(0)
    true = (predict == y).sum().item()
    acc = true / total
    
    return acc

def plot_loss_acc(loss_list, acc_list):
    x_1 = range(1, len(loss_list) + 1)
    x_2 = range(1, len(acc_list) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x_1, loss_list)
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(x_2, acc_list)
    plt.title('Training Accuracy')

    # save 
    plt.savefig('train_loss_acc.png', dpi=300, bbox_inches='tight')