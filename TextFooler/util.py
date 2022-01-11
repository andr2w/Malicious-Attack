import re
import torch
import matplotlib.pyplot as plt


def clean_text(text):
    """
    This function cleans the text in the following ways
    1. Replace websites with URL
    2. Replace 's with <space>'s (e.g., her's --> her 's)
    """
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "URL", text) # Replace urls with special token
    #text = text.replace("\'s", "")
    #text = text.replace("\'", "")
    #text = text.replace("n\'t", " n\'t")
    text = text.replace("@", "")
    text = text.replace(":", "")
    text = text.replace("#", "")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("&amp;", "")
    text = text.replace("&gt;", "")
    text = text.replace("\"", "")
    text = text.replace("$MENTION$", '')
    text = text.replace("$ URL $", '')
    text = text.replace("$URL$", '')
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("<end>", "")
    text = text.replace("|", "")
    text = text.lower()
    return text.strip()

def change_label(label):
    if label == 'non-rumours':
        label = 0
    elif label == 'rumours':
        label = 1
    
    return label

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def compute_accuracy(y_hat, y):
    _, predict = torch.max(y_hat.data, 1)
    total = y.size(0)
    true = (predict == y).sum().item()
    acc = true / total

    print('Total: ', total)
    print('true: ', true)
    print('----------------------------------')
    return acc

def plot_acc_loss(loss_list, acc_list):
    x_1 = range(1, len(loss_list) + 1)
    x_2 = range(1, len(acc_list) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_1, loss_list)
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(x_2, acc_list)
    plt.title('Training Accuracy')

    plt.show()

    