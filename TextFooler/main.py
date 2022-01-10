from model import Config
from data import dataloader

def main(): 
    config = Config()
    train_iter = dataloader(config)
    for X, y in train_iter:
        print(X)
        print(len(X))
        print(y)
        print(len(y))
        break

    # for X, y in test_iter:
    #     print(X)
    #     print(len(X))
    #     print(y)
    #     print(len(y))
    #     break

# The problem is in the train_test_split


if __name__ == '__main__':
    main()