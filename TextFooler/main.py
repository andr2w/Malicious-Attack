from model import Config
from data import dataloader

def main(): 
    config = Config()
    train_iter, test_iter = dataloader(config)
    
    for X, y in train_iter:
        # print(X)
        
        '''
        X.shape = [64, 35]
        where 64 is the batch_size;
        35 is the padding length == num of time steps
        \mathcal{X} = \{x_1, x_2, x_3, ..., x_35\}
        
        X.T.shape = [35, 64]

        The vocab size
        '''
        
        # print(X.shape)
        # print(X.T.shape)
        print('Hello World')
        
        break
    #for X, y in test_iter:
    #    print(X)
    #    print(len(X))
    #    print(y)
    #    print(len(y))
    #    break
# The problem is in the train_test_split


if __name__ == '__main__':
    main()