from re import I
from model import Config
from data import TokenEmbedding, return_data, Vocab, build_iterator

def main(): 
    config = Config()
    X_train, X_test, y_train, y_test = return_data(config.data_path)
    vocab = Vocab(X_train, min_freq=config.min_freq, reserved_tokens=['<pad>'])
    print(len(vocab))
    pretrain_embedding = TokenEmbedding(config.embedding_path)
    embeds = pretrain_embedding[vocab.idx_to_token]
    print(embeds.shape)
    
    train_iter, test_iter = build_iterator(X_train, X_test, y_train, y_test, vocab, config)    

        
    '''
        X.shape = [64, 35]
        where 64 is the batch_size;
        35 is the padding length == num of time steps
        \mathcal{X} = \{x_1, x_2, x_3, ..., x_35\}
        
        X.T.shape = [35, 64]

        The vocab size
        '''
        
   #     print(X.shape)
        #print(X.T.shape)
        #print('Hello World')
        
        #break
    ##for X, y in test_iter:
    #    print(X)
    #    print(len(X))
    #    print(y)
    #    print(len(y))
    #    break
# The problem is in the train_test_split

if __name__ == '__main__':
    main()

