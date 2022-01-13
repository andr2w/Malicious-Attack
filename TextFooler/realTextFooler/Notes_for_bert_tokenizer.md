# Tokenizer

A tokenizer is in charge of preparing the inputs for a model. 

The `PretrainedTokenizer` is doing the below things:

1. Tokenizing (splitting strings in sub-word token strings), converting tokens strings to ids and back. `This is actually what Vocab does in previous Recurrent Neural network`.