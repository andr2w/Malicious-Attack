from abc import ABC, abstractmethod
import torch
import numpy as np
import torch.nn as nn
import transformers


def batch_model_predict(model_predict, inputs, batch_size=32):
    '''
    Runs prediction on iterable ``inputs`` using batch size ``batch_size``.
    '''
    outputs = []
    i = 0
    while i < len(inputs):
        batch = inputs[i: i + batch_size]
        batch_preds = model_predict(batch)
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        if not isinstance(batch_preds, np.ndarray):
            batch_preds = np.array(batch_preds)
        outputs.append(batch_preds)
        i += batch_size
    
    return np.concatenate(outputs, axis=0)




class ModelWrapper(ABC):
    @abstractmethod
    def __call__(self, text_input_list, **kwargs):
        raise NotImplementedError()
    
    def get_grad(self, text_input):
        raise NotImplementedError()
    
    def _tokenize(self, inputs):
        raise NotImplementedError()

    def tokenize(self, inputs, strip_prefix=False):
        tokens = self._tokenize(inputs)
        if strip_prefix:
            strip_chars = ['##', "Ġ", "__"]

            def strip(s, chars):
                for c in chars:
                    s = s.replace(c, "")
                return s

            tokens = [[strip(t, strip_chars) for t in x] for x in tokens]
        
        return tokens



class PytorchModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"Pytorch model must be torch.nn.Module, got type {type(model)}"
            )
        
        self.model = model
        self.tokenizer = tokenizer

    def to(self, device):
        self.model.to(device)
    
    '''
    This allow me to call the model just like call a funciton
    '''
    def __call__(self, text_input_list, batch_size=32):
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer(text_input_list)
        ids = torch.tensor(ids).to(model_device)

        with torch.no_grad():
            outputs = batch_model_predict(
                self.model, ids, batch_size=batch_size
            )

        return outputs

    def get_grad(self, text_input, loss_fn=nn.CrossEntropyLoss()):
        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )

        if not isinstance(loss_fn, torch.nn.Module):
            raise ValueError("Loss function must be of type `torch.nn.Module`.")

        self.model.train()

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])
        
        emb_hook = embedding_layer.register_backward_hood(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer([text_input])
        ids = torch.tensor(ids).to(model_device)

        predictions = self.model(ids)

        output = predictions.argmax(dim=1)
        loss = loss_fn(predictions, output)
        loss.backward()

        grad = torch.transpose(emb_grads[0], 0, 1)[0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": ids[0].tolist(), "gradient": grad}
        
        return output


    def _tokenize(self, inputs):
        return [self.tokenizer.convert_ids_to_tokens(self.tokenizer(x)) for x in inputs]


    
class HuggingFaceModelWrapper(PytorchModelWrapper):
    def __init__(self, model, tokenizer):
        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list):
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        return outputs.logits

    def get_grad(self, text_input):
        '''
        Get gradient of loss with respect to input tokens.
        '''

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels = labels)[0]

        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss."
                "One cause for this might be if you instantiated your model using `transformer.AutoModel`"
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )
        
        loss.backward()

        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict['input_ids'], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)['input_ids'][0]
            )
            for x in inputs
        ]





