import numpy as np
import transformers
import torch
from transformers import BertJapaneseTokenizer,BertModel

class BERT_EMB_4():
  def __init__(self):
    self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.max_len = 128
    self.model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking',output_hidden_states=True)

  def get_sentence_embedding(self, text):

    inputs = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      padding='max_length',
      truncation='longest_first',
    )

    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    tokens_tensor = torch.LongTensor(ids).reshape(1,-1)
    attention_mask = torch.LongTensor(mask).reshape(1,-1)

    self.model.eval()
    with torch.no_grad():
        outputs = self.model(tokens_tensor,attention_mask =attention_mask)
    # can use last hidden state as word embeddings
        last_hidden_state = outputs[0]
        word_embed_1 = last_hidden_state
    # Evaluating the model will return a different number of objects based on how it's  configured in the `from_pretrained` call earlier. In this case, becase we set `output_hidden_states = True`, the third item will be the hidden states from all layers. See the documentation for more details:https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]
    # initial embeddings can be taken from 0th layer of hidden states
    word_embed_2 = hidden_states[0]
    # sum of all hidden states
    word_embed_3 = torch.stack(hidden_states).sum(0)
    # sum of second to last layer
    word_embed_4 = torch.stack(hidden_states[2:]).sum(0)
    # sum of last four layer
    word_embed_5 = torch.stack(hidden_states[-4:]).sum(0)
    # concatenate last four layers
    word_embed_6 = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1)
        
    features = word_embed_6[:,0,:].numpy()

    return features

