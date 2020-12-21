import numpy as np
import transformers
import torch
from transformers import BertJapaneseTokenizer,BertModel

class BERT_EMB():
  def __init__(self):
    self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.max_len = 128
    self.model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

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
        output = self.model(tokens_tensor,attention_mask =attention_mask)

    last_hidden_states = output[0]
    features = last_hidden_states[:,0,:].numpy()

    return features