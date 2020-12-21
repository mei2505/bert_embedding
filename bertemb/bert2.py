# 必要なライブラリをインポート
import numpy as np
import transformers
import torch
from transformers import BertJapaneseTokenizer,BertModel
from torch import cuda

class BERT_EMB_0():
  def __init__(self):
    self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.device = 'cuda' if cuda.is_available() else 'cpu'
    self.max_len = 128

  def get_sentence_embedding(self, text): 

    # デバイスの指定
    self.model.to(self.device)

    inputs = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      padding='max_length',
      truncation='longest_first',
    )
    ids = inputs['input_ids']
    tokens_tensor = torch.LongTensor(ids).reshape(1, -1)
    tokens_tensor = tokens_tensor.to(self.device)

    self.model.eval()
    with torch.no_grad():
        all_encoder_layers, _ = self.model(tokens_tensor)

    if torch.cuda.is_available():    
            return all_encoder_layers[0][0].cpu().detach().numpy() # 0番目は [CLS] token, 768 dim の文章特徴量
    else:
            return all_encoder_layers[0][0].detach().numpy()
    
