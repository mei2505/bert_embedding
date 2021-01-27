import numpy as np
import transformers
import torch
from transformers import BertJapaneseTokenizer,BertModel

class BERT_EMB_4():
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
        tokens_tensor = torch.LongTensor(ids).reshape(1,-1)

        self.model.eval()
        with torch.no_grad():
            _, _,  hidden_states  = self.model(tokens_tensor,output_hidden_states=True)
            # 最終４層の隠れ層からそれぞれclsトークンのベクトルを取得する
            vec1 = hidden_states[-1][:,0,:].view(-1, 768)
            vec2 = hidden_states[-2][:,0,:].view(-1, 768)
            vec3 = hidden_states[-3][:,0,:].view(-1, 768)
            vec4 = hidden_states[-4][:,0,:].view(-1, 768)

            # 4つのclsトークンを結合して１つのベクトルにする。
            sumvec = torch.cat([vec1, vec2, vec3, vec4], dim=1)
            feature = sumvec.numpy()

        return feature