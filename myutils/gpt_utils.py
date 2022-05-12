import os
import random
import numpy as np
import torch
import time

from tqdm.notebook import tqdm
from torch.utils.data.dataset import Dataset
from nltk.translate.bleu_score import sentence_bleu

###########################################################################################
# gpt2-text generation dataset 생성 
#
# => 입력 : 문장들.
# => 출력 : input_ids => bos_token + 문장 + eos_token
#           attention_maskes => 111111000000
###########################################################################################
def TextGeneration_tokenizer_seq(sentence, tokenizer, max_length):
    return tokenizer(tokenizer.bos_token + sentence + tokenizer.eos_token, truncation=True, max_length=max_length, padding="max_length")

class TextGeneration_Dataset(Dataset):
    def __init__(self, sentences, tokenizer, gpt2_type="gpt2", max_length=128):
        
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        
        for sentence in tqdm(sentences):
            encodings = TextGeneration_tokenizer_seq(sentence, tokenizer, max_length)
            #print(encodings) break
            self.input_ids.append(torch.tensor(encodings['input_ids']))
            self.attention_masks.append(torch.tensor(encodings['attention_mask']))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx]
###########################################################################################

###########################################################################################
# BLEU 스코어 구하는 함수
# =>입력된 문장에 대해, 일부 토큰을 제거한후 모델을 통해 생성된 문장과 비교하여 bleu 스코어를 구함
# => from nltk.translate.bleu_score import sentence_bleu 이용
#
# => input : tokenizer, 문장목록(리스트), 뒤에서 제거할 토큰 계수 
# => out :  bleu스코어(리스트)
#           candidate 문장 목록(리스트: 모델에서 출력 생성된 문장)
#           reference 문장 목록(리스트: 원래 문장)
###########################################################################################
def get_bleu_scores(model,                     # GPT-2 모델
                    tokenizer,                 # tokenizer
                    device,                    # device
                    sentences: list,           # 측정할 문장들(list)
                    remove_token_len: int=5,   # 뒤에서 제거할 토큰계수
                    show_text=False,           # True=input, candiate, reference, blue스코어등을 printf 함
                   ):
    model.eval()
    candidate_list = []
    reference_list = []
    #inps_list = []
    bleuscore_list = []
    count = 0
    
    for sentence in tqdm(sentences):
        input_seq = sentence
        sentence_encode = torch.tensor(tokenizer.encode(input_seq))
        #print(sentence_encode)
        sentence_encode_len = len(sentence_encode)
        #print(f'*sentence_encode_len:{sentence_encode_len}')
 
        # toekn 길이가 3보다 작으면 continue
        if sentence_encode_len < 3:
            continue
               
        # 문장 token 길이가 제거할 token+3 보다 크면=>뒤에 remove_token_len 만큼 토큰을 제거함
        if sentence_encode_len > remove_token_len + 3:
            input_ids = sentence_encode[:-(remove_token_len)]
        # 문장 token 길이가 제거할 token+3 보다 작거나 같으면=>뒤에 2개 token만 제거함 
        else:
            input_ids = sentence_encode[:-2]
        
        #print(f'*input_ids:{input_ids}')
        input_decode = tokenizer.decode(input_ids, skip_special_tokens=True)
        #print(f'*input:{input_decode}')
        
        # 모델 실행
        outputs = model.generate(input_ids.unsqueeze(0).to(device),
                                 max_length=sentence_encode_len,
                                 repetition_penalty=2.0,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id,
                                 bos_token_id=tokenizer.bos_token_id,
                                 use_cache=True)
        
        decode = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print(f'*decode:{decode}')
        
        #==============================================================
        # bleu score 구함
        ref_list = []
        ref_list.append(sentence)  #references는 리스트로 변환해야 함
        ref = list(map(lambda ref: ref.split(), ref_list))
        
        bleu = sentence_bleu(ref, decode.split())
        #==============================================================
        
        count += 1
        if show_text:
            print(f"*input    : {input_decode} *blue: {bleu} *-{count}")
            print(f"*reference: {sentence}")
            print(f"*candidate: {decode}\n")
       
        bleuscore_list.append(bleu)   
        #inp_list.append(input_decode)
        candidate_list.append(decode)   # 모델이 생성한 문장
        reference_list.append(sentence) # 실제 문장
    
    return candidate_list, reference_list, bleuscore_list
###########################################################################################
