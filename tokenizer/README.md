# GPT-2 Tokenizer
- GPT-2 에서는 **SententencePieceTokenzer(SentencePieceBPETokenzer)** 를 이용한다. 
- 허깅페이스에서도 GPT2TokenizerFast, PreTrainedTokenizerFast 를 이용하는데 모두 SentencePieceTokenizer를 지원한다.(*참고로 BERT는 BertWordpieceTokenizer 이용함)
- 여기서는 Tokenizer vocab을 신규 생성하는 방법과, 기존 vocab에 추가하는 방법에 대해 설명한다.

## 1. Scratch Tokenizer(신규 생성)
#### 1. SetnecePieceBPETokenzer 훈련
- 훈련시, corpora 목록, vocab_size, min_frequency 등을 설정함
```
from tokenizers import SentencePieceBPETokenizer

corpus_path = '../../korpora/kowiki_20190620/wiki_20190620_small.txt'
stokenizer = SentencePieceBPETokenizer(add_prefix_space=True)

stokenizer.train(
    files = [corpus_path],
    vocab_size = 52000,  # 최대 vocab 계수 
    special_tokens = ["<cls>", "<eos>", "<mask>", "<unk>", "<pad>"],  # speical token 지정
    min_frequency = 5,   # 빈도수 
    show_progress = True,
    #limit_alphabet=10000, 
)
```
#### 2. 훈련한 SetnecePieceBPETokenzer 를 PreTrainedTokenizerFast 와 연동
```
from transformers import PreTrainedTokenizerFast
transforer_tokenizer = PreTrainedTokenizerFast(tokenizer_object=stokenizer)
```

#### 3. PreTrainedTokenizerFast tokenizer 저장
- 정상적으로 저장되면, 해당 폴더에 3개 json 파일이 생성됨(tokenizer.json, special_tokens_map.json, tokenizer_config.json)
```
import os
OUT_PATH = './mytoken'
os.makedirs(OUT_PATH, exist_ok=True)
transforer_tokenizer.save_pretrained(OUT_PATH)
```
예제) [new_token.ipynb](https://github.com/kobongsoo/GPT-2/blob/master/tokenizer/new_token.ipynb)

## 2. 기존 vocab 추가하기
#### 1. 기존 tokenizer 불러오기
- bos_token, eos_token, unk_token, pad_token, mask_token 등은 기존 tokenizer.json 파일에 저장된 실제 token값들을 입력해야 함
```
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast

model_path='../../model/gpt-2/kogpt-2/'   #tokenizer 파일 경로
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path,
                                                   bos_token='</s>',
                                                   eos_token='</s>',
                                                   unk_token='<unk>',
                                                   pad_token='<pad>',
                                                   mask_token='<mask>')
```
#### 2. 신규 vocab 추가
- 추가할 vocab들은 목록(list)로 여러개 지정할 수 있음
```
new_vocab = ['<question>', '<answer>'] # 추가할 vocab 들
new_tokenizer = tokenizer.add_tokens(new_vocab)
```
#### 3. 추가한 tokenzier 저장
- 정상적으로 저장되면, 해당 폴더에 3개 json 파일이 생성됨(tokenizer.json, special_tokens_map.json, tokenizer_config.json)
```
import os
OUT_PATH = '../../model/gpt-2/kogpt-2/addvocab'
os.makedirs(OUT_PATH, exist_ok=True)
tokenizer.save_pretrained(OUT_PATH)
```
예제) [add_token.ipynb](https://github.com/kobongsoo/GPT-2/blob/master/tokenizer/add_token.ipynb)
