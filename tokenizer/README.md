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
