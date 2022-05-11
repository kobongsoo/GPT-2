# GPT-2 Tokenizer
- 허깅페이스 GPT2TokenizerFast, PreTrainedTokenizerFast 를 이용하면, 손쉽게 GPT2 Tokenizer를 불러올수 있다.(*참고로 BERT는 BertWordpieceTokenizer 이용함)
- 여기서는 Tokenizer vocab을 신규 생성하는 방법(SentencePieceBPETokenizer/ByteLevelBPETokenizer) 과, 기존 KoGPT2 모델 vocab에 추가하는 방법(SentencePieceBPETokenizer)에 대해 설명한다.

### GPT2: ByteLevelBPETokenizer  
- GPT-2는 tokenizer 로 **ByteLevelBPETokenizer** 를 이용한다. 
- ByteLevelBPETokenizer 는 모든 단어들은 **유니코드 바이트 수준**으로 토큰화 됨. (**한글 1자는 3개의 유니코드 바이트로 표현됨** 예: 안녕하세요 > ìķĪëħķíķĺìĦ¸ìļĶ, 영어 1자는 1개의 유니코드 바이트로 )
- GPT-2 는 파일로 vocab.json 와 merges.txt 가 있는데, **vocab.json 바이트 레벨 BPE의 어휘 집합**이며 **merges.txt는 바이그램 쌍의 병합 우선순위**를 나타낸다.

### KoGPT2: SentencePieceBPETokenizer 
- 한국어 KoGPT2는 tokenizer로 **SentencePieceBPETokenizer** 를 이용한다. 
- SentencePieceBPETokenizer는 **subword 수준**으로 토큰화 됨
- 한국어 KoGPT-2는 tokenizer.json(vocab.json과 동일) 만 있고, merges.txt 파일은 없다.
- **한국어 모델은 SentencePieceBPETokenizer 를 이용**하여 만들어야 한다.


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
#### 2. 훈련한 SentencePieceBPETokenzer 를 PreTrainedTokenizerFast 와 연동
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
|소스| 설명 | 기타 |
|:------------|:--------------------------|:---------------|
|[new_token.ipynb](https://github.com/kobongsoo/GPT-2/blob/master/tokenizer/new_token.ipynb)| SentencePieceBPETokenizer 이용한 vocab 생성 | KoGPT2 방식|
|[new_token_bytelevelbpeokenizer](https://github.com/kobongsoo/GPT-2/blob/master/tokenizer/new_token_bytelevelbpeokenizer.ipynb)| ByteLevelBPETokenizer 이용한 vocab 생성 | GPT2 방식|

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


## 참고 사이트
- [허깅페이스 tokenizer 예제](https://gist.github.com/lovit/e11c57877aae4286ade4c203d6c26a32)
- [nlpbook tokenizer 강좌](https://ratsgo.github.io/nlpbook/docs/preprocess/vocab/)
