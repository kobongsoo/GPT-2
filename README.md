# GPT-2

## 1.개요
- Generative Pretrained Transformer2
- Open AI 에서 개발한 모델로, transformers 의 decode 부분을 12개의 레이어를 쌓아서 만든 language model임
- **Fine-Tuning을 위한 layer 추가가 필요 없음**
- **BERT 처럼 다국어 모델이 없음**(Open AI에서 배포한 모델은 모두 영어 모델임)
- GPT-1, [GPT-2](https://github.com/openai/gpt-2)는 공개되었지만, GPT-3는 공개 안됨

#### [GPT 모델 종류]
|모델|파라메터수|출시일|기타|
|:--------|:----------|:-----:|-------------------|
|GPT-1|117M|2018||
|GPT-2 SMALL|117M|2019||
|GPT-2 MEDIUM|345M|2019||
|GPT-2 LARGE|762M|2019||
|GPT-2 EXTRA-LARGE|1,542M|2019||
|GPT-3|175,000M|2020||
|[KoGPT-2 Ver2.0](https://github.com/SKT-AI/KoGPT2)|125M|2021|SKT 한국어 GPT-2 모델|

## 2.  GPT-2 VS BERT
- GPT-2는 이전 단어들을 가지고, 다음 단어를 예측하는 **자기 회귀 모델(auto-regressive model)** 임 
- BERT는 해당 스텝에서 모든 단어를 고려하는 Self-Attention 사용하고,
반면, GPT-2는 해당스탭에서 오른쪽에 있는 단어들은 고려 하지 않는 **Masked Self-Attention(CLM(Causal language modeling) 방식)** 사용함
 ![image](https://user-images.githubusercontent.com/93692701/167518193-15bf7128-2e8c-427f-ba87-99bf4c11b936.png)

## 3. Fine-Tuning
- 한국어 [KoGPT-2 Ver2.0](https://github.com/SKT-AI/KoGPT2) 를 가지고 Fine-Tuning 하는 예시임
- KoGPT-2는 Tokenizer로 **SentencePiece** 방식 이용함.(GPT-2도 SentencePiece 방식 이용함)
- KoGPT-2는 vocab size(단어 계수) 가 51,200 개며, embedding 차원수는 768임

### 1. 텍스트 생성(Text Generation)
- gpt-2 모델 선언(GPT2LMHeadModel), tokenizer 선언(PreTrainedTokenizerFast)
- **StartToken + 문장 + EndToken** 식으로 된 훈련 dataset 생성
- 모델에 input_ids, lables, attention_mask 을 입력하여 훈련 시킴
- 원래 input_ids = 100,200,101,201,300,301 토큰이 입력된다면, labels은 input_ids 좌측으로 shift된 값 labels = 200,101,201,300,301 식으로 입력이 이루어 저야 하는데, **GPT2LMHeadModel 를 이용하면, labels = input_ids 와 똑같이 입력하면 내부적으로 label값을 shift 시킴**

### 2. 추상(생성)요약(Abstractive summarization)
