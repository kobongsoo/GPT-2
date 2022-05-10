# GPT-2

## 1.개요
- Generative Pretrained Transformer2
- Open AI 에서 개발한 모델로, transformers 의 decode 부분을 12개의 레이어를 쌓아서 만든 language model임
- **Fine-Tuning을 위한 layer 추가가 필요 없음**
- **각 Task에 맞게 입력데이터와 정의한 특수토큰들을 조합하여 훈련 시킴** (예: Q&A Fine-Tuing 훈련 데이터 = 지문 + <Question 토큰> + 질문 + <Answer 토큰> + 정답)
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
- [허깅페이스](https://huggingface.co/) 라이브러리를 이용함, KoGPT-2 모델도 허깅페이스 [여기](https://huggingface.co/skt/kogpt2-base-v2)에 등록되어 있음
- KoGPT-2는 Tokenizer로 **SentencePiece** 방식 이용함.(GPT-2도 SentencePiece 방식 이용함)
- KoGPT-2는 vocab size(단어 계수) 가 51,200 개며, embedding 차원수는 768임

### 1. 텍스트 생성(Text Generation)
- gpt-2 모델 선언(GPT2LMHeadModel), tokenizer 선언(PreTrainedTokenizerFast)
- **<Start토큰> + 문장 + <End토큰>** 식으로 된 훈련 dataset 생성
- 모델에 input_ids, lables, attention_mask 을 입력하여 훈련 시킴
- 원래 input_ids = 100,200,101,201,300,301 토큰이 입력된다면, labels은 input_ids 좌측으로 shift된 값 labels = 200,101,201,300,301 식으로 입력이 이루어 저야 하는데, **허깅페이스의 GPT2LMHeadModel 를 이용하면, labels = input_ids 와 똑같이 입력하면 내부적으로 label값을 shift 시킴**

|소스|내용|
|:--------|:-------------------------------|
|[kogpt2_text_generation_finetuning](https://github.com/kobongsoo/GPT-2/blob/master/kogpt2_text_generation_finetuning.ipynb)|kogpt2 모델을 이용한 한국어 text generaion Fine-Tuning 훈련 예시임|
|[kogpt2_text_generation_test](https://github.com/kobongsoo/GPT-2/blob/master/kogpt2_text_generation_test.ipynb)|kogpt2 모델을 이용한 한국어 텍스트 생성 하는 예제임, top_k, top_p 등의 샘플링 수치를 적용할수 있음, input_ids = StartToken + 단어|

### 2. 추상(생성)요약(Abstractive summarization)
- gpt-2 모델 선언(GPT2LMHeadModel), tokenizer 선언(PreTrainedTokenizerFast)
- **요약할 문장+<생성토큰>+요약문+<End토큰>** 식으로 된 훈련 dataset 생성
- 모델에 input_ids, lables 을 입력하여 훈련 시킴

|소스|내용|
|:--------|:-------------------------------|
|[kogpt2_summarizer_finetuning](https://github.com/kobongsoo/GPT-2/blob/master/kogpt2_summarizer_finetuning.ipynb)|kogpt2 모델을 이용한 한국어 생성 요약 Fine-Tuning 훈련 예시임|
|[kogpt2_summarizer_test](https://github.com/kobongsoo/GPT-2/blob/master/kogpt2_summarizer_test.ipynb)|kogpt2 모델을 이용한 한국어 생성 요약하는 예제임, top_k 샘플링 수치를 적용할수 있음, input_ids = 요약할 문장 + 생성token|

### 3. 추론(NLI:Natural Language Inference)
- gpt-2 모델 선언(GPT2LMHeadModel), tokenizer 선언(PreTrainedTokenizerFast)
- **문장1 + <문장구분토큰>+ 문장2 + <추론토큰> + 추론값(Entailment, Netural, Contradiction)** 식으로 된 훈련 dataset 생성

### 4. Q&A
- gpt-2 모델 선언(GPT2LMHeadModel), tokenizer 선언(PreTrainedTokenizerFast)
- **지문 + <Question토큰> + 질문 + <Answer토큰> + 정답** 식으로 된 훈련 dataset 생성

## 4. Scrach 훈련(새롭게 훈련)
#### 1. Sentencepiece tokenizer 생성 
 - 단어 vocab 생성 (52,000 개 정도가 좋다고 함)
 - Taks에 맞는 특수 토큰들 추가(bos, eos, summarize, question, answer, classification, nli 토큰 등)
 
#### 2. 빈껍데기 GPT-2 모델 생성
- embedding size는 token size와 동일 해야 함
```
configuration = GPT2Config(vocab_size=52_000)
model = GPT2LMHeadModel(config=configuration) 
```

#### 3. 훈련
- test generation Fine-tuing 방식과 동일하게 훈련

#### 4. 모델과 tokenizer 저장
```
model.save_pretrained(MODEL_OUT_PATH)
tokenizer.save_pretrained(MODEL_OUT_PATH)
```
