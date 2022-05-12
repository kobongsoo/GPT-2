# GPT Evaluation(평가) 방법 
- GPT는 Text 생성 하는 Language Model이므로, 딱히 Accuracy를 평가 할 방법이 없다.
- 분류 모델 처럼 labe이 정해져 있는 모델로 Fine-Tuning 하면, 분류 label과 실제 label를 비교하여 Accuracy를 구하면 된다.
- Language Model에서는 성능 측정 방법으로, **BLEU(Bilingual Evaluation Understudy)** 를 대표적으로 사용한다.

### 1. BLEU Score(Bilingual Evaluation Understudy Score)
- BLEU는 기계번역 성능 측정 스코어로, **실제 번역 문장(references)** 과 **모델이 생성한 문장(candidate)** 을 비교하여 스코어를 구한다
- 측정 기준은 n-gram에 기반하며, 0~1 사이값을 가짐. BLEU 값이 높을 수록 성능이 좋음을 의미함. (BLEU에 대한 자세한 내용은 [여기](https://wikidocs.net/31695) 참조)
- NLTK에 **bleu_score 라이브러리**를 이용하면 됨
- 참고로 **get_bleu_scores 함수** 를 구현해 놨음 ([소스](https://github.com/kobongsoo/GPT-2/blob/master/evaluate/bleuscore_test.ipynb) 참조)
- 

#### BLEU Score  예제-1
```
import nltk.translate.bleu_score as bleu

# 모델에서 출력한 값
candidate = "오늘은 날씨가 흐리고 비가 옵니다."

# 실제 추정 값
references = ['내일은 날씨가 흐리고 비가 옵니다.', '오늘은 날씨가 좋고 비가 옵니다.', '오늘은 날씨가 흐리고 눈이 옵니다']

bleu_score = bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)),candidate.split())
print('*bleu_score:{}'.format(bleu_score))
```
#### BLEU Score  예제-2
```
import statistics
from nltk.translate.bleu_score import sentence_bleu

scores = []
for i in range(len(references)):
    references_list = []
    references_list.append(references[i])  #references는 리스트로 변환해야 함
    candidate = candidates[i]
    inputtext = inputs[i]
    
    ref = list(map(lambda ref: ref.split(), references_list))
    bleu = sentence_bleu(ref, candidate.split())
    
    print(f"*input: {inputtext} *blue: {bleu}")
    print(f"*reference: {references_list[0]}")
    print(f"*candidate: {candidate}\n")
    
    scores.append(bleu)

# bleu 스코어 평균을 구함    
bleu_score = statistics.mean(scores)
```
|소스|내용|
|:--------|:-------------------------------|
|[bleuscore_test.ipynb](https://github.com/kobongsoo/GPT-2/blob/master/evaluate/bleuscore_test.ipynb)|kogpt2 모델을 이용한 BLEU 스코어 구하는 예제|
