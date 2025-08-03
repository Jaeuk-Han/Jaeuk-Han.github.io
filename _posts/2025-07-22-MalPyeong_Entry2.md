---
title: "[AI 말평 대회] 참여기 #2: 1주차(2) - 평가 지표 정리 (수식+EM 포함)"
date: 2025-07-22 21:00:00 +09:00
categories: [AI, NLP, 2025 말평]
tags: [말평대회, RAG, HuggingFace, NLP, 평가지표, ExactMatch]
toc: true
pin: true
math: true
image:
  path: /assets/img/for_post/MalPyeong/week1_metrics.png
  alt: "AI 말평 대회 평가 지표 정리"
---

# AI 말평 대회 참여기 #2: 1주차(2) - 평가 지표 정리

1주차 첫 번째 시간에는 **모델 탐색과 대회 이해**에 집중했다.  
이번 글에서는 가장 먼저 **대회 평가 지표**를 정리하고,  
**Exact Match(EM) + BLEURT + BERTScore + ROUGE-1**과  
**공식 코드 핵심 부분 및 수식**까지 살펴보기로 결정하였다.

공식 평가지표 코드는 GitHub에서 확인 가능하다.  
[평가 지표 코드 바로가기](https://github.com/teddysum/korean_evaluation/blob/main/evaluation.py#L373)

---

## 1. 대회 평가 흐름 요약

말평 대회에서는 단순 정확도가 아닌 **문장 품질과 의미 유사도**를 반영한 평가를 진행한다.

1. 제출 JSON과 정답 JSON의 ID를 매칭
2. 문제 유형에 따라 다른 지표 사용
   - **선택형/단답형** → Exact Match (EM) / Accuracy
   - **서술형/교정형** → BLEURT + BERTScore + ROUGE-1 평균
3. 최종 점수 = 유형별 점수의 평균

> 실제 대회 코드에서는 `evaluation_korean_contest_RAG_QA()` 함수로  
> RAG 과제 점수를 계산하며,  
> 정답 문장에서 **정답(Answer)**과 **이유(Reason)**를 분리해 각각 평가한다.

---

## 2. 사용된 핵심 지표

이번 대회에서 중요한 지표는 네 가지다.

| 지표        | 특징                               | 장점                          | 한계                     |
|-------------|-----------------------------------|-------------------------------|-------------------------|
| **EM**      | 예측과 정답이 완전히 일치 시 1점    | 간단하고 직관적               | 띄어쓰기·표기 차이도 0점 |
| BLEURT      | BERT 기반 문장 품질 평가           | 의미 유사도 반영, 사람 평가와 유사 | 사전학습 필요, 연산량 큼 |
| BERTScore   | BERT 임베딩 기반 의미 유사도       | 의미적 정밀 평가               | 긴 문장 처리 시 느림     |
| ROUGE-1     | 1-gram(단어) 중복률 기반 평가     | 직관적, 계산 빠름              | 의미 유사도 반영 못함   |

---

### 2-1. Exact Match (EM)

- **개념**  
  예측 문장이 정답 문장과 **완전히 동일**하면 1점, 아니면 0점  
  전체 점수는 샘플별 결과의 평균

- **수식**

$$
EM = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]
$$

- **대회 코드 예시**

```python
def calc_exact_match(true_data, pred_data):
    correct = 0
    total = len(true_data)

    for true, pred in zip(true_data, pred_data):
        # 여러 정답이 '#'로 구분될 경우 처리
        acceptable_answers = true.split('#')
        if any(pred.strip() == ans.strip() for ans in acceptable_answers):
            correct += 1

    return correct / total if total > 0 else 0
```

---

### 2-2. BLEURT

- **개념**  
  BLEURT(Bilingual Evaluation Understudy with Representations from Transformers)는  
  **BERT 기반 사전학습 모델**로 문장의 자연스러움과 의미 일치도를 평가한다.

- **수식 표현 (개념적)**

$$
BLEURT(\hat{y}, y) \approx MLP\big(\cos(\mathbf{h}_{\hat{y}}, \mathbf{h}_{y})\big)
$$

- $\mathbf{h}_{\hat{y}}$: 생성 문장 임베딩  
- $\mathbf{h}_{y}$: 참조 문장 임베딩

- **대회 코드 예시**

```python
from bleurt import score

checkpoint = "BLEURT-20"
scorer = score.BleurtScorer(checkpoint)

def calc_bleurt(true_data, pred_data):
    if type(true_data[0]) is list:
        true_data = list(map(lambda x: x[0], true_data))
    scores = scorer.score(references=true_data, candidates=pred_data, batch_size=64)
    return sum(scores) / len(scores)
```

---

### 2-3. BERTScore

- **개념**  
  생성 문장과 참조 문장의 **의미 유사도**를 BERT 임베딩 기반으로 평가한다.  
  Precision / Recall / F1 스코어를 제공하며, F1을 최종 점수로 활용.

- **수식**

$$
P = \frac{1}{|\hat{y}|} \sum_{x_i \in \hat{y}} \max_{y_j \in y} \cos(\mathbf{h}_{x_i}, \mathbf{h}_{y_j})
$$

$$
R = \frac{1}{|y|} \sum_{y_j \in y} \max_{x_i \in \hat{y}} \cos(\mathbf{h}_{y_j}, \mathbf{h}_{x_i})
$$

$$
F_1 = 2 \cdot \frac{P \cdot R}{P + R}
$$

- **대회 코드 예시**

```python
import evaluate
bert_scorer = evaluate.load('bertscore')
bert_model_type = 'bert-base-multilingual-cased'

def calc_bertscore(true_data, pred_data):
    if type(true_data[0]) is list:
        true_data = list(map(lambda x: x[0], true_data))
    scores = bert_scorer.compute(predictions=pred_data,
                                 references=true_data,
                                 model_type=bert_model_type)
    return sum(scores['f1']) / len(scores['f1'])
```

---

### 2-4. ROUGE-1

- **개념**  
  ROUGE(Recall-Oriented Understudy for Gisting Evaluation)는  
  **n-gram 기반 중복률**을 계산해 문장 유사도를 평가한다.  
  그중 **ROUGE-1**은 1-gram(단어 단위) 일치율만 계산한다.

- **수식**

$$
ROUGE\text{-}1 = \frac{|\text{생성 문장 단어} \cap \text{참조 문장 단어}|}{|\text{참조 문장 단어}|}
$$

- **대회 코드 예시**

```python
from rouge_metric import Rouge

def calc_ROUGE_1(true, pred):
    rouge_evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=1000,
        length_limit_type="words",
        use_tokenizer=True,
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # F1 score
        weight_factor=1.0,
    )
    scores = rouge_evaluator.get_scores(pred, true)
    return scores['rouge-1']['f']
```

---

## 3. 평가 함수 구조

대회 평가 핵심 함수는 `evaluation_korean_contest_RAG_QA()`로,  
정답과 예측 데이터를 받아 **정확도 + 생성 품질**을 모두 측정한다.

```python
def evaluation_korean_contest_RAG_QA(true_data, pred_data):
    scores = {
        "exact_match": 0,
        "rouge_1": 0,
        "bertscore": 0,
        "bleurt": 0,
        "descriptive_avg": 0,
        "final_score": 0
    }

    # 1. 정답/예측에서 답변과 이유를 분리
    # 2. Exact Match로 선택형 채점
    # 3. ROUGE-1, BERTScore, BLEURT 평균으로 서술형 채점
    # 4. 최종 점수 = (Exact Match + 서술형 평균) / 2

    return scores
```

---

## 4. 오늘 정리한 핵심

1. **EM(Exact Match)** → 예측과 정답이 완전히 같을 때만 1점  
2. **BLEURT** → 의미 중심 문장 품질 평가, 사람 평가와 유사  
3. **BERTScore** → 임베딩 기반 의미 유사도, F1 스코어 활용  
4. **ROUGE-1** → 단어 단위 일치율 기반, 빠르고 직관적  

> 다음 글에서는 **이 지표들을 실제로 활용해  
> 베이스라인 모델 성능을 평가**해볼 예정이다.
