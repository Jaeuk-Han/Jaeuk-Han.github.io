---
title: "[한국 전통주 RAG] 4편: 리트리버 파이프라인 구현"
date: 2025-08-22 22:00:00 +09:00
categories: [Project, KoreanSool_RAG]
tags: [KoreanLiquor, RAG, Retriever, BM25, MMR, CrossEncoder]
toc: true
pin: true
math: true
image:
  path: /assets/img/for_post/KoreanSool_RAG/retriever.png
  alt: "전통주 RAG 리트리버"
---

# 한국 전통주 RAG 프로젝트 4편: 리트리버 파이프라인 구현

데이터 전처리까지 끝났으니 이제는 본격적으로 **리트리버**를 설계할 차례다.

이번에는 내가 리트리버를 어떻게 구현 했는지에 대해서 정리해보려고 한다.

---

## 1. 말평 대회와 다른 점

지난 AI 말평 대회에서 가장 힘들었던 점은 **쿼리와 컨텍스트의 형식이 너무 달랐다**는 것이다.  
예를 들어,  

- 쿼리: "가축을 기를 때에는 {먹이량/먹이양}을 조절해 주어야 한다."  
- 컨텍스트: "한 음절의 한자어는 앞말이 고유어나 외래어일 때…"  

이렇게 매칭이 전혀 안 되는 경우가 많았다.

심지어 키워드가 겹치는 경우도 적어서 리트리버를 구성 과정에서 정말 난감했다.

이번 전통주 데이터는 그보다는 나았다.  

주류/누룩 레시피와 개념 위주라서 최소한 **키워드 레벨에서는 겹치는 부분**이 있었다.

여기서 한가지 아이디어를 떠올렸다.

말평 대회 당시에는 **Perplexity**를 이용해 Gold를 찾거나 컨텍스트를 추려냈다.

현재 데이터는 레시피, 절차 중심이므로 **Perplexity** 대신 키워드를 사용한 방법으로 유사한 작업을 효과적으로 수행 가능할 것이다. 

예를 들어, 쿼리가 **"백화주는 어떻게 만들어?"**라면, **"백화주"**라는 키워드만으로도 적절한 컨텍스트를 빠르게 찾을 수 있다.

때문에 이번 프로젝트에서는 첫 단계로 BM25 기반 키워드 검색을 적용하여 1차적으로 리트리브를 진행해보기로 했다.


---

## 2. BM25 (Okapi)

BM25는 대표적인 통계 기반 검색 기법이다.

문서 $d$와 쿼리 $q$에 대해 BM25 점수는 다음과 같이 계산된다.

$$
\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1+1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
$$

- $f(t, d)$ : 문서 $d$에서 토큰 $t$의 빈도  
- $\lvert d\rvert$ : 문서 길이  
- $\text{avgdl}$ : 전체 문서 평균 길이  
- $k_1, b$ : 하이퍼파라미터  

BM25의 특징은 **문서 길이를 보정한다는 점**이다. 

- **짧은 문서**는 평균보다 단어 수가 적으므로, 단어가 한 번만 등장해도 점수가 크게 올라간다.  
- **긴 문서**는 단어가 우연히 포함될 확률이 높으니, 점수를 깎아 과대평가를 막는다.  

여기서 **b**는 길이 보정의 강도를 조절하고, **$k_1$**는 같은 단어가 여러 번 나올 때 점수가 얼마나 빨리 포화되는지를 정한다.  

즉, BM25는 짧은 문서에서는 단어 한 번 등장만으로도 크게 반영해 주고, 긴 문서는 길이 때문에 불리하지도, 또 과도하게 유리하지도 않게 보정해준다.  

평균 문서 길이($\text{avgdl}$)가 100, $k_1=1.5$, $b=0.75$라고 가정해 보자.  

- **짧은 문서** (길이 20, 단어 1번 등장): 점수 약 **1.56**  
- **평균 문서** (길이 100, 단어 1번 등장): 점수 약 **1.00**  
- **긴 문서** (길이 300, 단어 1번 등장): 점수 약 **0.53**  

결론적으로 같은 단어가 문장에서 한번 등장 한건 동일해도 짧은 문서에서 점수가 더 크게 나오고, 긴 문서는 점수가 깎여 작게 나온다. 

내 코드는 `rank_bm25` 라이브러리를 사용해 구현했다.  

```python
from rank_bm25 import BM25Okapi

def tokenize_ko_en(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())

self._tokens = [tokenize_ko_en(p.text) for p in self.passages]
self._bm25 = BM25Okapi(self._tokens)
```

---

## 3. Dense Retriever (Dual Encoder)

이제 BM25 말고도 대회 때와 유사하게 **Dual Encoder**를 사용해 **의미적 유사도**를 잡아내기로 했다.

이 경우 BM25를 통한 리트리브에서 놓친 Gold 컨텍스트가 있는 경우에도 Dual Encoder가 그걸 잡아줄 확률이 높아진다.

임베딩 모델은 `intfloat/multilingual-e5-base`를 사용했고 쿼리는 `"query: ..."`, 문서는 `"passage: ..."` 프리픽스를 붙여 인코딩했다 .  

```python
def e5_encode_text_prefix(text: str, is_query: bool) -> str:
    if is_query:
        return f"query: {text.strip()}"
    return f"passage: {text.strip()}"
```

결론적으로 BM25와 Dense 결과를 합쳐 **Candidate Pool**을 만든 뒤, 그 중 상위 후보를 자르는 걸로 답변에 사용할 컨텍스트가 리트리브 된다.

---

## 4. MMR (Maximal Marginal Relevance)

한 가지 문제가 있었다.  

리트리브 결과를 보면 특정 섹션(예: "숙성")만 반복적으로 상위에 등장하는 현상이 나타났다.

우리의 데이터는 레시피의 형태이기에 다양한 단계의 레시피가 상위에 골고루 올라오는 것이 답변에 유리하다고 생각한다.

때문에 이를 해결하기 위해 방법을 찾아 보던 도중 **MMR (Maximal Marginal Relevance)**이라는 개념을 발견했다.

MMR은 **관련성(Relevance)**과 **다양성(Diversity)**를 동시에 고려한다.  

$$
\text{MMR} = \arg\max_{d_i \in D \setminus S} \left[ \lambda \cdot \text{Sim}(d_i, q) - (1-\lambda) \cdot \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]
$$

- $D$: 전체 후보 문서  
- $S$: 이미 선택된 문서 집합  
- $\text{Sim}(d_i, q)$: 문서와 쿼리 유사도  
- $\text{Sim}(d_i, d_j)$: 문서 간 유사도  
- $\lambda$: 관련성/다양성 가중치 (0.7 사용)

이 MMR을 통해서 리트리브 결과에 다양성을 추가해보려고 시도했다.

코드 구현은 다음과 같다 .  

```python
def mmr_select(query_vec, doc_vecs, cand_idxs, top_k, lam=0.7):
    rel = {i: float(doc_vecs[i] @ q) for i in cand_idxs}
    while len(selected) < min(top_k, len(cand_set)):
        for i in list(cand_set):
            div = max(float(doc_vecs[i] @ doc_vecs[j]) for j in selected) if selected else 0.0
            score = lam * rel[i] - (1.0 - lam) * div
```

MMR을 적용하니 다행이도 상위 컨텍스트에 **숙성/담금/여과 등 다양한 단계**가 고르게 포함되기 시작했다.  

---

## 5. Cross Encoder (선택적)

마지막으로, 필요에 따라 **Cross Encoder**를 붙여 리랭킹을 사용할 수 있게 했다.

로컬에서 구현을 진행중이고 GPU도 그리 좋지 않은터라 빠른 테스트를 위해 Cross Encoder는 선택적으로 적용 가능하게 구현 했다. 

현재 코드에서는 컴퓨터 자원이 충분하면 사용자가 선택해 MiniLM 기반 Cross Encoder로 rerank, 부족하면 그냥 Dense+MMR까지만 사용한다.  

```python
class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Passage]):
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
```

---

## 6. LLM 연결과 Gradio Demo

Retrieval 파트가 끝났으니 이제 LLM과 연결했다.

앞서 말했듯 내 PC 사양이 좋지 않아 이번에는 로컬 추론 대신 **API 방식**을 사용했다.

대회 때는 로컬 추론을 사용했으니 이참에 API도 한번 사용 해봤는데, 간편하고 성능도 상당히 잘 나와 역시 언어 모델은 크기가 중요하다는 것을 실감할 수 있었다.

마지막으로 Gradio UI를 입혀서 데모를 완성했다.  

- 좌측: 질문 입력 + 옵션 (Top-K, Cross Encoder 사용 여부)  
- 우측: Retrieved Context 카드와 메타데이터  

```python
def build_demo(engine: RAGEngine, llm: LLM):
    with gr.Blocks(title="Sool RAG — Portfolio Demo") as demo:
        chat = gr.Chatbot()
        qbox = gr.Textbox(label="질문")
        topk = gr.Slider(1, 12, value=5, step=1, label="Top-K")
        use_rr = gr.Checkbox(value=False, label="크로스 인코더 사용")
```

---

## 7. 정리

- **BM25**: 키워드 기반 1차 후보 생성  
- **Dense Retriever**: E5 임베딩으로 의미적 유사도 보강  
- **MMR**: 특정 섹션 쏠림 방지, 다양성 확보  
- **Cross Encoder**: 최종 rerank (선택)  
- **LLM + Gradio**: API 연결, 데모 완성  

> 전체 데모 파일은 [GitHub repo](https://github.com/Jaeuk-Han/korean-traditional-liquor-dataset)에 정리해 두었다.

---
 
다음 편에서는 이 파이프라인을 통해 **실제 추론 결과**를 확인하고, 어떤 한계와 개선점이 있었는지 기록해볼 예정이다.
