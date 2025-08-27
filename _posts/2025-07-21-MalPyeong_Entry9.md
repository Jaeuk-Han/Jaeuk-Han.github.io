---
title: "[AI 말평 대회] 참여기 #9: 3주차(1) - 커스텀 듀얼 인코더 파인튜닝과 전략 전환"
date: 2025-07-21 22:00:00 +09:00
categories: [2025_말평, 참여기]
tags: [말평대회, Retrieval, Dual-Encoder, LoRA, Quantization, MNR, CE-Loss, GRPO, RAG, Reinforcement-Learning]
toc: true
pin: true
math: true
image:
  path: /assets/img/for_post/MalPyeong/week3_dual_encoder.png
  alt: "AI 말평 대회 듀얼 인코더 파인튜닝"
---

# AI 말평 대회 참여기 #9: 3주차(1) - 커스텀 듀얼 인코더 파인튜닝과 전략 전환

지난 편(#8)에서는 **Sentence-Transformers 기반의 간단한 듀얼 인코더**로 Retrieval을 구현했다.  
이번에는 직접 **DualEncoder 모델을 정의하고 파인튜닝**을 시도했지만, 예상보다 성능이 낮게 나왔다.  
이 글에서는 그 이유와 앞으로의 방향성을 기록한다.

---

## 1) 데이터 준비

`split_qa.py`를 이용해 train / eval 세트를 분리한다:

```python
def split_dataset(input_path, train_path, eval_path, eval_ratio=0.2):
    data = load_json(input_path)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - eval_ratio))
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    save_json(train_data, train_path)
    save_json(eval_data, eval_path)
    print(f"[split] Total: {len(data)} | Train: {len(train_data)} | Eval: {len(eval_data)}")
```

---

## 2) DualEncoder 모델 정의

`model.py`에서 직접 구현한 구조는 다음과 같다:

```python
class DualEncoder(nn.Module):
    def __init__(self, model: AutoModel, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    def _encode(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.last_token_pool(output.last_hidden_state, attention_mask)
        return pooled

    def forward(self, input_ids_q, attention_mask_q,
                      input_ids_c, attention_mask_c,
                      labels=None):
        q = self._encode(input_ids_q, attention_mask_q)
        c = self._encode(input_ids_c.view(-1, input_ids_c.size(-1)),
                         attention_mask_c.view(-1, input_ids_c.size(-1))).view(*input_ids_c.shape[:2], -1)
        q, c = F.normalize(q, p=2, dim=-1), F.normalize(c, p=2, dim=-1)

        tau = torch.exp(self.log_tau)
        logits = (q.unsqueeze(1) * c).sum(dim=-1) / tau
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}
```

- `temperature` 학습 가능  
- `last_token_pool` 기반 query/context 임베딩  
- **CrossEntropy Loss** 기반 분류 학습

---

## 5) LoRA + 4bit Quantization

학습 효율을 위해 **QLoRA + LoRA** 방식을 적용할 수 있다:

```bash
python src/train.py   --model_name "Qwen/Qwen3-Embedding-4B"   --grammar_path "data/GrammarBook_structured.json"   --qa_path "data/qa_train.json"   --val_path "data/qa_eval.json"   --use_lora --lora_r 8 --lora_alpha 16   --use_quant --bnb_4bit_quant_type "nf4"
```

---

## 3) 학습 스크립트

`train.py`에서는 HuggingFace `Trainer`를 사용해 학습을 돌렸다:

```python
trainer = Trainer(
    model=emb_model,
    args=training_args,
    train_dataset=train_examples,
    eval_dataset=val_examples,
    data_collator=Dualcollator(tokenizer, max_length=args.max_length),
    tokenizer=tokenizer,
    compute_metrics=top_acc
)

trainer.train()
emb_model.model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
```

출력된 메트릭은 `top_acc`로 Top-1 / Top-5 / 평균 순위를 기록했다.

> 전체 구현 코드는 [GitHub repo](https://github.com/Jaeuk-Han/Korean_QA_RAG_2025)에서 확인할 수 있다.

---

## 구조 불일치 문제: Query vs Context

하지만 학습 성능은 기대보다 낮았다. 처음에는 코드 문제라고 생각해서 정말 열심히 점검을 하고 MNR(MultipleNegativesRanking) Loss나 Hard Negative Sampling 등을 적용해서 성능을 높이기 위해 노력했다.

그래도 성능은 잘 나오지 않았다......

![Face Palm...](/assets/img/for_post/MalPyeong/facepalm.png){: .w-70 .shadow .rounded}

고민 끝에 같은 팀의 석사과정 팀원분에게 도움을 청했다.

팀원분은 문제점으로 **쿼리와 콘텍스트 구조가 지나치게 다르다는 점을** 지적하셨다.

예시:

- **Query**:  
  `"가축을 기를 때에는 {먹이량/먹이양}을 조절해 주어야 한다." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.`

- **Context**:  
  `"한 음절의 한자어는 앞말이 고유어나 외래어일 때는 독립적인 한 단어로 인식하여 두음 법칙을 적용하고, 앞말이 한자어일 때는 하나의 단어로 인정하지 않아 두음 법칙을 적용하지 않는다."`

즉, 질문은 **시험 문제 지시문**이고, 문맥은 **규정 설명문**이라 포맷이 완전히 다르다.  
이런 상황에서는 단순 벡터 내적 기반 듀얼 인코더로는 좋은 성능을 기대하기 어렵다.

---

## 4) 팀의 전략 전환

결국 우리 팀은 다음과 같이 방향을 바꾸었다:

- **기존 PPL 기반 candidate retrieval 유지**  
- 그 안에서 **강화학습(RL)**으로 **gold 문맥을 선택**하는 방식으로 전환  

나는 이 과정에서 **GRPO 트레이너**를 활용해 **context 없이도 답변 포맷**을 잘 지키는 모델을 학습시키는 역할을 맡았다.

```text
"{선택·교정 문장}이/가 옳다. {이유}"
```

즉, 모델이 Retrieval을 통하지 않더라도 **출력 형식과 문법적 정답성**을 강화학습으로 학습하도록 한 것이다.

이러면 이후에 context만 붙여준다면 좋은 성능을 기대할 수 있을 것이다.

---

## 6) 배운 점

- 단순 듀얼 인코더로는 **쿼리/문맥 구조 불일치 문제**를 극복하기 어려움  
- MNR(MultipleNegativesRanking) Loss나 Hard Negative Sampling 등 Dual Encoder의 성능을 높이기 휘한 방법론들
- Retrieval 기반 후보는 PPL 방식으로 유지하되, **RL 기반 Gold 선택**으로 전환한 것이 합리적 선택  

그래도 이번 리트리버 구현은 실패했지만 그 과정에서 정말 많은걸 배웠다. 결과도 물론 중요하지만 그 과정 또한 무척이나 중요하다고 생각한다.

대회 기간이 얼마 남지 않아 지금 하지는 못하지만 언젠가 다시 이 문제에 대한 해결 방안을 찾아볼 것이다.

그리고 이번 경험을 통해 End to End로 RAG 시스템을 구현해보기로 결심했다.

적절한 주제를 찾아서 나만의 RAG 시스템을 만들어볼것이다.

---

## 다음 예고

다음 글에서는 내가 맡은 **GRPO 기반 강화학습 과정**에 대해서 다룰 예정이다.
