---
title: "[AI 말평 대회] 참여기 #10: 3주차(2) - GRPO 기반 강화학습과 DeepSeek의 사례"
date: 2025-07-23 22:00:00 +09:00
categories: [2025_말평]
tags: [말평대회, Reinforcement-Learning, GRPO, RLHF, RAG, DeepSeek, 강화학습]
toc: true
pin: true
math: true
image:
  path: /assets/img/for_post/MalPyeong/week3_rl.png
  alt: "AI 말평 대회 GRPO 강화학습"
---

# AI 말평 대회 참여기 #10: 3주차(2) - GRPO 기반 강화학습과 DeepSeek 사례

지난 글에서는 듀얼 인코더가 구조적으로 한계가 있음을 확인했고, 우리 팀은 **PPL 기반 candidate + 강화학습(RL)**으로 gold 문맥을 선택하는 전략을 채택했다.  

이번 글에서는 내가 맡은 GRPO 트레이너 파트에 대해 공부한 내용을 정리한다.

글에서는 크게 강화학습 개념들과 **GRPO (Group Relative Policy Optimization)** 알고리즘 그리고 이를 성공적으로 적용한 **DeepSeek의 사례**를 소개하려고 한다.

---

## 1) 강화학습(RL) 기초 개념

강화학습은 크게 네 가지 요소로 구성된다:

- **환경(Environment)**: 모델이 상호작용하는 대상 (우리 과제에서는 프롬프트)  
- **에이전트(Agent)**: 행동을 수행하는 모델 (LLM)  
- **행동(Action)**: 모델이 생성하는 응답 (텍스트 출력)  
- **보상(Reward)**: 응답의 품질을 평가하는 신호 (포맷 준수 여부 등)

LLM 강화학습에서 자주 쓰이는 형태는 **RLHF (Reinforcement Learning with Human Feedback)**다.  
RLHF의 전형적 구조는 다음과 같다:

1. **Supervised Fine-Tuning (SFT)**: 기본 응답 데이터로 초기 학습  
2. **Reward Model (RM)**: 응답 품질을 평가하는 모델 학습  
3. **PPO (Proximal Policy Optimization)**: SFT 모델을 RM 보상에 따라 조정

하지만 PPO는 critic 모델과 RM 학습이 필요해 **복잡하고 자원 소모적**이다.

---

## 2) GRPO: PPO의 대안

**GRPO (Group Relative Policy Optimization)**는 PPO의 한계를 개선한 방식이다.

- **Value 모델이 필수가 아님** → critic 없이 학습 가능  
- **그룹 내 상대적 비교** → 여러 응답을 생성해 보상을 상대 비교  
- **메모리 효율적**이고, **안정적 학습** 가능

공식 정의:  
> 한 프롬프트에서 여러 응답을 생성하고, 그 중 상대적으로 더 나은 응답이 어떤 것인지를 기준으로 정책을 업데이트한다.

---

## 2-1) DeepSeek 사례

최근 발표된 **DeepSeekMath** 연구에서는 GRPO를 활용해 **GPT-4 수준에 근접한 수학 추론 능력**을 확보했다.

- DeepSeekMath‑Instruct 7B 모델은 **MATH benchmark에서 51.7%** 성능을 달성 → GPT-4와 근접 ([Zhang et al., 2024](https://arxiv.org/abs/2402.03300))  
- GRPO 덕분에 **critic 모델이 필요 없었고, 메모리 효율적으로 학습** 가능  
- 단순 SFT 모델에서 GRPO RL을 거친 뒤 **reasoning 성능이 크게 향상**됨

또한 DeepSeek-R1‑Zero 실험에서는 **SFT 없이도 GRPO만으로 모델의 reasoning 능력이 발전**하는 사례가 보고되었다.  
이는 강화학습 자체가 모델의 자기 진화를 가능하게 한다는 점을 시사한다.

---

## 3) Hugging Face GRPOTrainer 사용법

Hugging Face TRL 라이브러리에서는 GRPOTrainer를 제공한다.

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# 데이터셋 로드
dataset = load_dataset("your_eval_prompts", split="train")

# 보상 함수 정의 (예시)
def reward_format(completions, **kwargs):
    scores = []
    for c in completions:
        if c.strip().startswith("정답") and "옳다." in c:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores

# 설정
training_args = GRPOConfig(
    output_dir="malpyeong-grpo",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    num_generations=4,   # 한 프롬프트 당 생성할 응답 수
    learning_rate=1e-5,
    gradient_accumulation_steps=2,
    logging_steps=20,
)

# 학습 실행
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=[reward_format],
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
```

### 주요 인자 설명
- **num_generations**: 한 프롬프트에서 생성할 응답 수 (그룹 크기)  
- **reward_funcs**: 보상 함수를 리스트로 제공 가능 (예: 포맷 보상 + 내용 보상)  
- **per_device_train_batch_size**: 배치 크기  
- **gradient_accumulation_steps**: GPU 메모리 최적화  
- **logging_steps** / **save_steps**: 학습 모니터링

---

## 4) 우리 팀의 보상 설계 아이디어

강화 학습을 진행하기 위해서는 **보상을 어떤 기준으로 줄지**가 가장 중요하다고 생각한다.

우리는 실제 대회 평가 지표들과 정해진 형식을 지키는지를 보상으로 삼으면 합리적일 것이라 판단했다.

- **Format Reward**:  
  출력이 반드시 `"{선택·교정 문장}이/가 옳다. {이유}"` 형태를 따르는지 확인  

- **대회 평가 지표 기반 Reward**:  
  - Exact Match (EM)  
  - ROUGE-1  
  - BERTScore  
  - BLEURT  


---

## 5) 정리 및 다음 계획

- GRPO는 PPO 대비 **단순하고 메모리 효율적인 강화학습 방법**  
- DeepSeek 사례는 GRPO가 실제로 reasoning 능력 강화를 가능케 함을 보여줌  
- 우리 팀은 대회 평가 지표 기반 보상 + Format reward를 조합하는 전략을 계획 중임  

👉 다음 글(#11)에서는 **우리 팀의 실제 GRPO 구현 코드**와 **학습 곡선 및 결과**를 정리한다.
