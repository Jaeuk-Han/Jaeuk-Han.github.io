---
title: "[AI 말평 대회] 참여기 #11: 3주차(3) - GRPO 학습 구현"
date: 2025-07-25 20:00:00 +09:00
categories: [2025_말평, NLP, AI]
tags: [말평대회, GRPO, RL, TRL, LoRA, InferenceCallback, ExactMatch, ROUGE, BERTScore]
toc: true
math: true
image:
  path: /assets/img/for_post/MalPyeong/week3_rl_impl.png
  alt: "GRPO 구현"
---

# #11. GRPO 학습 구현

> 저번에 **개념과 보상 아이디어**를 정리했다면, 이번 편은 실제 구현이다. 

>팀원분의 조언에 따라 콜백(CallBack)을 통해 **저장 시점마다 자동 검증 추론 → JSON 저장 → 한 번에 스코어링 → 최고 성능 체크포인트 선택** 파이프라인을 구성했다.

---

## 1) 데이터/프롬프트 파이프라인

- **CustomDataset**은 입력 JSON에서 `question`과 `answer`를 읽고, 지시문(`instruct`)과 **few-shot 예시**를 합쳐 인퍼런스용 입력을 구성한다.  
- few-shot은 **유형별 샘플링(Category_FewShotGenerater)** 을 사용해 선택형/교정형을 구분해 넣었다.

```python
# 데이터셋/지시문 준비 (요약)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
few_shot = Category_FewShotGenerater(args.few_shot_data_path, num_few_shot=args.num_few_shot_data)

with open(args.instruct_path, 'r', encoding='utf-8') as f:
    instruct = json.load(f)

train_dataset = CustomDataset(args.train_data_path, tokenizer, instruct, few_shot_generater=few_shot)
eval_dataset  = CustomDataset(args.eval_data_path,  tokenizer, instruct, few_shot_generater=few_shot)
```

---

## 2) 보상 함수(Reward) 구성

대회 지표를 그대로 보상으로 쓴다. **Format(“...이/가 옳다.”), EM, ROUGE‑1, BERTScore** 를 조합한다.

```python
def multi_reward(prompts, completions, references=None, **kwargs):
    r_bert  = BERTScore_reward_fn(prompts, completions, references)
    r_em    = EM_reward_fn(prompts, completions, references)
    r_form  = format_reward_fn(prompts, completions, references)
    r_rouge = ROUGE_1_reward_fn(prompts, completions, references)
    # 조합: (BERT + EM + (Format + 3*ROUGE)/2) / 3
    return [(b + e + (f + 3*r)/2) / 3 for b, e, f, r in zip(r_bert, r_em, r_form, r_rouge)]
```

> 포맷 보상은 정답 문장 형식을 정규식으로 체크하고, EM은 `정답문장` 일치(여러 허용 정답 `#` 분리)로 0/1 스코어,  
> ROUGE‑1/BERTScore는 **이유(reason)** 부분의 유사도를 평가한다.

---

## 3) GRPO 설정과 콜백 등록

학습은 **TRL의 `GRPOTrainer`** 를 사용하고, 저장 타이밍마다 2개의 콜백이 동작한다.

- **LoraSaveCallback**: LoRA 어댑터를 `output_dir/epoch_{E}`에 저장
- **InferenceCallback**: 방금 저장된 가중치를 즉시 로드해 **검증셋에 추론** → `inference_epoch{E}.json` 저장

```python
grpo_config = GRPOConfig(
    num_generations=args.num_generations,
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    logging_steps=args.logging_steps,
    save_strategy=args.save_strategy,    # "epoch" 권장
    eval_strategy=args.eval_strategy,    # "no" 또는 "epoch"
    load_best_model_at_end=False,        # 모델 선택은 별도 스크립트로
    remove_unused_columns=False,
    report_to="wandb"
)

trainer = GRPOTrainer(
    model=base_model,
    args=grpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    reward_funcs=[multi_reward],
    callbacks=[LoraSaveCallback(), inference_cb]  # 아래 InferenceCallback 참고
)
```

---

## 4) 메모리 관리

GRPO는 **한 프롬프트에서 여러 응답을 생성**하기 때문에, 메모리 사용량이 `num_generations(G)`에 **선형**으로 늘어난다.

**기호**
- **B** = `per_device_train_batch_size`
- **G** = `num_generations`
- **A** = `gradient_accumulation_steps`
- **D** = GPU 개수(world size)

**효과적 배치 크기**
- 한 옵티마이저 스텝 총 응답 수: `B × G × A × D`
- 한 옵티마이저 스텝 프롬프트 수: `(B ÷ G) × A × D`

> ⚠️ `B`는 반드시 `G`의 배수여야 한다.

정해진 GPU VRAM 제한(24GB) 내에서 학습을 돌리느라 OOM을 정말 많이 본것 같다. 
그 때문에 양자화와 LoRA등 정말 다양한 메모리 절약 방식을 적용하면서 많은걸 배웠다. 
학습을 돌리면서 이 부분을 특히 주의해서 파라미터 구성을 했던 경험이 생각나 공유해본다.

---

### 🔧 양자화 + LoRA 적용 코드

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# --- 양자화 설정 ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",   # bf16 연산
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_use_double_quant=True
)

# --- 모델 로드 (4bit 양자화 적용) ---
base_model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# --- LoRA 설정 ---
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# --- LoRA 어댑터 부착 ---
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

위 코드에서:  
- **양자화**: 4bit(`nf4`)로 메모리 절약, 연산은 `bfloat16`  
- **LoRA**: q/k/v/o projection 층만 학습 → 파라미터/VRAM 최소화  

이런 설정으로 24GB VRAM에서도 `K-intelligence/Midm-2.0-Base-Instruct`같은 큰 모델까지 학습이 가능했다.

---

## 5) InferenceCallback: 저장 직후, 검증셋 자동 추론

학습이 **저장(save)** 될 때마다 다음을 수행한다.

1) 가장 최근 **LoRA 어댑터 폴더(예: `epoch_3/`)** 경로를 구성  
2) 해당 가중치를 붙여 **검증셋에 generate**  
3) JSON(`..._epoch3.json`) 파일로 결과를 기록

```python
class InferenceCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        lora_path = os.path.join(args.output_dir, f"epoch_{epoch}")
        run_inference(
            input_path=args.eval_data_path,
            output_path=args.inference_output_path.replace(".json", f"_epoch{epoch}.json"),
            model_id=args.model_name,
            tokenizer_id=args.model_name,
            lora_weights_path=lora_path,  # 최신 LoRA
            prompt=instruct["prompt"],
            correction_prompt=instruct["correction_prompt"],
            selection_prompt=instruct["selection_prompt"],
            quant=args.quant_type if args.use_quant else None,
            few_shot_generater=few_shot
        )
```

`run_inference`는 **양자화 옵션/LoRA 부착/종료 토큰 설정** 후, few‑shot 프롬프트로 `generate`를 호출해 **각 문제의 답변과 정규화된 답변(normalized)** 을 JSON으로 쓴다.

```python
def run_inference(...):
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_cfg, ...)
    if lora_weights_path:
        model = PeftModel.from_pretrained(model, lora_weights_path)
    model.eval()

    dataset = CustomDataset(input_path, tokenizer, instruct=..., few_shot_generater=few_shot)
    for i in range(len(dataset)):
        outputs = model.generate(...)
        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        text = strip_prefixes_and_normalize(text)
        result[i]["output"] = {"answer": text, "normalized_answer": normalize_quotes(text)}
    json.dump(result, open(output_path, "w"), ensure_ascii=False, indent=4)
```

---

## 6) 한 번에 평가하고 **최고 체크포인트 선택**

에폭마다 생긴 `inference_epoch{E}.json` 들을 **한 디렉토리에 모아** 아래 스크립트로 평가한다.  
평가는 **ROUGE‑1, BERTScore, EM** 을 평균해 `Final Mean`을 만들고, **가장 높은 파일이 곧 Best Checkpoint**다.

```bash
python evaluate_json.py   --input_dir outputs/infer_results/ \  # *.json들이 들어있는 폴더
  --label_path data/labels_eval.json
```

출력 예시:

```
File: inference_epoch3.json
   - ROUGE-1        : 0.4132
   - BERTScore      : 0.6871
   - EM             : 0.5528
   - Final Mean     : 0.5510

Best Final Mean:
   File       : inference_epoch5.json
   Final Mean : 0.5742
```

---

## 7) 마무리

결론적으로 현재 학습 파이프라인에서 콜백으로 **학습‑저장‑검증‑선택** 루프를 자동화하여, 사람 손을 거의 쓰지 않고 **가장 좋은 체크포인트**를 고를 수 있었다.

다음 편은 학습을 돌려본 결과에 대해 다뤄볼 예정이다.
