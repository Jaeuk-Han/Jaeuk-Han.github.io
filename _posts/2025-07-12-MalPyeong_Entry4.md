---
title: "[AI 말평 대회] 참여기 #4: 1주차(4) - 디코딩 파라미터와 추론 코드 분석"
date: 2025-07-12 23:45:00 +09:00
categories: [2025_말평, 참여기]
tags: [말평대회, RAG, HuggingFace, NLP, Decoding, Inference, Quantization]
toc: true
pin: true
image:
  path: /assets/img/for_post/MalPyeong/week1_inference.png
  alt: "AI 말평 대회 추론 및 디코딩"
---

# AI 말평 대회 참여기 #4: 1주차(4) - 디코딩 파라미터와 양자화(Quantization)의 개념 그리고 추론코드 분석

테스트 해볼만한 모델을 찾았으니 본격적으로 모델 추론을 돌려보기로 결심하였다.
팀장님께서 추론을 돌리기 전 **디코딩 파라미터**와 **추론 코드**에 대한 공부를 하면 좋을 것 같다고 말씀하셔서
이번 글에서는 **Baseline 추론 과정**을 다루면서 **디코딩 파라미터 개념**, **추론 코드 분석**을 정리한다.

---

## 1. 디코딩 파라미터 상세 설명

LLM 추론에서 **디코딩(Decoding)**은 모델이 다음 토큰을 선택해 문장을 완성하는 과정이다.  
모델 출력 확률 분포에서 어떤 토큰을 선택하느냐에 따라 결과가 달라지며,  
이를 제어하는 핵심 요소가 **디코딩 파라미터**다.

### 1-1. `max_new_tokens`
- **설명**: 한 번에 생성할 최대 토큰 수
- **영향**
  - 값이 작으면 문장이 도중에 끊김
  - 값이 크면 불필요하게 긴 출력 가능
- **Tip**: QA/문법 교정 → 256~512, 장문 생성 → 1024 이상

### 1-2. `temperature`
- **설명**: 토큰 선택 확률을 조절
- **원리**: 모델 확률분포 \( p_i \)를 \( p_i^{1/T} \)로 변환 후 정규화
  - T < 1 → 확률 분포가 날카로움 → **보수적·일관된 답변**
  - T > 1 → 확률 분포가 평탄함 → **다양하고 창의적인 답변**
- **Tip**
  - QA·문법 교정 → 0.5~0.7
  - 창작형 → 0.8~1.0

### 1-3. `top_k` (Top-K 샘플링)
- **설명**: 다음 토큰 선택 시 확률 상위 k개만 후보로 사용
- **영향**
  - k가 작으면 보수적, k가 크면 다양성↑
- **예시**
  - top_k=1 → 사실상 Greedy
  - top_k=50 → 일반적인 설정

### 1-4. `top_p` (Nucleus Sampling)
- **설명**: 누적 확률이 p 이하인 후보만 샘플링
- **특징**
  - top_p=0.9 → 누적 확률 90% 차지하는 후보들 중 선택
  - 문장 품질 안정적, 의미 없는 토큰 제거

### 1-5. `repetition_penalty`
- **설명**: 동일 토큰 반복 억제 계수
- **영향**
  - 1.0 → 영향 없음
  - 1.05~1.2 → 반복 줄어듦, 과하면 어색
- **활용**: QA·문법 교정 1.05~1.1 권장

### 1-6. `num_beams` (Beam Search)
- **설명**: 여러 후보 경로를 탐색해 가장 확률 높은 문장 선택
- **원리**
  1. 초기 토큰에서 상위 k개(beam 수) 확장
  2. 각 후보를 계속 확장하며 누적 확률 비교
  3. 최종적으로 가장 높은 점수 경로 선택
- **특징**
  - 다양성 낮음, 높은 일관성
  - beam이 크면 속도 저하
- **Tip**
  - QA·문법 교정 1~4
  - 창작형 문장은 beam보다는 샘플링 선호

> 요약: **temperature/top_p/top_k**는 다양성 제어,  
> **repetition_penalty**는 반복 억제,  
> **max_new_tokens**는 길이 제어,  
> **beam search**는 높은 신뢰성.

---

## 2. 추론 코드 분석

우리 팀의 Baseline 추론 코드는 Hugging Face `transformers`의  
`AutoModelForCausalLM.generate()`를 기반으로 설계되었다.

### 2-1. 코드 실행 흐름

1. **모델 및 토크나이저 로딩**
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       args.model_id, quantization_config=quantization_config, **model_kwargs
   )
   tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
   tokenizer.pad_token = tokenizer.eos_token
   ```
   - `quantization_config`로 4bit/8bit/int8 양자화 적용 가능
   - `pad_token`을 EOS 토큰으로 설정해 배치 추론 시 패딩 처리

2. **데이터셋 로딩**
   ```python
   dataset = CustomDataset(file_test, tokenizer, prompt=args.prompt, ...)
   ```
   - JSON 파일을 불러와 `input_ids`로 변환
   - 선택형/교정형 문제에 맞춰 프롬프트 추가

3. **텍스트 생성**
   ```python
   outputs = model.generate(
       input_ids,
       max_new_tokens=args.max_new_tokens,
       num_beams=args.num_beams,
       do_sample=args.do_sample,
       temperature=args.temperature,
       top_p=args.top_p,
       top_k=args.top_k,
       repetition_penalty=args.repetition_penalty,
   )
   output_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
   ```
   - `generate()`에서 **디코딩 파라미터** 적용
   - `skip_special_tokens=True`로 EOS·패딩 토큰 제거

4. **후처리 & 결과 저장**
   ```python
   output_text = output_text.replace("[답변]", "")
   result[idx]["output"] = {"answer": output_text}
   ```
   - `[답변]`·`답변:` 접두어 제거
   - JSON 형태로 저장

5. **전체 추론 코드**
  
  ```python
  import argparse
  import json
  import tqdm

  import torch
  import numpy
  from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

  from src.data import CustomDataset
  from peft import (
      get_peft_model,
      prepare_model_for_kbit_training,
  )


  # fmt: off
  parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

  g = parser.add_argument_group("Common Parameter")
  g.add_argument("--input", type=str, required=True, help="input filename")
  g.add_argument("--output", type=str, required=True, help="output filename")
  g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
  g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
  g.add_argument("--device", type=str, required=True, help="device to load the model")
  g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")


  g.add_argument("--num_beams", type=int, default=1, help="number of beams for beam search")
  g.add_argument("--do_sample", default=True, help="whether to use sampling for text generation")
  g.add_argument("--top_p", type=float, default=0.9, help="top_p value for nucleus sampling")
  g.add_argument("--top_k", type=int, default=50, help="top_k value for sampling (0 for no top-k)")
  g.add_argument("--temperature", type=float, default=0.7, help="temperature for sampling")
  g.add_argument("--repetition_penalty", type=float, default=1.05, help="repetition penalty for text generation")
  g.add_argument("--max_new_tokens", type=int, default=1024, help="maximum number of new tokens to generate")
  # fmt: on

  g.add_argument("--prompt", type=str, default=None, help="prompt to use for the model")
  g.add_argument("--system_prompt", type=str, default="You are a helpful AI assistant. Please answer the user's questions kindly. \
            당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. \
            단, 동일한 문장을 절대 반복하지 마시오.", help="system 역할 메시지(최초 지침)")
  g.add_argument(
      "--correction_prompt", type=str, default=None, help="교정형 문제에 대한 프롬프트"
  )
  g.add_argument(
      "--selection_prompt", type=str, default=None, help="선택형 문제에 대한 프롬프트"
  )

  g.add_argument(
      "--quant",
      type=str,
      choices=["4bit", "8bit", "int8"],
      help="quantization type for the model (4bit, 8bit, int8)",
  )


  def main(args):
      quantization_config = None

      if args.quant is not None:
          print(f"[DBG] quantization type: {args.quant}")

          if args.quant == "4bit":
              quantization_config = BitsAndBytesConfig(
                  load_in_4bit=True,
                  bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.float16,
                  bnb_4bit_use_double_quant=True,
              )
          elif args.quant == "8bit":
              quantization_config = BitsAndBytesConfig(
                  load_in_8bit=True,
              )
          else:
              print(f"Unknown Quantization Type: {args.quant}")
              quantization_config = BitsAndBytesConfig(
                  load_in_4bit=True,
                  bnb_4bit_compute_dtype=torch.float16,
                  bnb_4bit_use_double_quant=True,
                  bnb_4bit_quant_type="nf4",
              )

      # Prepare model loading kwargs
      model_kwargs = {
          "torch_dtype": torch.bfloat16,
          "device_map": args.device,
      }
      if args.use_auth_token:
         model_kwargs["use_auth_token"] = args.use_auth_token
      model_kwargs["cache_dir"] = "/media/nlplab/hdd1/cache_dir"
      model_kwargs["trust_remote_code"] = True

    # DEBUG
      print("\n\n[DBG] model_id: ", args.model_id)
      if quantization_config is not None:
          model = AutoModelForCausalLM.from_pretrained(
              args.model_id, 
              quantization_config = quantization_config, 
              **model_kwargs)
          model.eval()
      else:
          model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

      if args.tokenizer == None:
          args.tokenizer = args.model_id

      # Prepare tokenizer loading kwargs
      tokenizer_kwargs = {}
      if args.use_auth_token:
          tokenizer_kwargs["use_auth_token"] = args.use_auth_token

      tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, **tokenizer_kwargs)
      tokenizer.pad_token = tokenizer.eos_token
      terminators = [
          tokenizer.eos_token_id,
          (
              tokenizer.convert_tokens_to_ids("<|eot_id|>")
              if tokenizer.convert_tokens_to_ids("<|eot_id|>")
              else tokenizer.convert_tokens_to_ids("<|endoftext|>")
          ),
      ]

      file_test = args.input
      dataset = CustomDataset(
          file_test,
          tokenizer,
          prompt=args.prompt,
          correction_prompt=args.correction_prompt,
          selection_prompt=args.selection_prompt,
      )

      with open(file_test, "r") as f:
          result = json.load(f)

      for idx in tqdm.tqdm(range(len(dataset))):
          inp = dataset[idx]
          with torch.no_grad():
              outputs = model.generate(
                  inp.to(args.device).unsqueeze(0),
                  max_new_tokens=args.max_new_tokens,
                  eos_token_id=terminators,
                  pad_token_id=tokenizer.eos_token_id,
                  repetition_penalty=args.repetition_penalty,
                  num_beams=args.num_beams,
                  do_sample=args.do_sample,
                  temperature=args.temperature,
                  top_p=args.top_p,
                  top_k=args.top_k,
                  use_cache=True
              )

          output_text = tokenizer.decode(
              outputs[0][inp.shape[-1] :], skip_special_tokens=True
          )

          # 출력에서 "답변: " 접두어 제거
          # if output_text.startswith("답변: "):
          #     output_text = output_text[4:]
          # elif output_text.startswith("답변:"):
          #     output_text = output_text[3:]
          if output_text.startswith("[답변]\n"):
              output_text = output_text.replace("[답변]\n", "", 1)
          elif output_text.startswith("[답변] "):
              output_text = output_text.replace("[답변] ", "", 1)
          elif output_text.startswith("[답변]"):
              output_text = output_text.replace("[답변]", "", 1)
          elif output_text.startswith("답변: "):
              output_text = output_text.replace("답변: ", "", 1)
          elif output_text.startswith("답변:"):
              output_text = output_text.replace("답변:", "", 1)

          result[idx]["output"] = {"answer": output_text}
        
          # DEBUG
          # 중간에 생성 결과를 직접 log로 확인
          print(f"\n\n[DBG] idx: {idx}, output: {output_text}")

      with open(args.output, "w", encoding="utf-8") as f:
          f.write(json.dumps(result, ensure_ascii=False, indent=4))


  if __name__ == "__main__":
      exit(main(parser.parse_args()))

```

---

## 3. 양자화(Quantization) 개념

양자화는 모델 파라미터를 **저정밀도 정수**로 변환해  
**메모리 사용량과 연산 속도를 줄이는 기법**이다.

- **4bit (NF4)**
  - 가중치를 4bit 정밀도로 저장 → 1/8 메모리
  - 최신 GPU에서 추론 속도 ↑, LoRA 미세 튜닝 가능
- **8bit**
  - 1/4 메모리 절감, 정확도 손실 거의 없음
  - 학습·추론 모두 안정적
- **int8**
  - CPU/GPU 범용 지원, 속도는 다소 낮음

**장점**
- VRAM 절감 → 대형 모델 로딩 가능
- 일부 환경에서 추론 속도 향상

**단점**
- 극단적 양자화(4bit)는 일부 정확도 손실 가능

> 이번 프로젝트에서는 **4bit 양자화 + bfloat16 연산**으로  
> RTX 3090 24GB 환경에서도 8B~13B 모델 추론을 수행할 수 있었다.

---

## 4. 다음 글 예고

다음 글에서는 본격적인 추론을 위해 필요한 것 중 하나인 프롬프트를 위한 
**프롬프트 엔지니어링** (선택형·교정형 Prompt 설계)와 **모델별 추론 결과 비교 & 분석**을 다룰 예정이다.
