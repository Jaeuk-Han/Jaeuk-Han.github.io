---
title: "[Poke-LLM] 1편: \"LLM은 메리프의 꿈을 꾸는가?\", LLM 포켓몬 트레이너 만들기 프로젝트 시작과 첫번째 테스트"
date: 2025-09-08 22:00:00 +09:00
categories: [Project, Poke-LLM]
tags: [Pokemon, LLM, poke-env, ReinforcementLearning, RAG]
toc: true
pin: true
image:
  path: /assets/img/for_post/poke_llm/intro.png
  alt: "Poke-LLM 프로젝트 시작"
---

## 1. 프로젝트를 시작한 이유

RAG 프로젝트를 하다 보니 문득 이런 생각이 들었다.

**"이걸 게임에도 써볼 수 있지 않을까?"**

그러다 [poke-env](https://poke-env.readthedocs.io/en/stable/)라는 파이썬 패키지를 발견했다.  

포켓몬 배틀 환경을 시뮬레이션할 수 있는 툴인데, 여기다 LLM을 붙이면 꽤 재밌는 실험이 될 것 같았다.

일단은 이 툴을 활용해 게임 환경의 정보를 받아서 마치 RAG와 같이 Context로 전달한다면 LLM이 말 그대로 **게임을 플레이 할 수 있는지**가 정말 궁금했다.

때문에 가장 먼저 이걸 확인하기 위해 API로 먼저 테스트를 진행해보기로 하였다.

그리고 만약 이게 가능하다면 이후에는 로컬에서 대회 때 사용했던 **GRPO 트레이너**를 사용해볼수도 있을것 같다.

기본적으로 poke-env가 강화학습 실험을 지원하는 만큼 LLM을 이용한 강화학습도 효과가 괜찮지 않을까?

---

## 2. 환경 세팅

[poke-env](https://poke-env.readthedocs.io/en/stable/)는 파이썬에서 **포켓몬 쇼다운(Pokémon Showdown) 배틀 환경**을 다룰 수 있게 해주는 시뮬레이션 라이브러리다.

Gym 스타일 API를 제공하기 때문에 강화학습(RL) 실험에도 자주 쓰이고, 간단히 에이전트를 만들어서 서로 싸움 붙일 수도 있다.

찾아봤더니 국내 자료는 거의 없는 것 같아 공식 docs를 확인하며 구현을 진행했다. 

크게 중요했던 점은 다음과 같다:

- **LLMPlayer** → poke-env의 `Player`를 상속해서 LLM이 move/switch를 선택하는 에이전트를 구현
- **RandomPlayer** → poke-env에서 기본 제공하는 무작위 플레이어를 상대로 사용
- **ServerConfiguration** → 로컬 서버(`LocalhostServerConfiguration`) 또는 공식 쇼다운 서버에 연결하는 설정  

즉, `LLMPlayer`와 `RandomPlayer`를 같은 포맷(`gen9randombattle`) 환경에서 붙여서 싸움을 시킨다.

---

### 기술 환경 구성

- **명령줄 인자 (argparse)**  
   실행할 때 배틀 수나 포맷을 바꾸고 싶으면 `--battles`, `--format` 같은 옵션을 준다:
   ```bash
   poetry run python src/agent/test_llm_vs_random.py --battles 5 --format gen9randombattle --debug-llm
   ```

   주요 인자:
   - `--battles`: 몇 판을 돌릴지
   - `--format`: 배틀 포맷 (예: `gen9randombattle`)
   - `--debug-llm`: 프롬프트/출력 로그 자세히 보기
   - `--open`: 관전 URL 자동 열기

---

### 게임 환경 정보 읽어오기

poke-env은 매 턴마다 **현재 가능한 행동**과 **배틀 상태**를 객체 형태로 제공한다.

LLM이 이 정보를 잘 파악하는 것이 아마 게임 플레이의 핵심일 것이다.

내 코드에서는 이를 간단하게 JSON 구조로 변환해서 LLM이 읽기 쉽게 정리했다.

- **기술 후보 (moves)**  
  ```python
  moves = list(battle.available_moves or [])
  move_rows = [_move_row(i, m, me_active) for i, m in enumerate(moves)]
  ```
  → 기술 이름, 타입, 위력(base_power), 명중률(accuracy), STAB(자속기) 여부 등 포함

- **교체 후보 (switches)**  
  ```python
  switches = list(battle.available_switches or [])
  switch_rows = [_switch_row(i, p) for i, p in enumerate(switches)]
  ```
  → 교체 가능한 포켓몬의 종, 타입, HP 비율, 상태를 요약

- **배틀 상태 (state)**  
  ```python
  state = {
      "turn": battle.turn,
      "force_switch": battle.force_switch,
      "my_active": dump_mon(me_active),
      "opp_active": dump_mon(opp_active),
      "weather": battle.weather.name if battle.weather else "none",
      "terrain": battle.terrain.name if battle.terrain else "none",
  }
  ```
  → 현재 턴, 강제 교대 여부, 내/상대 포켓몬 정보, 날씨/지형 등을 포함

이렇게 정리된 `state`와 `[moves + switches]` 후보 리스트를 LLM에 전달해서 **"어떤 기술을 쓸지, 어떤 포켓몬으로 교체할지"** 의사결정을 맡긴다.

말 그대로 LLM이 포켓몬 트레이너가 되는 것이다.

---

## 3. LLMPlayer

`llm_player.py`에서는 LLM이 게임을 플레이 할 수 있도록 하였다.

핵심은 **poke-env의 Player 클래스를 상속**해서, 매 턴마다 LLM이 `move`/`switch`중 어떤 행동을 할지를 JSON 형식으로 출력하고,

그 JSON 형식의 데이터를 받아 파싱해 실제 행동을 수행하는 것이다.

```python
class LLMPlayer(Player):
    """
    - LLM outputs JSON like:
      {"action":"move|switch","index":0,"reason":"short"}
    - If parsing fails → safe fallback
    """
    def choose_move(self, battle: Battle):
        moves = list(battle.available_moves or [])
        switches = list(battle.available_switches or [])

        # Send state + candidates to LLM
        decision = self._llm_decide(state, rows)

        # If fail → fallback to random move
        if not decision:
            return self.choose_random_move(battle)

        return self.create_order(moves[decision["index"]])
```

또한 LLM이 형식에 맞지 않는 출력을 한 경우를 대비해 fallback으로 랜덤한 기술을 사용하도록 세팅해 두었다.

그리고 여기서 중요한 부분을 하나 더 뽑자면 바로 시스템 프롬프트이다.

포켓몬이라는 게임에 대해 GPT와 같은 거대 모델은 사전 지식을 가지고 있을 확률이 

---

## 4. 테스트: LLM vs Random

`test_llm_vs_random.py`에서는 앞에서 구현한 LLMPlayer와 poke-env가 기본으로 제공하는 `RandomPlayer`를 배틀시켜보는 코드를 작성했다.

```bash
poetry run python src/agent/test_llm_vs_random.py --battles 3 --format gen9randombattle
```

실행하면 로그로 현재 환경 상태와 각각의 플레이어가 어떤 기술을 사용하는지 확인 가능하다:

```
--- TURN 1 ---
My: {...} Opp: {...}
[DECIDE] MOVE idx=1 (thunderbolt) | reason=fallback:expected-damage
```

그리고 마지막에는 승패에 대한 결과도 확인 가능하다:

```
Done. LLM won 2 / lost 1
```

테스트로 배틀을 돌려본 결과를 일부 공유하면 다음과 같다.

![배틀 스크린샷](/assets/img/for_post/poke_llm/run_example.png)

```json
{
  "event": "llm_ok",
  "turn": 15,
  "parsed": {
    "action": "move",
    "index": 0,
    "reason": "highest damage with STAB"
  },
  "state": {
    "my_active": {"species": "Baxcalibur", "hp_pct": 48},
    "opp_active": {"species": "Amoonguss", "hp_pct": 100}
  }
}
```

- `"event": "llm_ok"` → LLM 응답이 정상적으로 파싱됨
- `"turn": 15` → 15턴째 상황
- `"action": "move"` → 교체가 아니라 기술 사용을 선택
- `"index": 0` → 가능한 기술 중 첫 번째 기술을 선택 (예: Glaive Rush)
- `"reason": "highest damage with STAB"` → 같은 타입 보너스(STAB, Same-Type Attack Bonus) 덕분에 가장 큰 대미지를 줄 수 있다고 판단 (자속기)
- `"my_active" / "opp_active"` → 현재 필드에 있는 포켓몬과 HP 상태 요약

위는 테스트 도중 LLM의 출력 일부로, **턴 15에서 LLM이 드닐레이브(Baxcalibur)에게 어떤 행동을 명령했는지**를 보여준다.

---

## 5. 다른 LLM 추론 로그

이번에는 LLM이 남긴 JSON 로그 일부를 직접 확인해봤다.

```json
{"event": "llm_ok", "turn": 1,
 "parsed": {"action": "move", "index": 0, "reason": "highest damage with STAB"},
 "state": {"my_active": {"species": "sandslash"}, "opp_active": {"species": "volcanion"}}}

{"event": "llm_ok", "turn": 2,
 "parsed": {"action": "move", "index": 0, "reason": "high damage STAB"},
 "state": {"my_active": {"species": "sandslash"}, "opp_active": {"species": "volcanion"}}}

{"event": "llm_ok", "turn": 3,
 "parsed": {"action": "move", "index": 0, "reason": "highest damage with STAB"},
 "state": {"my_active": {"species": "sandslash"}, "opp_active": {"species": "meowscarada"}}}
```

로그를 확인 해보자 LLM이 요청한 형식에 맞춰서 답변을 잘 수행해 거의 fallback이 발생하지 않는 것을 확인 가능했다.

특히 **턴마다 "무슨 행동을 왜 선택했는지" 기록**을 남기는게 정말 흥미로웠다.

현재는 토큰을 아끼기 위해 시스템 프롬프트에 STAB(자속기)를 고르면 데미지가 강해진다는 정보만 넣어놔서 LLM은 대부분 자속기를 골랐다.

이후에 배틀에 대한 다양한 정보를 포함하면 더 다채로운 답변이 나올 것으로 예상된다.

---

## 6. 앞으로 하고 싶은 것들

현재 일단 일차적인 테스트는 성공했지만 아직 가야할 길이 멀다.

일단 가장 먼저 LLM에게는 사전 지식이 없을 확률이 높다고 생각한다.

API로 GPT 같은 거대 모델을 사용하는 지금은 아마 LLM이 포켓몬에 대해서 조금이라도 알고 있을 확률이 존재하지만, 이제 강화학습을 위해 로컬 추론으로 변경한다면 모델이 포켓몬에 대해 아예 사전 지식이 없을 가능성이 높다.

원활한 배틀을 위해서는 LLM에게 상성이나 기술정보 포켓몬 특성 등의 기술을 제공해야 한다.

API 테스트에서도 LLM이 사전 지식이 없어 기술 상성을 고려하지 않았고 **"브레이브 버드"**와 같은 자폭기를 사용해 오히려 배틀에서 지는 상황도 존재했다.

이걸 전부 프롬프트로 제공하자니 출력 형식 미준수나 Lost in the Middle과 같은 정보 소실 문제가 걱정되어서 고민이 많다.

대회 때의 경험을 통해 프롬프트가 너무 길어도 좋지 않다는 것은 정말 많이 느꼈다.

때문에 모델을 포켓몬 관련 데이터로 파인튜닝 시키거나 배틀에 꼭 필요한 정보만 모아서 제공하는 방안을 생각중이다.

최종적으로 포켓몬을 잘 하는 LLM 모델을 만들어 인간과 배틀을 해 보는 것이 최종 목표이다.

---

## 7. 마무리

그래도 시작이 나쁘지는 않다는 생각이 든다.

처음 써보고 또 생소한 poke-env라는 툴을 사용해서 일단 **LLM에게 게임을 시켜보는 것**은 성공했으니 반은 온 셈이다.

이제 문제는 LLM에게 포켓몬에 대한 어떤 사전 지식을 어떻게 줘야 하는지에 달린 것 같다.

힘든 과정이지만 무척이나 재미있는 프로젝트를 찾은 것 같아 마음에 든다.

LLM이 배틀을 잘 수행하는 것을 보면서 LLM 모델의 잠재력과 범용성을 다시금 느끼게 되는 것 같다.

프로젝트가 진행되어 다른 결과가 나오면 다시 글을 작성하겠다.

> 전체 데모 파일은 [GitHub repo](https://github.com/Jaeuk-Han/poke-llm)에 정리해 두었다.