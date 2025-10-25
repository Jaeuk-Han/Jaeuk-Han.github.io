---
title: "[동아리 강의 후기] 1편, 선형회귀: 개념과 실습"
date: 2024-05-30 21:00:00 +09:00
categories: [태광고등학교 동아리 강의, 후기]
tags: [후기, 선형회귀, 최소제곱법, 경사하강법, sklearn, 강의]
math: true
toc: true
pin: true
image:
  path: /assets/img/for_post/club_lecture/cover1.png
  alt: club lecture linear regression
---


> **일단 어떻게 시작되었나?**  

전역 이후 남은 휴학 기간에 뭘 할지 고민하던 어느 날 갑자기 고등학교 3학년 때 담임을 맡아주셨던 은사님께 전화가 왔다. 

반갑게 근황을 주고받던 중, 선생님이 "고등학교 동아리 시간에 보조 강사로 **인공지능 강의**를 해보면 어떻겠냐"라고 제안해 주셨다.  

내가 졸업한 고등학교는 일정 간격으로 수요일마다 3시간 정도 동아리 활동을 진행한다.

그 중 선생님은 "4차 산업과 수학"이라는 동아리를 맡고 계시고 이 동아리는 이름만 봐도 인공지능과 큰 관련이 있기 때문에 학생들에게 인공지능에 대해 설명해줄 사람이 있으면 좋겠다는 것이였다.

이 말을 들었을 때 굉장히 좋은 기회라는 생각이 들었다.

이런 기회가 절대로 인생에서 흔하지는 않다.

내가 언제 고등학교에서 학생들 앞에 서서 강의를 해보겠는가?

또한 남은 휴학 기간을 의미 없이 보내고 싶지는 않았다.

그리고 **내가 아는 걸 남에게 설명하는 과정에서 배우는 것**도 많은 법이다.

그래서 바로 해보겠다고 했다.

## 준비하면서 든 고민

가장 많이 고민한 건 **"고등학생에게 인공지능을 어떻게 설명할까?"**였다.

내가 생각해도 인공지능은 쉽지 않은 분야고, 나도 고등학생 때 이 내용을 쉽다고 느끼진 않았다.

인공지능에 대한 동아리 활동이나 발표를 준비할 때도 관련 내용을 이해하는데 무척이나 애를 먹었던 것 같다. 

그래서 첫 강의는 학생들이 이해하기 쉽게 **개념 위주**로 진행하기로 결심했다.

설명한 내용을 정리해보자면 다음과 같다.

## 인공지능을 어떻게 설명할까?

2016년 알파고 이후, 인공지능의 발전은 빠르게 진행되었다.

지금은 ChatGPT 같은 도구가 일상 속으로 들어왔으며, 학교 교육과정에서도 관련 내용을 다루기 때문에, 학생들에게도 인공지능 자체가 아예 낯설지는 않을 것이다.

그러나 **"인공지능이 정확히 무엇인가?"** 또는 **"기계는 어떻게 배우는가?"**에 대한 질문이 들어왔을 때 이에 대한 명확한 답변을 할 수 있는 학생들은 많이 없을 것이다.

사실 인공지능은 굉장히 포괄적인 용어라고 생각한다.

그럼에도 이에 대한 나름대로의 정의를 내려보자면, 나는 인공지능을 **현실의 데이터를 수학적 함수로 모델링**해 예측/분류를 수행하는 체계라고 생각한다.

학생들의 성적이나 과일 사진과 같은 정보들 즉, 이미지, 음성, 문자, 숫자와 같은 다양한 형태의 정보가 입력으로 들어왔을 때 그걸 수학적인 공식을 통해 모델링 한 것을 인공지능이라고 말할 수 있을 것 같다.   

그 함수를 사용해 새로운 데이터가 들어와도 예측이나 분류가 가능해진다.

다양한 형태의 함수 중 단언컨데 학생들에게 가장 익숙한 형태의 함수는 Y=Ax+b 즉, 일차 함수일 것이다.

때문에 **"인공지능이 정확히 무엇인가?"** 그리고 **"기계는 어떻게 배우는가?"**에 대한 설명을 하기 위해서 단순 선형 회귀 모델을 설명하기로 하였다.


## 기계는 어떻게 배우는가?

"기계는 어떻게 배우는가?"에 대해 말해보자면, 기계는 바로 데이터를 통해 배운다.

많은 양의 데이터를 통해 그 데이터들을 가장 잘 일반화 하는 공식을 찾는 것이다.

이게 어떻게 가능한가를 설명하기 위해서는 경사하강법에 대한 내용을 설명해야 하는데, 이를 이해하기 위해서는 미적분에 대한 지식이 필요하다.

이는 고등학생들에게는 이해가 쉽지 않은 개념이다.

고등학교 저학년들은 아직 미적분을 배우지 않았을테고 고학년들도 배운 개념을 바로 이러한 방법론에 적용하기는 어려울 것이다.

때문에 간단히 미분값이 현재 지점에서의 기울기를 의미하며, 이를 기반으로 아주 작은 값을 빼거나 더하면서 최적의 값을 찾는다는 사실을 매우 유명한 "눈을 감고 산을 내려간다"는 비유를 기반으로 내용을 설명하기로 했다.

추가로 단순 선형 회귀의 경우 최소제곱법을 통해 직접 최적의 값을 계산 가능하므로, 학생들이 주어진 간단한 데이터로 이 최적의 값을 직접 계산해보고 실제 코드로 구현된 경사하강법으로 구한 값과 근사한다는 것을 확인해보는 실습을 진행하면 이해에 도움이 될것 같아 직접 관련 코드를 ipynb 노트북으로 구현하였다.

## 실습 코드 스니펫

### 왜 이 순서로 설명했나
- **현장 제약**: 3시간, 설치 없이 바로 실행 → **Colab + 작은 데이터**
- **주제 선택**: 한 줄 모델 **선형회귀(ŷ=ax+b)** → 결과가 그래프로 즉시 보임
- **범위 관리**: 분류/군집은 다음 시간으로 보류하고 **연속값 예측**에 집중

전체 코드는 레포 참고

```py
# 수작업 OLS: 공부시간(x) vs 점수(y)
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 4, 6, 7])
y = np.array([30, 50, 70, 80, 90])

# 최소제곱(해석적) a,b
x_mean, y_mean = x.mean(), y.mean()
a = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - a * x_mean
print(f"[Manual OLS] a={a:.3f}, b={b:.3f}")

# 예측선 + MSE
y_cal = a * x + b
mse = np.mean((y - y_cal) ** 2)
print(f"[Manual OLS] MSE={mse:.3f}")

# 시각화
plt.figure(figsize=(6,4))
plt.scatter(x, y, label="data")
plt.plot(x, y_cal, "--", label="fitted line")
plt.xlabel("Time"); plt.ylabel("Score"); plt.legend()
plt.title("Study hours vs. Score (Manual OLS)")
plt.tight_layout(); plt.show()
```

```py
# scikit-learn 선형회귀: Diabetes 예제
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
Y = pd.Series(diabetes.target, name="target")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, Y_train)

train_pred = lr.predict(X_train)
test_pred  = lr.predict(X_test)

print("train MSE :", mean_squared_error(Y_train, train_pred))
print("test  MSE :", mean_squared_error(Y_test,  test_pred))

# 단일 특성 기준 산점도
if 'bmi' in X_test.columns:
    plt.figure(figsize=(6,4))
    plt.scatter(X_test['bmi'], Y_test, label='Test')
    plt.scatter(X_test['bmi'], test_pred, label='Predict')
    plt.legend(); plt.title("Diabetes: BMI vs Target (Test vs Predict)")
    plt.tight_layout(); plt.show()
```

> **백문이 불여일견**이라는 말이 있듯이 공식 설명만 듣는 것보다, **직접 최소제곱법으로 계산 해보고** → **경사하강으로 근사**해 보면 눈으로 확인된다.

실제로 학생들도 직접 계산한 값과 코드 결과가 유사하다는 것이 신기하다는 반응이 많았다.

대학에 가면 수학은 충분히 깊게 배우게 될 테니, 지금은 **흥미를 붙이는 것**이 가장 중요하다고 생각했다.

## 마무리(느낀 점)

고등학교 시절 수업을 듣던 장소에서 내가 수업을 하게 된다는 경험은 무척이나 이상한 느낌이었다.

듣는 입장에서 설명하는 입장이 되니 무척이나 어려움이 많았다.

강의를 준비하면서 **"어떻게 하면 학생들이 잘 이해할 수 있을까?"**를 무척이나 많이 고민한 것 같다.

많은 고민 끝에 나온 최소제곱법 계산과 경사하강법을 비교해보는 실습에 대해 학생들의 반응이 괜찮아서 다행이라는 생각이 든다.

동시에 배우는 점도 무척 많았다.

내용을 알기 쉽게 구성하는 과정에서 기초적인 내용을 돌아보게 되었고 나의 지식 기반을 단단히 다질 수 있었다. 

추가로 코드도 학생들이 구현해보면 좋을 것 같아서 구현까지 염두에 두고 노트북을 작성했지만, 생각보다 파이썬에 대해 익숙치 않은 학생들이 많아서 다음시간 딥러닝에 대한 설명 이후 파이썬에 대한 강의를 진행하면 좋을 것 같다는 생각이 들었다.

동아리의 많은 학생들이 소프트웨어 분야 진학을 희망하는 만큼 미리 코드를 작성할줄 알면 활동 선택의 폭이 넓어질 것이다.

다음 시간에는 딥러닝 강의 후기를 정리 해보겠다.

