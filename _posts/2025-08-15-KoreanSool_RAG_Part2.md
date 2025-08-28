---
title: "[한국 전통주 RAG] 2편: 데이터 크롤링과 파싱"
date: 2025-08-15 22:00:00 +09:00
categories: [Project, KoreanSool_RAG]
tags: [KoreanLiquor, RAG, Crawling, Dataset, AI]
toc: true
pin: true
image:
  path: /assets/img/for_post/KoreanSool_RAG/crawling.png
  alt: "전통주 RAG 크롤링"
---

# 한국 전통주 RAG 프로젝트 2편: 데이터 크롤링과 파싱

RAG 시스템을 만들려면 **데이터 확보**가 가장 먼저다.

나는 우연한 계기로 잘 정리된 데이터를 찾았으니 반은 온 셈이다.

이번 편에서는 내가 어떻게 한국술 고문헌 DB에서 데이터를 긁어왔는지, 그리고 그 데이터를 어떤 방식으로 파싱했는지 정리해보려고 한다.  

---

## 1. 처음엔 막막했던 구조

사이트 메인에 들어갔을 때는 검색창 하나만 있었다.

"특정 키워드로 검색을 하고 그 결과를 크롤링 해야 하나?" 싶었지만, 무작정 키워드를 넣어 긁는 건 비효율적이라 포기했다.  

---

## 2. 상세 주방문 페이지 발견

대신 **"상세 주방문" 페이지**를 찾아냈다.  

여기서는 셀렉트 박스로 문헌을 고르면, 그 문헌에 수록된 술 정보가 한 번에 표시되는 구조였다.  

셀레니움으로 간편하게 셀렉트 박스 값들을 순회하며, 크롤링을 진행 가능하다는 것을 이전에 배워둔 적이 있기 때문에 이 페이지를 활용하면 되겠다는 생각이 들었다.

그래서 실제로 셀레니움을 이용해 각 문헌을 선택 → 결과 페이지 HTML을 저장하는 방식으로 전체 데이터를 수집했다.  

```python
from selenium import webdriver
from selenium.webdriver.support.ui import Select

def get_book_options(driver):
    sel = Select(driver.find_element("name", "book"))
    return [
        {"value": opt.get_attribute("value"), "text": opt.text}
        for opt in sel.options
        if opt.get_attribute("value") not in ("", "@")
    ]

def save_current_html(driver, label, suffix):
    fname = f"{label}__{suffix}.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    return fname
```

---

## 3. 파싱을 하며 생긴 고민

결론적으로 셀렉트 박스를 순회하며 각 서적에 포함된 레시피들을 담은 HTML은 성공적으로 크롤링했다.

HTML을 모았으니 나니 이제 파싱을 해야 했다.

그런데 데이터를 까보니 생각보다 단순하지 않았다.  

앞서 말했듯 한국술 고문헌 DB에는 술 이름과 레시피뿐 아니라 다양한 정보가 있다.

- 고려시대 군인들에게 술을 공급하던 관청 기록  
- 앵무조개로 만든 술잔 같은 물품 정보  
- "신 술 고치는 법" 같은 양조 방법  

즉, **모든 데이터가 술은 아니다.**

이걸 어떻게 `술`/`누룩`/`기타`로 나눌지 고민이 컸다.

카테고리만 보자니 분류가 굉장히 많고 분명 술 레시피지만 카테고리 안에 `누룩`이 포함되는 경우가 많았다.

이름만으로 구분하기엔 엣지 케이스(예: 구주불비법 / 법주)가 있었고,    

그러다 중요한 규칙을 발견했다.  
> **카테고리의 첫 번째 값**이 항상 가장 큰 분류(발효주/과하주/혼성주 같은 주류 or 누룩/기타)를 의미한다.  

그래서 `parse_koreansool.py`에서는 첫 번째 카테고리를 기준으로 사이트 메인에 있던 분류 기준인 `sool`, `nuruk`, `other` 세 가지로 정리했다.

```python
LEVEL1_TO_MAIN = {
    "발효주": "sool",
    "증류주": "sool",
    "과하주": "sool",
    "혼성주": "sool",
    "누룩":   "nuruk",
    "기타":   "other",
}

def derive_main_category(liq, hanja, source_categories):
    level1 = source_categories[0] if source_categories else None
    if level1 in LEVEL1_TO_MAIN:
        return LEVEL1_TO_MAIN[level1]
    return "other"
```

코드에서는 레벨1 카테고리 즉, 가장 첫번째 카테고리를 기준으로 잡아 데이터를 분류하고 나머지를 저장한다.

이를 통해 메인 페이지에 있던 (술 2,876, 누룩 193, 기타 450) 카테고리에 해당하는 데이터들을 100%의 정확도로 성공적으로 분류할수 있었다.

> 전체 크롤링 & 파싱 로직은 [GitHub repo](https://github.com/Jaeuk-Han/korean-traditional-liquor-dataset)에 정리해 두었다.

---

## 4. 결과 구조

최종적으로 각 레코드는 대략 다음과 같은 JSON 구조를 가진다.  

```json
{
  "doc_id": "string",
  "book": "string",
  "title": "string",
  "liq": "string",
  "hanja": "string|null",
  "description": "string|null",
  "main_category": "sool",
  "entry_type": "sool",
  "text": "string|null",
  "metadata": {
    "layout": "string|null",
    "page_title": "string|null",
    "order": "number|null",
    "source_categories": ["string", "..."],
    "category_levels": {
      "level1": "string",
      "level2": "string|null",
      "level3": "string|null",
      "level4": "string|null"
    },
    "external_links": {
      "문헌 정보": "string",
      "상세 주방문": "string"
    },
    "grid_headers": ["string", "..."],
    "steps": [
      {
        "단계": "string",
        "일": "number|null",
        "발효": "string|boolean|null",
        "멥쌀": "number|null",
        "찹쌀": "number|null",
        "물": "number|null",
        "가공": "string|null",
        "누룩": "number|null",
        "누룩형태": "string|null",
        "메모": "string|null",
        "...": "..."
      }
    ],
    "memo_free": "string|null",
    "original": "string",
    "translation": "string",
    "year_guess": "number|null",
    "is_beverage": "boolean"
  }
}
```

어차피 이후에 전처리를 한번 할 예정이기에 HTML에서 가능한 많은 데이터를 추출하고자 했다.

---

## 5. 정리

- **크롤링 단계**: `crawler.py`로 셀렉트 박스를 순회하며 문헌별 HTML 저장  
- **파싱 단계**: `parse_koreansool.py`로 1분류 카테고리 기반 분류 (`sool/nuruk/other`)  
- **산출물**: 구조화된 JSON 데이터셋  

---

이제 기본 데이터셋이 확보되었으니, 다음 편에서는 이 데이터 중 필요한 데이터만 고르고 텍스트를 **RAG 학습에 맞게 전처리**하는 과정을 다뤄보려 한다.
