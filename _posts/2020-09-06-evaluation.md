---
title:  "분류 성능 평가 지표 정리 - 파이썬 머신러닝 완벽 가이드"
header:
  teaser: "/assets/images/book2.jpeg"

excerpt: "머신러닝 분류 모델의 성능 평가를 위한 지표를 정리하고자 한다."
categories:
  - Data Analysis
tags:
  - Python
  - evaluation
  - machine learning
last_modified_at: 2020-09-06T16:01:04-04:00
toc: true
toc_ads: true
toc_label: "On this page"

---
이 페이지 조회수 : [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Frlagywns0213.github.io%2Fdata%2520analysis%2Fevaluation%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=%EC%A1%B0%ED%9A%8C%EC%88%98&edge_flat=false)](https://hits.seeyoufarm.com)
## 0\. 들어가며

머신러닝은 데이터가공/변환, 모델 학습/예측, 그리고 **평가**의 프로세스로 구성된다.

비록 내가 데이터 수집, 전처리, 모델링까지 피나는 노력을 하여도 모델에 대한 평가지표를 알지 못하면 그것은 헛된 노력이 된다.

따라서, 분류 모델의 성능평가 지표를 '파이썬 머신러닝 완벽 가이드' 책을 읽고 정리하고자 한다.

**※  ADSP (데이터 분석 준전문가 자격) 시험 3과목에 무조건 출제되는 기출이다!! ※**

### 1\. 평가(Evaluation) 단계

-   '회귀' 모델 성능평가 지표
-   **'분류' 모델 성능평가 지표**

### 2\. 분류의 성능 평가 지표

**2.1 오차행렬(Confusion Matrix)**


![images](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb9IVyu%2FbtqH48uLrU5%2FJlaznfVZ4mo5kRw488loq1%2Fimg.png)

—>  이진 분류의 예측 오류가 얼마인지 + 어떠한 유형의 예측 오류가 발생하고 있는지 파악 가능

—>  TP, TN, FP, TN 값을 조합해 Classifier 의 성능을 측정할 수 있는 주요 지표인 정확도, 정밀도, 재현율 값을 확인 가능

**2.2 정확도(Accuracy)**

: 실제 데이터에서 예측 데이터가 얼마나 같은지를 판단하는 지표 → 직관적으로 모델 예측 성능 나타냄

> **" 정확도 = 예측 결과가 동일한 데이터 건수 / 전체 예측 데이터 건수 "  ( TN + TP ) / (TN + FP + FN + TP ) **

※ 단점 : 100개의 데이터에서 90개의 데이터 레이블이 0 , 10개의 데이터 레이블 1이라면 —>

무조건 0으로 예측 결과를 반환하는 모델이라면 정확도가 90%임. 즉, 예측값과 실제 값이 얼마나 동일한가에 대한 비율만으로 결정

따라서, _**'정확도'라는 지표 하나만으로 모델의 성능을 평가해서는 안됨!**_

**2.3 정밀도(Precision)**

: 예측을 Positive로 한 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율

> **" 정밀도 = (TP) / (FP + TP) " **

외우는 꿀팁 : Predict의 pre를 본딴 precision→ 참이라고 _예측_한 값중 참!

**2.4 재현율(Recall) = 민감도(Sensitivity) = TPR(True Positive Rate)**

: 실제 값이 Positive로 한 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율

> **" 재현율= (TP) / (FN + TP) "  
> **

—> 실제 Positive 양성 데이터를 Negative로 잘못 판단하게 되면 업무상 큰 영향 발생

외우는 꿀팁 : 재현해야되고 민감한 것 → 실제  참인 것들 중 참

**2.5 F1 SCORE**

: 정밀도와 재현율을 결합한 지표 (조화평균)

![images](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbzsRhX%2FbtqHYJi1uvN%2FBquZX41kJ9vTJzauOGRgS0%2Fimg.png)

**2.6 ROC 곡선 & AUC (Area Under Curve)**

: FPR( False Positive Rate){1-특이도}이 변할 때 TPR(True Positive Rate){민감도}이 어떻게 변하는지를 나타내는 곡선

> x축은 FPR(False Positive Rate)로, 틀린것을 맞았다고 잘못 예측한 수치(FPR = FP / (FP + TN))  
> y축은 TPR(True Positive Rate)로, 맞은것을 맞았다고 잘 예측한 수치 (TPR = TP / (TP + FN)) \[==Recall\]

![images](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbgHBQQ%2FbtqHWvSPYhF%2F9CITKubNzFK74nQ7Uj1990%2Fimg.png)

-   가운데 직선 : ROC 곡선의 최저 값
-   ROC 곡선이 가운데 직선에 가까우면 성능 떨어진 것, 멀어질수록 성능 뛰어난 것
-   AUC (Area Under Curve) : 그래프 아래의 면적값으로써 최대값은 1에 가까울 수록 좋은 모델

### 3\. 실습

파이썬 머신러닝 완벽 가이드의 피마 인디언 당뇨병 예측(p.174) 예제를 실습하였다.

```ruby
#필요한 라이브러리 import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#데이터 불러온다.
diabetes_data = pd.read_csv('datasets_228_482_diabetes.csv')
diabetes_data.head()
```

![images](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdONKok%2FbtqHZjEegG2%2FgOnCcvKotJJuQJaW3b4rHK%2Fimg.png)

▶ 종속변수로 사용될 Outcome 클래스의 결정값의 개수를 파악

```ruby
print(diabetes_data['Outcome'].value_counts())

#negative 500개
#positive 268개
```

![images](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FLVn15%2FbtqHXDCWX6A%2Fnd1hL3TzsAu892hNW55ij1%2Fimg.png)

> 전체 768개 데이터 중에서 **Negative값이 500개, Positive 값이 268개로 상대적으로 Negative가 많다.**  
> 만약 너무 편향된 데이터라면 업샘플링 / 다운샘플링 을 통하여 모델을 개선시킬 수 있다.   
> 추후에 업샘플링과 다운샘플링에 대해 블로그에 포스트 할 예정이다.

▶ feature 타입과 Null 개수 파악

```
diabetes_data.info()
```

![images](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FP3CJR%2FbtqH4lgUQ7D%2FOQDw0igkeTuFBg28G310YK%2Fimg.png)

> Null 값은 없으며 피처의 타입은 모두 숫자형  
> 임신 횟수, 나이와 같은 숫자형 피처 + 당뇨 검사 수치 피처로 구성된 특징을 볼 때, **별도의 피처 인코딩은 필요하지 않아보인다.  
> **

▶ **성능 평가 지표 출력하는 함수 만들기!! (분류 모델 성능 평가 가능)**

```ruby
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    #ROC-AUC 추가
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    #roc-auc print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
```

▶ 로지스틱 회귀 예측 모델

```ruby
X= diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 156, stratify=y)

#로지스틱 회귀로 학습, 예측 및 평가 수행
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred= lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:,1]

get_clf_eval(y_test, pred, pred_proba)
```

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbMB4c3%2FbtqH3DB1rfv%2FHsSub15u2kkiq3SuS5DCX1%2Fimg.png)

## 4\. 정리

지금까지 분류에 사용되는 정확도, 오차 행렬, 정밀도, 재현율, F1 스코어, ROC-AUC  의 성능 평가 지표를 살펴보았다.

특히 **이진 분류에서 레이블 값이 불균형하게 분포될 경우**는 단순히 _정확도만으로는_ 예측 성능 평가가 어렵다.

따라서, 정밀도 재현율을 결합한 평가 지표인 F1 스코어를 통해 성능 평가를 한다. 이 때, 정밀도와 재현율이 어느 한쪽으로 치우치지 않을 때 높은 지표값을 가지게 된다.

또한, ROC 곡선 밑의 면적인 AUC 가 1에 가까울 수록 좋은 모델이라고 평가 가능하다.

이렇게 정리해두면 매번 분류 예측 모델할 때마다 찾아보지 않아도 되고, 위에서 정의해둔 get\_clf\_eval 함수를 통하여 쉽게 평가 가능할 것이다!
