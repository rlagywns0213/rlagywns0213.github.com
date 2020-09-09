---
title:  "캐글 데이터 EDA -PortoSeguro"
header:
  teaser: "https://user-images.githubusercontent.com/28617444/92389061-4d51af80-f153-11ea-9711-5251e9f57a08.PNG"
excerpt: "캐글 데이터로 EDA, 데이터 전처리를 시도해보며 인사이트 도출하고자 한다"
categories:
  - Data Analysis
tags:
  - Python
  - EDA
  - kaggle
last_modified_at: 2020-09-07T16:01:04-04:00
toc: true
toc_label: "On this page"

---
## 0\. 들어가며

'캐글'의 안전 운전자 예측 경진대회의 데이터를 이용하여 EDA를 진행해보았다.

또한, 우승자 코드 및 notebook을 참고하여 간단한 예측 모델을 구성하였다.

캐글 홈페이지는 다음과 같다.

## 1\. EDA

**1.1 데이터 불러오기**

대회에서 제공하는 데이터는 결측치에 대해 '-1'의 값으로 지정해주었다고 명시되어 있다. 

이를 편하게 전처리 하기 위해서 na\_values 에 -1을 지정해주었다.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv',na_values = ['-1','-1.0'])
test = pd.read_csv('test.csv',na_values = ['-1','-1.0'])


```

**1.2 결측값 처리**

데이터의 결측치 갯수를 각 column별로 도출하여 그래프로 나타내는 함수를 만들었다.

```python
def describe_missing_values(df):
    na_percent = {}
    N = df.shape[0]
    for column in df:
        na_percent[column] = df[column].isnull().sum() * 100 / N

    na_percent = dict(filter(lambda x: x[1] != 0, na_percent.items()))
    plt.bar(range(len(na_percent)), na_percent.values())
    plt.ylabel('Percent')
    plt.xticks(range(len(na_percent)), na_percent.keys(), rotation='vertical')
    plt.show()

print("Missing values for Train dataset")
describe_missing_values(train)

print("Missing values for Test dataset")
describe_missing_values(test)
```

![image](https://blog.kakaocdn.net/dn/bG3vuc/btqH7byfNVK/N4PufDwWdbekKYLyA1YX81/img.png)

> 20%가 넘는 결측치가 존재 시, column 삭제  
> numeric 값은 평균값 대체, 나머지는 최빈값 대체

```python
binary = train.filter(like='bin').columns
category = train.filter(like='cat').columns
integer = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_calc_04',
       'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',
       'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14']
floats = ['ps_reg_01','ps_reg_02', 'ps_reg_03',
          'ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15',
          'ps_calc_01','ps_calc_02','ps_calc_03']
numeric = integer + floats

#20%가 넘는 결측값 삭제
train.drop([ "ps_car_03_cat", "ps_car_05_cat"], axis=1, inplace=True)
test.drop([ "ps_car_03_cat", "ps_car_05_cat"], axis=1, inplace=True)

# numeric 값은 평균값 대체, 나머지 최빈값
for col in binary:
    train[col].fillna(value=train[col].mode()[0], inplace=True)
    test[col].fillna(value=test[col].mode()[0], inplace=True)

for col in category:
    train[col].fillna(value=train[col].mode()[0], inplace=True)
    test[col].fillna(value=test[col].mode()[0], inplace=True)

for col in numeric:
    train[col].fillna(value=train[col].mean(), inplace=True)
    test[col].fillna(value=test[col].mean(), inplace=True)

print("Missing values for Train dataset")
describe_missing_values(train)

print("Missing values for Test dataset")
describe_missing_values(test)
```

![image](https://blog.kakaocdn.net/dn/Vf67l/btqH2ygVj84/9YobhhiAG8nRXyLkjGxCV0/img.png)

**1.3 종속변수 데이터 확인**

```Python
target0=train['target'].value_counts()[0]
target1=train['target'].value_counts()[1]
print('Class 0:', target0)
print('Class 1:', target1)
print('Proportion:', round(target0 / target1, 2), ': 1')

train['target'].value_counts().plot(kind='bar', title='Count (target)');
```
![image](https://user-images.githubusercontent.com/28617444/92388897-fcda5200-f152-11ea-8056-269e1cbd6693.PNG)

매우 편향된 데이터임을 확인할 수 있다. 오버샘플링 / 다운샘플링 을 시도해보고 평가 지표를 비교해본다.

**1.4 그래프를 통한 확인**

데이터가 column별로 편향되어있는지를 확인한다.

integer, numeric, category 별로 각각 확인하며 그 다음 코드로 압축하며 생략한다.

```python
for col in integer:

    fig = plt.figure(figsize=(18,12))
    sns.distplot(train[col], ax=plt.subplot(221))
    plt.xlabel(col, fontsize=14);
    # Y-axis Label
    plt.ylabel('Density', fontsize=14);
    # Adding Super Title (One for a whole figure)
    plt.suptitle('Plots for '+col, fontsize=18);

    #claim not
    sns.distplot(train.loc[train.target==0, col], color='red', label='Claim not filed', ax=plt.subplot(222))
    # Claim Filed hist
    sns.distplot(train.loc[train.target==1, col], color='blue', label='Claim filed', ax=plt.subplot(222))
    # Adding Legend
    plt.legend(loc='best')
    # X-axis Label
    plt.xlabel(col, fontsize=14)
    # Y-axis Label
    plt.ylabel('Density per Claim Value', fontsize=14)

    ### Average Column value per Claim Value
    sns.barplot(x="target", y=col, data=train, ax=plt.subplot(223));
    # X-axis Label
    plt.xlabel('Is Filed Claim?', fontsize=14);
    # Y-axis Label
    plt.ylabel('Average ' + col, fontsize=14);

    ### Boxplot of Column per Claim Value
    sns.boxplot(x="target", y=col, data=train, ax=plt.subplot(224));
    # X-axis Label
    plt.xlabel('Is Filed Claim?', fontsize=14);
    # Y-axis Label
    plt.ylabel(col, fontsize=14);
    # Printing Chart
    plt.show()

```
![image](https://user-images.githubusercontent.com/28617444/92389061-4d51af80-f153-11ea-9711-5251e9f57a08.PNG)

> 매우 편향된 데이터 "ps\_ind\_10\_bin", "ps\_ind\_11\_bin", "ps\_ind\_12\_bin", "ps\_ind\_13\_bin"를 파악함.  
> 위의 변수들을 제거해준다.
