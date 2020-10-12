---
title:  "[Python] fasttext 설치 및 실습 코드(예제)"
header:
  teaser: ""
excerpt: "fasttext 라이브러리를 윈도우에서 설치하는 법과 코드를 간단히 설명하고자 합니다."
categories:
  - NLP
tags:
  - fasttext
  - NLP

last_modified_at: 2020-09-26T16:01:04-04:00
toc: false
toc_ads: true
toc_label: "On this page"
---
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Frlagywns0213.github.io%2Fnlp%2FNLP-fasttext0%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=%EC%A1%B0%ED%9A%8C%EC%88%98&edge_flat=false)](https://hits.seeyoufarm.com)
## 0\. 설치

<br> fasttext 관련 코드 [github주소](https://github.com/facebookresearch/fastText)입니다.<br>

먼저, fasttext 라이브러리를 설치 위한 리눅스 형식의 설치법입니다.

```python
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
$ sudo python setup.py install
```
위의 코드는 [이 사이트](https://fasttext.cc/docs/en/support.html) 에서 구체적으로 확인할 수 있습니다.

그러나 위의 깃허브 주소에서 권장하는 fasttext 라이브러리 설치 방법은 윈도우에서 사용하기는 까다로워 보입니다.
따라서,
[다음 사이트](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext)에 들어가 파이썬 버전에 맞는 whl파일을 다운받습니다.

    > Example)
    python3.6/64bit : fasttext-0.9.1-cp36-cp36m-win_amd64.whl
    python3.7/64bit : fasttext-0.9.1-cp37-cp37m-win_amd64.whl
    python3.8/64bit : fasttext-0.9.1-cp38-cp38-win_amd64.whl

  > 저는 파이썬 3.7 버전과 윈도우 64비트를 사용중이라 "fasttext‑0.9.2‑cp37‑cp37m‑win_amd64.whl" 이 파일을 다운받았습니다!

디렉토리에 다운 받은 파일을 넣고, cmd 창에 "**pip install fasttext‑0.9.2‑cp37‑cp37m‑win_amd64.whl**" 를 입력하시면 fasttext 라이브러리가 설치됩니다.

확인을 위해, 쥬피터 노트북이나 cmd 창에서 python을 열고 import fasttext를 하시면 에러 없는 것을 확인할 수 있습니다!


## 1\. fasttext 파라미터

fasttext에서 제공하는 [코드](https://fasttext.cc/docs/en/support.html) 를 실습해 보았습니다.

1. train_unsupervised('data.txt') <br>
파라미터로 모델 지정 가능 ('skipgram', 'cbow')

#### train_unsupervised parameters
```
input             # training file path (required)
model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
lr                # learning rate [0.05]
dim               # size of word vectors [100]
ws                # size of the context window [5]
epoch             # number of epochs [5]
minCount          # minimal number of word occurences [5]
minn              # min length of char ngram [3]
maxn              # max length of char ngram [6]
neg               # number of negatives sampled [5]
wordNgrams        # max length of word ngram [1]
loss              # loss function {ns, hs, softmax, ova} [ns]
bucket            # number of buckets [2000000]
thread            # number of threads [number of cpus]
lrUpdateRate      # change the rate of updates for the learning rate [100]
t                 # sampling threshold [0.0001]
verbose           # verbose [2]
```

  2. train_supervised('data.train.txt')

#### train_supervised parameters
```
input             # training file path (required)
lr                # learning rate [0.1]
dim               # size of word vectors [100]
ws                # size of the context window [5]
epoch             # number of epochs [5]
minCount          # minimal number of word occurences [1]
minCountLabel     # minimal number of label occurences [1]
minn              # min length of char ngram [0]
maxn              # max length of char ngram [0]
neg               # number of negatives sampled [5]
wordNgrams        # max length of word ngram [1]
loss              # loss function {ns, hs, softmax, ova} [softmax]
bucket            # number of buckets [2000000]
thread            # number of threads [number of cpus]
lrUpdateRate      # change the rate of updates for the learning rate [100]
t                 # sampling threshold [0.0001]
label             # label prefix ['__label__']
verbose           # verbose [2]
pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []
```

Model object functions
```
get_dimension           # Get the dimension (size) of a lookup vector (hidden layer).
                        # This is equivalent to `dim` property.
get_input_vector        # Given an index, get the corresponding vector of the Input Matrix.
get_input_matrix        # Get a copy of the full input matrix of a Model.
get_labels              # Get the entire list of labels of the dictionary
                        # This is equivalent to `labels` property.
get_line                # Split a line of text into words and labels.
get_output_matrix       # Get a copy of the full output matrix of a Model.
get_sentence_vector     # Given a string, get a single vector represenation. This function
                        # assumes to be given a single line of text. We split words on
                        # whitespace (space, newline, tab, vertical tab) and the control
                        # characters carriage return, formfeed and the null character.
get_subword_id          # Given a subword, return the index (within input matrix) it hashes to.
get_subwords            # Given a word, get the subwords and their indicies.
get_word_id             # Given a word, get the word id within the dictionary.
get_word_vector         # Get the vector representation of word.
get_words               # Get the entire list of words of the dictionary
                        # This is equivalent to `words` property.
is_quantized            # whether the model has been quantized
predict                 # Given a string, get a list of labels and a list of corresponding probabilities.
quantize                # Quantize the model reducing the size of the model and it's memory footprint.
save_model              # Save the model to the given path
test                    # Evaluate supervised model using file given by path
test_label              # Return the precision and recall score for each label.
```

## 2\. fasttext 실습
```python
import fasttext
model = fasttext.train_unsupervised('review.sorted.uniq.refined.tsv.text.tok',model='skipgram', epoch=5,lr = 0.1)
print(model.words)   # list of words in dictionary
```
![image](https://user-images.githubusercontent.com/28617444/95245656-5e730680-084e-11eb-97bd-d35ad0b2c4fb.png)

```python
print(model['행사']) # get the vector of the word '행사'
```
단어 '행사' 에 대한 vector를 도출할 수 있습니다. <br><br>
![image](https://user-images.githubusercontent.com/28617444/95245774-8e220e80-084e-11eb-8b96-325a8abb6e01.png)

## Importance of character n-grams

서브단어 정보를 사용하면 모르는 단어의 벡터도 도출할 수 있습니다.
```python
model.get_word_vector("보아즈")
```
![image](https://user-images.githubusercontent.com/28617444/95246650-beb67800-084f-11eb-9fb0-e69c8f588db6.png)

이처럼 데이터에 없는 단어(vocabulary에 없는 단어)도 벡터 출력 가능합니다.<br>
이를 out of vocabulary (oov)라 합니다.

## Nearest neighbor queries

word vector의 퀄리티를 평가하며 벡터의 의미 정보 유형을 직관적으로 보여줍니다.

```python
model.get_nearest_neighbors('반짝반짝')
```
![image](https://user-images.githubusercontent.com/28617444/95246297-3afc8b80-084f-11eb-8261-c8651fa57df2.png)

## Measure of similarity
단어 사이의 유사성을 계산해 최근접 이웃을 찾을 수 있을 수 있습니다.
모든 단어들을 계산해서 가장 유사한 단어 10개를 나타내고 해당 단어가 있다면, 상단에 표시되고 유사도는 1입니다.

## Word analogies
다음과 같이 세 단어의 관계를 유추할 수 있습니다.

```python
model.get_analogies("저렴","싸","할인")
```
![image](https://user-images.githubusercontent.com/28617444/95246464-7dbe6380-084f-11eb-80ac-8b01a474512f.png)
