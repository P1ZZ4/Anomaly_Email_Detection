# 해당 코드는 jupyter lab에서 돌린 code이며, 편의상 py로 올려놓음 

import pandas as pd
data = pd.read_csv('int_file_6500.csv',encoding='utf-8')

pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

del data['Unnamed: 0']

# v1 == X-spam-type
# v2 == Subject 

X_data = data['Subject']
y_data = data['X-spam-type']

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# 정규표현식을 써서, 단어가 아니면 공백으로
#spam_data['v2'] = spam_data['v2'].str.replace("[^\w]|br", " ")

# 혹시 공백이 있으면 제거
#spam_data['v2'] = spam_data['v2'].replace("", np.nan)
#spam_data['v1'] = spam_data['v1'].replace("", np.nan)
# null array 없애는 함수
#spam_data = spam_data.dropna(axis=1)
#spam_data = spam_data.dropna(how='any')
#spam_data.columns = ["label", "mail"]

review_train, review_test, y_train, y_test = train_test_split(data2['Subject'], data2['X-spam-type'], test_size=0.25, shuffle=False, random_state=23)

print("# split done")

stopwords = ['a', 'an']

X_train = []
for stc in review_train:
    token = []
    words = stc.split()
    for word in words:
        if word not in stopwords:
            token.append(word.lower())
    X_train.append(token)

X_test = []
for stc in review_test:
    token = []
    words = stc.split()
    for word in words:
        if word not in stopwords:
            token.append(word.lower())
    X_test.append(token)

print("# tokenization done")

#머신러닝에 사용될 수 있도록 각 단어를 숫자정보로 변경한다.
#추후 Word Embedding 과정에 들어가 적절히 임베딩될 수 있도록 하는 것이다.

from tensorflow.keras.preprocessing.text import Tokenizer

# 인덱스 개수 기준
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
print(len(tokenizer.word_counts))

#Tokenizer 메소드를 불러와 객체를 만들고, X_train에 fitting한다.
#해당 객체의 word_counts에는 인덱싱한 총 단어가 적혀있다.
#우리는 전체 단어를 인덱싱하지 않는다. 빈도수가 낮은 단어까지 학습에 이용할 필요는 없기 때문이다.
#따라서 적절한 값을 탐색한다.

count = 0
for word, word_count in tokenizer.word_counts.items():
    if word_count > 1:
        count += 1
print(count)

#빈도수가 1초과, 즉 적어도 2번 이상 등장한 단어만 카운팅한다.
#1237, 약 2000개 정도만 인덱싱하기로 하자.
#tokenizer 객체가 X_train 내부의 4000개 단어를 인덱싱하도록 fitting 했다.

tokenizer = Tokenizer(2000)
tokenizer.fit_on_texts(X_train)

from tensorflow.keras.preprocessing.sequence import pad_sequences

# 부여된 정수 인덱스로 변환
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print("# int_encoding done")

# 최종적으로 데이터에 대해 정수 인코딩을 진행한다.
# X_train에 대해서만 fitting을 진행했기 때문에 X_test도 그를 기준으로 인덱싱될 것이다.
# X_train과 X_test로 나누기 전에 전체 X에 대해 fitting을 시켜줘도 프로세스 상 큰 문제는 없다.

################# 데이터 패딩 #################

# 각 벡터가 크기가 다르면 원활한 진행을 할 수 없으므로 크기를 통일화해준다.

# 패딩 결정, 임베딩 레이어로 넣을 벡터 길이를 정함
# 해당 벡터길이는 최대길이 or 평균길이
max_len = 15
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

#pad_sequences 메소드를 활용해 max_len값만큼 데이터를 늘리거나 줄인다. (늘릴 경우 앞을 0으로 채운다.)
#보통 최대길이 혹은 평균길이를 활용한다. 각각 장단점이 있다.
#최대길이 : 데이터 손실은 없지만, 효율 떨어짐
#평균길이 : 데이터 손실은 있지만, 효율이 좋음
#본 코드에서는 최대길이도 평균길이 사이 정도의 값을 선택했다.

################# 모델 구축 ##################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Embedding

# 모델 구축
# 레이어들을 쌓을 모델을 생성
model = Sequential()
model.add(Embedding(2000, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

#Sequential에 각 레이어를 쌓는다.
#지금까지 구축한 데이터(단어들이 인덱싱된 데이터)가 Embedding에 들어가 워드 임베딩화 된다. (4000개의 단어를 32차원으로 보낸다.)
#그렇게 임베딩화된 데이터가 LSTM을 통화하며 특정 값들을 도출한다. (입력은 32차원으로 보내온 임베딩 데이터)
#그들을 sigmoid함수에 통과시켜 최종 값을 찾는다.

################# 최적 모델 찾기 #################

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 테스트 데이터 손실함수값(val_loss)이 patience회 이상 연속 증가하면 학습을 조기 종료하는 콜백
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# 훈련 도중 테스트 데이터 정확도(val_acc)가 높았던 순간을 체크포인트로 저장해 활용하는 콜백
model_check = ModelCheckpoint('the_best.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# 학습을 무조건적으로 많이 시키면 과적합 -> 테스트 데이터에 대한 손실함수와 정확도가 떨어짐

################# 모델 학습 ####################

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64, callbacks=[early_stop, model_check])

#손실함수는 교차엔트로피(binary_crossentropy), 최적화방법은 아담(adam), 평가기준은 정확도(acc)로 한다.
#validation_data엔 검증 데이터가 들어간다.
#앞서 만든 콜백도 callbacks 변수에 넣어준다.

# 정확도 측정
# 출력하면 [loss, acc]
print(model.evaluate(X_test, y_test))
