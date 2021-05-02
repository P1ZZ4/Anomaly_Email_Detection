# 해당 코드는 jupyter lab에서 돌린 code이며, 편의상 py로 올려놓음 
# Test용 코드로 데이터셋의 일부로만 실험


#Preprocessing

import pandas as pd
data = pd.read_csv('spam_int.csv',encoding='utf-8')

pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

#print("sample: ", len(data))
#data[:10]

#del data['Unnamed: 0']

#data.dtypes

data['X-spam-type'] = data['X-spam-type'].replace(['RELAY','SPAM'],[0,1])

"""
#file['X-spam-type'].astype(int)

## 이거 되려면 결측치를 다 채워줘야함 

data.fillna(1.0) # 결측치를 모두 1.0으로 채움

여긴 결국 실패 
"""

data.isnull().values.any() # 결측값 확인 

pd.isnull(data) # 결측값이 어디에 존재하는지 확인

data.isnull().sum() # 결측값 갯수 확인 

data = data.dropna(axis=0) # 결측치가 있는 행 제거 

# float형을 int형으로 변환
data['X-spam-type'] = data['X-spam-type'].astype(int) 

#int_file = data.to_csv('int_file.csv') # 중간에 파일 저장

# bar형으로 spam(1), ham(0) 시각화 
data['X-spam-type'].value_counts().plot(kind='bar');

# 갯수로 spam(1), ham(0) 확인
print(data.groupby('X-spam-type').size().reset_index(name='count'))

"""
본문 수: 937
스팸 값 수: 937
"""


# Vanilla RNN
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data) # 937개의 행을 가진 X의 각 행에 토큰화를 수행
sequences = tokenizer.texts_to_sequences(X_data) # 단어를 숫자값, 인덱스로 변환하여 저장

#print(sequences[:5])

word_to_index = tokenizer.word_index
#print(word_to_index)

threshold = 2
total_cnt = len(word_to_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합(vocabulary)에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

threshold = 2
total_cnt = len(word_to_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합(vocabulary)에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

max_len = 22
# 전체 데이터셋의 길이는 max_len으로 맞춥니다.
data = pad_sequences(X_data, maxlen = max_len)
print("훈련 데이터의 크기(shape): ", data.shape)

X_test = data[n_of_train:] #X_data 데이터 중에서 뒤의 n개의 데이터만 저장
y_test = np.array(y_data[n_of_train:]) #y_data 데이터 중에서 뒤의 n개의 데이터만 저장
X_train = data[:n_of_train] #X_data 데이터 중에서 앞의 n개의 데이터만 저장
y_train = np.array(y_data[:n_of_train]) #y_data 데이터 중에서 앞의 n개의 데이터만 저장

from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(vocab_size, 32)) # 임베딩 벡터의 차원은 32
model.add(SimpleRNN(32)) # RNN 셀의 hidden_size는 32
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=4, batch_size=64, validation_split=0.2)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

