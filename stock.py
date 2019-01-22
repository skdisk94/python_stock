import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

tf.set_random_seed(777)

# 가장 작은 값을 0으로하고 0을 기준으로 정규화 (숫자가 크기 때문에)
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min())/(x_np.max() - x_np.min() + 1e-7)

def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

seq_length = 21
input_dim = 5
output_dim =1
hidden_dim = 10  # 각 셀의 (hidden)출력 크기

# 데이터 불러오기
dataframe = pd.read_csv('stock_data.csv', encoding='euc-kr')
name = dataframe['종목명'][1]
del dataframe['날짜']
del dataframe['종목명']

# 데이터 정규화
df = min_max_scaling(dataframe)

# x : 입력 데이터 , y : 출력 데이터(종가)
x = df
y = x[:, [0]]

dataX = []
dataY = []

#_x : i부터 seq_length 기간 동안의 입력 데이터, _y : seq_length 다음날의 출력 데이터(종가)
for i in range(0, len(y) - seq_length):
    _x = x[i: i + seq_length]
    _y = y[i + seq_length]
    dataX.append(_x)
    dataY.append(_y)

# 학습 데이터 : 70% , 테스트 데이터 : 30%
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size

trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size: len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size: len(dataY)])

# placeholder, 행렬의 차원[seq_length:input]
X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
Y = tf.placeholder(tf.float32, [None, 1])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# ***fully_connected(fc)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn = None) # Y의 예측 값

loss = tf.reduce_sum(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    _, l = sess.run([train, loss], feed_dict={X:trainX, Y:trainY})
    if i % 100 == 0:
        print('epoch : {0}, loss : {1}'.format(i, l))

testPredict = sess.run(Y_pred, feed_dict={X:testX})

path = 'c:Windows\Fonts\malgunbd.ttf'
fontprop = fm.FontProperties(fname=path, size=18)

plt.plot(testY,'skyblue')
plt.plot(testPredict,'orange')
plt.title('종목명 : ' + name, fontproperties=fontprop)
plt.xlabel("Time Period")
plt.ylabel("Stock Price") # 정규화 되어있는 주가
plt.show()

recent_data = np.array([x[len(x) - seq_length:]])
testPredict = sess.run(Y_pred, feed_dict={X:recent_data})
testPredict = reverse_min_max_scaling(dataframe, testPredict)  # 금액데이터 역정규화
print("Tomorrow's stock price", testPredict[0])  # 예측한 주가를 출력