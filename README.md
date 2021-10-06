# deeplearning

딥러닝

- Sequential 함수로 딥러닝의 구조를 한층 씩 쌓아올릴 수 있도록 만들어 줌.
- Model.add를 사용하여 필요한 층을 차례로 추가하게 된다.

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

import numpy as np

import tensorflow as tf

import pandas as pd

(데이터 불러오기)

data = pd.read\_csv(&#39;dataset/ThoraricSurgery(1).csv&#39;, encoding = &#39;utf-8&#39;)

(feature, target 분리)

X = data\_set[:, 0:17]

y = data\_set[:, 17]

np.random.seed(3)

tf.random.set\_seed(3)

data\_set = np.loadtxt(&#39;dataset/ThoraricSurgery.csv&#39;, delimiter = &#39;,&#39;)

**실제 딥러닝 시행부분**

(딥러닝 구조 결정)

model = Sequential()

model.add(Dense(30, input\_dim = 17, activation = &#39;relu&#39;))

model.add(Dense(1, activation = &#39;sigmoid&#39;))

(딥러닝 실행)

model.compile(loss=&#39;binary\_crossentropy&#39;, optimizer=&#39;adam&#39;, metrics=[&#39;accuracy&#39;])

model.fit(X, y, epochs = 100, batch\_size = 10)

층을 얼마나 쌓을지 결정하는 것은 데이터의 형태에 따라 다르다. 하지만, 캐라스는 model.add로 필요한 만큼의 층을 쉽게 쌓을 수 있다는 장점을 가지고 있다.

- Activation : 다음 층으로 어떻게 값을 넣을지 정하는 것. (relu, sigmoid 함수를 사용)
- Loss : 한번 신경망이 시행될 때마다 오차를 추적하는 함수
- Optimizer : 오차를 줄이는 함수

딥러닝을 시행할 때는, 위의 5줄 형태로만 시행하면 된다. (Sequential, add, compile, fit으로 구현하는 것)

구조 이해

- Input (X : 17개의 feature)
  - Weight : 가중치로 Input에 곱해지는 값이다. 일정한 가중치를 부여하고 입력된 값을 변형하고 다음 layer에 넘긴다. 다음 layer에서도 Input \* weight으로 예측값 Y\*가 나온다.
  - Loss function : 실제 Y와 예측 Y를 비교하여 Loss score를 도출한다. 여기서 Optimizer를 활용하여 가중치를 다시 조정하게 된다.

NL 해보기

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

가상데이터 생성

X = data = np.linspace(1,2,200)

y = X\*4 + np.random.randn(200)\*0.3

model = Sequential()

model.add(Dense(1, input\_dim=1, activation = &#39;linear&#39;))

# tf.keras.models.Sequential, tf.keras.layer.Dense -\&gt; from import로생략

model.compile(loss = &#39;mse&#39;, optimizer = &#39;sgd&#39;, metrics = [&#39;mse&#39;])

model.fit(X, y, batch\_size = 1, epochs = 30)

predict = model.predict(data)

plt.plot(data, predict, &#39;b&#39;, data, y, &#39;k.&#39;)

plt.show()

![](RackMultipart20211006-4-if6336_html_73978768738fad7b.png)

model.fit(X, y, batch\_size = 1, epochs = 30)

output

Epoch 1/30

200/200 [==============================] - 1s 1ms/step - loss: 1.2804 - mse: 1.2804

Epoch 2/30

200/200 [==============================] - 0s 1ms/step - loss: 0.1637 - mse: 0.1637

…

Epoch 30/30

200/200 [==============================] - 0s 1ms/step - loss: 0.1033 - mse: 0.1033

위에서는 epochs를 30으로 했지만, 만약 100으로 지정한다면 loss값이 감소하다가 향상되는 것을 확인할 수 있음. 즉, Epochs를 늘린다고 해서 loss값이 무조건 줄어드는 것은 아니다. 이런 것을 over-fitting이라고 한다.

Loss : **회귀선**** (blue line) **과** 실제 값****(black dot)**의 차이를 의미한다.

Mse : Mean Squared Error로 총 오차 값을 의미한다. (해당 출력에서는 loss = mse)

또한, 그래프를 보면, 첫번째 그래프가 파란 선 (선형 회귀)으로 도출된 것을 볼 수 있다. 즉, NL이 선형 회귀와 관련성이 있다고 볼 수 있는 것이다. (검은 점은 실제 y값임.) 여기서 선형 그래프는 y = ax+b라고 할수 있다.

- &#39;a&#39;는 weight, 가중치라고 한다.
- &#39;b&#39;는 bias, 편향이라고 한다.

기울기(a), 절편(b)를 조정하여 오차를 최소화하는 것이 필요함. 그런데, loss는 2차 함수로 표현되는 것을 알 수 있음. (mse=loss이므로) 그런데 2차 함수의 형태는 볼록한 형태임. 2차함수의 최솟값(오차의 최솟값)을 도출하는 과정은 미분을 통해서 할 수 있음. Y = ax2의 최솟값은 미분한다면, y&#39;=2ax로 도출할 수 있고 x=0일 때, y&#39;=0이 된다. 이런 형태로 loss 값을 구할 수 있는 것이다. (참고로 미분 값은 접선의 기울기이기도 함)

그런데, 변수가 여러 개 있는 경우라면 편미분의 개념을 도입해야 한다.

- F(x,y) = x2 +yx + a
- Y에 대한 편미분 : x
- X에 대한 편미분 : 2x+y

![](RackMultipart20211006-4-if6336_html_f2be31edfd975f4e.png)

- 시그모이드: 로지즈틱회귀와같은분류문제를확률적으로표현하는데사용함.
  - 선형함수의결과를0~1 사이의비선형형태로변형해 줌.
  - 하지만, hidden layer의깊이가커지면기울기소멸문제가발생함. (미분을계속시행하다 보면, 값이작아지고결국기울기가0으로만들어지는문제임)
  - 시그모이드를미분한다면, 0에서가장높은값을가지고멀어질수록0에수렴함
- 하이퍼볼릭탄젠트(tanh) : -1, 1 사이의비선형으로바꿔 줌.
  - 0이아닌양수로만들어줬지만, 기울기소멸문제잔존
- ReLu
  - 기울기소멸의문제가발생하지않음. 입력이음수라면0, 양수이면x를출력함.
  - Tanh보다6배정도높은성능을자랑함
  - 하지만음수일 때0을출력하기에Leaky ReLu를도입하여한계를보완함.
- 소프트맥스
  - 입력값을0 ~ 1 로출력되도록정규화 함. 출력의총합은항상1이되도록형성
  - 전체경우의수중에서특정사건이발생할확률로표현된다.

sigmoid 함수는 0, 1으로 결국 분리할 수 있음. X가 무한히 작거나, 크면 sigmoid에서는 0, 1으로 표현되게 된다.

선형 회귀

[https://github.com/gilbutITbook/080228](https://github.com/gilbutITbook/080228) (git clone + Link) 후 &#39;code .&#39; or &#39;jupyter notebook&#39; 실행

# -\*- coding: utf-8 -\*-

import numpy as np

# x 값과y값

x=[2, 4, 6, 8]

y=[81, 93, 91, 97]

# x와y의평균값

mx = np.mean(x)

my = np.mean(y)

print(&quot;x의평균값:&quot;, mx)

print(&quot;y의평균값:&quot;, my)

# 기울기공식의분모

divisor = sum([(mx - i)\*\*2for i in x])

# 기울기공식의분자

deftop(x, mx, y, my):

d = 0

for i inrange(len(x)):

d += (x[i] - mx) \* (y[i] - my)

return d

dividend = top(x, mx, y, my)

print(&quot;분모:&quot;, divisor)

print(&quot;분자:&quot;, dividend)

# 기울기와y 절편구하기

a = dividend / divisor

b = my - (mx\*a)

# 출력으로확인

print(&quot;기울기a =&quot;, a)

print(&quot;y 절편b =&quot;, b)

output

x의평균값: 5.0

y의평균값: 90.5

분모: 20.0

분자: 46.0

기울기a = 2.3

y 절편b = 79.0

여기서 평균제곱오차(MSE)는 로 y-hat = y-pred라고. 생각하면 됨.

경사 하강법 (gradient descent)

- Loss 의 최솟값을 찾기 위해서 임의의 a(가중치)를 대입하고 loss가 가장 작은 지점을 구해나가는 방식을 의미함.
- 애초에 loss함수는 이므로 2차함수의 형태임. 여기서 임의의 a를 넣고 미분을 한다면, 2차함수의 기울기를 알 수 있음. 기울기가 0이 될 때, 오차가 최솟값이 되기 때문에 일일이 대입해보고 기울기를 0으로 만드는 지점을 찾는 것이 경사 하강법임.
- 그 ![](RackMultipart20211006-4-if6336_html_154bf09f4a8b45f6.png)
 래프를 보면 알 수 있음.
- 오차가 최소가 되는 Final Value를 찾아가는 과정에서 학습률 개념이 도입된다.
  - 학습률은 얼마나 이동할 것인지 이동거리를 결정하는 법
  - 여기서 a 값을 조정하는 법이 optimizer라고 한다. (adam 등의 여러 방식이 있음)
- 물론 b를 구하는 방식에도 경사하강법이 사용된다.

y-hat = ax + b이므로 에 대입하고 편미분을 진행하면 된다. 그리고 경사하강법을 도입하여 최적의 a, b를 찾게 된다. (기울기를 낮추기 위해 a간격을 조정하는 법이 optimizer, 이동거리가 학습률, learning rate)

y-hat = a \* x + b, Error = y – y-hat

a\_diff : MSE를 a로 편미분한 값

a = a – lr(학습률) \* a\_diff(편미분 계수)

(실습)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

data = [[2,81], [4,93], [6,91], [8,97]]

X = [i[0] for i in data]

y = [i[1] for i in data]

plt.figure(figsize=(8,5))

plt.scatter(X,y)

plt.show()

![](RackMultipart20211006-4-if6336_html_c1681fb9898e7295.png)(산점도 출력 결과)

#리스트-\&gt; np array로변환(인덱스부여로하나씩계산이가능)

X\_data = np.array(X)

y\_data = np.array(y)

# 기울기a와절편b의값을초기화합니다.

a = 0

b = 0

#학습률을정합니다.

lr = 0.03

#몇번반복될지를설정합니다.

epochs = 2001

#경사하강법for 문

for i inrange(epochs): # epoch 수만큼반복

y\_hat = a \* X\_data + b # y추정치를구하는식

error = y\_data - y\_hat # 실제y와추정한값의차이

a\_diff = -(2/len(X\_data)) \* sum(X\_data \* (error)) # a로미분한error

b\_diff = -(2/len(X\_data)) \* sum(error) # b로미분한error

a = a - lr \* a\_diff # 학습률을곱해기존의a값을업데이트합니다.

b = b - lr \* b\_diff # 학습률을곱해기존의b값을업데이트합니다.

if i % 100 == 0: # 100번반복될때마다현재의a값, b값을출력합니다.

print(&quot;epoch=%5.f, 기울기(a, 가중치)=%.04f, 절편(b, bias)=%.04f, \

MSE=%.04f&quot; % (i, a, b, sum(error)/4))

Output

epoch= 0, 기울기(a, 가중치)=27.8400, 절편(b, bias)=5.4300, MSE=90.5000

epoch= 100, 기울기(a, 가중치)=7.0739, 절편(b, bias)=50.5117, MSE=4.6644

epoch= 200, 기울기(a, 가중치)=4.0960, 절편(b, bias)=68.2822, MSE=1.7548

epoch= 300, 기울기(a, 가중치)=2.9757, 절편(b, bias)=74.9678, MSE=0.6602

epoch= 400, 기울기(a, 가중치)=2.5542, 절편(b, bias)=77.4830, MSE=0.2484

epoch= 500, 기울기(a, 가중치)=2.3956, 절편(b, bias)=78.4293, MSE=0.0934

epoch= 600, 기울기(a, 가중치)=2.3360, 절편(b, bias)=78.7853, MSE=0.0352

epoch= 700, 기울기(a, 가중치)=2.3135, 절편(b, bias)=78.9192, MSE=0.0132

epoch= 800, 기울기(a, 가중치)=2.3051, 절편(b, bias)=78.9696, MSE=0.0050

epoch= 900, 기울기(a, 가중치)=2.3019, 절편(b, bias)=78.9886, MSE=0.0019

epoch= 1000, 기울기(a, 가중치)=2.3007, 절편(b, bias)=78.9957, MSE=0.0007

epoch= 1100, 기울기(a, 가중치)=2.3003, 절편(b, bias)=78.9984, MSE=0.0003

epoch= 1200, 기울기(a, 가중치)=2.3001, 절편(b, bias)=78.9994, MSE=0.0001

epoch= 1300, 기울기(a, 가중치)=2.3000, 절편(b, bias)=78.9998, MSE=0.0000

epoch= 1400, 기울기(a, 가중치)=2.3000, 절편(b, bias)=78.9999, MSE=0.0000 (나머지는 생략)

그래프 그려보기

y\_pred = a \* X\_data + b

plt.scatter(X, y)

plt.plot([min(X\_data), max(X\_data)], [min(y\_pred), max(y\_pred)])

plt.show()

![](RackMultipart20211006-4-if6336_html_f810b8506e27f217.png)
