#### Coursera의 강의 "DeepLearining AI Tensorflow Developer" 를 보고 작성한 글입니다.

:exclamation:   [GitHub - https-deeplearning-ai/tensorflow-1-public](https://github.com/https-deeplearning-ai/tensorflow-1-public)  에 있는 파일을 colab에서 연 뒤 사본 저장 후 실습하며 정리한 글이므로 딱히 도움이 되지 않을 수 있습니다.   :exclamation:

______

- C1_W1_Lab_1_hello_world_nn.ipynb
  
  - 처음 텐서플로우를 사용하려고 한다면 어느 프로그래밍 언어에서도 그렇듯 코드 맨 처음에 이걸 쓴다고 알려줘야 한다. 그리고 배열같은 여러 데이터도 같이 처리해야 하므로 numpy도 추가해야 하고 여러 개의 layer를 겹쳐 nn을 만들 것이기 때문에 tensorflow 안의 keras도 추가해준다. 여기서 as 라고 뒤에 따로 써준 것은 뒤에 코드를 짤 때 더 편하기 위해 각각 라이브러리의 닉네임을 정해준 거라고 생각하면 된다.
    
    ```python
    import tensorflow as tf
    import numpy as np
    from tensorflow import kears
    
    
    print(tf.__version__)
    ```
  
  -  이제 input 값도 1개고 layer도 1개인 아주아주 간단한 nn을 하나 만들어보자. 이건 keras의 Sequential 클래스를 이용하면 아주 쉽다. 이 클래스는 이름 그대로 연속된 layer로 네트워크를 만들 때 사용한다. 
    
    ```python
    model = tf.keras.Sequential([kears.layers.Dense(units=1, input_shape=[1])])
    ```
    
    units은 output 공간의 차원을 말하며 Dense 작업의 output = activation(dot(input, kernel) + bias) 가 된다. 이 때 kernel와 bias는 layer가 만들어내는 가중치 행렬과 편향 벡터 값이다.
  
  - 이제 loss와 optimizer function을 정해줘야 한다. 이게 무슨 일을 하는 것이냐면 예를 들어보자. y=2x-1 이라는 x와 y의 관계를 정해놓고 이걸 컴퓨터로 하여금 맞추게 한다고 가정해보자. 컴퓨터가 맨 처음 y=10x+10 이라고 하면 loss function은 컴퓨터에게 실제 정답과의 오차를 알려주고 optimizer function은 이 loss 값을 바탕으로 컴퓨터에게 다른 가정을 만들어준다. 이 과정을 반복하며 loss 값을 최소로 줄여 정답을 맞추는 것이 nn의 목적인 것이다. 고맙게도 tensorflow엔 이런 function이 이미다 있으니 우리는 그때 그때 좋을 것 같은 것들을 골라 쓰기만 하면 된다!
    
    ```python
    #Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')
    ```
    
    여기서 sgd는 stochastic gradient descent를 말한다. 이게 뭔지 모르겠으면 [GitHub - seoyounji/DeepLearning-Basic: 딥러닝 개발을 위한 기본 이론](https://github.com/seoyounji/DeepLearning-Basic) 여길 보면 된다구!
  
  - 이제 nn을 훈련시키기 위한 데이터를 넣어볼 차례다. 가볍게 x랑 y 6개씩만 넣어보자. 데이터를 넣을 때도 배열을 지켜서 넣어줘야 하는데 이럴 때 사용하려고 선언한 numpy! 
    
    ```python
    trainingX = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    trainingY = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    ```
  
  - 준비가 다 되었으니 이제 nn을 학습시킬 차례다. 




