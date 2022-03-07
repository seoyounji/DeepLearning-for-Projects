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
  
  - 이제 nn을 훈련시키기 위한 데이터를 넣어볼 차례다. 가볍게 x랑 y 6개씩만 넣어보자. 데이터를 넣을 때도 배열을 지켜서 넣어줘야 하는데 이럴 때 사용하려고 선언한 게 바로  numpy! 
    
    ```python
    trainingX = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    trainingY = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    ```
  
  - 준비가 다 되었으니 이제 nn을 학습시킬 차례다. x와 y 사이의 관계를 학습시키기 위해선 model.fit() 이라는 함수를 불러와야한다. 이 함수는 설정한 epoch 동안 위에서 설정한 optimizer를 이용해 위에서 설정한 loss를 최소한으로 줄여주는 x와 y의 관계를 도출하는 역할을 한다. 위 코드를 돌리면 각 epoch 마다의 loss가 출력되고 이 loss가 점점 줄어드는 것을 확인할 수 있다. loss가 줄어든다는 것은 보통 학습이 잘 되었다는 뜻이지만 over fitting이 된 건 아닌지 확인해야 한다! 이 부분은 [GitHub - seoyounji/DeepLearning-Basic: 딥러닝 개발을 위한 기본 이론](https://github.com/seoyounji/DeepLearning-Basic) 여기에 더 자세히 설명이 돼있다구!
    
    ```python
    model.fit(trainingX, trainingY, epochs=500)
    ```
    
  - 위 코드를 돌리면 각 epoch 마다의 loss가 출력되고 이 loss가 점점 줄어드는 것을 확인할 수 있다. loss가 줄어든다는 것은 보통 학습이 잘 되었다는 뜻이지만 over fitting이 된 건 아닌지 확인해야 한다! 이 부분은 [GitHub - seoyounji/DeepLearning-Basic: 딥러닝 개발을 위한 기본 이론](https://github.com/seoyounji/DeepLearning-Basic) 여기에 더 자세히 설명이 돼있다구!
  
  * 암튼 이렇게 x와 y의 관계를 학습시킨 모델을 하나 얻게 되었다. 최종적으로 이 모델의 성능을 체크하기 위해 x,y 값 한 쌍을 넣어보자
  
    ```python
    print(model.predict([10,0]))
    ```
  
    원래대로라면 19가 정학하게 나와야하지만 실제로 나온 값은 18.976002 이다. 왜 아주 약간 작은 값이 나온걸까? NN은 가능성을 처리하는 네트워크다. 네트워크는 학습으로 y=2x-1 이라는 식을 충분히 도출해냈겠지만 겨우 6개의 학습 데이터만을 가지고 이 관계식이 확실한 것인지는 확신할 수는 없었을 것이다. 그렇기 때문에 10을 넣었을 때 정확히 19가 아닌, 19에 한없이 가까운 값을 내놓은 것이다. 



* C1_W2_Lab_1_beyond_hello_world.ipynb

  * 사실 위에서 한 y=2x-1 같은 예제를 보면 굳이 컴퓨터를 학습시킬 필요 없이 그냥 사람이 공식을 알려주는 게 훨씬 편하다. 하지만 머신러닝은 이런 문제를 해결하기 위해 발전된 게 아니다. 사람이 공식으로 규명할 수 없는 상황에 대해 가정해보자. 컴퓨터가 서로 다른 10가지의 옷 종류를 인식하도록 학습시키려고 한다면? 공식으로는 불가능하다. 이게 바로 컴퓨터 비전 분야에서 머신러닝이 특히 발달한 이유다. 자 그럼 이제 필요한 라이브러리와 데이터셋을 가져와보자. 

    ```python
    import tensorflow as tf
    
    print(tf.__version__)
    
    fminst = tf.keras.datasets.fashion_mnist
    ```

    여기서 사용할 데이터셋은 Fashion MNIST dataset이라는 건데 28x28 픽셀로 이루어진 흑백 옷 이미지이고 각각의 이미지는 0부터 9까지의 라벨이 붙어있는데 각 라벨은 서로 다른 옷 종류를 뜻한다. 이 데이터셋은 tf.keras.datasets API에서 바로 다운받아 사용할 수 있는데 위와 같이 쓰면 된다. 

  * 이제 데이터를 가져와보자. 이미 데이터를 가져온거 아니냐고? 그건 그렇지만 우리가 학습에 사용할 형태로 바꿔줘야 한다. load_data() 를 이용하면 각각 2개의 list로 이루어진 2개의 tuple이 return 되는데 각각 training 용과 test 용이다.

    ```python
    (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
    ```

  * 데이터셋이 실제로 어떻게 생겼는지 이미지와 numpy 배열로 출력해보자. 

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    
    index = 0
    np.set_printoptions(linewidth=320)
    
    print(f'LABEL: {training_labels[index]}')
    print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')
    
    # Visualize the image
    plt.imshow(training_images[index])
    ```

    training dataset이 6만개이므로 index엔 0부터 59999 까지의 수를 넣어주면 되고 numpy 배열로 출력할 때 한 라인의 너비를 320으로 설정해준 것이다. 이 때 라인 너비 값은 값이 짤리지 않기 위해 널널하게 320으로 설정한 것일뿐 더 작게 해도 상관없다. 그리고 파이썬 버전 3.6부터 사용할 수 있는 f-string 포매팅 방식을 사용해 numpy 배열을 출력할 수 있고 이미지는 matplotlib 를 이용해 출력해줄 수 있다.

  * 출력되는 numpy 배열의 값을 보면 모두 0에서 255 사이인 것을 확인할 수 있는데 normalize해서 0에서 1 사이 값으로 만드려면 numpy 특성상 for loop 대신 나누기만 해주면 된다.

    ```python
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    ```

    그런데 왜 데이터셋을 normalize 시켜줘야 하는 걸까? 이것 또한 [GitHub - seoyounji/DeepLearning-Basic: 딥러닝 개발을 위한 기본 이론](https://github.com/seoyounji/DeepLearning-Basic) 여기에 더 자세히 설명이 되어 있으니 한 번 봐달라구!

  * 이제 모델을 만들어보자. 

    ```python
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    ```

    Sequential 함수는 위에서도 말했다시피 NN을 연속된 layer로 정의하는 함수이다. Flatten은 2차원 이미지를 1차원 배열로 쭉 펴주는 함수인데 여기서는 기초부터 배우는 것이기 때문에 썼지만 요즘 이미지 처리에서는 이 함수를 쓰지 않는다. 그 이유는 바로 2차원 이미지를 1차원으로 펴버리면 이미지 고유의 공간 정보가 사라지기 때문에! 그래서 요즘 이미지 처리 딥러닝에서는 Convolution layer를 사용한 CNN을 주로 쓴다. 아무튼! Dense는 layer를 추가해주는 단순한 함수인데 이때 우리는 각 layer에 있는 값들에 적용할 activation function을 정해주어야 한다. layer 값이 activation function을 거쳐 
