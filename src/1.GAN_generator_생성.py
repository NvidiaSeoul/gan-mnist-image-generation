from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential ,Model
import numpy as np
import matplotlib.pyplot as plt

# 생성자 모델 생성
# LeakyRelu() 와  tanh 활성화 함수 사용 이유는 실험적으로 좋은 결과를
# 만들어 주어서 선호
generator = Sequential()                      # input_dim=100 : 100차원 랜덤벡터 입력 (변경가능 )
generator.add(Dense(7*7*128, input_dim=100)) # 128 : 임의로 정한 노드 수 (충분한 노드로 변경가능 )
generator.add(BatchNormalization())          # 7 * 7 : 초기이미지크기(7x7x채널)==>28x28이미지 확장고려
generator.add(Activation(LeakyReLU(0.2)))
generator.add(Reshape((7,7,128)))
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=5, padding='same')) # 활성화함수 없을경우 필터연산만 수행
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))
#generator.summary()

#판별자 모델 생성
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28,28,1), padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
#discriminator.summary()

# 생성자와 판별자를 연결하여 최종 GAN 모델 생성
discriminator.trainable = False
ginput = Input(shape=(100,)) # input_dim=100 : 100차원 랜덤벡터 입력
dis_output = discriminator(generator(ginput)) # 생성자가 출력한 g(z)를 Fake데이터를 판별자에 입력
GanModel = Model(ginput, dis_output)  # Model()함수에 입력과 출력(판별자의 출력) 정의
GanModel.compile(loss='binary_crossentropy', optimizer='adam')
GanModel.summary()

# GAN 신경망 실행함수
def gen_train(epoch, batch_size, save_interval):

    # MNIST 데이터 셋 로드
    (X_train, _),(_,_) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
    # -1 ~ 1 값으로 정규화
    X_train = (X_train - 127.5)/127.5
    true_date = np.ones((batch_size,1))  # 실제 타깃 : 1
    fake_date = np.zeros((batch_size,1)) # 거짓 타깃 : 0

    discriminator_losslist = []
    gan_losslist = []
    for i in range(epoch):
        # 실제 데이터 판별자에 입력
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx] # batch_size 크기만큼 랜덤 실제 데이터 추출
        d_loss_real = discriminator.train_on_batch(imgs, true_date) # 실제데이터 배치사이즈만큼 한번 훈련

        #noise데이터로 생성자에서 만들어낸 가짜 이미지 데이터를 판별자에 입력
        noise = np.random.normal(0,1, (batch_size,100))
        gen_imgs = generator.predict(noise) # 배치사이즈만큼 가짜 이미지 생성 (32, 28,28,1)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_date) # 가짜데이터 배치사이즈만큼 한번 훈련

        #판별자와 Gan모델 오차계산
        d_loss = 0.5*np.add(d_loss_real, d_loss_fake)
        gan_loss = GanModel.train_on_batch(noise, true_date) # 판별자는 동결상태임으로 생성자만 갱신되면서 훈련

        print('epoch : {}, d_loss: {:.4f} , gan_loss: {:.4f}'.format(i, d_loss, gan_loss))
        # 판별자와 생성자는 적대적 관계로서
        # 판별자의 loss가 감소해 성능이 좋다면 생성자는 loss증가 성능 감소
        # 생성자의 loss가 감소해 성능이 좋다면 판별자는 loss증가 성능 감소 , 즉 판별자를 잘 속이는
        # 이미지를 생성자가 생성했다는 의미
        discriminator_losslist.append(d_loss)
        gan_losslist.append(gan_loss)

        # 200 epoch 구간마다 생성자가 만들낸 이미지 출력 및 저장
        if i % save_interval == 0:
            noise = np.random.normal(0,1,(25,100))
            gen_imgs = generator.predict(noise)
            print(gen_imgs.shape)
            # 스케일 조정 : 0 ~ 1 로
            gen_imgs = 0.5 * gen_imgs + 0.5
            gen_imgs = gen_imgs.reshape(25, 28,28)
            fig, axes = plt.subplots(5, 5)
            count = 0
            for low in range(5):
                for col in range(5):
                    axes[low, col].imshow(gen_imgs[count], cmap='gray')
                    axes[low, col].axis('off')
                    count += 1
                    fig.savefig('gan_images/gan_mnist_%d.png' %i) # <== plt.show() 효과
                    plt.close(fig) # fig 메모리 해제

    plt.figure(figsize=(10,5))
    plt.plot(discriminator_losslist, label='Disc')
    plt.plot(gan_losslist, label='gan')
    plt.legend()
    plt.show()



gen_train(4001, 32, 200)


