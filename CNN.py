import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import DensityController as DC 

Mode=1 # 0:Test 1:Train 2:Train and Test
TrainDate='_201111'

def TrainGraph():
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],'b-',label='loss')
    plt.plot(history.history['val_loss'],'r--',label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'],'g-',label='accuracy')
    plt.plot(history.history['val_accuracy'],'k--',label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0.7,1)
    plt.legend()
    plt.show()
def CheckResalt(model, testX, testY, iterator):
    for i in range(iterator):
       te1 = np.argmax(model.predict(np.array(testX[i]).reshape(1,3,DC.hyper,1)))
       te2 = testY[i]
       if te1==te2:
           print(te1, te2, 'Right Answer', i+'/'+str(iterator))
           break


        
if Mode==0:
    print('Loading CNN Model')
    model= DC.tf.keras.models.load_model(DC.FilePath+'IMU_Gesture_Recognition_TrainModel'+'/GestureRecognitionModel'+str(DC.hyper)+TrainDate+'.h5')
    model.summary()

    print('Test Start')
    DC.DataIO.GetTest(model) #TestMode

elif Mode==1:
    (trainX,trainY),(testX,testY)=DC.DataIO.DataRead('CombinedData'+str(DC.hyper)+DC.Date,3,DC.hyper,True)

    trainX=np.array(trainX)
    trainY=np.array(trainY)
    testX=np.array(testX)
    testY=np.array(testY)

    model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=(3,DC.hyper,1),kernel_size=(3,3),filters=64,padding='same',activation='relu'),
                                 tf.keras.layers.Conv2D(kernel_size=(1,3),filters=128,padding='same',activation ='relu'),
                                 tf.keras.layers.MaxPooling2D(strides=(1,2)),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Conv2D(kernel_size=(1,3),padding='valid',filters=256,activation='relu'),
                                 tf.keras.layers.Conv2D(kernel_size=(1,3),padding='valid',filters=512,activation='relu'),
                                 tf.keras.layers.MaxPooling2D(strides=(1,2)),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(units=1024, activation='relu'),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(units=512, activation='relu'),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(units=11, activation = 'softmax')
                                 ])

    model.compile(optimizer= tf.keras.optimizers.Adam(),
                  loss= 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit((trainX),(trainY),epochs=50,validation_split=0.01)

    model.save(DC.FilePath+'IMU_Gesture_Recognition_TrainModel'+'/GestureRecognitionModel'+str(DC.hyper)+TrainDate+'.h5')

    resalt=model.evaluate((testX),(testY),verbose=1)
    print('loss :', resalt[0], 'correntness:',resalt[1]*100,"%")
    TrainGraph() #TrainMode

    if Mode==2:
        print('Test Start')
        DC.DataIO.GetTest(model) #TestMode