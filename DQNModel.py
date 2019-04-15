from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D

img_rows = 84
img_cols = 84
#Convert image into Black and white
'''
The original screen size is 512 × 288 pixels in three
channels, but we convert the image captured from the
screen to grayscale, crop it to 340 × 288 pixels, and
downsample it by a factor of 0.3, resulting in a 102 × 86
pixel image. It is then rescaled to 84 × 84 pixels and
normalized from [0, 255] to [0, 1]

'''
img_channels = 4 #We stack 4 frames

class DQNModel:
    
    
    def __init__(self,learning_rate):
        
        '''
            input_shape is a parameter for Keras CNN which defines the shape of the images shown to the DQN.
            input_shape = (img_rows,img_cols,img_channels)
            
            This network takes as input a 84 × 84 × historyLength image and has a single output for every 
            possible action. 
        '''
        print("Now we build the model")
        self.model = Sequential()
        #The first layer is a convolution layer with 32 filters of size 8 × 8 with stride 4,followed by a rectified nonlinearity.
        self.model.add(Conv2D(32,(8, 8), strides=4, padding='same',input_shape=(img_rows,img_cols,img_channels)))  #84*84*4
        self.model.add(Activation('relu'))
        #The second layer is also a convolution layer of 64 filters of size 4 × 4 with stride 2, followed by another rectified linear unit.
        self.model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        self.model.add(Activation('relu'))
        #The third convolution layer has 64 filters of size 3 × 3 with stride 1 followed by a rectified linear unit.
        self.model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        #Following that is a fully connected layer with 512 outputs
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        #the output layer (also fully connected) with a single output for each action.
        self.model.add(Dense(2))
        #use adam optimizer
        adam = Adam(lr=learning_rate)
        self.model.compile(loss='mse',optimizer=adam)
        self.model.summary()
        print("We finish building the model")

#         