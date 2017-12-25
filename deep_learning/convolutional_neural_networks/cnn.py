from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
# number of feature detectors
# it creates 32 feature detectors with size of 3row x 3col
# input_shape , size and format of our image
# input_shape 3 is for number of channels RGB
# input_shape dimensions of array (256,256)
# input_shape last input is activation function (relu for get nonlinearity)
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation = 'relu'))

# Step 2 - Pooling
# stride size is the size of square that we are goig to user for pooling
# max pooling -> get the maximum when featured detectors match
# this pool_size will reduce out feature map by half withou reducing its performance
classifier.add(MaxPooling2D(pool_size = (2,2)))


# Adding a second convolution for improve accuracy
classifier.add(Convolution2D(32,3,3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))



# Step 3 - Flattening
# this will flaten all the feature maps in the poolinglayer into huge single vector
classifier.add(Flatten())

# Step 4 - Full connection
# good number for input is somthing between number of input nodes and number of output nodes
# output_dim -> not too small to make the classifier a good model and not too big to not make it to highly compute
# its a common practice to be a number that is power of 2
classifier.add(Dense(output_dim = 128 ,activation = 'relu'))

# Here we are using sigmoid function for outcome because the outcome is binary (cat or dog)
# but if we have more that two categories we should use softmax activation function
classifier.add(Dense(output_dim = 1 ,activation = 'sigmoid'))

# Compiling the CNN
# loss -> we are using because we have binary outcome(cat,dog) otherwise we use categorical
classifier.compile(optimizer='adam' , loss = 'binary_crossentropy' , metrics=['accuracy'])

# image augmentation -> reduce overfitting , to enrich our data set without adding training data
# code from keras documentation
from keras.preprocessing.image import ImageDataGenerator

# apply zoom , flip , .. to our images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), #because of line 17
        batch_size=32,
        class_mode='binary')



test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64), # because of line 17
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

# to improve accuracy we have two option
# 1 - Add another convolutional layer
# 2 - Add another fully connected layer
