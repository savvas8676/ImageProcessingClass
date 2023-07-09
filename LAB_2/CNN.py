#--------------- Basic Libraries and Utils
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from utils import LOAD_YAML,plot_metric,plot_images,encode_labels,ConfusionMatrixMine,plot_metric_total_set
#------------- Tensor Flow Keras ------------------------
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers as opt
from tensorflow.keras.layers  import Conv2D, AveragePooling2D, Dense, Flatten
from tensorflow.keras.layers import (
    BatchNormalization)
import tensorflow as tf
#--------------------SK learn-----------------------
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



#--------------- Load yml config file --------------#
config = LOAD_YAML()




#End import section------------------------------
accuracy_list = []
loss_list = []
#------------ CNN class ----------------------------

class CNN():
    kernel_size = config['fit']['kernel_size']
    padding = 0
    shape = (28, 28, 1)
    stride = config['fit']['stride']
    num_filters_1 = config['fit']['layer_1_filters']
    num_filters_2 = config['fit']['layer_2_filters']
    learning_rate = config['fit']['l_rate']
    epochs = config['fit']['epochs']
    batch_size = config['fit']['batch']
    model = None

    #Layers activation functions
    conv1 = config['activation']['Conv_1']
    conv2 = config['activation']['Conv_2']
    dense1 = config['activation']['Dense_1']
    dense2 = config['activation']['Dense_2']
    final = config['activation']['Final']
    def __init__(self) -> None:
        #Creating model sequential in the constructor
        
        self.model = Sequential()
        self.model.add(Conv2D(self.num_filters_1, (self.kernel_size, self.kernel_size), activation=self.conv1, input_shape=self.shape ,strides = (self.stride,self.stride)))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D((2,2),strides=(2,2)))
        self.model.add(Conv2D(self.num_filters_2, (self.kernel_size, self.kernel_size), activation=self.conv2,strides = (self.stride,self.stride)))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D((2,2),strides=(2,2)))
        # Flatten to feed to the NN
        self.model.add(Flatten())
        #Regular NN for classification
        self.model.add(Dense(config['fit']['dense_1'], activation=self.dense1))
        self.model.add(Dense(config['fit']['dense_2'], activation=self.dense2))
        #Last Layer
        self.model.add(Dense(config['fit']['final_layer'], activation=self.final))

        


    def Compile_and_fit(self,X:np.array,y:np.array,callback)-> None:
        '''
        X: Images to be fitted to train the neural net 
        y: Labels of said images
        callback: callback to stop training if 99.5 accuracy is achieved

        Output:
        Model fitted and trained
        
        
        '''
        #Compiling and fitting the CNN model we just created
        
                
        #Compile model
        sgd = opt.SGD(learning_rate=self.learning_rate)#momentum
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    #categorical_crossentropy

        history = self.model.fit(X, y, epochs=self.epochs, batch_size = self.batch_size,callbacks = [callback], workers=4,shuffle = True,validation_split=0.1)

        return self.model,history





#There are 10 classes representing numbers 0-9

#------- Class that inherits Callback API class of keras in order to stop training when acceptable limit on accuracy is reached
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>config['fit']['limit']):
      print(f"\nReached {100*config['fit']['limit']}% accuracy aborting further training.")
      self.model.stop_training = True


    loss, accuracy = self.model.evaluate(test_X, test_y_one_hot,verbose = config['report']['model_evaluate_verbose'],use_multiprocessing=True,workers = 4)
    accuracy_list.append(accuracy*100)
    loss_list.append(loss)
    #print('Accuracy: %.2f' % (accuracy*100))
     


#Creating function for encoding again one hot encoded labels



#---------------- Main Script----------------------------

#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()


#Visualizing one image from each class for report
visualize = config['report']['visualize']
if visualize==True:
    plot_images(train_X=train_X,train_y=train_y)


# Normalizing the images because models tend to train faster as per standard practice
train_X=train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1) # shape 60.000x28x28x1
train_X=train_X /255.0
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
test_X=test_X/255.0


# One hot encoding the output values
train_y_one_hot = tf.one_hot(train_y.astype(np.int32), depth=10)
test_y_one_hot = tf.one_hot(test_y.astype(np.int32), depth=10)







#Affirming that we have 60000 training set images and each one is greyscale image of size 28x28
print(train_X.shape)

#Instatiating CNN model
NN = CNN()

#Instatiating a callback function as specified in its class 
callbacks = myCallback()


model,history = NN.Compile_and_fit(train_X,train_y_one_hot,callbacks)

# evaluate the keras model
_, accuracy = model.evaluate(train_X, train_y_one_hot,verbose = config['report']['model_evaluate_verbose'],use_multiprocessing=True,workers = 4)
print('Accuracy after last evaluate: %.2f' % (accuracy*100))


predictions = model.predict(test_X)
#Endoding predictions in order to compute confusion matrix
encoded_predictions = encode_labels(predictions)




#Classification results Confusion matrix
conf_matrix = ConfusionMatrixMine(test_y,encoded_predictions)
print(conf_matrix)

#Visualize Confusion Matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.title("Heat map of Confusion Matrix.")
plt.xlabel("Predicted class")
plt.ylabel("True class")

#--------------------- Plotting Results ---------------------
# Plotting loss and accuracy over all epochs of training set
if not config['report']['test_set']:
    plot_metric(history,'accuracy')
    plot_metric(history,'loss')
elif config['report']['test_set']:
    # Plotting loss and acccuracy over all epochs of test set 
    plot_metric_total_set(accuracy_list,'accuracy')
    plot_metric_total_set(loss_list,'loss')


pass