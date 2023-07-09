import yaml
import os
import matplotlib.pyplot as plt
from numpy import unique,zeros,argmax

def LOAD_YAML()->dict:
    '''
        Function to load yaml file into a dictionary 
        Output dictionary with yaml variables

    '''

    config_path = os.path.join(os.getcwd(), "config.yml")

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
        return config

def plot_metric(history, metric):
    '''
        Function to plot Results of Cnn training
    '''
    plt.figure(figsize=(10, 8))
    train_metrics = history.history[metric]
    val_metrics = history.history[f'val_{metric}']
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs,val_metrics)
    plt.title('Training '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric,"validation_"+metric])
    plt.show( block = False)
    plt.savefig(f'trainandvalidation_{metric}.png')

def ConfusionMatrixMine(test_y:None,predicted_test_y: None)-> None:
    '''
    Function to compute the confusion matrix based on label vector test_y and predicted labels array predicted_test_y
    Input: test_y: numpy array with labels of test set initial
           predicted_test_y: numpy array with predicted labels
    Output: Numpy array containing the confusion matrix
    
    '''
    labels = unique(test_y)#get labels
    conf_matrix = zeros((len(labels),len(labels)))
    for i in range(len(test_y)):#Confusion matrix of sklearn has true labels as rows and predicted labels as collumns
        conf_matrix[test_y[i],predicted_test_y[i]] = conf_matrix[test_y[i],predicted_test_y[i]] + 1

    return conf_matrix.astype(int)

def plot_images(train_X:None,train_y:None)->None:
    '''
        Function that plots in a single figure one image from each class
        Input: 
            train_X: Numpy array containing all training images 
            train_y: Numpy array containing all labels of training images
    '''

    #plot one image per class
    class_ = 0
    rows  =  2
    columns = 5
  

    fig = plt.figure()
    
    for i in range(train_X.shape[0]):
        if train_y[i] == class_:
            fig.add_subplot(rows, columns, class_+1)
            class_  = class_+1
        
            plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
            plt.title(f"Image for class {class_}")  
        elif class_ == 10:
            break


def encode_labels(vector:None)->None:
    '''
    Input: vector numpy array of one hot encoded values to be encoded
    Output: encoded values in a numpy array
    
    '''
    encoded =  zeros(vector.shape[0])
    for i in range(vector.shape[0]):
        encoded[i] = argmax(vector[i,:], axis=-1)

    return encoded.astype(int)

def plot_metric_total_set(history, metric):
    '''
        Function to plot Results of Cnn training
    '''
    plt.figure(figsize=(10, 8))
    train_metrics = history
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.title('Test '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["Test_"+metric])
    plt.show( block = False)
    plt.savefig(f'test_{metric}.png')