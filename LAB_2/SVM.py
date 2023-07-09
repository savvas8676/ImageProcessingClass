#------------------ General Imports ----------------
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from HOG import HOG
from utils import LOAD_YAML,ConfusionMatrixMine

#------------------ Dataset Import ------------------------
from keras.datasets import mnist

#--------------------- Metrics and HOG --------------------
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
#----------------- SVM Imports -------------------------------
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#End import section------------------------------
#There are 10 classes representing numbers 0-9

#--------------- Load yml config file --------------#
config = LOAD_YAML()


#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors 
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

#To affirm that they are numpy arrays
print(type(train_X))


#Extracting Hog features using python scikit-image function for train first and test second
patch  = config['HOG']['patch'] #patch size
block_sz = config['HOG']['block_size'] #block size
bins = config['HOG']['orientations'] #number of bins in each cells histogram

#Initializing variables
X_test = np.array([])
X_train = np.array([])

if not config['HOG']['mine']: #SKIMAGE HOG IMPLEMENTATION
    #Step 1: extract HOG features for train data
    hog_features_train = []
    for image in range(train_X.shape[0]):
        fd,hog_image = hog(train_X[image,:,:], orientations=bins, pixels_per_cell=(patch,patch),cells_per_block=(block_sz, block_sz),block_norm= 'L2',visualize=True)
        
        hog_features_train.append(fd) #Each feture vector will be size 

    #convert list to numpy array
    X_train = np.array(hog_features_train)

    #Step 2: extract HOG features for test data
    hog_features_test = []
    for image in range(test_X.shape[0]):
        fd,hog_image = hog(test_X[image,:,:], orientations=bins, pixels_per_cell=(patch,patch),cells_per_block=(block_sz, block_sz),block_norm= 'L2',visualize=True)
        
        hog_features_test.append(fd) 

    #convert list to numpy array
    X_test = np.array(hog_features_test)

else: # MY implemetation of HOG
    model = HOG()
    #Step 1: extract HOG features for train data
    hog_features_train = []
    for image in range(train_X.shape[0]):
        model.image = train_X[image,:,:]
        fd = model.process()
        
        hog_features_train.append(fd) #Each feture vector will be size 

    #convert list to numpy array
    X_train = np.array(hog_features_train)

    #Step 2: extract HOG features for test data
    hog_features_test = []
    for image in range(test_X.shape[0]):
        model.image = test_X[image,:,:]
        fd = model.process()
        
        hog_features_test.append(fd) 

    #convert list to numpy array
    X_test = np.array(hog_features_test)

##APPLY SIMPLE CLASSIFIER without oprimization using gridsearch
C = config['classifier']['SVC']['C']
gamma = config['classifier']['SVC']['gamma']
kernel  = config['classifier']['SVC']['kernel']

svm = SVC(kernel=kernel , C = C,gamma=gamma)
svm.fit(X_train,train_y )
predictions = svm.predict(X_test)


#print('Confusion Matrix:')
#confusion_m = confusion_matrix(test_y, predictions)
#print(confusion_m)

print("---------------------------------------------------------------------- MINE")
conf_matrix = ConfusionMatrixMine(test_y,predictions)
print(conf_matrix)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.title("Heat map of Confusion Matrix.")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.savefig("Histogram_SVM")
print("---------------------------------------------------------------------- MINE")

print('Classification report:')
print(classification_report(test_y, predictions))






## APPLY OPTIMIZED CLASSIFIER using gridsearch cv
#Finished the Hog feature extraction going forward with classification using SVM and optimizing it using grid search CV

if config['fine_tune']['allow']:
    #Setting Parameter values for grid search

    params_grid = config['fine_tune']['param_grid']#{'C': [0.1, 1,10], 'gamma': [1], 'kernel': ['rbf']}
    verbose_level = config['fine_tune']['verbose']
    cv =  config['fine_tune']['cv'] #number of folds for cross-validation of grid search
    # instantiating the GridSearchCV object



    grid = GridSearchCV(estimator=SVC(), param_grid=params_grid, verbose=verbose_level,refit=True,cv = cv)#cv is for number of folds
    #Executing grid search to find the best parameter values a multitute of kernels and hyperparameter values are used as seen in the.yml file
    grid.fit(X_train,train_y )
    #C is penalty parameter, we try to create a grid with values that differ at least one order of metric
    #Show best parameters and best score
    print(grid.best_params_,grid.best_score_)
    #Show the full spectrum of the best estimator
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test)


    print('Confusion Matrix:')
    confusion_m_grid = ConfusionMatrixMine(test_y, grid_predictions)
    print(confusion_m_grid)


    print('Classification report:')
    print(classification_report(test_y,grid_predictions))

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_m_grid ,display_labels=grid.best_estimator_.classes_)
    disp.plot()
