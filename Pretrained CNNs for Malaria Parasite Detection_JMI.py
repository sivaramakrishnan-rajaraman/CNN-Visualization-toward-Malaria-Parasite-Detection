
# coding: utf-8

# This code is meant to use pretrained models to extract the features from the optimal layer for the parasitized and uninfected cells to aid in improved malaria disease screening. However, you can use these codes as the skeleton to make use of pretrained models as feature extractors for your task of interest.Simply use this skeleton and extract the features from the most optimal layer from the model of your interest for the underlying data. You shall optimize the model hyperparameters to suit your data.

# To begin with, let us define a few functions to load the data and convert them to Keras compatible targets. We will load the libraries to begin with.

# In[ ]:


# load libraries
import cv2
import numpy as np
import os
from keras.utils import np_utils
import matplotlib.pyplot as plt
import itertools
import time
import densenet
from keras.layers import Dense
from keras.models import Model
from sklearn.metrics import log_loss
from keras.optimizers import SGD
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.metrics import average_precision_score
from keras import backend as K
#from keras.models import load_model

# We performed 5-fold cross validation at the patient level. we had train and test splits for each fold to ensure that none of the patienet information in the training data leaks into the test data. We randomly split 10% of the training data for validation. For simplicity, we used a single fold here to show how to run the script.

# In[ ]:


#define data directories
train_data_dir = 'f1_mal/train'
valid_data_dir = 'f1_mal/valid'
test_data_dir = 'f1_mal/test'

# declare the number of samples in each category
nb_train_samples = 19808 #  modify for your dataset
nb_valid_samples = 4952 #  modify for your dataset
nb_test_samples = 2730 # modify for your dataset
num_classes = 2 # binary classification 
img_rows_orig = 100 # modify these values depending on your requirements
img_cols_orig = 100 # modify these values depending on your requirements


# Lets define functions to load and resize the training, validation and test data.

# In[ ]:


def load_training_data():
    labels = os.listdir(train_data_dir)
    total = len(labels)
    X_train = np.ndarray((nb_train_samples, img_rows_orig, img_cols_orig, 3), dtype=np.uint8)
    Y_train = np.zeros((nb_train_samples,), dtype='uint8')
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    j = 0
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        total = len(image_names_train)
        print(label, total)
        for image_name in image_names_train:
            img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
            img = np.array([img])
            X_train[i] = img
            Y_train[i] = j
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        j += 1    
    print(i)                
    print('Loading done.')
    print('Transform targets to keras compatible format.')
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    np.save('imgs_train.npy', X_train, Y_train) #save as numpy files
    return X_train, Y_train
    
def load_validation_data():
    # Load validation images
    labels = os.listdir(valid_data_dir)
    X_valid = np.ndarray((nb_valid_samples, img_rows_orig, img_cols_orig, 3), dtype=np.uint8)
    Y_valid = np.zeros((nb_valid_samples,), dtype='uint8')
    i = 0
    print('-'*30)
    print('Creating validation images...')
    print('-'*30)
    j = 0
    for label in labels:
        image_names_valid = os.listdir(os.path.join(valid_data_dir, label))
        total = len(image_names_valid)
        print(label, total)
        for image_name in image_names_valid:
            img = cv2.imread(os.path.join(valid_data_dir, label, image_name), cv2.IMREAD_COLOR)
            img = np.array([img])
            X_valid[i] = img
            Y_valid[i] = j
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        j += 1
    print(i)            
    print('Loading done.')
    print('Transform targets to keras compatible format.');
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)
    np.save('imgs_valid.npy', X_valid, Y_valid) #save as numpy files
    return X_valid, Y_valid

def load_test_data():
    labels = os.listdir(test_data_dir)
    X_test = np.ndarray((nb_test_samples, img_rows_orig, img_cols_orig, 3), dtype=np.uint8)
    Y_test = np.zeros((nb_test_samples,), dtype='uint8')
    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    j = 0
    for label in labels:
        image_names_test = os.listdir(os.path.join(test_data_dir, label))
        total = len(image_names_test)
        print(label, total)
        for image_name in image_names_test:
            img = cv2.imread(os.path.join(test_data_dir, label, image_name), cv2.IMREAD_COLOR)
            img = np.array([img])
            X_test[i] = img
            Y_test[i] = j
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        j += 1
    print(i)            
    print('Loading done.')
    print('Transform targets to keras compatible format.');
    Y_test = np_utils.to_categorical(Y_test[:nb_test_samples], num_classes)
    np.save('imgs_test.npy', X_test, Y_test) #save as numpy files
    return X_test, Y_test


# We will define functions to resize the original images to that dimensions required for the pretrained models using the functions defined below.

# In[ ]:


def load_resized_training_data(img_rows, img_cols):

    X_train, Y_train = load_training_data()
    X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
    
    return X_train, Y_train
    
def load_resized_validation_data(img_rows, img_cols):

    X_valid, Y_valid = load_validation_data()
    X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])
        
    return X_valid, Y_valid   

def load_resized_test_data(img_rows, img_cols):

    X_test, Y_test = load_test_data()
    X_test = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_test[:nb_test_samples,:,:,:]])
    
    return X_test, Y_test


# An evaluation script has been written to compute the confusion matrix for the performance of the trained model. This function prints and plots the confusion matrix. Normalization can be applied by setting 'normalize=True'.

# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False, #if true all values in confusion matrix is between 0 and 1
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# We will now proceed to extract the features from our dataset using the pretrained modelsand visualize the confusion matrix, ROC and AUC curves.

# In[ ]:

if __name__ == '__main__':
    with K.tf.device('/gpu:0'):
        img_rows=100
        img_cols=100
        channel = 3
        num_classes = 2 
        batch_size = 8 
        nb_epoch = 3
   
    #declare a weight function to store the weights
        model_final_weights_fn = 'custom_densenet_malaria.h5'
   
    # Load our model
        base_model = densenet.DenseNet(input_shape=(100,100,3), include_top=False, classes=num_classes, depth=40, nb_dense_block=4, growth_rate=12, 
			  bottleneck=True, reduction=0.5, weight_decay=1e-4, weights='imagenet', activation='softmax')
            
        base_model.summary()
        
        
        x = base_model.output
        predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()

        # Load data
    
        X_train, Y_train = load_resized_training_data(img_rows, img_cols)
        X_valid, Y_valid = load_resized_validation_data(img_rows, img_cols)
        
        print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)

    # Start Fine-tuning
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) #try varying this for your task and see the best fit
        model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        print('-'*30)
        print('Start Training the Custom DenseNet on the Malaria Cell Dataset...')
        print('-'*30)
        t=time.time()
        hist=model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )
        print('Training time: %s' % (time.time()-t))
        train_loss=hist.history['loss']
        val_loss=hist.history['val_loss']
        train_acc=hist.history['acc']
        val_acc=hist.history['val_acc']
        xc=range(nb_epoch)
        
        plt.figure(1,figsize=(20,10), dpi=100)
        plt.plot(xc,train_loss)
        plt.plot(xc,val_loss)
        plt.xlabel('num of Epochs')
        plt.ylabel('loss')
        plt.title('train_loss vs val_loss')
        plt.grid(True)
        plt.legend(['train','val'])
        plt.style.use(['classic'])
        
        plt.figure(2,figsize=(20,10), dpi=100)
        plt.plot(xc,train_acc)
        plt.plot(xc,val_acc)
        plt.xlabel('num of Epochs')
        plt.ylabel('accuracy')
        plt.title('train_acc vs val_acc')
        plt.grid(True)
        plt.legend(['train','val'])
        plt.style.use(['classic'])
        
        model.save_weights(model_final_weights_fn)
        model.save('custom_densenet_malaria_model.h5')
        
#Test

if __name__ == '__main__':
    with K.tf.device('/gpu:0'):

        model_final_weights_fn = 'custom_densenet_malaria.h5'
    
#    # Load our model
#        # model = load_model('custom_densenet_malaria_model.h5') #this can be used when i save the trained model
#        model = densenet.DenseNet(input_shape=(100,100,3), classes=num_classes, depth=40, nb_dense_block=4, growth_rate=12, 
#			  bottleneck=True, reduction=0.5, weight_decay=1e-4, weights=None, activation='softmax')
#        model.summary()
        model.load_weights(model_final_weights_fn, by_name=True) 
    
        X_test, Y_test = load_resized_test_data(img_rows, img_cols)
        print(X_test.shape, Y_test.shape)
    
    # Make predictions
        print('-'*30)
        print('Predicting on test data...')
        print('-'*30)
        Y_test_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
        
        #compute the ROC-AUC values
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_test_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_test_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        #Plot ROC curves
        plt.figure(figsize=(20,10), dpi=100)
        lw = 1
        plt.plot(fpr[1], tpr[1], color='red',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[1])
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristics')
        plt.legend(loc="lower right")
        plt.show()
        
        
        # compute the accuracy
        Test_accuracy = accuracy_score(Y_test.argmax(axis=-1),Y_test_pred.argmax(axis=-1))
        print("Test_Accuracy of DenseNet Model is: = ",Test_accuracy)
    
     # Cross-entropy loss score
        score = log_loss(Y_test, Y_test_pred)
        print(score)
        
        # compute the average precision score
        prec_score = average_precision_score(Y_test,Y_test_pred)  
        print(prec_score)
    
        print('Generating the ROC_AUC_Scores') #Compute Area Under the Curve (AUC) from prediction scores
        print(roc_auc_score(Y_test,Y_test_pred)) #this implementation is restricted to the binary classification task or multilabel classification task in label indicator format.
    
    # transfer it back
        Y_test_pred = np.argmax(Y_test_pred, axis=1)
        Y_test = np.argmax(Y_test, axis=1)
    
        print(Y_test_pred)
        print(Y_test)
    
        np.savetxt('malaria_Y_test_pred.csv',Y_test_pred,fmt='%i',delimiter = ",")
        np.savetxt('malaria_Y_test.csv',Y_test,fmt='%i',delimiter = ",")
    
        target_names = ['class 0(parasitic)', 'class 1(normal)']
        print(classification_report(Y_test,Y_test_pred,target_names=target_names))
        print(confusion_matrix(Y_test,Y_test_pred))
        cnf_matrix = (confusion_matrix(Y_test,Y_test_pred))
        np.set_printoptions(precision=4)
        plt.figure()
    
    # Plot non-normalized confusion matrix
        plt.figure(figsize=(20,10), dpi=100)
        plot_confusion_matrix(cnf_matrix, classes=target_names,
                          title='Confusion matrix')
        plt.show()