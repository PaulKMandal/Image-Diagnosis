
# coding: utf-8

# In[ ]:


import numpy as np
#np.set_printoptions(threshold = np.nan, linewidth = 115)
import pickle
import os

base = os.path.dirname(os.getcwd()) + '/fastai/'

# Load with pickle instead of processing images again
training_img_1 = pickle.load(open(base + '1vAll_img_res_Infiltration_1st_half.p', 'rb'))
training_img_2 = pickle.load(open(base + '1vAll_img_res_Infiltration_2nd_half.p', 'rb'))

training_img_one = np.append(training_img_1, training_img_2, axis=0)

training_img_3 = pickle.load(open(base + '1vAll_img_res_Infiltration_3rd_half.p', 'rb'))
training_img_4 = pickle.load(open(base + '1vAll_img_res_Infiltration_4th_half.p', 'rb'))

training_img_two = np.append(training_img_3, training_img_4, axis=0)

training_img = np.append(training_img_one, training_img_two, axis=0)

training_img.shape

val_img = training_img[:2269]
val_img = np.append(val_img, training_img[34029:], axis=0)
training_img = training_img[2269:34029]
print(len(val_img))
print(len(training_img))
print(len(training_img) + len(val_img))

labels_1 = pickle.load(open(base + '1vAll_labels_res_Infiltration_1st_half.p', 'rb'))
labels_2 = pickle.load(open(base + '1vAll_labels_res_Infiltration_2nd_half.p', 'rb'))
labels_3 = pickle.load(open(base + '1vAll_labels_res_Infiltration_3rd_half.p', 'rb'))
labels_4 = pickle.load(open(base + '1vAll_labels_res_Infiltration_4th_half.p', 'rb'))

training_labels = np.append(labels_1, np.append(labels_2, np.append(labels_3, labels_4, axis = 0), axis = 0), axis = 0)

val_labels = training_labels[:2269]
val_labels = np.append(val_labels, training_labels[34029:], axis=0)
training_labels = training_labels[2269:34029]
print(len(val_labels))
print(len(training_labels))
print(len(training_labels) + len(val_labels))

test_img = pickle.load(open(base + '1vAll_test_img.p', 'rb'))
test_labels = pickle.load(open(base + '1vAll_test_labels.p', 'rb'))

print('Labels shape: ', training_labels.shape)
print('Length of test_labels: ', len(test_labels))
print('No. of Infiltration Diagnoses: ', sum(training_labels))


# In[ ]:


import keras
from keras import models, optimizers, layers, regularizers, metrics, losses
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ReLU, ThresholdedReLU
from keras.layers.core import Dense, Dropout, SpatialDropout2D, Activation
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.models import model_from_json, Sequential
from keras.callbacks import Callback

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

import matplotlib.pyplot as plt

IMG_SIZE = 256

# Save Comparison model
def save_model(model_name, hist_str, model_str):

    pickle.dump(model_name.history, open('Training Histories/'+ hist_str + '.p', 'wb'))
    
    print("Saved " + hist_str + " to Training Histories folder")
    
    # serialize model to JSON
    model_name = model.to_json()
    with open("CNN Models/" + model_str + ".json", "w") as json_file:
        json_file.write(model_name)

    # serialize weights to HDF5
    model.save_weights("CNN Models/" + model_str + ".h5")
    print("Saved " + model_str + " and weights to CNN Models folder")
    
# Load model architecture and weights NOTE: must compile again
def load_model():
    model_str = str(input("Name of model to load: "))

    # load json and create model
    json_file = open('CNN Models/' + model_str + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("CNN Models/" + model_str + ".h5")
    print("Loaded " + model_str + " and weights from CNN Models folder")
    
    return loaded_model
    
# Load history object
def load_history():
    hist_str = str(input("Name of history to load: "))

    loaded_history = pickle.load(open('Training Histories/' + hist_str + '.p', 'rb'))
    
    print("Loaded " + hist_str + " from Training Histories folder")
    
    return loaded_history

class True_Eval(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.total_accuracy = []
        self.i_accuracy = []
        self.e_accuracy = []
    
    def ie_real_acc(self, prediction):
        y_true = self.validation_data[1]
        i_acc = 0
        i_total = 0
        e_acc = 0
        e_total = 0
        for i in range(0, len(prediction)):
            if (y_true[i].round() == 0):
                if (prediction[i].round() == y_true[i]):
                    i_acc += 1
                i_total += 1
            else:
                if (prediction[i].round() == y_true[i]):
                    e_acc += 1
                e_total += 1
        return (i_acc/i_total), (e_acc/e_total)

    def on_epoch_end(self, epoch, logs={}):
        x_val = self.validation_data[0]
        y_pred = self.model.predict(x_val)
        i_real_acc, e_real_acc = self.ie_real_acc(y_pred)
        print ("T Acc: %f" % i_real_acc)
        print ("F Acc: %f" % e_real_acc)
        self.i_accuracy.append(i_real_acc)
        self.e_accuracy.append(e_real_acc)
        

class MultiLabel_Acc(Callback):
    
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.accuracy = []
        
    def getAccuracy(self, prediction):
        y_true = self.validation_data[1]
        
        correct = []
        total = []
        accuracy = []
        
        for i in range(0, len(y_true[0])):
            correct.append([0, 0])
            total.append([0, 0])
        
        for sample in range(0, len(prediction)):
            for neuron in range(0, len(prediction[sample])):
                
                if (y_true[sample][neuron] == 0.0):
                    if (round(prediction[sample][neuron]) == y_true[sample][neuron]):
                        correct[neuron][0] += 1
                    total[neuron][0] += 1
                    
                if (y_true[sample][neuron] == 1.0):
                    if (round(prediction[sample][neuron]) == y_true[sample][neuron]):
                        correct[neuron][1] += 1
                    total[neuron][1] += 1
        
        for neuron in range(0, len(correct)):
            accuracy.append((correct[neuron][0]/total[neuron][0],
                                  correct[neuron][1]/total[neuron][1]))
        return accuracy
                    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        epoch_data = self.getAccuracy(y_pred)
        self.accuracy.append(epoch_data)
        
        for i in range(0, len(epoch_data)):
            print("         Neuron: #"+ str(i + 1))
            print("Zeroes Accuracy:", epoch_data[i][0])
            print("  Ones Accuracy:", epoch_data[i][1])
            
            
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                         normalize = False,
                         title = 'Confusion Matrix',
                         cmap=plt.cm.Blues):
    _fontsize = 'xx-large'
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix without Normalization")
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=_fontsize)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, fontsize=_fontsize) 
    plt.yticks(tick_marks, classes, fontsize=_fontsize)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.min() + 0.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment = 'center',
                color='white' if cm[i, j] > thresh else 'black', 
                fontsize=_fontsize)
    
    plt.ylabel('True Labels', fontsize=_fontsize)
    plt.xlabel('Predicted Labels', fontsize=_fontsize)
    plt.tight_layout()
    plt.savefig('Plot Images/' + title + '_Mx.png')

# Metric Analysis
def _1vAll_accuracy(y_test, pred):
    
    pred = np.squeeze(pred, axis = -1)
    pred = np.round_(pred)
    pred = pred.astype(dtype = 'uint8')
    
    ft = pred == y_test
    
    accuracy = sum(ft)/len(ft)
        
    print('\t Complete Label Accuracy: %.2f' % round((accuracy * 100), 2), '%')
    
    print('Sum of Fully Correct Predictions: ', sum(ft))
    print('\t\t    Total Labels: ', len(ft))
    
    return accuracy


# In[ ]:


'''
Deep Residual Neural Network - XVNet for Paperspace

'''     

img_height = 256
img_width = 256
img_channels = 1

#
# network params
#

cardinality = 8


def residual_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        #y = layers.SpatialDropout2D(0.125)(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.ReLU()(y)

        return y

    # conv block 1
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = add_common_layers(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same')(x)
    
    # residual block
    for i in range(3):
        project_shortcut = (i == 0)
        x = residual_block(x, 16, 32, _project_shortcut=project_shortcut)

    # conv block 2
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = add_common_layers(x)    
    x = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same')(x)
    
    # conv block 3
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = add_common_layers(x)    
    x = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same')(x)
    
    
    
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 32, 64, _strides=strides)

    # conv4
    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 64, 128, _strides=strides)

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 128, 256, _strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(16)(x)
    x = layers.Dense(1)(x)

    return x


image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
network_output = residual_network(image_tensor)
  
model = models.Model(inputs=[image_tensor], outputs=[network_output])
print(model.summary())

# Last model: XVNet_250e
#model = load_model()
#model.summary()

model.compile(optimizer = optimizers.RMSprop(lr = 1e-4), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

custom_metrics = MultiLabel_Acc((val_img, val_labels))

model_obj = model.fit(training_img, training_labels, 
                      epochs = 250, initial_epoch = 0, 
                      validation_data = (val_img, val_labels), 
                      batch_size = 128, verbose = 1, 
                      callbacks = [custom_metrics])

Predictions = model.predict(test_img)

Accuracy = _1vAll_accuracy(test_labels, Predictions)

history_str = 'XVNet_250e_history'
model_str   = 'XVNet_250e'
    
save_model(model_obj, history_str, model_str)

acc = model_obj.history['acc']
val_acc = model_obj.history['val_acc']
loss = model_obj.history['loss']
val_loss = model_obj.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure(figsize=(15, 10))
plt.plot(epochs, acc, 'bd', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontsize='xx-large')
plt.xticks(fontsize='xx-large')
plt.xlabel('Epochs', fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.ylabel('Accuracy', fontsize='xx-large')
plt.legend(fontsize='xx-large')
plt.savefig('Plot Images/' + model_str +'_Acc.png')
plt.show()

plt.figure(figsize=(15, 10))
plt.plot(epochs, loss, 'rd', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss', fontsize='xx-large')
plt.xticks(fontsize='xx-large')
plt.xlabel('Epochs', fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.ylabel('Loss', fontsize='xx-large')
plt.legend(fontsize='xx-large')
plt.savefig('Plot Images/' + model_str +'_Loss.png')
plt.show()

# Plot Confusion Matrix
cm = confusion_matrix(test_labels, np.round_(Predictions))
cm_plot_labels = ['Not Infiltration', 'Infiltration']

plt.clf()
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm, cm_plot_labels, title = model_str, normalize=True)

