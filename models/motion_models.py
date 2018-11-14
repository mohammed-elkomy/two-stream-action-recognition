"""
********************************
*   Created by mohammed-alaa   *
********************************
This Contains two keras models as motion stream:
motion streams expects data tensors in the form [batch_size,height,width,stacked_frames(u/v=10*2)]
1) Xception model
2) resnet 50
"""
"""
To understand what is going look at this https://keras.io/applications/
"""

import h5py
import numpy as np
import tensorflow.keras.backend as K


from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import *
#
from tensorflow.keras.models import Model

from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.engine.saving import load_attributes_from_hdf5_group
from tensorflow.python.keras.utils import get_file

# from keras.applications.resnet50 import WEIGHTS_PATH_NO_TOP can't be imported in newer versions so I copied it
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# from keras_applications.xception import TF_WEIGHTS_PATH_NO_TOP can't be imported in newer versions so I copied it
TF_WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.4/'
    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

def is_same_shape(shape1, shape2):
    """Checks if two structures[could be list or single value for example] have the same shape"""
    if len(shape1) != len(shape2):
        return False
    else:
        for i in range(len(shape1)):
            if shape1[i] != shape2[i]:
                return False

        return True


# This piece of code is inspired by keras source
def compare_layers_weights(first_model_layers, second_model_layers):
    """Compare layers weights: I use them to test the pre trained models are loaded correctly"""
    for i in range(len(first_model_layers)):
        weights1 = first_model_layers[i].get_weights()
        weights2 = second_model_layers[i].get_weights()
        if len(weights1) == len(weights2):
            if not all([is_same_shape(weights2[w].shape, weights1[w].shape) and np.allclose(weights2[w], weights1[w]) for w in range(len(weights1))]):
                print(first_model_layers[i].name, "!=", second_model_layers[i].name)
        else:
            print(first_model_layers[i].name, "!=", second_model_layers[i].name)


# This piece of code is inspired by keras source
def get_symbolic_filtered_layer_weights_from_model(model):
    """For the given model get the symbolic(tensors) weights"""
    symbolic_weights = []
    for layer in model.layers:
        if layer.weights:
            symbolic_weights.append(layer.weights)
    return symbolic_weights  # now you can load those weights with tensorflow feed


# This piece of code is inspired by keras source
def get_named_layer_weights_from_h5py(h5py_file):
    """decodes h5py for a given model downloaded by keras and gets layer weight name to value mapping"""
    with h5py.File(h5py_file) as h5py_stream:
        layer_names = load_attributes_from_hdf5_group(h5py_stream, 'layer_names')

        weights_values = []
        for name in layer_names:
            layer = h5py_stream[name]
            weight_names = load_attributes_from_hdf5_group(layer, 'weight_names')
            if weight_names:
                weight_values = [np.asarray(layer[weight_name]) for weight_name in weight_names]
                weights_values.append((name, weight_values))
    return weights_values


# This piece of code is inspired by keras source
def load_layer_weights(weight_values, symbolic_weights):
    """loads weight_values which is a list ot tuples from get_named_layer_weights_from_h5py()
        into symbolic_weights obtained from get_symbolic_filtered_layer_weights_from_model()
    """
    if len(weight_values) != len(symbolic_weights):  # they must have the same length of layers
        raise ValueError('number of weights aren\'t equal', len(weight_values), len(symbolic_weights))
    else:  # similar to keras source code :D .. load_weights_from_hdf5_group
        print("length of layers to load", len(weight_values))
        weight_value_tuples = []

        # load layer by layer weights
        for i in range(len(weight_values)):  # list(layers) i.e. list of lists(weights)
            assert len(symbolic_weights[i]) == len(weight_values[i][1])
            # symbolic_weights[i] : list of symbolic names for layer i
            # symbolic_weights[i] : list of weight ndarrays for layer i
            weight_value_tuples += zip(symbolic_weights[i], weight_values[i][1])  # both are lists with equal lengths (name,value) mapping

        K.batch_set_value(weight_value_tuples)  # loaded a batch to be efficient


def cross_modality_init(in_channels, kernel):
    """
        Takes a weight computed for RGB and produces a new wight to be used by motion streams which need about 20 channels !
        kernel is (x, y, 3, 64)
    """
    # if in_channels == 3:  # no reason for cross modality
    #   return kernel
    print("cross modality kernel", kernel.shape)
    avg_kernel = np.mean(kernel, axis=2)  # mean (x, y, 64)
    weight_init = np.expand_dims(avg_kernel, axis=2)  # mean (x, y, 1, 64)
    return np.tile(weight_init, (1, 1, in_channels, 1))  # mean (x, y, in_channels, 64)


def CrossModalityResNet50(num_classes, pre_trained, cross_modality_pre_training, input_shape):
    """Pretrained Resnet50 model from keras which uses cross modality pretraining to obtain a convolution weight which suits 20 channels needed by motion stream"""
    cross_modality_pre_training = cross_modality_pre_training and pre_trained

    # create the model
    model = ResNet50(classes=num_classes, weights=None, input_shape=input_shape, include_top=True)
    channels = input_shape[2]

    # load weight file >>> downloads some file from github
    weights_path = get_file(
        'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        md5_hash='a268eb855778b3df3c7506639542a6af')

    # get the named weights of each layer
    weight_values_ = get_named_layer_weights_from_h5py(weights_path)
    # get the symbolic weights of each layer
    symbolic_weights_ = get_symbolic_filtered_layer_weights_from_model(model)[:len(weight_values_)]

    if cross_modality_pre_training:  # use a pretrained convolution weight
        # update it (name,[kernel,bias])
        # cross modality pre-training for kernel
        # leave bias as is of course
        weight_values_[0] = ("conv1_cross_modality",
                             [cross_modality_init(kernel=weight_values_[0][1][0], in_channels=channels),  # 0 = first layer , 1 = weight_value , 0 = kernel
                              weight_values_[0][1][1]]  # 0 = first layer , 1 = weight_value , 1 = bias
                             )

    else:  # start the first convolution layer as random glorot
        symbolic_weights_ = symbolic_weights_[1:]
        weight_values_ = weight_values_[1:]

    if pre_trained:
        # do weight loading
        load_layer_weights(weight_values=weight_values_, symbolic_weights=symbolic_weights_)

    return model


class ResNet50MotionCNN:
    """
    ResNet model used for motion stream which is (input layer >> norm layer >> resnet50 model)
    """
    """
      pretrained+adam:
      scratch+adam:

      pretrained+MSGD:80%
      scratch+MSGD:
      """

    def __init__(self, num_classes, is_tesla_k80, stacked_frames, pre_trained=True, cross_modality_pre_training=True):
        self.is_teslaK80 = is_tesla_k80
        # input layer
        self.inputs = Input(shape=(224, 224, 2 * stacked_frames), name="input_motion")

        # data normalization
        self.data_norm = BatchNormalization(3, name='data_norm', center=False, scale=False)
        # create the base pre-trained model
        self.resnet = CrossModalityResNet50(num_classes=num_classes, pre_trained=pre_trained, cross_modality_pre_training=cross_modality_pre_training, input_shape=(224, 224, 2 * stacked_frames))

    def get_keras_model(self):
        # keras functional api
        return Model(self.inputs, self.resnet(self.data_norm(self.inputs)), name="motion_resnet")

    def get_loader_configs(self):
        return {"width": 224, "height": 224, "batch_size": 28 if self.is_teslaK80 else 24}


def CrossModalityXception(num_classes, pre_trained, cross_modality_pre_training, input_shape, include_feature_fields=False):
    cross_modality_pre_training = cross_modality_pre_training and pre_trained

    # create the model
    model = Xception(classes=num_classes, weights=None, input_shape=input_shape, include_top=True)
    channels = input_shape[2]

    # load weight file >>> downloads some file from github
    weights_path = get_file(
        'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
        TF_WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='b0042744bf5b25fce3cb969f33bebb97')

    weight_values_ = get_named_layer_weights_from_h5py(weights_path)
    symbolic_weights_ = get_symbolic_filtered_layer_weights_from_model(model)[:len(weight_values_)]

    if cross_modality_pre_training:  # use a pretrained convolution weight
        # update it (name,[kernel,bias])
        # cross modality pre-training for kernel
        # leave bias as is of course
        weight_values_[0] = ("conv1_cross_modality",
                             [cross_modality_init(kernel=weight_values_[0][1][0], in_channels=channels),  # 0 = first layer , 1 = weight_value , 0 = kernel
                              # Xception has no bias
                              ]
                             )

    else:  # start the first convolution layer as random glorot
        symbolic_weights_ = symbolic_weights_[1:]
        weight_values_ = weight_values_[1:]

    if pre_trained:
        # do weight loading
        load_layer_weights(weight_values=weight_values_, symbolic_weights=symbolic_weights_)

    if include_feature_fields:
        return Model(model.inputs, [layer.output for layer in model.layers[-2:]])
    else:
        return model


class XceptionMotionCNN:
    """
    Xception model used for motion stream which is (input layer >> norm layer >> xception model)
    """
    """
      pretrained+adam: 84.4%
      scratch+adam:

      pretrained+MSGD:
      scratch+MSGD:
      """

    def __init__(self, num_classes, is_tesla_k80, stacked_frames, pre_trained=True, cross_modality_pre_training=True, include_feature_fields=False):
        self.is_teslaK80 = is_tesla_k80
        # input layer
        self.inputs = Input(shape=(299, 299, 2 * stacked_frames), name="input_motion")
        # data normalization
        self.data_norm = BatchNormalization(3, name='data_norm', center=False, scale=False)

        # create the base pre-trained model
        self.xception = CrossModalityXception(num_classes=num_classes, cross_modality_pre_training=cross_modality_pre_training, pre_trained=pre_trained, input_shape=(299, 299, 2 * stacked_frames), include_feature_fields=include_feature_fields)

    def get_keras_model(self):
        # keras functional api
        return Model(self.inputs, self.xception(self.data_norm(self.inputs)), name="motion_xception")

    def get_loader_configs(self):
        return {"width": 299, "height": 299, "batch_size": 28 if self.is_teslaK80 else 28}


if __name__ == '__main__':
    # test :D
    model1 = ResNet50MotionCNN(num_classes=101, stacked_frames=10, is_tesla_k80=True)
    model2 = ResNet50MotionCNN2(num_classes=101, stacked_frames=10, is_tesla_k80=True)
    model3 = ResNet50()
    print(model1.layers)
    print(model2.layers)
    print(model3.layers)
    print(" ")
    compare_layers_weights(model1.layers[1].layers, model2.layers[1].layers)
    print(" ")
    compare_layers_weights(model3.layers, model2.layers[1].layers)
    print(" ")
    compare_layers_weights(model3.layers, model1.layers[1].layers)
    print(" ")

    print("xception test")
    model4 = Xception(input_shape=(299, 299, 3))
    model5 = XceptionMotionCNN(num_classes=101, is_tesla_k80=True, stacked_frames=10)

    print(model4.layers)
    print(model5.layers)
    compare_layers_weights(model4.layers, model5.layers[1].layers)

    print("values")
    print(model4.layers[1].weights)
    print(model4.layers[1].get_weights()[0][0, 0, :, 0])
    print(model5.layers[1].layers[1].get_weights()[0][0, 0, :, 0])
