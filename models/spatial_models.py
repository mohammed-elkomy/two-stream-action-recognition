"""
********************************
*   Created by mohammed-alaa   *
********************************
This Contains four keras models as spatial stream:
motion streams expects data tensors in the form [batch_size,height,width,3)]
1) Xception model
2) resnet 50
3) VGG19
4) MobileNet
"""
"""
To understand what is going look at this https://keras.io/applications/
"""
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Reshape, Activation, Dropout, GlobalAveragePooling2D, Conv2D, Flatten, Dense, BatchNormalization


class ResNet50SpatialCNN:
    """
    ResNet model used for spatial stream which is
     (input layer >> norm layer >> resnet50 without prediction layers(look at keras docs https://keras.io/applications/) >> flatting >> softmax projection)
    """
    """
    pretrained+adam: 80 ~ 81.2
    scratch+adam: 0.42215174 !!! imagenet pre training is really important

    pretrained+MSGD: 78.5 ~ 80
    scratch+MSGD:
    """

    def __init__(self, num_classes, is_tesla_k80, pre_trained=True):
        self.is_teslaK80 = is_tesla_k80

        # input layer
        self.inputs = Input(shape=(224, 224, 3), name="input_spatial")
        # data normalization
        self.data_norm = BatchNormalization(3, name='data_norm', center=False, scale=False)

        # create the base pre-trained model
        self.resnet = ResNet50(weights='imagenet' if pre_trained else None, include_top=False)

        # print(self.base_model.get_layer('avg_pool').__dict__)
        self.flat = Flatten(name="flatten")

        # self.drop_out_fc = keras.layers.Dropout(.75)
        self.fc_custom = Dense(num_classes, name="fc_custom", activation="softmax")

    def get_keras_model(self):
        # keras functional api
        def model(inputs):
            return self.fc_custom(self.flat(self.resnet(self.data_norm(inputs))))

        return Model(self.inputs, model(self.inputs), name="spatial_resnet50")

    def get_loader_configs(self):
        return {"width": 224, "height": 224, "batch_size": 76 if self.is_teslaK80 else 48}


class XceptionSpatialCNN:
    """
    ResNet model used for spatial stream which is
     (input layer >> norm layer >> xception without prediction layers (look at keras docs https://keras.io/applications/) >> GlobalAveragePooling2D >> softmax projection)
    """
    """
    pretrained+adam: 86.12% <3
    scratch+adam:

    pretrained+MSGD:82%
    scratch+MSGD:
    """

    def __init__(self, num_classes, is_tesla_k80, pre_trained=True):
        self.is_teslaK80 = is_tesla_k80
        # input layer
        self.inputs = Input(shape=(299, 299, 3), name="input_spatial")
        # data normalization
        self.data_norm = BatchNormalization(3, name='data_norm', center=False, scale=False)

        # create the base pre-trained model
        self.xception = Xception(weights='imagenet' if pre_trained else None, include_top=False, input_shape=(299, 299, 3))

        self.GlobalAveragePooling2D = GlobalAveragePooling2D(name='avg_pool')

        # self.drop_out_fc = keras.layers.Dropout(.75)
        self.fc_custom = Dense(num_classes, name="predictions", activation="softmax")

    def get_keras_model(self):
        # print(inputs)
        def model(inputs):
            return self.fc_custom(self.GlobalAveragePooling2D(self.xception(self.data_norm(inputs))))

        return Model(self.inputs, model(self.inputs), name="spatial_xception")

    def get_loader_configs(self):
        return {"width": 299, "height": 299, "batch_size": 28 if self.is_teslaK80 else 28}  # 28


class VGGSpatialCNN:
    """
    VGG19 model used for spatial stream which is
     (input layer >> norm layer >> VGG19 without prediction layers (look at keras docs https://keras.io/applications/) >> GlobalAveragePooling2D >> softmax projection)
    """
    """
    pretrained+adam:
    scratch+adam:

    pretrained+MSGD: 70%
    scratch+MSGD:
    """

    def __init__(self, num_classes, is_tesla_k80, pre_trained=True):
        self.is_teslaK80 = is_tesla_k80
        # input layer
        self.inputs = Input(shape=(224, 224, 3), name="input_spatial")
        # data normalization
        self.data_norm = BatchNormalization(3, name='data_norm', center=False, scale=False)

        # create the base pre-trained model
        self.vgg19_no_top = VGG19(weights='imagenet' if pre_trained else None, include_top=False)

        self.flat = Flatten(name='flatten')
        self.Dense_1 = Dense(4096, activation='relu', name='fc1')
        self.Dense_2 = Dense(4096, activation='relu', name='fc2')
        self.Dense_3 = Dense(num_classes, activation='softmax', name='predictions')

    def get_keras_model(self):
        # print(inputs)
        def model(inputs):
            x = self.vgg19_no_top(self.data_norm(inputs))
            x = self.flat(x)
            x = self.Dense_1(x)
            x = self.Dense_2(x)
            prediction = self.Dense_3(x)
            return prediction

        return Model(self.inputs, model(self.inputs), name="spatial_vgg19")

    def get_loader_configs(self):
        return {"width": 224, "height": 224, "batch_size": 40 if self.is_teslaK80 else 40}


class MobileSpatialCNN:
    """
    MobileNet model used for spatial stream which is
     (input layer >> norm layer >> MobileNet without prediction layers (look at keras docs https://keras.io/applications/) >> GlobalAveragePooling2D >> softmax projection)
    """
    """
    pretrained+adam:
    scratch+adam:

    pretrained+MSGD:
    scratch+MSGD:
    """

    def __init__(self, num_classes, is_tesla_k80, alpha=1, dropout=1e-3, pre_trained=True):
        self.is_teslaK80 = is_tesla_k80

        # input layer
        self.inputs = Input(shape=(224, 224, 3), name="input_spatial")
        # data normalization
        self.data_norm = BatchNormalization(3, name='data_norm', center=False, scale=False)

        # create the base pre-trained model
        self.mobile_net = MobileNet(weights='imagenet' if pre_trained else None, include_top=False)

        self.GlobalAveragePooling2D = GlobalAveragePooling2D()

        shape = (1, 1, int(1024 * alpha))
        self.Reshape_1 = Reshape(shape, name='reshape_1')
        self.Dropout = Dropout(dropout, name='dropout')
        self.Conv2D = Conv2D(num_classes, (1, 1), padding='same', name='conv_preds')
        self.Activation = Activation('softmax', name='act_softmax')
        self.Reshape_2 = Reshape((num_classes,), name='reshape_2')

    def get_keras_model(self):
        def model(inputs):
            x = self.mobile_net(self.data_norm(inputs))
            x = self.GlobalAveragePooling2D(x)
            x = self.Reshape_1(x)
            x = self.Dropout(x)
            x = self.Conv2D(x)
            x = self.Activation(x)
            prediction = self.Reshape_2(x)
            return prediction

        return Model(self.inputs, model(self.inputs), name="spatial_mobilenet")

    def get_loader_configs(self):
        return {"width": 224, "height": 224, "batch_size": 100 if self.is_teslaK80 else 100}
