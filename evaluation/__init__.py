"""
********************************
*   Created by mohammed-alaa   *
********************************
This contains helper functions needed by evaluation
"""
import json
import logging
import os

import h5py
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine.saving import model_from_config, load_weights_from_hdf5_group

is_tesla_k80 = os.path.isdir("/content")


# from tensorflow.keras.models import load_model # 1.11.1.rc2
# load model in the new version of tensorflow doesnt work for me and i can't re install older tensorflow-gpu with older cuda for every colab machine :DDD
def legacy_load_model(filepath, custom_objects=None, compile=True):  # pylint: disable=redefined-builtin
    """
    legacy load model since my pretrained models could't be loaded to newer versions of tensorflow
    """
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')

    if not custom_objects:
        custom_objects = {}

    def convert_custom_objects(obj):
        if isinstance(obj, list):
            deserialized = []
            for value in obj:
                deserialized.append(convert_custom_objects(value))
            return deserialized
        if isinstance(obj, dict):
            deserialized = {}
            for key, value in obj.items():
                deserialized[key] = convert_custom_objects(value)
            return deserialized
        if obj in custom_objects:
            return custom_objects[obj]
        return obj

    opened_new_file = not isinstance(filepath, h5py.File)
    if opened_new_file:
        f = h5py.File(filepath, mode='r')
    else:
        f = filepath

    try:
        # instantiate model
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model found in config file.')
        model_config = json.loads(model_config.decode('utf-8'))
        model = model_from_config(model_config, custom_objects=custom_objects)

        # set weights
        load_weights_from_hdf5_group(f['model_weights'], model.layers)

        if compile:
            # instantiate optimizer
            training_config = f.attrs.get('training_config')
            if training_config is None:
                logging.warning('No training configuration found in save file: '
                                'the model was *not* compiled. Compile it manually.')
                return model
            training_config = json.loads(training_config.decode('utf-8'))
            optimizer_config = training_config['optimizer_config']
            optimizer = optimizers.deserialize(
                optimizer_config, custom_objects=custom_objects)

            # Recover loss functions and metrics.
            loss = convert_custom_objects(training_config['loss'])
            metrics = convert_custom_objects(training_config['metrics'])
            sample_weight_mode = training_config['sample_weight_mode']
            loss_weights = training_config['loss_weights']

            # Compile model.
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                sample_weight_mode=sample_weight_mode)

            # Set optimizer weights.
            if 'optimizer_weights' in f:
                # Build train function (to get weight updates).
                model._make_train_function()
                optimizer_weights_group = f['optimizer_weights']
                optimizer_weight_names = [
                    n.decode('utf8')
                    for n in optimizer_weights_group.attrs['weight_names']
                ]
                optimizer_weight_values = [
                    optimizer_weights_group[n] for n in optimizer_weight_names
                ]
                try:
                    model.optimizer.set_weights(optimizer_weight_values)
                except ValueError:
                    logging.warning('Error in loading the saved optimizer '
                                    'state. As a result, your model is '
                                    'starting with a freshly initialized '
                                    'optimizer.')
    finally:
        if opened_new_file:
            f.close()
    return model


def get_batch_size(model_restored, spatial):
    """
    Helper function to get batch size per model
    """
    if spatial:
        if model_restored.layers[2].__dict__["_name"] == 'resnet50':
            batch_size = 76 if is_tesla_k80 else 48
        elif model_restored.layers[2].__dict__["_name"] == 'xception':
            batch_size = 24 if is_tesla_k80 else 24
        elif model_restored.layers[2].__dict__["_name"] == 'vgg19':
            batch_size = 36 if is_tesla_k80 else 36
        else:
            batch_size = 100 if is_tesla_k80 else 100
    else:
        if model_restored.layers[2].__dict__["_name"] == 'resnet50':
            batch_size = 20 if is_tesla_k80 else 20
        else:
            batch_size = 18 if is_tesla_k80 else 18

    return batch_size
