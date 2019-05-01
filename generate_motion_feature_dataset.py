"""
********************************
*   Created by mohammed-alaa   *
********************************
Here I'm generating visual features from my pretrained motion stream having 84% top-1 accuracy
these visual features are just the layer below the softmax prediction, layer have 2048 features for each input image
the data generated are stored into a big list and then dumped as pickle file for each epoch of data (more data augmentation)
then this data are fed into a recurrent network implemented in recurrent_fusion_trainer.py file to train a video level classifier instead of frame level classier

I expect a pre-trained xception model here to be downloaded from drive
-------------------
In this file I do what I call model surgery which is removing or adding some layer from a model
Here I load my trained model whose architecture is

Input_image > batch_norm  >> xception model as layer >>> softmax layer of 101 classes


so I can change the model a little bit and make it have 2 outputs which are the features just below the softmax and the softmax
so the model becomes

Input_image > batch_norm  >> xception model as layer >>> softmax layer of 101 classes
#                                                    >>> feature field of 2048 features

which are two outputs now
"""
import pickle

from tensorflow.python.keras import Model, Input

import frame_dataloader
from evaluation import legacy_load_model
from evaluation.evaluation import *
from utils.drive_manager import DriveManager

#####################################################
feature_field_size = 2048
testing_samples_per_video = 19
#####################################################
"""Managed"""
evaluate = False
generate_test = False

drive_manager = DriveManager("motion_feature_dataset")
drive_manager.download_file('1O8OM6Q01az_71HdMQmWM3op1qJhfsQoI', "motion.zip")  # the id of the zip file contains my network

motion_model_restored = legacy_load_model(filepath="motion.h5", custom_objects={'sparse_categorical_cross_entropy_loss': sparse_categorical_cross_entropy_loss, "acc_top_1": acc_top_1, "acc_top_5": acc_top_5})
motion_model_restored.summary()
# xception here is a layer
# The architecture summary is
# input_image > batch_norm > xception layer
xception_rebuilt = Model(
    motion_model_restored.layers[-1].layers[0].input,  # input image to xception layer itself not my wrapper model
    [layer.output for layer in motion_model_restored.layers[-1].layers[-2:]]  # two outputs of xception layer itself visual features, softmax output
)

motion_model_with_2_outputs = Model(
    motion_model_restored.inputs[0],  # input of my wrapper model
    xception_rebuilt(motion_model_restored.layers[1](motion_model_restored.inputs[0]))  # the two outputs obtained from xception layer are connected to the original input of the wrapper model

)

data_loader = frame_dataloader.MotionDataLoaderVisualFeature(
    num_workers=workers, samples_per_video=19,
    width=int(motion_model_restored.inputs[0].shape[1]), height=int(motion_model_restored.inputs[0].shape[2])
    , use_multiprocessing=True, augmenter_level=0, # heavy augmentation
)
train_loader, test_loader = data_loader.run()

"""
Evaluate and check
"""
if evaluate:
    progress = tqdm.tqdm(test_loader, total=len(test_loader))
    inp = Input(shape=(2048,), name="dense")
    dense_layer = Model(inp, motion_model_restored.layers[-1].layers[-1](inp))

    video_level_preds_np = np.zeros((len(progress), num_actions))  # each video per 101 class (prediction)
    video_level_labels_np = np.zeros((len(progress), 1))

    for index, (video_frames, video_label) in enumerate(progress):  # i don't need frame level labels
        feature_field, frame_preds = motion_model_with_2_outputs.predict_on_batch(video_frames)
        assert np.allclose(frame_preds, dense_layer.predict(feature_field))

        video_level_preds_np[index, :] = np.mean(frame_preds, axis=0)
        video_level_labels_np[index, 0] = video_label

    video_level_loss, video_level_accuracy_1, video_level_accuracy_5 = keras.backend.get_session().run(
        [val_loss_op, acc_top_1_op, acc_top_5_op], feed_dict={video_level_labels_k: video_level_labels_np, video_level_preds_k: video_level_preds_np})

    print("Motion Model validation", "prec@1", video_level_accuracy_1, "prec@5", video_level_accuracy_5, "loss", video_level_loss)

"""
Generate the data and save into pickles
"""
##############################################################################
# test data generation
if generate_test:
    test_progress = tqdm.tqdm(test_loader, total=len(test_loader))

    samples, labels = np.zeros([len(test_loader), testing_samples_per_video, feature_field_size], dtype=np.float32), np.zeros([len(test_loader), ], dtype=np.float32)

    last_access = 0
    for index, (video_frames, video_label) in enumerate(test_progress):  # i don't need frame level labels
        feature_field, _ = motion_model_with_2_outputs.predict_on_batch(video_frames)
        samples[index] = feature_field
        labels[index] = video_label
        last_access = index

    print("test samples:", samples.shape)
    print("test labels:", labels.shape)
    assert last_access == len(test_progress) - 1

    with open("test_features_motion.pickle", 'wb') as f:
        pickle.dump((samples, labels), f)

    del samples, labels
    drive_manager.upload_project_file("test_features_motion.pickle")

##############################################################################
# train data generation
for epoch in range(1):
    train_progress = tqdm.tqdm(train_loader, total=len(train_loader))
    samples, labels = np.zeros([len(train_loader), testing_samples_per_video, feature_field_size], dtype=np.float32), np.zeros([len(train_loader), ], dtype=np.float32)

    last_access = 0
    for index, (video_frames, video_label) in enumerate(train_progress):  # i don't need frame level labels
        feature_field, _ = motion_model_with_2_outputs.predict_on_batch(video_frames)
        samples[index] = feature_field
        labels[index] = video_label
        last_access = index

    print("train samples:", samples.shape)
    print("train labels:", labels.shape)
    assert last_access == len(train_loader) - 1

    with open("train_features_motion.pickle", 'wb') as f:
        pickle.dump((samples, labels), f)

    del samples, labels
    drive_manager.upload_project_file("train_features_motion.pickle")
##############################################################################
