"""
********************************
*   Created by mohammed-alaa   *
********************************
This contains helper functions needed to
evaluate the model while training
evaluate the model loaded from the disk
evaluate prediction file in pickle format
"""

import multiprocessing
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tqdm
from tensorflow import keras

""" Global variables for evaluation """
num_actions = 101
workers = min(multiprocessing.cpu_count(), 4)
is_tesla_k80 = os.path.isdir("/content")  # this is true if you are on colab :D

# keras placeholder used for evaluation
video_level_labels_k = keras.backend.placeholder([None, 1], dtype=tf.float32)
video_level_preds_k = keras.backend.placeholder([None, num_actions], dtype=tf.float32)

# tensors representing top-1 top-5 and cost function in symbolic form
val_loss_op = keras.backend.mean(keras.metrics.sparse_categorical_crossentropy(video_level_labels_k, video_level_preds_k))
acc_top_1_op = keras.metrics.sparse_top_k_categorical_accuracy(video_level_labels_k, video_level_preds_k, k=1)
acc_top_5_op = keras.metrics.sparse_top_k_categorical_accuracy(video_level_labels_k, video_level_preds_k, k=5)


def acc_top_5(y_true, y_pred):
    """Helper function for top-5 accuracy reported in UCF"""
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)


def acc_top_1(y_true, y_pred):
    """Helper function for top-1 accuracy/(traditional accuracy) reported in UCF"""
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=1)


# compile the model (should be done *after* setting layers to non-trainable)
def sparse_categorical_cross_entropy_loss(y_true, y_pred):
    """Custom loss function:I changed it a little bit but observed no difference"""
    return keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


def eval_model(model, test_loader, test_video_level_label, testing_samples_per_video):
    """
    runs a progressor showing my custom validation per epoch, returning the metrics
    """
    progress = tqdm.tqdm(test_loader, total=len(test_loader))
    test_video_level_preds = defaultdict(lambda: np.zeros((num_actions,)))

    for video_names, sampled_frame in progress:  # i don't need frame level labels

        frame_preds = model.predict_on_batch(sampled_frame)
        _batch_size = frame_preds.shape[0]  # last batch wont be batch_size :3

        for video_id in range(_batch_size):  # in batch
            video_name = video_names[video_id]  # ApplyMakeup_g01_c01 for example
            test_video_level_preds[video_name] += frame_preds[video_id]

    video_level_loss, video_level_accuracy_1, video_level_accuracy_5 = video_level_eval(test_video_level_preds=test_video_level_preds,
                                                                                        test_video_level_label=test_video_level_label,
                                                                                        testing_samples_per_video=testing_samples_per_video)

    return video_level_loss, video_level_accuracy_1, video_level_accuracy_5, test_video_level_preds


def video_level_eval(test_video_level_preds, test_video_level_label, testing_samples_per_video):
    """
    video level validation applying accuracy scoring top-5 and top-1 using predictions and labels feeded as dictionaries
    """
    video_level_preds_np = np.zeros((len(test_video_level_preds), num_actions))  # each video per 101 class (prediction)
    video_level_labels_np = np.zeros((len(test_video_level_preds), 1))

    for index, video_name in enumerate(sorted(test_video_level_preds.keys())):  # this should loop on test videos = 3783 videos
        video_summed_preds = test_video_level_preds[video_name] / testing_samples_per_video  # average on
        video_label = test_video_level_label[video_name]  # 0 based label

        video_level_preds_np[index, :] = video_summed_preds
        video_level_labels_np[index, 0] = video_label

    video_level_loss, video_level_accuracy_1, video_level_accuracy_5 = keras.backend.get_session().run(
        [val_loss_op, acc_top_1_op, acc_top_5_op], feed_dict={video_level_labels_k: video_level_labels_np, video_level_preds_k: video_level_preds_np})

    return video_level_loss, video_level_accuracy_1, video_level_accuracy_5
