"""
********************************
*   Created by mohammed-alaa   *
********************************
Spatial Dataloader implementing sequence api from keras (defines how to load a single item)
this loads batches of images for each iteration it returns [batch_size, height, width ,3] ndarrays
"""
import copy
import random

import cv2
import numpy as np
import tensorflow.keras as keras

from .UCF_splitting_kernel import *
from .helpers import get_training_augmenter, get_validation_augmenter


class SpatialSequence(keras.utils.Sequence):
    def __init__(self, data_to_load, data_root_path, batch_size, is_training, augmenter):
        """get data structure to load data"""
        # list of (video names,frame/max_frame,label)
        self.data_to_load = copy.deepcopy(data_to_load)
        self.batch_size = batch_size
        self.is_training = is_training

        self.augmenter = copy.deepcopy(augmenter)

        self.data_root_path = data_root_path

        self.video_names, self.frames, self.labels = [list(one_of_three_tuples) for one_of_three_tuples in zip(*self.data_to_load)]  # three lists

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return (len(self.video_names) + self.batch_size - 1) // self.batch_size  # ceiling div

    def get_actual_length(self):
        """Denotes the total number of samples"""
        return len(self.video_names)

    def __getitem__(self, batch_start):
        """Gets one batch"""
        batch_video_names = self.video_names[batch_start * self.batch_size:(batch_start + 1) * self.batch_size]
        batch_frames = self.frames[batch_start * self.batch_size:(batch_start + 1) * self.batch_size]
        batch_y = np.array(self.labels[batch_start * self.batch_size:(batch_start + 1) * self.batch_size])

        batch_x = []  # could be less or equal batch size
        #
        for vid_id, _ in enumerate(batch_y):
            if self.is_training:  # max frame is given
                frame_id = random.randint(1, batch_frames[vid_id])  # random frame (one based)
            else:
                frame_id = batch_frames[vid_id]  # just as selected

            batch_x.append(
                cv2.cvtColor(cv2.imread(os.path.join(self.data_root_path, "v_" + batch_video_names[vid_id], 'frame{}'.format(str(frame_id).zfill(6)) + '.jpg')), cv2.COLOR_BGR2RGB)
            )

        if self.is_training:
            return np.array(self.augmenter.augment_images(batch_x), dtype=np.float32) / 255.0, batch_y
        else:
            # no label needed since (test_video_to_label mapping) (dictionary of name to label) is returned
            return batch_video_names, np.array(self.augmenter.augment_images(batch_x), dtype=np.float32) / 255.0

    def shuffle_and_reset(self):
        """
        new data for the next epoch
        """
        random.shuffle(self.data_to_load)
        self.video_names, self.frames, self.labels = [list(one_of_three_tuples) for one_of_three_tuples in zip(*self.data_to_load)]  # shuffle all


class SpatialDataLoader:
    def __init__(self, batch_size, testing_samples_per_video, width, height, log_stream=open("/tmp/null.log", "w"), augmenter_level=1, data_root_path='./jpegs_256/', ucf_list_path='./UCF_list/', ucf_split='01'):
        """
        get the mapping and initialize the augmenter
        """
        self.batch_size = batch_size
        self.width, self.height = width, height
        self.data_root_path = data_root_path
        self.testing_samples_per_video = testing_samples_per_video
        self.log_stream = log_stream
        # split the training and testing videos
        data_util_ = DataUtil(path=ucf_list_path, split=ucf_split)
        self.train_video_to_label, self.test_video_to_label = data_util_.get_train_test_video_to_label_mapping()  # name without v_ or .avi and small s .. name to numeric label starts at 0

        # get video frames
        self.video_frame_count = data_util_.get_video_frame_count()  # name without v_ or .avi and small s

        self.augmenter_level = augmenter_level

    def run(self):
        """
        get the data structure for training and validation
        """
        train_loader = self.get_training_loader()
        val_loader = self.get_testing_loader()

        return train_loader, val_loader, self.test_video_to_label

    def get_training_data_structure(self):
        """
        get the data structure for training
        """
        training_data_structure = []  # list of (video names,frame/max_frame,label)
        for video_name in self.train_video_to_label:  # sample from the whole video frames
            training_data_structure.append((video_name, self.video_frame_count[video_name], self.train_video_to_label[video_name]))

        return training_data_structure

    def get_testing_data_structure(self):
        """
        get the data structure for validation
        """
        test_data_structure = []  # list of (video names,frame/max_frame,label)
        for video_name in self.test_video_to_label:
            nb_frame = self.video_frame_count[video_name]
            interval = nb_frame // self.testing_samples_per_video

            if interval == 0:  # for videos shorter than self.testing_samples_per_video
                interval = 1

            # range is exclusive add one to be inclusive
            # 1 >  self.testing_samples_per_video * interval
            for frame_idx in range(1, min(self.testing_samples_per_video * interval, nb_frame) + 1, interval):
                test_data_structure.append((video_name, frame_idx, self.test_video_to_label[video_name]))

        return test_data_structure

    def get_training_loader(self):
        """
        an instance of sequence loader for spatial model for parallel dataloading using keras sequence
        """
        loader = SpatialSequence(data_to_load=self.get_training_data_structure(),
                                 data_root_path=self.data_root_path,
                                 batch_size=self.batch_size,
                                 is_training=True,
                                 augmenter=get_training_augmenter(height=self.height, width=self.width, augmenter_level=self.augmenter_level),
                                 )

        print('==> Training data :', len(loader.data_to_load), 'videos', file=self.log_stream)
        print('==> Training data :', len(loader.data_to_load), 'videos')
        return loader

    def get_testing_loader(self):
        """
        an instance of sequence loader for spatial model for parallel dataloading using keras sequence
        """

        loader = SpatialSequence(data_to_load=self.get_testing_data_structure(),
                                 data_root_path=self.data_root_path,
                                 batch_size=self.batch_size,
                                 is_training=False,
                                 augmenter=get_validation_augmenter(height=self.height, width=self.width),
                                 )

        print('==> Validation data :', len(loader.data_to_load), 'frames', file=self.log_stream)
        print('==> Validation data :', len(loader.data_to_load), 'frames')
        return loader


if __name__ == '__main__':
    data_loader = SpatialDataLoader(batch_size=64, use_multiprocessing=True,  # data_root_path="data",
                                    ucf_split='01',
                                    testing_samples_per_video=19, width=224, height=224, num_workers=2)
    train_loader, test_loader, test_video_level_label = data_loader.run()

    print(len(train_loader))
    print(len(test_loader))

    print(train_loader.get_actual_length())
    print(test_loader.get_actual_length())

    print(train_loader.sequence[0][0].shape, train_loader.sequence[0][1].shape)
    print(train_loader[0][0].shape, train_loader[0][1].shape)
    # import tqdm
    # progress = tqdm.tqdm(train_loader.get_epoch_generator(), total=len(train_loader))

    # for (sampled_frame, label) in progress:
    #     pass

    import matplotlib.pyplot as plt


    # preview raw data
    def preview(data, labels):
        # 3 channels
        fig, axeslist = plt.subplots(ncols=8, nrows=8, figsize=(10, 10))

        for i, sample in enumerate(data):
            axeslist.ravel()[i].imshow(data[i])
            axeslist.ravel()[i].set_title(labels[i])
            axeslist.ravel()[i].set_axis_off()

        plt.subplots_adjust(wspace=.4, hspace=.4)


    print("train sample")
    for batch in train_loader.get_epoch_generator():
        print(batch[0].shape, batch[1].shape)
        print(batch[1])
        preview(batch[0], batch[1])

        break
    print("test sample")  # same name will be displayed testing_samples_per_video with no shuffling
    for batch in test_loader.get_epoch_generator():
        print(batch[1].shape, batch[2].shape)
        print(batch[0], batch[2])
        preview(batch[1], batch[2])

        break
