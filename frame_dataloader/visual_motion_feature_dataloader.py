"""
********************************
*   Created by mohammed-alaa   *
********************************
Motion visual feature Dataloader implementing sequence api from keras (defines how to load a single item)
we sample "samples_per_video" per video on equal intervals for validation or randomly for training
this loads batches of stacked images(representing a video) for each iteration it returns [samples_per_video, height, width ,stacked_frames*2] ndarrays
2*stacked_frames since we have u,v optical flow
"""
import random
import sys

import numpy as np
import tensorflow.keras as keras

from .UCF_splitting_kernel import *
from .helpers import get_training_augmenter, get_validation_augmenter, stack_opticalflow


class MotionSequenceFeature(keras.utils.Sequence):
    def __init__(self, data_to_load, data_root_path, samples_per_video, is_training, augmenter, stacked_frames):
        """get data structure to load data"""
        # list of (video names,[frame]/max_frame,label)
        self.data_to_load = data_to_load
        self.samples_per_video = samples_per_video
        self.is_training = is_training

        self.augmenter = augmenter

        self.data_root_path = data_root_path
        self.stacked_frames = stacked_frames
        self.video_names, self.frames, self.labels = [list(one_of_three_tuples) for one_of_three_tuples in zip(*self.data_to_load)]  # three lists

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.video_names)

    def get_actual_length(self):
        """Denotes the total number of samples"""
        return len(self)

    def __getitem__(self, video_index):
        """Gets one sample"""  # (samples_per_video,h,w,stacked*2)
        video_label = self.labels[video_index]
        video_name = self.video_names[video_index]

        # start index of each frame so i will stack 19 samples each sample is 20 frames stacked ._.
        if self.is_training:  # max frame is given
            video_frames_start_idx = sorted(random.sample(range(1, self.frames[video_index] + 1), self.samples_per_video))  # sample random frames (samples_per_video) and sort them
        else:
            video_frames_start_idx = self.frames[video_index]  # just as selected list of samples_per_video

        video_frames = []  # could be less or equal batch size

        for video_frame_start_idx in video_frames_start_idx:  # for each sample here
            video_frames.append(stack_opticalflow(start_frame_index=video_frame_start_idx, video_name=video_name, data_root_path=self.data_root_path, stacked_frames=self.stacked_frames))  # append one sample which is (h,w,stacked*2)

        return np.array(self.augmenter.augment_images(video_frames), dtype=np.float32) / 255.0, video_label

    def shuffle_and_reset(self):
        """
        new data for the next epoch
        """
        random.shuffle(self.data_to_load)
        self.video_names, self.frames, self.labels = [list(one_of_three_tuples) for one_of_three_tuples in zip(*self.data_to_load)]  # shuffle all


class MotionDataLoaderVisualFeature:
    def __init__(self, samples_per_video, width, height, num_workers, use_multiprocessing, augmenter_level=0, log_stream=open("/tmp/null.log", "w"), data_root_path='./tvl1_flow/', ucf_list_path='./UCF_list/', ucf_split='01', queue_size=10, stacked_frames=10):
        """
        get the mapping and initialize the augmenter
        """
        self.samples_per_video = samples_per_video
        self.use_multiprocessing = use_multiprocessing
        self.queue_size = queue_size
        self.num_workers = num_workers

        self.width, self.height = width, height
        self.stacked_frames = stacked_frames
        self.data_root_path = data_root_path
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

        return train_loader, val_loader

    def get_training_data_structure(self):
        """
        get the data structure for training
        """
        training_data_structure = []  # list of (video names,[frame]/max_frame,label)
        for video_name in self.train_video_to_label:  # sample from the whole video frames
            training_data_structure.append((video_name, self.video_frame_count[video_name] - self.stacked_frames + 1, self.train_video_to_label[video_name]))  # we need 10 frames to stack together

        return training_data_structure

    def get_testing_data_structure(self):
        """
        get the data structure for validation
        """
        test_data_structure = []  # list of (video names,[frame]/max_frame,label)
        for video_name in self.test_video_to_label:
            nb_frame = self.video_frame_count[video_name] - self.stacked_frames + 1  # we need 10 frames to stack together (this will be inclusive)
            interval = nb_frame // self.samples_per_video

            if interval == 0:  # for videos shorter than self.testing_samples_per_video
                interval = 1

            # range is exclusive add one to be inclusive
            # 1 >  self.testing_samples_per_video * interval inclusive
            sampled_frames = []
            for frame_idx in range(1, min(self.samples_per_video * interval, nb_frame) + 1, interval):
                sampled_frames.append(frame_idx)

            test_data_structure.append((video_name, sampled_frames, self.test_video_to_label[video_name]))

        return test_data_structure

    def get_training_loader(self):
        """
        an instance of sequence loader for motion model for parallel dataloading using keras sequence
        """
        loader = MotionSequenceFeature(data_to_load=self.get_training_data_structure(),
                                       data_root_path=self.data_root_path,
                                       samples_per_video=self.samples_per_video,
                                       is_training=True,
                                       augmenter=get_training_augmenter(height=self.height, width=self.width, augmenter_level=self.augmenter_level),
                                       stacked_frames=self.stacked_frames)

        print('==> Training data :', len(loader.data_to_load), 'videos', file=self.log_stream)
        print('==> Training data :', len(loader.data_to_load), 'videos')
        return loader

    def get_testing_loader(self):
        """
        an instance of sequence loader for motion model for parallel dataloading using keras sequence
        """
        loader = MotionSequenceFeature(  data_to_load=self.get_testing_data_structure(),
                                       data_root_path=self.data_root_path,
                                       samples_per_video=self.samples_per_video,
                                       is_training=False,
                                       augmenter=get_validation_augmenter(height=self.height, width=self.width),
                                       stacked_frames=self.stacked_frames
                                       )
        print('==> Validation data :', len(loader.data_to_load), 'frames', file=self.log_stream)
        print('==> Validation data :', len(loader.data_to_load), 'frames')
        return loader


if __name__ == '__main__':
    data_loader = MotionDataLoaderVisualFeature(samples_per_video=19, use_multiprocessing=True, ucf_list_path='../UCF_list/',
                                                width=224, height=224, num_workers=1, log_stream=sys.stdout, heavy=False)
    train_loader, test_loader, test_video_level_label = data_loader.run()

    print(len(train_loader))
    print(len(test_loader))

    print(train_loader.get_actual_length())
    print(test_loader.get_actual_length())

    print(train_loader[0][0].shape, train_loader[0][1])

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
            axeslist.ravel()[i].imshow(data[i], cmap='gray')
            axeslist.ravel()[i].set_title(labels[i])
            axeslist.ravel()[i].set_axis_off()

        plt.subplots_adjust(wspace=.4, hspace=.4)


    for batch in test_loader.get_epoch_generator():
        print(batch[0], batch[1].shape, batch[2].shape)
        preview(batch[1][:, :, :, 0], batch[2])
        preview(batch[1][:, :, :, 1], batch[2])
        preview(batch[1][:, :, :, 2], batch[2])
        preview(batch[1][:, :, :, 3], batch[2])

        preview(batch[1][:, :, :, 10], batch[2])
        preview(batch[1][:, :, :, 11], batch[2])
        preview(batch[1][:, :, :, 12], batch[2])
        preview(batch[1][:, :, :, 13], batch[2])
        break

    for batch in train_loader.get_epoch_generator():
        print(batch[0].shape, batch[1].shape)
        preview(batch[0][:, :, :, 0], batch[1])
        preview(batch[0][:, :, :, 1], batch[1])
        preview(batch[0][:, :, :, 2], batch[1])
        preview(batch[0][:, :, :, 3], batch[1])

        preview(batch[0][:, :, :, 10], batch[1])
        preview(batch[0][:, :, :, 11], batch[1])
        preview(batch[0][:, :, :, 12], batch[1])
        preview(batch[0][:, :, :, 13], batch[1])
        break
