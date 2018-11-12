"""
********************************
*   Created by mohammed-alaa   *
********************************
This Contains:
Helper function for data loaders and augmentation
the sequence loader class: multiprocess/multithread approach for dataloading
"""
import os

import cv2
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from tensorflow.python.keras.utils import OrderedEnqueuer


def stack_opticalflow(start_frame_index, video_name, data_root_path, stacked_frames):  # returns numpy (h,w,stacked*2) = one sample
    """
    Stacks "stacked_frames" u/v frames on a single numpy array : (h,w,stacked*2)
    """
    first_optical_frame_u = cv2.imread(os.path.join(data_root_path, "u", "v_" + video_name, 'frame{}'.format(str(start_frame_index).zfill(6)) + '.jpg'), cv2.IMREAD_GRAYSCALE)  # horizontal
    first_optical_frame_v = cv2.imread(os.path.join(data_root_path, "v", "v_" + video_name, 'frame{}'.format(str(start_frame_index).zfill(6)) + '.jpg'), cv2.IMREAD_GRAYSCALE)  # vertical

    stacked_optical_flow_sample = np.zeros(first_optical_frame_u.shape + (2 * stacked_frames,), dtype=np.uint8)  # with channel dimension of  stacked_frames(u)+ stacked_frames(v)

    stacked_optical_flow_sample[:, :, 0] = first_optical_frame_u
    stacked_optical_flow_sample[:, :, 0 + stacked_frames] = first_optical_frame_v

    for index, optical_frame_id in enumerate(range(start_frame_index + 1, start_frame_index + stacked_frames), 1):  # index starts at 1 placed after the first one
        stacked_optical_flow_sample[:, :, index] = cv2.imread(os.path.join(data_root_path, "u", "v_" + video_name, 'frame{}'.format(str(optical_frame_id).zfill(6)) + '.jpg'), cv2.IMREAD_GRAYSCALE)
        stacked_optical_flow_sample[:, :, index + stacked_frames] = cv2.imread(os.path.join(data_root_path, "v", "v_" + video_name, 'frame{}'.format(str(optical_frame_id).zfill(6)) + '.jpg'), cv2.IMREAD_GRAYSCALE)

    return stacked_optical_flow_sample


def get_noise_augmenters(augmenter_level):
    """
    Gets an augmenter object of a given level
    """
    # 0 heavy , 1 medium,2 simple
    if augmenter_level == 0:
        ####################################################### heavy augmentation #########################################################################
        return [iaa.Sometimes(0.9, iaa.Crop(
            percent=((iap.Clip(iap.Normal(0, .5), 0, .6),) * 4)  # random crops top,right,bottom,left
        )),
                # some noise
                iaa.Sometimes(0.9, [iaa.GaussianBlur(sigma=(0, 0.3)), iaa.Sharpen(alpha=(0.0, .15), lightness=(0.5, 1.5)), iaa.Emboss(alpha=(0.0, 1.0), strength=(0.1, 0.2))]),
                iaa.Sometimes(0.9, iaa.Add((-12, 12), per_channel=1))]  # rgb jittering
    elif augmenter_level == 1:
        ####################################################### medium  augmentation #######################################################################
        return [iaa.Sometimes(0.9, iaa.Crop(percent=((0.0, 0.15), (0.0, 0.15), (0.0, 0.15), (0.0, 0.15)))),  # random crops top,right,bottom,left
                # some noise
                iaa.Sometimes(0.5, [iaa.GaussianBlur(sigma=(0, 0.25)), iaa.Sharpen(alpha=(0.0, .1), lightness=(0.5, 1.25)), iaa.Emboss(alpha=(0.0, 1.0), strength=(0.05, 0.1))]),
                iaa.Sometimes(.7, iaa.Add((-10, 10), per_channel=1))]  # rgb jittering
    elif augmenter_level == 2:
        ######################################################## simple augmentation #######################################################################
        return [iaa.Sometimes(0.6, iaa.Crop(percent=((0.0, 0.1), (0.0, 0.1), (0.0, 0.1), (0.0, 0.1)))),  # random crops top,right,bottom,left
                # some noise
                iaa.Sometimes(0.35, [iaa.GaussianBlur(sigma=(0, 0.17)), iaa.Sharpen(alpha=(0.0, .07), lightness=(0.35, 1)), iaa.Emboss(alpha=(0.0, .7), strength=(0.1, 0.7))]),
                iaa.Sometimes(.45, iaa.Add((-7, 7), per_channel=1))]  # rgb jittering
        ###################################################################################################################################################


def get_validation_augmenter(height, width):
    """
    for validation we don't add any stochasticity just resize them to height*width
    """
    aug = iaa.Sequential([
        iaa.Scale({"height": height, "width": width})
    ])

    return aug


def get_training_augmenter(height, width, augmenter_level):
    """
    Get validation augmenter according to the level of stochasticity added
    """
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        *get_noise_augmenters(augmenter_level),  # noisy heavy or simple
        iaa.Scale({"height": height, "width": width})
    ], random_order=True)  # apply augmenters in random order

    return aug


class SequenceLoader:
    """
    efficient pre fetching multiprocessing wrapper
    """

    def __init__(self, sequence_class, num_workers, queue_size, use_multiprocessing, do_shuffle, **kwargs):
        self.sequence_class = sequence_class
        self.kwargs = kwargs
        self.sequence = None
        self.parallel_generator = None
        self.shuffle = do_shuffle
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.use_multiprocessing = use_multiprocessing

        self.reset()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.sequence)

    def get_actual_length(self):
        """Denotes the total number of samples"""
        return self.sequence.get_actual_length()

    def __getitem__(self, index):  # this is linear and inefficient, don't use me
        return self.sequence[index]

    def reset(self):
        self.sequence = self.sequence_class(**self.kwargs)
        enqueuer = OrderedEnqueuer(self.sequence, use_multiprocessing=self.use_multiprocessing)
        enqueuer.start(workers=self.num_workers, max_queue_size=self.queue_size)
        self.parallel_generator = enqueuer.get()

        if self.shuffle:
            self.sequence.shuffle_and_reset()

    def get_epoch_generator(self):  # you can loop on this or send it to generator based training
        for _ in range(self.__len__()):  # makes sure of length
            yield next(self.parallel_generator)

        if self.shuffle:
            self.sequence.shuffle_and_reset()

    def get_sustaining_generator(self):  # you can loop on this or send it to generator based training
        while True:  # any number of epochs
            for _ in range(self.__len__()):  # makes sure of length
                yield next(self.parallel_generator)

            if self.shuffle:
                self.sequence.shuffle_and_reset()

    def get_epochs_generator(self, epochs):  # you can loop on this or send it to generator based training
        for _ in range(epochs):  # any number of epochs
            for _ in range(self.__len__()):  # makes sure of length
                yield next(self.parallel_generator)

            if self.shuffle:
                self.sequence.shuffle_and_reset()
