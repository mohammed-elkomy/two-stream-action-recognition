"""
********************************
*   Created by mohammed-alaa   *
********************************
This contains :
the class responsible for train-test split(video to label mapping) given by UCF101 authors

look at the notes below
"""

import os
import pickle


class DataUtil:
    """
    Gets video name to label mapping using UCF101 splits
    """

    def __init__(self, path, split):
        self.path = path
        self.split = split

        self.action_to_label = {}
        self.init_action_to_label_mapping()

    def init_action_to_label_mapping(self):
        with open(os.path.join(self.path, 'classInd.txt')) as f:
            class_index_mapping = f.readlines()
            class_index_mapping = [x.strip('\n') for x in class_index_mapping]

        for line in class_index_mapping:
            label, action = line.split(' ')
            self.action_to_label[action] = int(label) - 1  # without v_ or avi(has HandstandPushups) # make it zero based

    def get_train_test_video_to_label_mapping(self):
        train_to_label = self.get_video_to_label_mapping("trainlist")
        test_to_label = self.get_video_to_label_mapping("testlist")

        return train_to_label, test_to_label

    def get_video_to_label_mapping(self, file):
        """warning: trainlist, testlist contains video names called v_HandStandPushups_g16_c03.avi"""
        with open(os.path.join(self.path, '{file}{split}.txt'.format(file=file, split=self.split))) as f:
            content = f.readlines()
            content = [x.strip('\n') for x in content]

        each_video_to_label = {}
        for line in content:
            video_name = line.split('/', 1)[1]  # get video name after /
            video_name = video_name.split(' ', 1)[0]  # ignore class number 0>1>..> 101 (only trainlist)
            video_name = video_name.split('_', 1)[1]  # remove v_
            video_name = video_name.split('.', 1)[0]  # remove .avi
            video_name = video_name.replace("HandStandPushups", "HandstandPushups")  # look at the warning <
            label = self.action_to_label[line.split('/')[0]]  # get label index from video_name.. [without v_ or avi get (has HandstandPushups)]
            each_video_to_label[video_name] = label  # zero based now
        return each_video_to_label

    def get_video_frame_count(self):
        with open(os.path.join(self.path, "..", "frame_dataloader/dic/frame_count.pickle"), 'rb') as file:
            old_video_frame_count = pickle.load(file)  # has HandstandPushups_g25_c01 for example (small)

        video_frame_count = {}
        for old_video_name in old_video_frame_count:
            new_video_name = old_video_name.split('_', 1)[1].split('.', 1)[0]  # remove v_ and .avi
            video_frame_count[new_video_name] = int(old_video_frame_count[old_video_name])  # name without v_ or .avi (has HandstandPushups)

        return video_frame_count


if __name__ == '__main__':
    path = '../UCF_list/'
    split = '01'
    data_util = DataUtil(path=path, split=split)
    train_video, test_video = data_util.get_train_test_video_to_label_mapping()
    print(len(train_video), len(test_video))

    frames = data_util.get_video_frame_count()

    frame_test, frame_train = {}, {}

    test, train, other = 0, 0, 0
    for key, value in frames.items():
        if key in test_video:
            test += value
            frame_test[key] = value
        elif key in train_video:
            train += value
            frame_train[key] = value
        else:
            other += value
    print(test, train, other)

    print(sum(value for key, value in frames.items()))
    print(sorted(frame_train.values())[:20])
    print(sorted(frame_test.values())[:20])

    # SequenceLoader(sequence_class=CustomSequence, queue_size=100, num_workers=4, use_multiprocessing=True, do_shuffle=True, data=list(range(5)))


"""Some Important Notes to understand the conflict between the datafolders and splitfile.txt"""
##########################
# HandstandPushups/v_HandStandPushups_g01_c01.avi (in actual data)
# HandstandPushups/v_HandStandPushups_g01_c01.avi 37 (in train list) <<<< make me small to work with the frame and processed data on disk
##########################
# v_HandstandPushups_g01_c01.avi(in frame count dict)
# HandstandPushups_g01_c01 (in valid and train dictionaries)
# v_HandstandPushups_g01_c01 (in processed data)
##########################
# Trainin: mini-batch stochastic gradient descent with momentum (set to 0.9). At each iteration, a mini-batch
# of 256 samples is constructed by sampling 256 training videos (uniformly across the classes), from
# each of which a single frame is randomly selected. In spatial net training, a 224 × 224 sub-image is
# randomly cropped from the selected frame; it then undergoes random horizontal flipping and RGB
# jittering. The videos are rescaled beforehand, so that the smallest side of the frame equals 256. We
# note that unlike [15], the sub-image is sampled from the whole frame, not just its 256 × 256 center.
# In the temporal net training, we compute an optical flow volume I for the selected training frame as
# described in Sect. 3. From that volume, a fixed-size 224 × 224 × 2L input is randomly cropped and
# flipped.
##########################
# Testing. At test time, given a video, we sample a fixed number of frames (25 in our experiments)
# with equal temporal spacing between them. From each of the frames we then obtain 10 ConvNet
# inputs [15] by cropping and flipping four corners and the center of the frame. The class scores for the
# whole video are then obtained by averaging the scores across the sampled frames and crops therein.
##########################
# v = vertical
# u = horizontal
