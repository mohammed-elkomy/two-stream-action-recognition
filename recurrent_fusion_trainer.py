"""
********************************
*   Created by mohammed-alaa   *
********************************
Here I'm training video level network based on recurrent networks(frames from CNN are concatenated into a 3d tensor and feed to RNN):
1. setting configs (considering concatenation will have feature of 4096 = 2048 *2)
2. generating experiment_identifier and creating files
3. downloading pickled data from drive each pickled file is a big numpy array whose shape is [instances, samples per video,features(2048 or 4096) ]
"""
import glob
import pickle
import random
import shutil

from tensorflow.keras import backend as K
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Softmax, GRU

import utils.training_utils as eval_globals
from evaluation.evaluation import *
from utils import log
from utils.drive_manager import DriveManager

################################################################################
"""Configs"""
lr = 1e-6
hidden_state = 128
feature_field = 2048
testing_samples_per_video = 19
epochs = 1200
save_every = 25
batch_size = 64

num_training_samples = 9537
num_testing_samples = 3783

is_spatial = True
is_motion = True

if is_spatial and is_motion:
    feature_field *= 2
################################################################################
"""Files, paths & identifier"""
suffix = ""  # put your name or anything :D
experiment_identifier = suffix + "recurrent_fusion_selu_atten_simple" + str(lr)
################
log_file = experiment_identifier + ".log"
log_stream = open(log_file, "a")
checkpoint_dir = "./fusion/"
checkpoints = checkpoint_dir + "fusion_chk"
try:
    shutil.rmtree(checkpoint_dir)
except:
    pass
drive_manager = DriveManager(experiment_identifier)
checkpoint_found, zip_file_name = drive_manager.get_latest_snapshot()
################################################################################
"""sanity check"""
if not is_motion and not is_spatial:
    exit()
################################################################################
"""Downloads the files and makes sure files aren't re-downloaded with every run if no one is missed"""
if is_spatial:
    drive_manager_spatial = DriveManager("spatial_feature_dataset")
    test_spatial = drive_manager_spatial.search_file("test_features_spatial.pickle")
    train_spatial = drive_manager_spatial.search_file("train_features_spatial.pickle")

    if len(test_spatial) == 0:
        print("Please run 'generate_spatial_feature_dataset.py' and generate 'test_features_spatial.pickle'..this file will be saved to your drive in '<YOUR FOLDER:{}>/spatial_feature_dataset'".format(drive_manager_spatial.personal_dfolder))
        exit()

    if len(train_spatial) == 0:
        print("Please run 'generate_spatial_feature_dataset.py' and generate 'train_features_spatial.pickle'..those files will be saved to your drive in '<YOUR FOLDER:{}>/spatial_feature_dataset'".format(drive_manager_spatial.personal_dfolder))
        exit()

    drive_manager_spatial.download_file(test_spatial[0]["id"], "test_features_spatial.pickle", unzip=False)

    if len(glob.glob("train_features_spatial.pickle*")) != len(train_spatial):
        drive_manager_spatial.download_files_list(train_spatial, False, False)

if is_motion:
    drive_manager_motion = DriveManager("motion_feature_dataset")

    test_motion = drive_manager_motion.search_file("test_features_motion.pickle")
    train_motion = drive_manager_motion.search_file("train_features_motion.pickle")

    if len(test_motion) == 0:
        print("Please run 'generate_motion_feature_dataset.py' and generate 'test_features_motion.pickle'..this file will be saved to your drive in '<YOUR FOLDER:{}>/motion_feature_dataset'".format(drive_manager_motion.personal_dfolder))
        exit()

    if len(train_motion) == 0:
        print("Please run 'generate_motion_feature_dataset.py' and generate 'train_features_motion.pickle'..those files will be saved to your drive in '<YOUR FOLDER:{}>/motion_feature_dataset'".format(drive_manager_motion.personal_dfolder))
        exit()

    drive_manager_motion.download_file(test_motion[0]["id"], "test_features_motion.pickle", unzip=False)

    if len(glob.glob("train_features_motion.pickle*")) != len(train_motion):
        drive_manager_motion.download_files_list(train_motion, False, False)
################################################################################
seen_spatial_files = set()
seen_motion_files = set()


def train_generator():
    while True:
        train_samples_spatial, train_labels_spatial, train_samples_motion, train_labels_motion = [0] * 4
        """Choose file to read while being downloaded then read files"""

        """load spatial data"""
        if is_spatial:
            spatial_features_files = glob.glob("train_features_spatial.pickle*")
            if len(spatial_features_files) == len(seen_spatial_files):
                seen_spatial_files.clear()

            while True:
                spatial_features_file = random.sample(spatial_features_files, k=1)[0]
                if spatial_features_file not in seen_spatial_files:

                    try:
                        with open(spatial_features_file, 'rb') as f:
                            train_samples_spatial, train_labels_spatial = pickle.load(f)

                        # print("chose:", spatial_features_file)
                        seen_spatial_files.add(spatial_features_file)
                        break
                    except:
                        pass

        """load motion data"""
        if is_motion:
            motion_features_files = glob.glob("train_features_motion.pickle*")
            if len(motion_features_files) == len(seen_motion_files):
                seen_motion_files.clear()

            while True:
                motion_features_file = random.sample(motion_features_files, k=1)[0]
                if motion_features_file not in seen_motion_files:

                    try:
                        with open(motion_features_file, 'rb') as f:
                            train_samples_motion, train_labels_motion = pickle.load(f)

                        # print("chose:", motion_features_file)
                        seen_motion_files.add(motion_features_file)
                        break
                    except:
                        pass

        """generation loop"""
        permutation = list(range((num_training_samples + batch_size - 1) // batch_size))
        random.shuffle(permutation)

        if is_spatial != is_motion:  # xor
            # single stream motion or spatial
            if is_spatial:
                train_samples, train_labels = train_samples_spatial, train_labels_spatial
                assert train_samples_spatial.shape[0] == num_training_samples
            else:
                train_samples, train_labels = train_samples_motion, train_labels_motion
                assert train_samples_motion.shape[0] == num_training_samples

            for batch_index in permutation:
                yield train_samples[batch_index * batch_size:(batch_index + 1) * batch_size], train_labels[batch_index * batch_size:(batch_index + 1) * batch_size]
        else:
            # concatenate samples from motion and spatial
            assert np.allclose(train_labels_spatial, train_labels_motion)
            assert train_samples_spatial.shape[0] == num_training_samples
            assert train_samples_motion.shape[0] == num_training_samples

            for batch_index in permutation:
                yield np.concatenate([train_samples_spatial[batch_index * batch_size:(batch_index + 1) * batch_size], train_samples_motion[batch_index * batch_size:(batch_index + 1) * batch_size]], axis=2), train_labels_spatial[batch_index * batch_size:(batch_index + 1) * batch_size]


def test_generator():
    """load spatial test data"""
    if is_spatial:
        with open("test_features_spatial.pickle", 'rb') as f:
            test_samples_spatial, test_labels_spatial = pickle.load(f)

    """load motion test data"""
    if is_motion:
        with open("test_features_motion.pickle", 'rb') as f:
            test_samples_motion, test_labels_motion = pickle.load(f)

    while True:
        if is_spatial != is_motion:  # xor
            # single stream motion or spatial
            if is_spatial:
                # noinspection PyUnboundLocalVariable
                test_samples, test_labels = test_samples_spatial, test_labels_spatial
                assert test_samples_spatial.shape[0] == num_testing_samples
            else:
                # noinspection PyUnboundLocalVariable
                test_samples, test_labels = test_samples_motion, test_labels_motion
                assert test_samples_motion.shape[0] == num_testing_samples

            for batch_index in range((test_samples.shape[0] + batch_size - 1) // batch_size):
                yield test_samples[batch_index * batch_size:(batch_index + 1) * batch_size], test_labels[batch_index * batch_size:(batch_index + 1) * batch_size]

        else:
            # concatenate samples from motion and spatial
            assert np.allclose(test_labels_motion, test_labels_spatial)
            assert test_samples_spatial.shape[0] == num_testing_samples
            assert test_samples_motion.shape[0] == num_testing_samples

            for batch_index in range((num_testing_samples + batch_size - 1) // batch_size):
                yield np.concatenate([test_samples_spatial[batch_index * batch_size:(batch_index + 1) * batch_size], test_samples_motion[batch_index * batch_size:(batch_index + 1) * batch_size]], axis=2), test_labels_spatial[batch_index * batch_size:(batch_index + 1) * batch_size]


class saver_callback(tf.keras.callbacks.Callback):
    """
    save checkpoint with tensorflow saver not h5py since my model implementation is supclass api not function >> function implementation is left as TODO
    also logging model state and uploading the file
    """

    def on_epoch_end(self, epoch, logs={}):
        epoch_one_based = epoch + 1
        if epoch_one_based % save_every == 0 and epoch_one_based > 0:
            log("=" * 100 + "\n(Training:)Epoch", epoch_one_based, "prec@1", logs["acc_top_1"], "prec@5", logs["acc_top_5"], "loss", logs["loss"], file=log_stream)
            log("(Validation:)Epoch", epoch_one_based, "prec@1", logs["val_acc_top_1"], "prec@5", logs["val_acc_top_5"], "loss", logs["val_loss"], file=log_stream)

            if logs["val_acc_top_1"] > eval_globals.best_video_level_accuracy_1:
                log("Epoch", epoch_one_based, "Established new baseline:", logs["val_acc_top_1"], file=log_stream)
                eval_globals.best_video_level_accuracy_1 = logs["val_acc_top_1"]

                # save the model and pickle
                #
            else:
                log("Epoch", epoch_one_based, "Baseline:", eval_globals.best_video_level_accuracy_1, "but got:", logs["val_acc_top_1"], file=log_stream)

            saver.save(tf.keras.backend.get_session(), checkpoints)

            drive_manager.upload_project_files(
                files_list=[log_file],
                dir_list=[checkpoint_dir],
                snapshot_name=str(epoch_one_based) + "-" + "{0:.5f}".format(eval_globals.best_video_level_accuracy_1) + "-" + "{0:.5f}".format(logs["val_acc_top_1"]))


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.gru_1 = GRU(hidden_state, return_sequences=True, input_shape=(testing_samples_per_video, feature_field), dropout=.5)  # recurrent layer
        # self.gru_2 = GRU(hidden_state, return_sequences=True)

        self.attention_layer = Dense(1)  # gets attention weight for time step
        self.attention_normalizer = Softmax(axis=1)  # normalizes the 3d tensor to give weight for each time step

        self.FC_1 = Dense(hidden_state // 2, activation='selu')
        # recurrent_fusion_model.add(BatchNormalization())
        # self.FC_2 = Dense(hidden_state // 4, activation='selu')
        # self.BN_1 = BatchNormalization()
        self.classification_layer = Dense(num_actions, activation='softmax')

    def call(self, input_visual_feature, training=None, mask=None):
        internal = self.gru_1(input_visual_feature)  # returns a sequence of vectors of dimension feature_field
        # in self attention i will return_sequences of course
        # internal = self.gru_2(internal)  # returns a sequence of vectors of dimension feature_field

        un_normalized_attention_weights = self.attention_layer(internal)
        normalized_attention_weights = self.attention_normalizer(un_normalized_attention_weights)  # normalize on timesteps dimension
        internal = normalized_attention_weights * internal
        print(internal)
        attention_vector = K.sum(internal, axis=1)  # sum on timesteps
        print(attention_vector)
        # recurrent_fusion_model.add(Dense(hidden_state // 2, activation='relu'))
        # recurrent_fusion_model.add(BatchNormalization())
        internal = self.FC_1(attention_vector)
        # internal = self.FC_2(internal)
        final_output = self.classification_layer(internal)

        return final_output


# create the model
recurrent_fusion_model = Model()
recurrent_fusion_model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss=sparse_categorical_cross_entropy_loss, metrics=[acc_top_1, acc_top_5])

# build internal tensors
recurrent_fusion_model.fit(*next(train_generator()), batch_size=1, epochs=1, verbose=0)

# get tensorflow saver ready > will be used if a checkpoint found on drive
saver = tf.train.Saver(recurrent_fusion_model.variables)

if checkpoint_found:
    # restore the model from the checkpoint
    log("Model restored")
    eval_globals.best_video_level_accuracy_1 = float(zip_file_name.split("-")[1])
    log("Current Best", eval_globals.best_video_level_accuracy_1)

    saver.restore(tf.keras.backend.get_session(), checkpoints)  # use tensorflow saver
    initial_epoch = int(zip_file_name.split("-")[0])  # get epoch number
else:
    # init the model from scratch, it's already done
    log("Starting from scratch")
    # expected input data shape: (batch_size, timesteps, data_dim)
    recurrent_fusion_model.summary()
    initial_epoch = 0

# training
recurrent_fusion_model.fit_generator(train_generator(), use_multiprocessing=False,
                                     epochs=epochs, steps_per_epoch=(num_training_samples + batch_size - 1) // batch_size,
                                     validation_data=test_generator(), validation_steps=(num_testing_samples + batch_size - 1) // batch_size,
                                     callbacks=[saver_callback(), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=50, verbose=1, min_lr=lr / 10)],
                                     initial_epoch=initial_epoch)
