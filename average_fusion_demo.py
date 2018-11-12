"""
********************************
*   Created by mohammed-alaa   *
********************************
this is a demo fusion of the output predictions of the two streams (the softmax outputs are summed and used for the final score)
those predictions are obtained from the model trained on colab
"""
import pickle

from evaluation.evaluation import video_level_eval
from frame_dataloader import DataUtil


def eval_pickles(pickle_files, weights):
    if not isinstance(pickle_files, list):
        pickle_files = [pickle_files]

    initialized = False
    test_video_level_preds = {}
    testing_samples_per_video = 0
    for index, pickle_file in enumerate(pickle_files):
        with open(pickle_file, 'rb') as f:
            test_video_level_preds_, testing_samples_per_video_ = pickle.load(f)
            if initialized:
                if testing_samples_per_video_ != testing_samples_per_video or len(test_video_level_preds) != len(test_video_level_preds_) or set(test_video_level_preds.keys()) != set(test_video_level_preds_.keys()):
                    print("Pickles doesn't match")
                    return
                else:
                    for key in test_video_level_preds:
                        test_video_level_preds[key] += weights[index] * test_video_level_preds_[key]
            else:
                initialized = True
                test_video_level_preds = test_video_level_preds_
                for key in test_video_level_preds_:
                    test_video_level_preds_[key] *= weights[index]
                testing_samples_per_video = testing_samples_per_video_

    for key in test_video_level_preds:
        test_video_level_preds[key] /= len(pickle_files)

    data_util = DataUtil(path='./UCF_list/', split='01')
    _, test_video_to_label_ = data_util.get_train_test_video_to_label_mapping()

    video_level_loss, video_level_accuracy_1, video_level_accuracy_5 = video_level_eval(test_video_level_preds=test_video_level_preds,
                                                                                        test_video_level_label=test_video_to_label_,
                                                                                        testing_samples_per_video=testing_samples_per_video)

    print("prec@1", video_level_accuracy_1, "prec@5", video_level_accuracy_5, "loss", video_level_loss)


if __name__ == '__main__':
    # Epoch 10 prec@1 0.86122125 prec@5 0.9698652 loss 0.52952474
    eval_pickles("../pickles/mot-xception-adam-5e-05-imnet-0.84140.preds", [1])
    eval_pickles("../pickles/spa-xception-adam-5e-05-imnet-0.86122.preds", [1])
    print("")
    eval_pickles("../pickles/mot-xception-adam-5e-05-imnet-0.84140.preds", [5])
    eval_pickles("../pickles/spa-xception-adam-5e-05-imnet-0.86122.preds", [5])
    print("")
    eval_pickles(["../pickles/mot-xception-adam-5e-05-imnet-0.84140.preds"] * 10, [1] * 10)
    eval_pickles(["../pickles/spa-xception-adam-5e-05-imnet-0.86122.preds"] * 10, [1] * 10)
    print("")
    eval_pickles(["../pickles/mot-xception-adam-5e-05-imnet-0.84192.preds", "../pickles/spa-xception-adam-5e-05-imnet-0.86122.preds"], [1] * 2)
    eval_pickles(["../pickles/mot-xception-adam-5e-05-imnet-0.84192.preds", "../pickles/spa-xception-adam-5e-06-imnet-0.85964.preds"], [1] * 2)
    eval_pickles(["../pickles/mot-xception-adam-5e-05-imnet-0.84192.preds", "../pickles/spa-xception-adam-5e-06-imnet-0.86016.preds"], [1] * 2)
    # eval_model_from_disk("spatial.h5", spatial=True, testing_samples_per_video=19)
    # eval_model_from_disk("motion.h5", spatial=False, testing_samples_per_video=19)
