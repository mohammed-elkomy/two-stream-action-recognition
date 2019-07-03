"""
  Created by mohammed-alaa
"""
import pickle

import tensorflow as tf

from evaluation.evaluation import eval_model
from utils import log

"""Evaluation best value over the course of training"""
best_video_level_accuracy_1 = 0
last_video_level_loss = 5.0


def get_validation_callback(log_stream, validate_every, model, test_loader, test_video_level_label, testing_samples_per_video, log_file, pred_file, h5py_file, drive_manager):
    """
    Validation callback: keeps track of validation over the course of training done by keras
    """

    class ValidationCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs={}):
            if self.display == 0:
                metrics_log = ''
                for k in self.params['metrics']:
                    if k in logs:
                        val = logs[k]
                        if abs(val) > 1e-3:
                            metrics_log += ' - %s: %.4f' % (k, val)
                        else:
                            metrics_log += ' - %s: %.4e' % (k, val)
                print('{}/{} ... {}'.format(self.seen,
                                            self.params['samples'],
                                            metrics_log))
                print("="*50)

        def on_epoch_end(self, epoch, logs=None):
            """
            View validation metrics every "validate_every" epochs
            since training epoch is just very short compared to validation epoch >> frame level training >> video level validation
            """
            global best_video_level_accuracy_1
            global last_video_level_loss
            epoch_one_based = epoch + 1
            log("Epoch", epoch_one_based, file=log_stream)

            if epoch_one_based % validate_every == 0 and epoch_one_based > 0:
                video_level_loss, video_level_accuracy_1, video_level_accuracy_5, test_video_level_preds = eval_model(model=model,
                                                                                                                      test_loader=test_loader,
                                                                                                                      test_video_level_label=test_video_level_label,
                                                                                                                      testing_samples_per_video=testing_samples_per_video)  # 3783*(testing_samples_per_video=19)= 71877 frames of videos
                if video_level_accuracy_1 > best_video_level_accuracy_1:
                    log("Epoch", epoch_one_based, "Established new baseline:", video_level_accuracy_1, file=log_stream)
                    best_video_level_accuracy_1 = video_level_accuracy_1

                    # save the model and pickle
                    #
                else:
                    log("Epoch", epoch_one_based, "Baseline:", best_video_level_accuracy_1, "but got:", video_level_accuracy_1, file=log_stream)

                last_video_level_loss = video_level_loss

                log("=" * 100 + "\n(Training:)Epoch", epoch_one_based, "prec@1", logs["acc_top_1"], "prec@5", logs["acc_top_5"], "loss", logs["loss"], file=log_stream)
                log("(Validation:)Epoch", epoch_one_based, "prec@1", video_level_accuracy_1, "prec@5", video_level_accuracy_5, "loss", video_level_loss, file=log_stream)

                logs['val_loss'] = video_level_loss

                log_stream.flush()
                with open(pred_file, 'wb') as f:
                    pickle.dump((dict(test_video_level_preds), testing_samples_per_video), f)
                model.save(h5py_file)

                drive_manager.upload_project_files(
                    files_list=[log_file, pred_file, h5py_file],
                    snapshot_name=str(epoch_one_based) + "-" + "{0:.5f}".format(best_video_level_accuracy_1) + "-" + "{0:.5f}".format(video_level_accuracy_1))

            else:
                logs['val_loss'] = last_video_level_loss
            log_stream.flush()

    return ValidationCallback()  # returns callback instance to be consumed by keras
