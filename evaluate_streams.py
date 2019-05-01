"""
********************************
*   Created by mohammed-alaa   *
********************************
Evaluate motion and spatial streams
"""
import frame_dataloader
from evaluation import legacy_load_model, get_batch_size
from evaluation.evaluation import *
from utils.drive_manager import DriveManager

"""
Evaluate spatial stream
"""
# download
drive_manager = DriveManager("spa-xception-adam-5e-06-imnet")
drive_manager.download_file('1djGzpxAYFvNX-UaQ7ONqDHGgnzc8clBK', "spatial.zip")

# load into ram
print("Spatial stream")
spatial_model_restored = legacy_load_model(filepath="spatial.h5", custom_objects={'sparse_categorical_cross_entropy_loss': sparse_categorical_cross_entropy_loss, "acc_top_1": acc_top_1, "acc_top_5": acc_top_5})
spatial_model_restored.summary()

# evaluate
_, spatial_test_loader, test_video_level_label = frame_dataloader.SpatialDataLoader(

    width=int(spatial_model_restored.inputs[0].shape[1]), height=int(spatial_model_restored.inputs[0].shape[2]), batch_size=get_batch_size(spatial_model_restored, spatial=True), testing_samples_per_video=19
).run()

video_level_loss, video_level_accuracy_1, video_level_accuracy_5, test_video_level_preds = eval_model(spatial_model_restored, spatial_test_loader, test_video_level_label, 19)
print("Spatial Model validation", "prec@1", video_level_accuracy_1, "prec@5", video_level_accuracy_5, "loss", video_level_loss)

"""
Evaluate motion stream
"""
# download
drive_manager = DriveManager("heavy-mot-xception-adam-1e-05-imnet")
drive_manager.download_file('1kvslNL8zmZYaHRmhgAM6-l_pNDDA0EKZ', "motion.zip")  # the id of the zip file contains my network

# load into ram
print("Motion stream")
motion_model_restored = legacy_load_model(filepath="motion.h5", custom_objects={'sparse_categorical_cross_entropy_loss': sparse_categorical_cross_entropy_loss, "acc_top_1": acc_top_1, "acc_top_5": acc_top_5})
motion_model_restored.summary()

# evaluate
_, motion_test_loader, test_video_level_label = frame_dataloader.MotionDataLoader(

    width=int(motion_model_restored.inputs[0].shape[1]), height=int(motion_model_restored.inputs[0].shape[2])
    ,
    batch_size=get_batch_size(motion_model_restored, spatial=True)
    , testing_samples_per_video=19).run()

video_level_loss, video_level_accuracy_1, video_level_accuracy_5, _ = eval_model(motion_model_restored, motion_test_loader, test_video_level_label, 19)

print("Motion Model validation", "prec@1", video_level_accuracy_1, "prec@5", video_level_accuracy_5, "loss", video_level_loss)
