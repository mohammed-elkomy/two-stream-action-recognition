"""
********************************
*   Created by mohammed-alaa   *
********************************
Configs for motion trainer
comment/uncomment one of these blocks
this includes: pretrained and from scratch resnet/xception/vgg19/mobile net hyper parameters
"""
###############################################################################
""" medium,adam,pretrained,5e-5,resnet  """
# is_adam = True
# pretrained = True
# testing_samples_per_video = 19
# lr = 5e-5
# model_name = "resnet"  # resnet xception vgg mobilenet
# epochs = 100
# validate_every = 5
# stacked_frames = 10
# augmenter_level = 1  # 0 heavy , 1 medium,2 simple
###############################################################################
""" medium,sgd,pretrained,5e-5,resnet  """
# is_adam = False
# pretrained = True
# testing_samples_per_video = 19
# lr = 5e-5
# model_name = "resnet"  # resnet xception vgg mobilenet
# epochs = 100
# validate_every = 5
# stacked_frames = 10
# augmenter_level = 1  # 0 heavy , 1 medium,2 simple
###############################################################################
""" medium,adam,scratch,5e-5,resnet  """
# is_adam = True
# pretrained = False
# testing_samples_per_video = 19
# lr = 5e-5
# model_name = "resnet"  # resnet xception vgg mobilenet
# epochs = 100
# validate_every = 5
# stacked_frames = 10
# augmenter_level = 1  # 0 heavy , 1 medium,2 simple
###############################################################################
""" medium,adam,pretrained,5e-5,xception """
# is_adam = True
# pretrained = True
# testing_samples_per_video = 19
# lr = 5e-5
# model_name = "xception"  # resnet xception vgg mobilenet
# epochs = 200
# validate_every = 5
# stacked_frames = 10
# augmenter_level = 1  # 0 heavy , 1 medium,2 simple
###############################################################################
""" medium,sgd,pretrained,5e-5,xception """
# is_adam = False
# pretrained = True
# testing_samples_per_video = 19
# lr = 5e-5
# model_name = "xception"  # resnet xception vgg mobilenet
# epochs = 100
# validate_every = 5
# stacked_frames = 10
# augmenter_level = 1  # 0 heavy , 1 medium,2 simple
###############################################################################
""" medium,adam,pretrained,5e-6,xception"""
# is_adam = True
# pretrained = True
# testing_samples_per_video = 19
# lr = 5e-6
# model_name = "xception"  # resnet xception vgg mobilenet
# epochs = 350
# validate_every = 5
# stacked_frames = 10
# augmenter_level = 1  # 0 heavy , 1 medium,2 simple
###############################################################################
""" heavy,adam,pretrained,10e-6,xception"""
is_adam = True
pretrained = True
testing_samples_per_video = 19
lr = 10e-6
model_name = "xception"  # resnet xception vgg mobilenet
epochs = 350
validate_every = 1
stacked_frames = 10
augmenter_level = 0  # 0 heavy , 1 medium,2 simple
###############################################################################
""" medium,sgd,pretrained,5e-6,xception"""
# is_adam = False
# pretrained = True
# testing_samples_per_video = 19
# lr = 5e-6
# model_name = "xception"  # resnet xception vgg mobilenet
# epochs = 100
# validate_every = 5
# stacked_frames = 10
# augmenter_level = 1  # 0 heavy , 1 medium,2 simple
