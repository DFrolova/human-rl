import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__),"humanrl"))

if __name__  == "__main__":

    import argparse

    import numpy as np

    from humanrl import pong_catastrophe
    from humanrl.classifier_tf import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="")


    episode_paths = frame.episode_paths("logs/pong/episodes")
    np.random.seed(seed=42)
    np.random.shuffle(episode_paths)
    data_loader = DataLoader(pong_catastrophe.PongBlockerLabeller())
    datasets = data_loader.split_episodes(episode_paths, 1, 1, 1)
    common_hparams = dict(use_action=True, expected_positive_weight=0.05)


    args = parser.parse_args()
    if args.logdir == "":
        logdir = get_unused_logdir("models/tmp/pong1blocker")
    else:
        logdir = args.logdir
    hparams_list = [
#         dict(batch_size=64, positive_weight_target=0.5, expected_positive_weight=0.05),
#         dict(batch_size=64, positive_weight_target=0.9, expected_positive_weight=0.05),
#         dict(batch_size=64, positive_weight_target=0.5, label_smoothing=True, expected_positive_weight=0.05),
#         dict(batch_size=64, positive_weight_target=0.1, expected_positive_weight=0.05),
        dict(batch_size=128),
        # dict(batch_size=1024),
        # dict(convolution2d_stack_args=[(16, [3, 3], [2, 2])] * 5,
        #      fully_connected_stack_args=[20, 20], keep_prob=0.5),

        # dict(image_crop_region=((34,34+160),(0,160)), batch_size=1),
        # dict(convolution2d_stack_args=[(4, [3, 3], [2, 2])] * 5, batch_size=1),
        # dict(image_crop_region=((34,34+160),(0,160)), convolution2d_stack_args=[(4, [3, 3], [2, 2])] * 5, batch_size=1),
        # dict(batch_size=5),
        # dict(image_crop_region=((34,34+160),(0,160)), batch_size=5),
        # dict(convolution2d_stack_args=[(4, [3, 3], [2, 2])] * 5, batch_size=5),
        # dict(image_crop_region=((34,34+160),(0,160)), convolution2d_stack_args=[(4, [3, 3], [2, 2])] * 5, batch_size=5),
    ]

    run_experiments(logdir, data_loader, datasets, common_hparams, hparams_list, steps=2000, log_every=50)
