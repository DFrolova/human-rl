import argparse
import os
import shutil
import sys

import numpy as np

from build_feature_files import build_feature_files
from humanrl import frame
from humanrl.classifier_tf import (DataLoader, TensorflowClassifierHparams,
                                   get_unused_logdir, run_experiments, 
                                   HumanOfflineBlockerLabeller)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="")
    parser.add_argument('--input_dir', type=str, default="logs/freeway/episodes")
    parser.add_argument('--labels_dir', type=str, default="logs/freeway/final_labels")
    parser.add_argument('--blocker_dir', type=str, default="logs/freeway/label")
    # parser.add_argument('--regenerate_feature_files', action="store_true")
    parser.add_argument('--steps', type=int, default=20000)
    # parser.add_argument('--block_radius', type=int, default=1)
    args = parser.parse_args()

    common_hparams = dict(
        use_action=True,
        batch_size=1,
        input_processes=1,
        image_shape=[210, 160, 3],
        expected_positive_weight=0.013,
        image_crop_region=((0, 210), (0, 160), ))

    num_episodes = 2000
    negative_example_keep_prob = 0.1
    # if args.regenerate_feature_files and os.path.exists(args.labels_dir):
    #     shutil.rmtree(args.labels_dir, ignore_errors=True)

    if not os.path.exists(args.labels_dir):
        print("Writing feature files")
        data_loader = DataLoader(
            hparams=TensorflowClassifierHparams(**common_hparams),
            labeller=HumanOfflineBlockerLabeller())
        label_counts = build_feature_files(args.input_dir, args.labels_dir, data_loader,
                                           num_episodes, negative_example_keep_prob)
    paths = frame.feature_file_paths(args.labels_dir)

    assert len(paths) > 0, "assert len(paths) > 0, {}, {}".format(len(paths), num_episodes)

    data_loader = DataLoader(hparams=TensorflowClassifierHparams(**common_hparams))
    datasets = data_loader.split_episodes(paths, 1, 1, 0, use_all=True, seed=42)

    if args.logdir == "":
        logdir = get_unused_logdir("logs/freeway/blocker/")
    else:
        logdir = args.logdir
    hparams_list = [
        # dict(),
        # dict(label_smoothing=True),
        dict(
            convolution2d_stack_args=[(16, [3, 3], [2, 2])] * 6,
            fully_connected_stack_args=[20, 20],
            keep_prob=0.5,
            positive_weight_target=0.5),
    ]

    run_experiments(
        logdir,
        data_loader,
        datasets,
        common_hparams,
        hparams_list,
        steps=args.steps,
        log_every=100,
        predict_episodes=False)
