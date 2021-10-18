import logging
import os
import pickle
from collections import namedtuple

import gym
import numpy as np

from catastrophe_wrapper import *
from catastrophe_wrapper import CatastropheWrapper
from classifier_tf import (SavedCatastropheBlockerTensorflow,
                           SavedCatastropheClassifierTensorflow)
                           
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CATASTROPHE_ZONE = [[168, 195], [30, 52]]
CHICKEN_COLOR = 252
ROAD_COLOR = 142


def chicken_on_the_first_lane(cat_zone):
    cat_zone = cat_zone[:-10, 14:-1, 0]
    if (cat_zone == CHICKEN_COLOR).sum() > 17:
        return True
    return False


def car_on_the_first_lane(cat_zone):
    cat_zone = cat_zone[3:-12, 6:, 0]
    if (cat_zone != ROAD_COLOR).sum() > 40:
        return True
    return False


def chicken_at_start(cat_zone):
    cat_zone = cat_zone[18:, 14:-1]
    if (cat_zone == CHICKEN_COLOR).sum() > 17:
        return True
    return False


def is_catastrophe(observation):
    cat_zone = observation[CATASTROPHE_ZONE[0][0]:CATASTROPHE_ZONE[0][1], 
                           CATASTROPHE_ZONE[1][0]:CATASTROPHE_ZONE[1][1]]
    if chicken_on_the_first_lane(cat_zone) and car_on_the_first_lane(cat_zone):
        return True
    return False


def should_block(observation, action):
    if observation is None:
        return False
    if is_catastrophe(observation):
        return False
    
    cat_zone = observation[CATASTROPHE_ZONE[0][0]:CATASTROPHE_ZONE[0][1], 
                           CATASTROPHE_ZONE[1][0]:CATASTROPHE_ZONE[1][1]]
    if action == 2 and chicken_at_start(cat_zone) and car_on_the_first_lane(cat_zone): # UP
        return True
    return False


def print_observation(observation):
    L = []
    for row in observation:
        for value in row:
            found = False
            for i, v in enumerate(L):
                if (v == value).all():
                    found = True
                    print(i, end="")
                    break
            if not found:
                print(len(L), end="")
                L.append(value)
        print()
    for i, v in enumerate(L):
        print(i, v)


def allowed_actions_heuristic(obs):
    return [0]

class CatastropheClassifierHeuristic(object):

    def is_catastrophe(self, obs):       
        return is_catastrophe(obs)

    def is_catastrophe_with_score(self, obs):
        return is_catastrophe(obs), 0


class CatastropheBlockerHeuristic(object):

    def should_block(self, obs, action):
        return should_block(obs, action)

    def should_block_with_score(self, obs, action):
        return should_block(obs, action), 0

    def allowed_actions(self, obs):
        return allowed_actions_heuristic(obs)


class FreewayClassifierLabeller(object):
    def __init__(self):
        pass

    def label(self, features, episode):
        images = (frame.image for frame in episode.frames if frame.action is not None)
        labels = np.array([is_catastrophe(image) for image in images])
        return features, labels


class FreewayBlockerLabeller(object):
    def __init__(self, block_radius=0):
        self.block_radius = block_radius

    def label_and_build_mask(self, episode):
        is_catastrophe_array = np.array(
            [is_catastrophe(frame.image) for frame in episode.frames if frame.action is not None])

        labels = np.full(len(is_catastrophe_array), fill_value=False, dtype=np.bool)
        mask = np.full(len(is_catastrophe_array), fill_value=True, dtype=np.bool)
        
        print(len(labels), len(mask))

        for i in range(len(is_catastrophe_array)):
            if i + self.block_radius + 1 >= len(is_catastrophe_array):
                mask[i] = False
                continue
            if is_catastrophe_array[i]:
                mask[i] = False
                continue
            for j in range(self.block_radius + 1):
                if is_catastrophe_array[i + j + 1]:
                    labels[i] = True
                    break
        return labels, mask

    def label(self, features, episode):
        labels, mask = self.label_and_build_mask(episode)
        labels = labels[mask]
        for key, value in features.items():
            features[key] = features[key][mask]
            assert (len(labels) == len(features[key])), "{} {}".format(
                len(labels), len(features[key]))
        return features, labels
