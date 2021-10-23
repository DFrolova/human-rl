# Human intervention reinforcement learning

Code for final project of the course Reinforcement Learning 2021 at Skoltech. It is a modification of the research code for the paper "*Trial without Error: Towards Safe Reinforcement Learning via Human Intervention*" ([arxiv](https://arxiv.org/pdf/1707.05173.pdf)) (2017)

## Overview

This repository contains the code for human intervention reinforcement learning in Atari environments (based on OpenAI's Gym). The `humanrl` package contains various Gym environment wrappers and utilities that allow modifying Atari environments to include catastrophes.

Compared to the original code of the paper, this repository has the following updates:
- A lot of updates to docker image in order to run it.
- Modification of `humanrl/pong_catastrophe.py` file for a two-sided symmetric catastrophe for Pong environment.
- New code for FreeWay Atari environment with modification of several files. The main are `humanrl/freeway_catastrophe.py` and `universe_starter_agent/envs.py`.

`scripts/human_feedback.py` is a script that allows a human to intervene during offline or online training of an RL agent.

## Installation and use

Firstly, you need to prepare docker container. To use, first install docker: https://docs.docker.com/engine/installation/

To build and start the docker image:

```
docker build -t base_new -f base.docker .
docker build -t main -f main.docker .
```

On Ubuntu:
```
docker run --name=human-rl -t -i -v /var/run/docker.sock:/var/run/docker.sock --net=host -v `pwd`:/mnt/human-rl/ main
```

On OS X (works on 10.12.2):
```
docker run --privileged -p 5901:5900 -v /usr/bin/docker:/user/bin/docker -v /var/run/docker.sock:/var/run/docker.sock -v `pwd`:/mnt/human-rl -e DOCKER_NET_HOST=172.17.0.1 -t -i main
open vnc://localhost:5901
```

Which launches a command line version of the docker container

and to restart the docker container later:

`docker start human-rl`

`docker attach human-rl`

(Note: the -v /var/run/docker.sock:/var/run/docker.sock --net=host options are necessary to allow the universe to use automatic remotes. This may not work outside of ubuntu. In this case, you may need to manually start universe remotes and point openai gym at them, see https://github.com/openai/universe/blob/master/doc/remotes.rst#how-to-start-a-remote)

It also opens a vnc server on port 5900. To view gym environments, you can run the training from the vnc session. (password is openai)
To attach to a vnc session, you need to install vncviewer. I used TigerVNC:

`apt install tigervnc-viewer`

Then you may attach to vnc session:

`vncviewer localhost:5900` (password is openai).

There you can view your training and label episodes in order to train blocker.

## Training

The following pipeline should be done one-by-one.

### No penalties/blocking (just saving frames)
To run A3C without any catastrophe penalties/blocking:

```
cd universe_starter_agent
python train.py --num-workers 4 --env-id Pong --log-dir $log_dir --catastrophe_reward 0
```

The script `train.py` starts the workers (and is not modified from the original). This calls `worker.py` which creates a gym env and runs A3C on the env. The script `envs.py` is where the env is constructed and where catastrophe wrappers are added. In the above command (where we didn't set a `catastrophe_type` argument), the only wrapper used is `frame.FrameSaveWrapper`, which just saves the frames.

Note: in `envs.py`, the function `make_env` converts 'Pong' to 'PongDeterministic-v3' and does the same for the other games. The deterministic versions are easier.

### Label frames
It is in case you use human labelling.
Start vncviewer:
`vncviewer localhost:5900` (password is openai).

Start labeling mode:
`python scripts/human_feedback.py --label_mode block -i 2 -f $logdir/episodes -o $logdir/labels`

Then you need to press 'b' to block action.

### Train classifier and blocker
Using classifier and blocker heuristics. You can modify code to use human-labelled data as training data.

```
python train/pong_classifier1.py
python train/pong_blocker1.py
```

### Penalties using catastrophe labeller
Penalties can be provided either by a hand-coded labeller or a TF classifier. For the Pong trained classifiers for blocker and catastrophe classifier we use:

```
cd universe_starter_agent
python train.py --num-workers=4 --env-id Freeway --log-dir $log_dir --catastrophe_reward -1 --blocker_file $blocker_file --classifier_file $classifier_file --catastrophe_type 1 --blocking_mode action_replacement
```

## Other info

See [the human feedback README](https://github.com/gsastry/human-rl/tree/master/scripts/README.md) for directions on providing human feedback with the OpenAI universe starter agent.

See the [catastrophe wrapper](https://github.com/gsastry/human-rl/blob/master/humanrl/catastrophe_wrapper.py) for a general purpose way to add catastrophes to Gym environments.
