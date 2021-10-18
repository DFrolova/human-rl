Build docker

Run docker

run vncviewer localhost:5900. There run the command:
python scripts/human_feedback.py --online --label_mode block -i 4.0 -f logs/pong/episodes -o logs/pong/label

in docker do:
cd universe_starter_agent
python train.py --num-workers=1 --online True --online_blocking_mode action_pruning --log-dir ../logs/pong && sleep 3 && tail -f ../logs/pong/logs/w-0.txt 

python train.py --num-workers=4 --env-id Pong --log-dir=../logs/pong_new/ --catastrophe_reward -1 --blocker_file ../models/pong/b2/0/final.ckpt --catastrophe_type 1 --blocking_mode action_replacement - worked









freeway
cd univer
python train.py --num-workers 4 --env-id Freeway --log-dir ../logs/freeway --catastrophe_reward 0

python scripts/human_feedback.py --label_mode block --blocking_mode action_replacement --safe_action 0 -i 3.5 -f logs/freeway/episodes -o logs/freeway/label --env-id Freeway

python train.py --num-workers 4 --env-id Freeway --log-dir ../logs/freeway_block --catastrophe_reward -1 --catastrophe_type 1 --blocking_mode action_replacement
