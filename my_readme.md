Build docker

Run docker

run vncviewer localhost:5900. There run the command:
python scripts/human_feedback.py --online --label_mode block -i 4.0 -f logs/pong/episodes -o logs/pong/label

in docker do:
cd universe_starter_agent
python train.py --num-workers=1 --online True --online_blocking_mode action_pruning --log-dir ../logs/pong && sleep 3 && tail -f ../logs/pong/logs/w-0.txt 




