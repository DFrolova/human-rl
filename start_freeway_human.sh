#!/bin/bash
python scripts/human_feedback.py --label_mode block --blocking_mode action_replacement --safe_action 0 -i 3.5 -f logs/freeway/episodes -o logs/freeway/label --env-id Freeway
