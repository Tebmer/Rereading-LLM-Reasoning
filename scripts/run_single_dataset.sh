set -x
set -e

data="gsm"
model="gpt-3.5-turbo-0613"
# model="gpt-4o-mini-2024-07-18"

# Baseline
python main.py --dataset $data --temperature 0.0 --acts vanilla cot --model $model 
# Re2 with read_times=2
python main.py --dataset $data --temperature 0.0 --acts vanilla cot --model $model --read_times 2