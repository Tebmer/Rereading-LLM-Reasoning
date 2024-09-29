set -x
set -e

datas=("gsm" "svamp" "asdiv" "aqua" "multiarith" "mawpssingleeq" "mawpsaddsub" "commonsenseqa" "strategyqa" "arc_easy" "arc_challenge" "date_understanding" "coin_flip")
# model="gpt-4o-mini-2024-07-18"
model="gpt-3.5-turbo-0613"

IS_MULTI_THREAD=0

if [ $IS_MULTI_THREAD -eq 1 ]; then
    echo "Using multi-thread"
    for data in ${datas[@]}; do
        python main.py --dataset $data --temperature 0.0 --acts vanilla cot --model $model --multithread --num_threads 40
        python main.py --dataset $data --temperature 0.0 --acts vanilla cot --model $model --read_times 2 --multithread --num_threads 40
    done
else
    echo "Not using multi-thread"
    for data in ${datas[@]}; do
        python main.py --dataset $data --temperature 0.0 --acts vanilla cot --model $model
        python main.py --dataset $data --temperature 0.0 --acts vanilla cot --model $model --read_times 2
    done
fi
