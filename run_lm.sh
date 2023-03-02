##! /bin/bash

# run_generation.py
python run_language_modeling.py     --output_dir=output/first   \
 --model_type=gpt2     --model_name_or_path=gpt2    --do_train  \
    --train_data_file=train.txt     --do_eval     --eval_data_file=valid.txt   \
    --layer='first'     --overwrite_output_dir=1 --evaluation_strategy="steps" --logging_steps=50  \
    --report_to="tensorboard" --logging_dir="./tblog/first"

python run_language_modeling.py     --output_dir=output/last    \
 --model_type=gpt2     --model_name_or_path=gpt2    --do_train  \
    --train_data_file=train.txt     --do_eval     --eval_data_file=valid.txt   \
    --layer='last'     --overwrite_output_dir=1 --evaluation_strategy="steps" --logging_steps=50  \
    --report_to="tensorboard" --logging_dir="./tblog/last"

python run_language_modeling.py     --output_dir=output/middle    \
 --model_type=gpt2     --model_name_or_path=gpt2    --do_train  \
    --train_data_file=train.txt     --do_eval     --eval_data_file=valid.txt   \
    --layer='middle'     --overwrite_output_dir=1 --evaluation_strategy="steps" --logging_steps=50  \
    --report_to="tensorboard" --logging_dir="./tblog/middle"