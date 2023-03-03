##! /bin/bash

#add  --overwrite_output_dir=1 (if you want to overwrite a output directory)

#baby test 
python run_language_modeling.py     --output_dir=output/test    \
 --model_type=gpt2     --model_name_or_path=gpt2-medium    --do_train  \
    --train_data_file=train_baby.txt     --do_eval     --eval_data_file=valid_baby.txt   \
    --layer='first'     --overwrite_output_dir=1 --evaluation_strategy="steps" --logging_steps=10  \
    --report_to="tensorboard" --logging_dir="./tblog/test"

# ft first layer
python run_language_modeling.py     --output_dir=output/first_medium   \
 --model_type=gpt2     --model_name_or_path=gpt2-medium    --do_train  \
    --train_data_file=train.txt     --do_eval     --eval_data_file=valid.txt   \
    --layer='first' --evaluation_strategy="steps" --logging_steps=500  \
    --report_to="tensorboard" --logging_dir="./tblog/first_medium"

# ft last layer
python run_language_modeling.py     --output_dir=output/last_medium    \
 --model_type=gpt2     --model_name_or_path=gpt2-medium    --do_train  \
    --train_data_file=train.txt     --do_eval     --eval_data_file=valid.txt   \
    --layer='last'  --evaluation_strategy="steps" --logging_steps=500  \
    --report_to="tensorboard" --logging_dir="./tblog/last_medium"

# ft middle layer
python run_language_modeling.py     --output_dir=output/middle_medium    \
 --model_type=gpt2     --model_name_or_path=gpt2-medium    --do_train  \
    --train_data_file=train.txt     --do_eval     --eval_data_file=valid.txt   \
    --layer='middle'  --evaluation_strategy="steps" --logging_steps=500  \
    --report_to="tensorboard" --logging_dir="./tblog/middle_medium"