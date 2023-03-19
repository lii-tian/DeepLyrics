##! /bin/bash

# run_generation.py
# python prompt_evaluation.py --model_type gpt2 --train_file /home/ubuntu/DeepLyrics/train_info.txt --test_file /home/ubuntu/DeepLyrics/test_info_baby.txt --mode naive 
# python prompt_evaluation.py --model_type gpt2 --train_file /home/ubuntu/DeepLyrics/train_info.txt --test_file /home/ubuntu/DeepLyrics/test_info_baby.txt --mode random
python prompt_evaluation.py --model_type gpt2 --train_file /home/ubuntu/DeepLyrics/train_info.txt --test_file /home/ubuntu/DeepLyrics/test_info_baby.txt --mode similar