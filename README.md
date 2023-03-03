# fine-tuning-GPT2

This repo contains the code for the Medium Article: [Fine-tuning GPT2 for Text Generation Using Pytorch](https://towardsdatascience.com/fine-tuning-gpt2-for-text-generation-using-pytorch-2ee61a4f1ba7).

The `run_language_modeling.py` and `run_generation.py` are originally from Huggingface with tiny modifications.


Hugging Face Transformer tutorial : https://github.com/lii-tian/transformers

Run script: https://huggingface.co/transformers/v2.5.0/examples.html

export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

#run genertion

python run_generation.py --model_type=gpt2 --model_name_or_path=output/test/ --length=50 --prompt="This is the lyrics of a love song by Drake: I just can't sleep tonight,"

#run ft
python run_language_modeling.py     --output_dir=output/test    \
 --model_type=gpt2     --model_name_or_path=gpt2-medium    --do_train  \
    --train_data_file=train_baby.txt     --do_eval     --eval_data_file=valid_baby.txt   \
    --layer='first'     --overwrite_output_dir=1 --evaluation_strategy="steps" --logging_steps=10  \
    --report_to="tensorboard" --logging_dir="./tblog/test"

#run tb
tensorboard --logdir './tblog'