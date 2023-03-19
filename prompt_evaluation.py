import argparse
import logging

import numpy as np
import torch
from evaluate import load
import json
from numpy.random import shuffle

from utils import MODEL_CLASSES
from utils import set_seed, adjust_length_to_model, compute_bertscore
from utils import format_augmented_prompt, format_naive_prompt, generate_lyrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    bertscore = load("bertscore")
    perplexity = load("perplexity")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2-medium",
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--train_file",
        default='/home/ubuntu/DeepLyrics/train.txt',
        type=str,
        required=True,
        help="Path to file to the file that we can sample answered prompts from"
    )

    parser.add_argument(
        "--test_file",
        default='/home/ubuntu/DeepLyrics/train.txt',
        type=str,
        required=True,
        help="Path to file that we want to generate lyrics for"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="choose among naive, random, similar"
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=2.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    # iterate through each line of test file
    with open(args.test_file, 'r') as test_file:
        lines = [line.rstrip() for line in test_file]

    # prepare prompts and ground truth/labels for lyrics
    # compute generated text according to prompt techniques
    if args.mode == 'naive':
        generated_seqs, gt = [], []
        for line in lines:
            artist, genre, lyrics = line.split('||')
            # format the prompt into "Generate <genre> song lyrics by <singer>:"
            prompt_text = format_naive_prompt(artist, genre)
            gt.append(lyrics[:100])
            generated_seqs.append(
                generate_lyrics(                        
                    prompt_text, 
                    model, tokenizer, 
                    device=args.device, 
                    max_length=args.length, temperature=args.temperature, 
                    top_k=args.k, top_p=args.p, 
                    repetition_penalty=args.repetition_penalty, 
                    num_return_sequences=args.num_return_sequences,
                    stop_token=args.stop_token)[0]
                    )
    elif args.mode == 'random':
        # need to sample two answered prompts from training data

        # read training data
        with open(args.train_file, 'r') as train_file:
            train_info = [line.rstrip() for line in train_file]

        generated_seqs, gt = [], []
        seed = 0
        for line in lines:
            artist, genre, lyrics = line.split('||')

            np.random.seed(seed)
            shuffle(train_info) # shuffle
            seed += 1

            # format the prompt into 
            # "artist, genre, lyrics, artist, genre, lyrics, artist, genre..."
            sampled_lines = train_info[:2]
            prompt_text = format_augmented_prompt(sampled_lines, artist, genre)
            gt.append(lyrics[:100])
            generated_seqs.append(
                generate_lyrics(                        
                    prompt_text, 
                    model, tokenizer, 
                    device=args.device, 
                    max_length=args.length, temperature=args.temperature, 
                    top_k=args.k, top_p=args.p, 
                    repetition_penalty=args.repetition_penalty, 
                    num_return_sequences=args.num_return_sequences,
                    stop_token=args.stop_token)[0]
                    )
    elif args.mode == 'similar':
        # need to sample two similar answered prompts from training data

        # read training data
        with open(args.train_file, 'r') as train_file:
            train_info = [line.rstrip() for line in train_file]

        generated_seqs, gt = [], []
        seed = 0
        for line in lines:
            artist, genre, lyrics = line.split('||')

            np.random.seed(seed)
            shuffle(train_info) # shuffle
            seed += 1

            sampled_lines = []
            # === choose the first two from the same genre === 
            i = 0
            while len(sampled_lines) < 2 and i < len(train_info):
                _, train_genre, _ = train_info[i].split('||')
                if train_genre == genre:
                    sampled_lines.append(train_info[i])
                i += 1
            # ======
            # format the prompt into 
            # "artist, genre, lyrics, artist, genre, lyrics, artist, genre..."
            prompt_text = format_augmented_prompt(sampled_lines, artist, genre)
            gt.append(lyrics[:100])
            generated_seqs.append(
                generate_lyrics(                        
                    prompt_text, 
                    model, tokenizer, 
                    device=args.device, 
                    max_length=args.length, temperature=args.temperature, 
                    top_k=args.k, top_p=args.p, 
                    repetition_penalty=args.repetition_penalty, 
                    num_return_sequences=args.num_return_sequences,
                    stop_token=args.stop_token)[0]
                    )

    else:
        raise ValueError("mode not implemented")

    # compute perplexity
    perp_score = perplexity.compute(predictions=generated_seqs, model_id='gpt2-medium')
    print("mean perp score", perp_score['mean_perplexity'])

    with open(f"prompting/perplexity/{args.mode}.json", "a+") as outfile:
        json.dump(perp_score, outfile)

    # compute bertscore
    bertscores = {'precision': [], 'recall': [], 'f1': []}
    for pred, ref in zip(generated_seqs, gt):
        max_len = min(len(pred), len(ref))
        score = compute_bertscore([pred[:max_len]], [ref[:max_len]], bertscore)
        for key in score.keys():
            if key == 'hashcode':
                continue
            bertscores[key].append(score[key][0])
            with open(f"prompting/bertscores/{args.mode}_scores.json", "a+") as outfile:
                json.dump(score, outfile)
    for key in bertscores.keys():
        bertscores[key] = np.mean(bertscores[key])
    print(f'Average bertscore for {args.mode} is: {bertscores}')
    with open(f"prompting/bertscores/{args.mode}.json", "w+") as outfile:
        json.dump(bertscores, outfile)
    return 
        

if __name__ == "__main__":
    main()
