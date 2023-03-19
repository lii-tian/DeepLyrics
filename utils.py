import argparse
import logging

import numpy as np
import torch
from evaluate import load
import json

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

MAX_LENGTH = int(1024)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def compute_bertscore(pred, ref, bertscore, lang="en"):
    return bertscore.compute(predictions=pred, references=ref, lang=lang)

def generate_lyrics(
    prompt_text, 
    model, tokenizer, 
    device, 
    max_length, temperature,
    top_k, top_p,
    repetition_penalty,
    num_return_sequences,
    stop_token):

    print('prompt text len', len(prompt_text))
    # prompt_text = tokenizer(prompt_text, truncation=True, max_length=max_length)
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=max_length)
    encoded_prompt = encoded_prompt.to(device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length + len(encoded_prompt[0]),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
        print("total_sequence", total_sequence)

    return generated_sequences

def format_naive_prompt(artist, genre):
    return f"Generate {genre} song lyrics by {artist}:"


def format_augmented_prompt(sampled_lines, artist, genre):
    res = ''
    for sample in sampled_lines:
        a, g, l = sample.split('||')
        l = l[:min(100, len(l))] # concat to the first 100 char
        res += ','.join([a, g, l])
        res += ';'
    
    res += f'{artist},{genre},:'

    return res