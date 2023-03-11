import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def format_lyrics_info(x):
    return '||'.join([x['artist'], x['genre'], x['clean_lyrics']])


if __name__ == '__main__':
    df = pd.read_csv('/home/ubuntu/DeepLyrics/processed_lyrics.csv')
    df['info'] = df.apply(format_lyrics_info, axis=1)

    train_test_ratio = 0.9
    train_valid_ratio = 7/9
    df_full_train, df_test = train_test_split(df, train_size = train_test_ratio, random_state = 1)
    df_train, df_valid = train_test_split(df_full_train, train_size = train_valid_ratio, random_state = 1)

    with open('train_info.txt', 'w+') as train:
        train.write('\n'.join(list(df_train['info'])))
    with open('valid_info.txt', 'w+') as valid:
        valid.write('\n'.join(list(df_valid['info'])))
    with open('test_info.txt', 'w+') as test:
        test.write('\n'.join(list(df_test['info'])))
    with open('test_info_baby.txt', 'w+') as test:
        test.write('\n'.join(list(df_test['info'])[:10]))

    
