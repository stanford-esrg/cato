import jsonlines
import argparse
import pandas as pd
import os
import glob
import csv
import time
import toml
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from helper import consts
from helper import utils
from measure import retina

def convert_raw_dataset(dataset_dir, pkt_depth="all"):
    """
    Convert raw dataset from Retina output files to CSV.
    :param dataset_dir: str directory containing files of raw collected features*.jsonl datasets
    :param pkt_depth: packet depth to collect at in case raw features need collecting
    :return: DataFrame containing raw dataset concatenated together
    """
    dataset_csv = os.path.join(dataset_dir, 'dataset.csv')
    if not os.path.exists(dataset_csv):
        print(utils.CYAN + "Converting raw features" + utils.RESET)
        data_files = glob.glob(f"{dataset_dir}/features*.jsonl")
        if not data_files:
            # no raw features collected, go collect using Retina
            print(utils.YELLOW + f"Collecting {dataset_dir}/features_a.jsonl" + utils.RESET)
            retina.collect_raw_dataset(pkt_depth, outfile_name="features_a.jsonl", duration=30, buckets=64, filter=r"ipv4 and tcp and tls")

            print(utils.YELLOW + f"Collecting {dataset_dir}/features_b.jsonl" + utils.RESET)
            retina.collect_raw_dataset(pkt_depth, outfile_name="features_b.jsonl", duration=300, buckets=256, filter=r"ipv4 and tcp and tls and (tls.sni ~ 'zoom\\.us' or tls.sni ~ 'nflxvideo\\.net' or tls.sni ~ 'ttvnw\\.net' or tls.sni ~ 'teams\\.microsoft\\.com' or tls.sni ~ 'facebook\\.com' or tls.sni ~ 'fbcdn\\.net' or tls.sni ~ 'twitter\\.com' or tls.sni ~ 'twimg\\.com')")
        
        data_files = glob.glob(f"{dataset_dir}/features*.jsonl")
        first_row = True
        start_ts = time.time()
        with open(dataset_csv, 'w', newline='') as writer:
            overall_cnt = 0
            for data_file in data_files:
                cnt = 0
                print(f"Reading {data_file}...")
                with jsonlines.open(data_file) as reader:
                    csv_writer = csv.writer(writer)
                    for obj in reader:
                        keys = list(obj.keys())
                        if first_row:
                            csv_writer.writerow(keys)
                            first_row = False
                        csv_writer.writerow(obj[key] for key in keys)
                        cnt += 1
                overall_cnt += cnt
        end_ts = time.time()
        print(f"Rows: {overall_cnt}, elapsed: {end_ts - start_ts}s")
    return pd.read_csv(dataset_csv)

def assign_label(x):
    sni = x["sni"]
    app_lst = toml.load("app_labels.toml")
    for app_label, sni_patterns in app_lst['app_labels'].items():
        regex_patterns = [re.escape(pattern).replace(r'\*', r'.*') for pattern in sni_patterns]
        matches = any(re.match(pattern, sni) for pattern in regex_patterns)
        if matches:
            return app_label
    return "other"

def add_labels(df):
    df['label_name'] = df.apply(lambda x: assign_label(x), axis=1)
    df = df.dropna()
    df = df.drop('sni', axis=1)
    print(df['label_name'].value_counts())
    return df

def encode(df):
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label_name'])
    return df

def preprocess(raw_dataset):
    print(utils.CYAN + "Preprocessing dataset" + utils.RESET)
    raw_dataset['sni'].fillna("", inplace=True)
    df = raw_dataset.fillna(-1)
    df = add_labels(df)
    df = encode(df)
    return df

def preprocess_and_split(dataset_dir, pkt_depth):
    train_dataset_csv = os.path.join(dataset_dir, 'train_dataset.csv')
    test_dataset_csv = os.path.join(dataset_dir, 'test_dataset.csv')
    if not os.path.exists(train_dataset_csv) or not os.path.exists(test_dataset_csv):
        df = convert_raw_dataset(dataset_dir, pkt_depth)
        df = preprocess(df)
        df_train, df_test = train_test_split(
            df, 
            test_size=0.2, 
            random_state=0, 
            shuffle=True,
            stratify=df['label']
        )
        df_train.to_csv(train_dataset_csv, index=False)
        df_test.to_csv(test_dataset_csv, index=False)
        return df_train, df_test
    return pd.read_csv(train_dataset_csv), pd.read_csv(test_dataset_csv)

def main(args):
    pkt_depth = args.pkt_depth
    dataset_dir = os.path.join(consts.dataset_dir, f"pkts_{pkt_depth}")
    df_train, df_test = preprocess_and_split(dataset_dir, pkt_depth)
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process live traffic data."
    )
    parser.add_argument("pkt_depth", help="Packet depth, or 'all'")
    main(parser.parse_args())