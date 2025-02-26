import jsonlines
import argparse
import pandas as pd
import os
import glob
import csv
import time

from sklearn.model_selection import GroupShuffleSplit

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
            print(utils.YELLOW + f"Collecting {dataset_dir}/features.jsonl" + utils.RESET)
            retina.collect_raw_dataset(pkt_depth)
        
        data_files = glob.glob(f"{dataset_dir}/features*.jsonl")
        first_row = True
        start_ts = time.time()
        existing_session_ids = set()  # Set to track session_ids
        with open(dataset_csv, 'w', newline='') as writer:
            overall_cnt = 0
            for data_file in data_files:
                cnt = 0
                print(f"Reading {data_file}...")
                with jsonlines.open(data_file) as reader:
                    csv_writer = csv.writer(writer)
                    for obj in reader:
                        session_id = obj.get('session_id')
                        if session_id not in existing_session_ids:
                            keys = list(obj.keys())
                            if first_row:
                                csv_writer.writerow(keys)
                                first_row = False
                            csv_writer.writerow(obj[key] for key in keys)
                            existing_session_ids.add(session_id)
                            cnt += 1
                overall_cnt += cnt
        end_ts = time.time()
        print(f"Rows: {overall_cnt}, elapsed: {end_ts - start_ts}s")
    return pd.read_csv(dataset_csv)


def add_startup_delays(df):
    label_file = os.path.join("/path/to/labels", "video_labels.csv")
    startup_df = pd.read_csv(label_file)
    delay_dict = dict(zip(startup_df.session_id, startup_df.startup_time))

    df['startup_delay'] = df['session_id'].map(delay_dict)
    df = df.dropna()
    return df


def preprocess(raw_dataset):
    print(utils.CYAN + "Preprocessing dataset" + utils.RESET)
    df = raw_dataset.fillna(-1)
    df = add_startup_delays(df)
    df = df.loc[(df['startup_delay'] > 0) & (df['startup_delay'] < 20000)]
    df = df.reset_index(drop=True)
    return df

def preprocess_and_split(dataset_dir, pkt_depth):
    train_dataset_csv = os.path.join(dataset_dir, 'train_dataset.csv')
    test_dataset_csv = os.path.join(dataset_dir, 'test_dataset.csv')
    if not os.path.exists(train_dataset_csv) or not os.path.exists(test_dataset_csv):
        df = convert_raw_dataset(dataset_dir, pkt_depth)
        df = preprocess(df)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_idx, test_idx = next(gss.split(df, groups=df['session_id']))
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

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
        description="Process startup delay data."
    )
    parser.add_argument("pkt_depth", help="Packet depth, or 'all'")
    main(parser.parse_args())