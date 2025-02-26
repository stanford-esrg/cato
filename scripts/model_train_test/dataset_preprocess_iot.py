import numpy as np
import jsonlines
import argparse
import pandas as pd
import os
import glob
import csv
import time

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
            print(utils.YELLOW + f"Collecting {dataset_dir}/features.jsonl" + utils.RESET)
            retina.collect_raw_dataset(pkt_depth)
        
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

def get_device_label(x, mac_to_device):
    s_mac = x['s_mac']
    d_mac = x['d_mac']
    if s_mac == 'ff:ff:ff:ff:ff:ff':
        label_mac = d_mac
    elif d_mac == 'ff:ff:ff:ff:ff:ff':
        label_mac = s_mac
    elif s_mac == '14:cc:20:51:33:ea':
        label_mac = d_mac
    elif d_mac == '14:cc:20:51:33:ea':
        label_mac = s_mac
    elif s_mac not in mac_to_device:
        label_mac = d_mac
    elif d_mac not in mac_to_device:
        label_mac = s_mac
    elif s_mac in mac_to_device and d_mac in mac_to_device:
        label_mac = s_mac
    else:
        return None
    return mac_to_device[label_mac]

def add_device_labels(df):
    device_lst = os.path.join("/path/to/labels", "List_Of_devices.txt")
    devices = pd.read_csv(device_lst, sep='\t+', engine='python')
    devices['MAC ADDRESS'] = devices['MAC ADDRESS'].str.strip()

    mac_addresses = set(df.s_mac.unique()).union(set(df.d_mac.unique()))
    mac_to_device = {}
    for m in mac_addresses:
        dev_name = devices[devices['MAC ADDRESS'] == m]['List of Devices'].values
        if len(dev_name) == 0:
            mac_to_device[m] = None
        else:
            mac_to_device[m] = dev_name[0]
    df['device_name'] = df.apply(lambda x: get_device_label(x, mac_to_device), axis=1)
    df = df.dropna()
    df = df.drop(['s_mac', 'd_mac'], axis=1)
    return df

def get_class_counts(df, n):
    class_counts = []
    for item in df['device_name'].value_counts().items():
        class_counts.append(item)
    print(f"Number of classes: {len(class_counts)}")
    return class_counts[:n]

def encode(df, class_counts):
    label_arr = np.zeros(df.shape[0])
    df['label'] = label_arr.astype(int)
    for i, (label, _cnt) in enumerate(class_counts):
        idx = df[df.device_name == label].index
        df.loc[idx, 'label'] = int(i+1)
    df = df.drop('device_name', axis=1)
    return df

def balance(df, class_counts):
    # samples_per_class = class_counts[-1][1]
    # do partial balancing
    samples_per_class = 1000
    df = df.sample(frac=1, random_state=0)
    df = df.groupby(df.label).head(samples_per_class)
    df = df.apply(pd.to_numeric, errors='raise')
    df = df.reset_index(drop=True)
    return df

def preprocess(raw_dataset):
    print(utils.CYAN + "Preprocessing dataset" + utils.RESET)
    df = raw_dataset.fillna(-1)
    df = add_device_labels(df)
    class_counts = get_class_counts(df, 100)
    df = encode(df, class_counts)
    df = balance(df, class_counts)
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
        description="Process iot data."
    )
    parser.add_argument("pkt_depth", help="Packet depth, or 'all'")
    main(parser.parse_args())