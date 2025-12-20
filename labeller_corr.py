import pandas as pd
import numpy as np
import os
from tqdm import tqdm

raw_dt = 'raw_data'
old_dt = 'images'
new_dt = 'images_labelled_properly'
wsize = 12

os.makedirs(new_dt)
price_dt = {}
avg_rt_dt = {}

csv_files = sorted([f for f in os.listdir(raw_dt)])
for csv_file in csv_files:
    coin_name = csv_file.split('_')[0]
    df = pd.read_csv(os.path.join(raw_dt, csv_file))
    close_prices = df['close'].values
    price_dt[coin_name] = close_prices
    avg_returns = []
    
    for i in range(len(close_prices)):
        current_price = close_prices[i]
        future_rt = []
        for step in range(1, 6):
            if i + step < len(close_prices):
                future_price = close_prices[i + step]
                pct_change = (future_price - current_price) / current_price * 100
                future_rt.append(pct_change)
        if len(future_rt) > 0:
            avg_return = np.mean(future_rt)
        else:
            avg_return = np.nan
        avg_returns.append(avg_return)
    
    avg_rt_dt[coin_name] = np.array(avg_returns)
    
    valid_rets = [r for r in avg_returns if not np.isnan(r)]
all_rets = []
for coin_name in avg_rt_dt.keys():
    valid_rets = [r for r in avg_rt_dt[coin_name] if not np.isnan(r)]
    all_rets.extend(valid_rets)

all_rets = np.array(all_rets)
first = np.percentile(all_rets, 33) 
second = np.percentile(all_rets, 66)


for npz_file in sorted([f for f in os.listdir(old_dt)]):
    coin_name = npz_file.split('.')[0]
    data = np.load(os.path.join(old_dt, npz_file))
    images = data['images']
    original_labels = data['labels']
    coin_avg_returns = avg_rt_dt[coin_name]
    candle_indices = np.arange(len(original_labels)) + wsize - 1 + 33
    label_returns = []
    for idx in candle_indices:
        if idx < len(coin_avg_returns):
            label_returns.append(coin_avg_returns[idx])
        else:
            label_returns.append(np.nan)
    label_returns = np.array(label_returns)
    
    new_labels = np.zeros(len(original_labels), dtype=np.uint8)
    for i in range(len(original_labels)):
        if np.isnan(label_returns[i]):
            new_labels[i] = 1  
        elif label_returns[i] >= second:
            new_labels[i] = 2  
        elif label_returns[i] >= first:
            new_labels[i] = 1  
        else:
            new_labels[i] = 0
    output_file = os.path.join(new_dt, f"{coin_name}.npz")
    np.savez_compressed(
        output_file,
        images=images,
        labels=new_labels,
        returns=label_returns,
        original_labels=original_labels
    )
    


