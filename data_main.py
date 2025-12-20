import pandas as pd
import numpy as np
import ta
import os
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view


wsize = 12          
iheight = 15         
max_window = 11          
i_folder = 'raw_data'
o_folder = 'images2' 

def compute_indicators(df):
    df = df.copy()
    
    df['volume'] = df['volume'].replace(0, 1)

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df['bb_pct'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()

    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    df['vol_sma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_sma'] + 1e-8)

    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx.adx()

    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()

    df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()

    ema20 = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_div'] = (df['close'] - ema20) / ema20

    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    df.dropna(inplace=True)
    
    features = [
        'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_width', 'bb_pct',
        'stoch_k', 'atr', 'vol_ratio', 'adx', 'cci', 'mfi', 'roc',
        'ema_div', 'log_ret'
    ]
    return df[features], df['close']


def hill_valley_labels(close: pd.Series, wsize: int = 11) -> pd.Series:
    #taken from the paper, is shit did not work
    n = len(close)
    labels = np.full(n, 0, dtype=np.uint8)
    w = wsize
    for end in range(w - 1, n):
        start = end - (w - 1)
        window = close.iloc[start:end+1].values
        min_idx = window.argmin() + start
        max_idx = window.argmax() + start
        mid_idx = (start + end) // 2
        if max_idx == mid_idx: labels[mid_idx] = 2
        elif min_idx == mid_idx: labels[min_idx] = 1
    return pd.Series(labels, index=close.index)


def image_gen(coin_file):
    print("generating imgs")
    path = os.path.join(i_folder, coin_file)
    df = pd.read_csv(path)
    features_computed, closes = compute_indicators(df)
    data_vals = features_computed.values
    windows = sliding_window_view(data_vals, window_shape=(wsize, iheight))
    windows = windows.squeeze()
    raw_vals = windows.transpose(0, 2, 1)
    means = raw_vals.mean(axis=(1, 2), keepdims=True)
    stds = raw_vals.std(axis=(1, 2), keepdims=True)
    norm_val = (raw_vals - means) / (stds + 1e-8)
    norm_val = norm_val.clip(-3, 3)
    val_Z = ((norm_val + 3) / 6 * 255).astype(np.uint8)
    close_vals = closes.iloc[wsize-1:]
    labels = hill_valley_labels(close_vals, wsize=max_window).values
    min_len = min(len(val_Z), len(labels))
    val_Z = val_Z[:min_len]
    labels = labels[:min_len]
    coin_name = coin_file.split('_')[0]
    os.makedirs(o_folder, exist_ok=True)
    output_path = os.path.join(o_folder, f"{coin_name}.npz")
    np.savez_compressed(output_path, images=val_Z, labels=labels)
    print("Saved")

if __name__ == "__main__":
    files = [f for f in os.listdir(i_folder)]
    for f in files:
        image_gen(f)
