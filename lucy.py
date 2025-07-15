import json
import MetaTrader5 as mt5
import telebot
import os, pickle
import numpy as np
from os.path import exists
from datetime import datetime, timedelta
import hashlib
import signal
import requests
import qrcode
from base64 import b64encode
import random, socket
import threading
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

global set_tp, set_lot, trailing, global_trailiing_usd, global_trailiing_set, ai_for_trade
ai_for_trade = 2
global_trailiing_set = 0.50
global_trailiing_usd = 0.30
set_tp = None
set_lot = None
trailing = True

def ensure_folder_exists(filepath):
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

def get_atr_from_array(highs, lows, closes, period=14):
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(highs))]
    return sum(trs[-period:])/period if len(trs)>=period else 0

def get_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None

    gains = np.zeros(period)
    losses = np.zeros(period)
    for i in range(1, period + 1):
        change = prices[i] - prices[i - 1]
        gains[i - 1] = max(change, 0)
        losses[i - 1] = abs(min(change, 0))
    avg_gain = gains.mean()
    avg_loss = losses.mean()

    if avg_loss == 0:
        rsi = 100
    else:
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

    for i in range(period + 1, len(prices)):
        change = prices[i] - prices[i - 1]
        gain = max(change, 0)
        loss = abs(min(change, 0))
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            rsi = 100
        else:
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
    return rsi

def get_macd(prices, fast=12, slow=26, signal=9):
    def ema(arr, span):
        alpha = 2/(span+1)
        ema_arr = [arr[0]]
        for price in arr[1:]:
            ema_arr.append(alpha*price + (1-alpha)*ema_arr[-1])
        return np.array(ema_arr)

    if len(prices) < slow:
        return 0
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, signal)
    return (macd_line[-1] - macd_signal[-1]) / 0.01

def get_bollinger_bandwidth(prices, period=20):
    if len(prices) < period:
        return 0
    ma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper_band = ma + (2 * std)
    lower_band = ma - (2 * std)
    return (upper_band - lower_band) / ma

def get_adx(highs, lows, closes, period=14):
    if len(highs) < period+1:
        return 0
    plus_dm = []
    minus_dm = []
    tr_list = []
    for i in range(1, len(highs)):
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
        tr_list.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    tr14 = np.sum(tr_list[-period:])
    plus_di = (np.sum(plus_dm[-period:]) / tr14) * 100 if tr14 != 0 else 0
    minus_di = (np.sum(minus_dm[-period:]) / tr14) * 100 if tr14 != 0 else 0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8) * 100
    adx = np.mean(dx) if isinstance(dx, np.ndarray) else dx
    return adx/100

def get_features_and_label(symbol, timeframe, lookahead_bars=12, pips_threshold=20):
    if not mt5.symbol_select(symbol, True):
        print(f"[!] Failed to select {symbol}")
        return [], []

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 5000)
    if rates is None or len(rates) < lookahead_bars + 20:
        return [], []

    features = []
    labels = []

    for i in range(20, len(rates) - lookahead_bars):
        close_prices = [r['close'] for r in rates[i-20:i]]
        highs = [r['high'] for r in rates[i-20:i]]
        lows = [r['low'] for r in rates[i-20:i]]
        volumes = [r['tick_volume'] for r in rates[i-20:i]]

        feature = [
            get_rsi(close_prices),
            get_atr_from_array(highs, lows, close_prices),
            get_macd(close_prices),
            get_bollinger_bandwidth(close_prices),
            get_adx(highs, lows, close_prices),
            np.std(close_prices),        # volatilitas
            np.mean(volumes) / 10000.0,  # volume yg normal
            rates[i]['spread'] / 100.0   # spread normal
        ]

        future_close = rates[i + lookahead_bars]['close']
        current_close = rates[i]['close']
        pips_movement = (future_close - current_close) * 10000  # pips

        if pips_movement > pips_threshold:
            label = 'buy'
        elif pips_movement < -pips_threshold:
            label = 'sell'
        else:
            label = 'hold'

        features.append(feature)
        labels.append(label)

    return features, labels

def save_dataset(symbols, filename="/LucyRFX/dataset_real.json"):
    print("[SOLOAIV2] Setting up datasets for AI from market price action... ( This may take a while )")
    ensure_folder_exists(filename)
    dataset = []
    for symbol in symbols:
        feats, lbls = get_features_and_label(symbol, mt5.TIMEFRAME_M15)
        if not feats:
            continue
        for f, l in zip(feats, lbls):
            if f and len(f) == 8:
                feat29 = extract_features(symbol)
                dataset.append({
                    "features": feat29.tolist(),
                    "target": l
                })
    if len(dataset) == 0:
        raise ValueError("[!] Dataset kosong.")
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"[✔] Dataset saved to {filename} with {len(dataset)} samples!")

def check_symbol_ready(symbol):
    selected = mt5.symbol_select(symbol, True)
    if not selected:
        print(f"[!] Symbol {symbol} gagal dipilih atau tidak aktif!")
        return False
    info = mt5.symbol_info(symbol)
    if info is None or not info.visible:
        print(f"[!] Symbol {symbol} tidak tersedia/terlihat di Market Watch!")
        return False
    return True

scaler = StandardScaler()
global repeat
os.system('clear' if os.name == 'posix' else 'cls')

print("""
 __                _____       _         _____ __ __
|  |   _ _ ___ _ _| __  |___ _| |___ ___|   __|  |  |
|  |__| | |  _| | |    -| .'| . | .'|  _|   __|-   -|
|_____|___|___|_  |__|__|__,|___|__,|_| |__|  |__|__|
              |___|

MrSanZz LucyNetwork, MetaTrader5 AI Based Robot Trading
---------------------------------------------------------------
Version      : 2.2.0
Release Date : 2025-04-10
Tool ID      : MrSanZz041025-V2
Ador4net™

We Are Social:
* https://tiktok.com/@ador4net
* https://t.me/MrSanZzXe
---------------------------------------------------------------
""")

disclaimer = """
---------------------------------------------------------------
DISCLAIMER:
This tool is provided "as is" without any representations or warranties,
express or implied. All trading decisions and associated financial risks
are solely the responsibility of the user. Under no circumstances will the
developer be held liable for any losses incurred, whether direct or indirect,
arising out of the use of this tool. It is strongly advised that you perform
thorough due diligence before engaging in any trading activities.

COPYRIGHT & LICENSE:
All intellectual property rights, including copyright and licensing for this
tool, are exclusively held by MrSanZz and N.A.S. Unauthorized reproduction, distribution,
or modification of this software is strictly prohibited without explicit
written permission from the copyright holder.

Contact Admin at WhatsApp:
-
---------------------------------------------------------------
"""

def setup_config(bot_token, user_id, login, password, server, name, threshold):
    config_template = '{\n'
    config_template += '    "telegram": {\n'
    config_template += f'        "token": "{bot_token}",\n'
    config_template += f'        "user_id": {user_id}\n'
    config_template += '    },\n'
    config_template += '    "mt5": {\n'
    config_template += r'        "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",'+'\n'
    config_template += '        "accounts": [\n'
    config_template += '            {\n'
    config_template += f'                "login": {login},\n'
    config_template += f'                "password": "{password}",\n'
    config_template += f'                "server": "{server}",\n'
    config_template += f'                "name": "{name}"\n'
    config_template += '            }\n'
    config_template += '        ]\n'
    config_template += '    },\n'
    config_template += '    "preferences": {\n'
    config_template += '        "active_account_index": 0\n'
    config_template += '    },\n'
    config_template += '    "trade_rules": {\n'
    config_template += '        "risk_percent": 1,\n'
    config_template += '        "min_risk_reward": 2,\n'
    config_template += '        "max_sl_percent": 3,\n'
    config_template += '        "atr_period": 14\n'
    config_template += '    },\n'
    config_template += '    "monitoring": {\n'
    config_template += '        "momentum_rsi_threshold": 30,\n'
    config_template += '        "interval": 60,\n'
    config_template += f'        "fast_market_threshold": {threshold}\n'
    config_template += '    },\n'
    config_template += '    "risk_management": {\n'
    config_template += '        "risk_percent": 1,\n'
    config_template += '        "reward_ratio": 3,\n'
    config_template += '        "pip_value": 10\n'
    config_template += '    }\n'
    config_template += '}'

    return config_template

if exists('config.json'):
    with open('config.json') as f:
        config = json.load(f)
else:
    print('[!] config.json not found, setting up..')
    bot_token = input("[~] Telegram bot token: ")
    user_id = input("[~] Telegram user id: ")
    login = input("[~] Mt5 login: ")
    password = input("[~] Mt5 password: ")
    server = input("[~] Mt5 broker server: ")
    name = input("[~] Mt5 account name: ")
    threshold = input("[~] Market threshold [e.g: 1.0(fast), 0.001(medium), 0.0001(slow)]: ")
    config = setup_config(bot_token, user_id, login, password, server, name, threshold)
    with open('config.json', 'a') as f:
        f.write(config)
    config = json.dump(config)
    pass

recorded = False
recorded2 = False
edited = False

bot = telebot.TeleBot(config['telegram']['token'])
user_id = config['telegram']['user_id']

LICENSE_FILE = 'license.lic'
SECRET_KEY = "rahasia123-soloAIV2"
PAYPAL_CLIENT_ID = "AeQw6dK794PDcRSZUYLINJ2ZhRXS50X_9Vq77bqAPcNS_3OlxX_eG_v9K3Vd2K4N8UhfD7iL3Rmi_I5Y"
PAYPAL_SECRET = "EOesMwnnaUsnAEojvyi50yZRvAsysVvE81qnvQIVvW9i-5iL8nzV5d5vNK4i33_jgYw5YPZcieHQJg7A"
PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com"

def format_ribuan(angka):
    if isinstance(angka, int) or isinstance(angka, float):
        angka_int = int(angka)
        angka_str = str(angka_int)
    elif isinstance(angka, str) and angka.isdigit():
        angka_int = int(angka)
        angka_str = angka
    else:
        raise ValueError("Input harus berupa angka atau string digit.")

    hasil = ''
    while len(angka_str) > 3:
        hasil = '.' + angka_str[-3:] + hasil
        angka_str = angka_str[:-3]
    hasil = angka_str + hasil

    def konversi_singkat(n):
        simbol = ['', 'K', 'M', 'B', 'T', 'Q']
        i = 0
        while n >= 1000 and i < len(simbol) - 1:
            n /= 1000.0
            i += 1
        return f"{round(n, 1)}{simbol[i]}"

    dalam_singkat = konversi_singkat(angka_int)

    return hasil, dalam_singkat

def requestor(license, online):
    url_base = 'https://lucynet.serveo.net/accept_api/logging_data.php'
    try:
        test = requests.get('https://lucynet.serveo.net/accept_api/database.json', headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36","Content-Type": "application/json"})
        if license in test.text:
            response = requests.post(url_base, headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36","Content-Type": "application/json"}, json={"License-ID": license, "Online": online})
        else:
            response = requests.post('https://lucynet.serveo.net/accept_api/status_response.php', headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36","Content-Type": "application/json"}, json={"License-ID": license, "Online": online})
        return response.text
    except:
        with open(LICENSE_FILE, 'r') as f:
            data = json.load(f)
        return data

def check_new_expire():
    return requests.get('https://lucynet.serveo.net/accept_api/database.json', headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36","Content-Type": "application/json"}).json()

def im_online(online):
    try:
        requests.post('https://lucynet.serveo.net/status/online', headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36","Content-Type": "application/json"}, json={"OnlineUser": online})
    except:
        pass

def check_user():
    try:
        response = requests.get('https://lucynet.serveo.net/status/online', headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36","Content-Type": "application/json"})
        return response.text
    except:
        return '0'

def get_status_update(target_id):
    try:
        response = requests.get('https://lucynet.serveo.net/accept_api/database.json')
        data = response.json()

        for license in data['licenses']:
            if license.get("License-ID") == target_id:
                return license

        print("License-ID not found.")
        return None
    except:
        with open(LICENSE_FILE, 'r') as f:
            data = json.load(f)
        if 'hash' in data:
            return {"Status-Paid": True}
        else:
            return {"Status-Paid": False}

def get_access_token():
    url = f"{PAYPAL_API_BASE}/v1/oauth2/token"
    auth = b64encode(f"{PAYPAL_CLIENT_ID}:{PAYPAL_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def create_paypal_order(amount, currency="USD"):
    token = get_access_token()
    url = f"{PAYPAL_API_BASE}/v2/checkout/orders"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "intent": "CAPTURE",
        "purchase_units": [{
            "amount": {
                "currency_code": currency,
                "value": f"{amount:.2f}"
            }
        }],
        "application_context": {
            "return_url": "https://101dd6c78a9c39fcaabd5be6e5448dd6.serveo.net/accept_api/accept.php",
            "cancel_url": "https://101dd6c78a9c39fcaabd5be6e5448dd6.serveo.net/accept_api/cancel.php"
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    order = response.json()
    for link in order["links"]:
        if link["rel"] == "approve":
            return link["href"]
    raise Exception("Link pembayaran tidak ditemukan.")

def generate_qr_ascii(data):
    qr = qrcode.QRCode(border=1)
    qr.add_data(data)
    qr.make(fit=True)
    qr.print_ascii(invert=True)

def generate_license_secure():
    global edited
    edited = False

    license_data = {
        "License-ID": ''.join(random.choice('ab1234567890=') for _ in range(18))
    }

    with open(LICENSE_FILE, 'w') as f:
        json.dump(license_data, f, indent=4)
    print(f"[✔] License created")
    licenses = LID()
    requestor(licenses, online=False)

def regenerate_license_secure(amount_paid, expire_str):
    global edited
    edited = False

    raw = f"{amount_paid}-{expire_str}-{SECRET_KEY}"
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    licenses = LID()

    license_data = {
        "paid": amount_paid,
        "expire": expire_str,
        "hash": hashed,
        "License-ID": f'{licenses}'
    }

    with open(LICENSE_FILE, 'w') as f:
        json.dump(license_data, f, indent=4)
    print(f"[✔] License created")

#def generate_license_secure(amount_paid, expire_str):
#    global edited
#    edited = False
#    days = int(amount_paid) * 4
#    expire_date = datetime.now() + timedelta(days=days)
#    expire_str = expire_date.strftime("%Y-%m-%d %H:%M:%S")
#
#    raw = f"{amount_paid}-{expire_str}-{SECRET_KEY}"
#    hashed = hashlib.sha256(raw.encode()).hexdigest()
#
#    license_data = {
#        "paid": amount_paid,
#        "expire": expire_str,
#        "hash": hashed,
#        "License-ID": {random.choice('ab1234567890=') for _ in range(18)}
#    }
#
#    with open(LICENSE_FILE, 'w') as f:
#        json.dump(license_data, f, indent=4)
#    print(f"[✔] License created until {expire_str}")

def verify_license_secure():
    global edited
    if not os.path.exists(LICENSE_FILE):
        print("[!] License file not found.")
        return False

    with open(LICENSE_FILE, 'r') as f:
        data = json.load(f)

    raw = f"{data['paid']}-{data['expire']}-{SECRET_KEY}"
    expected_hash = hashlib.sha256(raw.encode()).hexdigest()

    if data['hash'] != expected_hash:
        edited = True
        print("[✘] License detected edited. Please restore license.lic as it was!")
        return False

    expire_time = datetime.strptime(data['expire'], "%Y-%m-%d %H:%M:%S")
    if datetime.now() > expire_time:
        print("[✘] License expired")
        print("[+] Purchase it at: https://lucynet.serveo.net/payments")
        return False

    print(f"[✔] License valid until {expire_time}")
    return True

def LID():
    with open(LICENSE_FILE, 'r') as f:
        data = json.load(f)
    return data['License-ID']

def expire():
    with open(LICENSE_FILE, 'r') as f:
        data = json.load(f)
    return data['expire']

def check_full():
    with open(LICENSE_FILE, 'r') as f:
        data = json.load(f)
    if 'paid' in data:
        return True
    else:
        return False

def fetch_economic_calendar():
    try:
        r = requests.get("https://faireconomy.media/ff_calendar_thisweek.json", timeout=5)
        events = r.json()
        return [e for e in events if e.get('impact') == 3]
    except:
        return []

##########################################################################################################

class DeepNN:
    def __init__(self, layer_dims, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, dropout_rate=0.0, l2_lambda=0.001):
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.t = 0
        self.params = {}
        self.adam_m = {}
        self.adam_v = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        np.random.seed(42)
        for l in range(1, self.L + 1):
            fan_in = self.layer_dims[l-1]
            self.params[f'W{l}'] = np.random.randn(fan_in, self.layer_dims[l]) * np.sqrt(2. / fan_in)
            self.params[f'b{l}'] = np.zeros((1, self.layer_dims[l]))
            self.adam_m[f'W{l}'] = np.zeros_like(self.params[f'W{l}'])
            self.adam_m[f'b{l}'] = np.zeros_like(self.params[f'b{l}'])
            self.adam_v[f'W{l}'] = np.zeros_like(self.params[f'W{l}'])
            self.adam_v[f'b{l}'] = np.zeros_like(self.params[f'b{l}'])

    def relu(self, Z): return np.maximum(0, Z)
    def tanh(self, Z): return np.tanh(Z)
    def relu_derivative(self, Z): return (Z > 0).astype(float)
    def tanh_derivative(self, Z): return 1 - np.tanh(Z) ** 2

    def forward(self, X, training=True):
        A = X.copy()
        cache = {'A0': A}
        for l in range(1, self.L):
            Z = A.dot(self.params[f'W{l}']) + self.params[f'b{l}']
            A = self.relu(Z)
            if training and self.dropout_rate > 0:
                D = (np.random.rand(*A.shape) > self.dropout_rate).astype(float)
                A *= D / (1 - self.dropout_rate)
                cache[f'D{l}'] = D
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        ZL = A.dot(self.params[f'W{self.L}']) + self.params[f'b{self.L}']
        AL = self.tanh(ZL)
        cache[f'Z{self.L}'] = ZL
        cache[f'A{self.L}'] = AL
        return AL, cache

    def compute_loss(self, predictions, targets):
        m = targets.shape[0]
        mse = np.mean((predictions - targets) ** 2)
        l2 = 0
        for l in range(1, self.L+1):
            l2 += np.sum(self.params[f'W{l}'] ** 2)
        l2 *= (self.l2_lambda / (2 * m))
        return mse + l2

    def backward(self, cache, predictions, targets):
        grads = {}
        m = targets.shape[0]
        dAL = 2 * (predictions - targets) / m
        dZ = dAL * self.tanh_derivative(cache[f'Z{self.L}'])
        grads[f'dW{self.L}'] = cache[f'A{self.L-1}'].T.dot(dZ) + (self.l2_lambda / m) * self.params[f'W{self.L}']
        grads[f'db{self.L}'] = np.sum(dZ, axis=0, keepdims=True)
        for l in range(self.L-1, 0, -1):
            dA = dZ.dot(self.params[f'W{l+1}'].T)
            if self.dropout_rate > 0 and f'D{l}' in cache:
                dA *= cache[f'D{l}'] / (1 - self.dropout_rate)
            dZ = dA * self.relu_derivative(cache[f'Z{l}'])
            grads[f'dW{l}'] = cache[f'A{l-1}'].T.dot(dZ) + (self.l2_lambda / m) * self.params[f'W{l}']
            grads[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True)
        return grads

    def update_parameters(self, grads):
        self.t += 1
        for l in range(1, self.L + 1):
            self.adam_m[f'W{l}'] = self.beta1 * self.adam_m[f'W{l}'] + (1 - self.beta1) * grads[f'dW{l}']
            self.adam_v[f'W{l}'] = self.beta2 * self.adam_v[f'W{l}'] + (1 - self.beta2) * (grads[f'dW{l}'] ** 2)
            m_hat_W = self.adam_m[f'W{l}'] / (1 - self.beta1 ** self.t)
            v_hat_W = self.adam_v[f'W{l}'] / (1 - self.beta2 ** self.t)
            self.params[f'W{l}'] -= self.learning_rate * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            self.adam_m[f'b{l}'] = self.beta1 * self.adam_m[f'b{l}'] + (1 - self.beta1) * grads[f'db{l}']
            self.adam_v[f'b{l}'] = self.beta2 * self.adam_v[f'b{l}'] + (1 - self.beta2) * (grads[f'db{l}'] ** 2)
            m_hat_b = self.adam_m[f'b{l}'] / (1 - self.beta1 ** self.t)
            v_hat_b = self.adam_v[f'b{l}'] / (1 - self.beta2 ** self.t)
            self.params[f'b{l}'] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def predict(self, X):
        preds, _ = self.forward(X, training=False)
        return preds

    def train(self, X, Y, epochs=10, batch_size=64, validation_data=None,
              checkpoint_path=None, checkpoint_freq=1, verbose=True):
        m = X.shape[0]
        for epoch in range(1, epochs+1):
            perm = np.random.permutation(m)
            X_shuf, Y_shuf = X[perm], Y[perm]
            epoch_loss = 0
            for i in range(0, m, batch_size):
                xb = X_shuf[i:i+batch_size]
                yb = Y_shuf[i:i+batch_size]
                preds, cache = self.forward(xb, training=True)
                loss = self.compute_loss(preds, yb)
                epoch_loss += loss * xb.shape[0]
                grads = self.backward(cache, preds, yb)
                self.update_parameters(grads)
            epoch_loss /= m
            if verbose:
                msg = f"[AI-HK02] Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}"
                if validation_data:
                    Xv, Yv = validation_data
                    pv, _ = self.forward(Xv, training=False)
                    val_loss = self.compute_loss(pv, Yv)
                    mae = np.mean(np.abs(pv - Yv))
                    ss_res = np.sum((Yv - pv)**2)
                    ss_tot = np.sum((Yv - np.mean(Yv))**2)
                    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0
                    msg += f" | Val Loss: {val_loss:.6f}, MAE: {mae:.4f}, R2: {r2:.4f}"
                print(msg)
            if checkpoint_path and epoch % checkpoint_freq == 0:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(self.params, f)
        return self

    def print_model_summary(self):
        total_params = 0
        for l in range(1, self.L + 1):
            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            total_params += np.prod(W.shape) + np.prod(b.shape)

        total_neurons = sum(self.layer_dims[1:])
        c1, c2 = format_ribuan(int(total_params))
        print(f"[+] Total Parameters: {c1} - {c2}")
        print(f"[+] Total Neurons: {int(total_neurons)}")

def load_or_generate_dataset_multifeature():
    dataset_file = "/LucyRFX/ai_training_dataset_multifeature.json"
    if os.path.exists(dataset_file):
        with open(dataset_file, "r") as f:
            dataset = json.load(f)
    else:
        dataset = []
        for i in range(1_000_000):
            rsi = 50 + i % 51 - 25
            norm_rsi = (rsi - 50) / 50.0
            atr = 0.1 + (i % 190) / 100.0
            norm_atr = 2 * ((atr - 1.05) / 1.9)
            price_diff = (i % 2001 - 1000) / 100.0
            norm_pd = price_diff / 10.0

            norm_rsi_h1 = norm_rsi * 0.9 + np.sin(i / 500) * 0.1
            macd = norm_pd * 10 + np.sin(i / 300)
            stoch = 50 + (np.sin(i/200)*50)
            sma_dev = price_diff / 100.0
            volume_ratio = 1.0 + np.cos(i / 1000) * 0.2
            ema_cross = (np.cos(i / 300) * 0.5)

            features = [
                norm_rsi, norm_atr, norm_pd,
                norm_rsi_h1,
                macd / 100.0,
                (stoch - 50) / 50.0,
                sma_dev,
                volume_ratio,
                ema_cross
            ]

            raw_target = 0.5 * norm_rsi + 0.5 * norm_pd
            target = np.clip(raw_target * 10, -1.0, 1.0)

            dataset.append({
                "features": features,
                "target": target
            })

        with open(dataset_file, "w") as f:
            json.dump(dataset, f)
    return dataset

def load_layer_dims():
    path = "/LucyRFX/layer_config.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return [9, 512, 256, 64, 1]

def save_layer_dims(dims):
    path = "/LucyRFX/layer_config.json"
    ensure_folder_exists(path)
    with open(path, "w") as f:
        json.dump(dims, f)

def train_ai_model_multifeature(epochs=10, batch_size=10000):
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    dataset = load_or_generate_dataset_multifeature()
    dataset = [d for d in dataset if abs(d["target"]) <= 1.0 and all(np.isfinite(d["features"]))]
    dataset_size = len(dataset)
    X_raw = np.array([d["features"] for d in dataset])
    Y = np.array([[d["target"]] for d in dataset])

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    ensure_folder_exists("/LucyRFX/scaler.pkl")
    with open("/LucyRFX/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    input_size = X.shape[1]
    base = input_size
    layer_dims = [base]
    while base < 512:
        base *= 2
        layer_dims.append(base)
    layer_dims.append(1)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

    input_size = X.shape[1]
    layer_dims = [input_size, 2048, 1024, 512, 256, 128, 1]

    save_layer_dims(layer_dims)
    model = DeepNN(layer_dims=layer_dims, learning_rate=0.0001)

    ensure_folder_exists("/LucyRFX/ai_model_multifeature_parameters_improved.json")
    params_file = "/LucyRFX/ai_model_multifeature_parameters_improved.json"
    if os.path.exists(params_file):
        with open(params_file, "r") as f:
            params = json.load(f)
        for l in range(1, model.L + 1):
            model.params['W' + str(l)] = np.array(params['W' + str(l)])
            model.params['b' + str(l)] = np.array(params['b' + str(l)])

    num_batches = dataset_size // batch_size
    for epoch in range(epochs):
        permutation = np.random.permutation(dataset_size)
        X_shuffled = X[permutation]
        Y_shuffled = Y[permutation]
        epoch_loss = 0.0

        for i in range(num_batches):
            batch_X = X_shuffled[i*batch_size:(i+1)*batch_size]
            batch_Y = Y_shuffled[i*batch_size:(i+1)*batch_size]
            predictions, cache = model.forward(batch_X)
            loss = np.mean((predictions - batch_Y) ** 2)
            epoch_loss += loss
            grads = model.backward(cache, predictions, batch_Y)

            max_norm = 5.0
            for key in grads:
                norm = np.linalg.norm(grads[key])
                if norm > max_norm:
                    grads[key] *= max_norm / norm

            model.update_parameters(grads)

        avg_loss = epoch_loss / num_batches
        print(f"[AI-HK02] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    model.train(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))

    new_params = {key: value.tolist() for key, value in model.params.items()}
    changed = True
    if os.path.exists(params_file):
        with open(params_file, "r") as f:
            old_params = json.load(f)
        changed = False
        for key in new_params:
            old = np.array(old_params[key])
            new = np.array(new_params[key])
            diff = np.mean(np.abs(old - new))
            if diff > 1e-6:
                changed = True
                print(f"[+] Change detected in {key}, Avg diff: {diff:.8f}")
            else:
                print(f"[-] No significant change in {key}, Avg diff: {diff:.8f}")

    if changed:
        with open(params_file, "w") as f:
            json.dump(new_params, f, indent=4)
        print("[AI-HK02] Model updated & saved!")
    else:
        print("[AI-HK02] No change in parameters. Model not saved.")

def load_trained_model_multifeature():
    params_file = "/LucyRFX/ai_model_multifeature_parameters_improved.json"
    if not os.path.exists(params_file):
        return train_ai_model_multifeature()
    with open(params_file, "r") as f:
        params = json.load(f)
    model = DeepNN(layer_dims=[9, 512, 256, 128, 1], learning_rate=0.01)
    for l in range(1, model.L + 1):
        model.params['W' + str(l)] = np.array(params['W' + str(l)])
        model.params['b' + str(l)] = np.array(params['b' + str(l)])
    return model

def ai_market_executor_with_model(symbol):
    model = load_trained_model_multifeature()

    rates_h4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)
    if rates_h4 is None or len(rates_h4) == 0:
        print(f"[AI-HK02] No H4 data for {symbol}")
        return

    close_prices_h4 = [r['close'] for r in rates_h4]
    rsi_value = get_rsi(close_prices_h4, period=14)
    if rsi_value is None:
        print(f"[AI-HK02] RSI failed for {symbol}")
        return
    norm_rsi = (rsi_value - 50) / 50.0

    atr_value = calculate_atr(symbol, period=14, timeframe=mt5.TIMEFRAME_H4)
    if atr_value is None:
        print(f"[AI-HK02] ATR failed for {symbol}")
        return
    norm_atr = 2 * ((atr_value - 1.05) / 1.9)

    rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
    if rates_h1 is None or len(rates_h1) == 0:
        print(f"[AI-HK02] No H1 data for {symbol}")
        return

    close_prices_h1 = [r['close'] for r in rates_h1]
    current_price = close_prices_h1[-1]
    ema_value = get_ema(close_prices_h1, 20)
    if ema_value is None:
        print(f"[AI-HK02] EMA failed for {symbol}")
        return
    price_diff = current_price - ema_value
    norm_pd = price_diff / (ema_value * 0.1)
    norm_pd = np.clip(norm_pd, -1.0, 1.0)

    highs = [r['high'] for r in rates_h1]
    lows = [r['low'] for r in rates_h1]
    volumes = [r['tick_volume'] for r in rates_h1]
    rsi_h1 = get_rsi(close_prices_h1)
    macd = get_macd(close_prices_h1)
    stoch = get_stochastic(close_prices_h1, highs, lows)
    sma20 = get_ma(close_prices_h1, 20)
    ema10 = get_ema(close_prices_h1, 10)
    ema50 = get_ema(close_prices_h1, 50)

    volume_ratio = volumes[-1] / (np.mean(volumes[-20:]) + 1e-8)
    ema_cross = (ema10 - ema50) / (ema50 + 1e-8)

    features = np.array([[
        (rsi_value - 50) / 50.0,
        norm_atr,
        norm_pd,
        (rsi_h1 - 50) / 50.0,
        macd / 100.0,
        (stoch - 50) / 50.0,
        (current_price - sma20) / sma20,
        volume_ratio,
        ema_cross
    ]])
    with open("/LucyRFX/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    features = scaler.transform(features)

    prediction = model.forward(features)[0][0, 0]
    dynamic_threshold = max(0.015, atr_value / current_price / 2)

    if prediction > dynamic_threshold:
        print("[AI-HK02] Prediction: BUY")
        open_position(symbol, "buy")
    elif prediction < -dynamic_threshold:
        print("[AI-HK02] Prediction: SELL")
        open_position(symbol, "sell")
    else:
        print(f"[AI-HK02] No good signal detected ({prediction:.4f}) // HOLD {symbol}")

def detect_break_of_structure(highs, lows, pivot_window=3):
    structure = []
    for i in range(pivot_window, len(highs)-pivot_window):
        window_high = max(highs[i-pivot_window:i+pivot_window+1])
        window_low = min(lows[i-pivot_window:i+pivot_window+1])
        if highs[i] == window_high:
            structure.append(('HH', i, highs[i]))
        if lows[i] == window_low:
            structure.append(('LL', i, lows[i]))
    if len(structure) >= 2:
        prev, curr = structure[-2], structure[-1]
        if prev[0]=='HH' and curr[0]=='LL': return 'CHoCH'
        if prev[0]=='LL' and curr[0]=='HH': return 'BOS'
    return None

def find_fvg(highs, lows):
    fvg = []
    for i in range(len(highs)-2):
        if highs[i]<lows[i+1] and highs[i+1]<lows[i+2]:
            fvg.append((highs[i], lows[i+2]))
    return fvg[-3:]

def is_bullish_engulfing(c1, c2): return c2['open']<c2['close'] and c2['open']<c1['close'] and c2['close']>c1['open']
def is_bearish_engulfing(c1, c2): return c2['open']>c2['close'] and c2['open']>c1['close'] and c2['close']<c1['open']
def is_hammer(c): return c['close']>c['open'] and (c['low']<c['open'] and (c['close']-c['low'])>2*(c['close']-c['open']))

def candlestick_signals(rates):
    sig = []
    for i in range(1, len(rates)):
        if is_bullish_engulfing(rates[i-1], rates[i]): sig.append(('bull_engulf', i))
        if is_bearish_engulfing(rates[i-1], rates[i]): sig.append(('bear_engulf', i))
        if is_hammer(rates[i]): sig.append(('hammer', i))
    return sig[-3:]

def extract_features(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if rates is None or len(rates) < 20:
        return np.zeros(29)

    close_prices = [r['close'] for r in rates]
    open_prices = [r['open'] for r in rates]
    highs = [r['high'] for r in rates]
    lows = [r['low'] for r in rates]
    volumes = [r['tick_volume'] for r in rates]

    structured_features = [
        np.mean(close_prices[-5:]) - close_prices[-1],
        np.mean(close_prices[-10:]) - close_prices[-1],
        highs[-1] - lows[-1],
        np.var(close_prices[-20:]),
        np.std(close_prices[-20:]),
        np.max(close_prices[-20:]) - np.min(close_prices[-20:]),
    ]
    while len(structured_features) < 21:
        structured_features.append(0.0)

    rsi_value = get_rsi(close_prices)
    if rsi_value is None:
        rsi_value = 50.0

    real_features = [
        rsi_value / 100,
        get_atr_from_array(highs, lows, close_prices),
        get_macd(close_prices),
        get_bollinger_bandwidth(close_prices),
        get_adx(highs, lows, close_prices),
        np.std(close_prices),
        np.mean(volumes) / 10000.0,
        rates[-1]['spread'] / 100.0
    ]

    final_features = np.array(structured_features + real_features, dtype=float)

    return final_features


def find_order_blocks(rates, direction='bull'):
    obs = []
    for i in range(1, len(rates)):
        prev = rates[i-1]; curr = rates[i]
        if direction=='bull' and prev['close']>prev['open'] and curr['close']<curr['open']:
            obs.append((prev['high'], prev['time']))
        if direction=='bear' and prev['close']<prev['open'] and curr['close']>curr['open']:
            obs.append((prev['low'], prev['time']))
    return obs[-3:]

##########################################################################################################

class ActionNN(DeepNN):
    def __init__(self, layer_dims, learning_rate=0.005, **kwargs):
        super().__init__(layer_dims, learning_rate, **kwargs)

    def forward(self, X, training=True):
        AL, cache = super().forward(X, training)
        if AL.ndim == 1:
            AL = AL.reshape(-1, 1)
        elif AL.shape[1] != 1:
            AL = AL[:, :1]
        logits = np.concatenate([-AL, np.zeros_like(AL), AL], axis=1)
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        cache['probs'] = probs
        return probs, cache


    def compute_loss(self, probs, labels):
        m = labels.shape[0]
        y_onehot = np.zeros_like(probs)
        for i, l in enumerate(labels.flatten()):
            y_onehot[i, l] = 1
        log_likelihood = -np.log(probs[np.arange(m), labels.flatten()] + 1e-8)
        loss = np.sum(log_likelihood) / m
        l2 = 0
        for l in range(1, self.L+1):
            l2 += np.sum(self.params[f'W{l}']**2)
        loss += (self.l2_lambda/(2*m)) * l2
        return loss

    def backward(self, cache, preds, labels):
        m = labels.shape[0]
        dZ = preds.copy()
        for i, l in enumerate(labels.flatten()):
            dZ[i, l] -= 1
        dZ /= m
        dAL = -dZ[:,0] + dZ[:,2]
        dAL = dAL.reshape(-1,1)
        return super().backward(cache, dAL, None)

    def train(self, X, y, epochs=10, batch_size=64, verbose=True):
        m = X.shape[0]
        for epoch in range(1, epochs+1):
            perm = np.random.permutation(m)
            X_shuf, y_shuf = X[perm], y[perm]
            epoch_loss = 0
            correct = 0
            for i in range(0, m, batch_size):
                xb = X_shuf[i:i+batch_size]
                yb = y_shuf[i:i+batch_size]
                probs, cache = self.forward(xb, True)
                loss = self.compute_loss(probs, yb)
                epoch_loss += loss * xb.shape[0]
                preds = np.argmax(probs, axis=1)
                correct += np.sum(preds == yb.flatten())
                grads = self.backward(cache, probs, yb)
                self.update_parameters(grads)
            acc = correct/m
            if verbose:
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss/m:.6f}, Acc: {acc:.4f}")
        return self

def load_or_generate_dataset_structured():
    fn = '/LucyRFX/dataset_structured_deterministic.json'
    try:
        with open(fn, 'r') as f:
            return json.load(f)
    except:
        data = []
        symbols = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'XAUUSD', 'USDCAD', 'CHFJPY', 'AUDUSD', 'XAGUSD', 'ETHUSD', 'GBPJPY']
        for sym in symbols:
            for i in range(10):
                feat = [
                    np.sin(i / 3.0),
                    np.cos(i / 4.0),
                    np.tan(i / 5.0) % 1,
                    (-1)**i,
                    0.5 + 0.1 * np.sin(i),
                    0.3 + 0.1 * np.cos(i),
                    i % 4
                ] * 3  # → jadi 21 fitur, masing masing 7
                target = (
                    'buy' if feat[0] < -0.3 and feat[3] > 0 and feat[-1] > 0
                    else 'sell' if feat[0] > 0.3 and feat[3] < 0 and feat[-1] > 0
                    else 'hold'
                )
                data.append({'features': feat, 'target': target})
        with open(fn, 'w') as f:
            json.dump(data, f, indent=2)
        return data

def train_action_model(epochs=10):
    data = load_or_generate_dataset_structured()
    X = np.array([d['features'] for d in data]); Y = np.array([{'sell':0,'hold':1,'buy':2}[d['target']] for d in data])
    model = ActionNN(layer_dims=[len(X[0]),64,32,3], learning_rate=0.005)
    for e in range(epochs):
        probs, cache = model.forward(X)
        pass
    return model

def ai_executor_structured(symbol):
    feat = extract_features(symbol).reshape(1, -1)
    model = load_trained_model_multifeature()
    action, conf = model.predict_action(feat)
    if conf > 0.45:
        open_position(symbol, action)
    else:
        print(f"[AI-HYBD] Hold or low confidence ({conf:.2f}) for {symbol}")


##########################################################################################################

def load_real_dataset(filename="/LucyRFX/dataset_real.json"):
    ensure_folder_exists(filename)
    with open(filename, "r") as f:
        data = json.load(f)
    if not data:
        raise ValueError("[!] Dataset kosong, tidak bisa training model.")
    X = np.array([d['features'] for d in data])
    y = np.array([{'sell':0, 'hold':1, 'buy':2}[d['target']] for d in data])
    return X, y

class HybridNN:
    def __init__(self, input_dim, hidden_dim=512, output_dim=3, learning_rate=0.001):
        self.lr = learning_rate
        self.branch_ind = self.init_layer([input_dim, 512, 256, 1])
        self.branch_tex = self.init_layer([input_dim, 512, 256, 1])
        self.Wh = np.random.randn(2, hidden_dim) * np.sqrt(2 / 2)
        self.bh = np.zeros((1, hidden_dim))
        self.Wo = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.bo = np.zeros((1, output_dim))

    def init_layer(self, dims):
        layer = []
        for i in range(len(dims)-1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2/dims[i])
            b = np.zeros((1, dims[i+1]))
            layer.append((W, b))
        return layer

    def forward_branch(self, X, branch):
        A = np.atleast_2d(X)
        for i, (W, b) in enumerate(branch):
            if A.shape[1] != W.shape[0]:
                raise ValueError(f"Shape mismatch: A{A.shape}, W{W.shape} at layer {i}")
            Z = A @ W + b
            A = np.maximum(0, Z)  # ReLU
        return A


    def relu(self, Z):
        return np.maximum(0, Z)

    def forward(self, X):
        cont = self.forward_branch(X, self.branch_ind)
        tex = self.forward_branch(X, self.branch_tex)

        feats = np.hstack([cont, tex])
        Z1 = feats @ self.Wh + self.bh
        A1 = self.relu(Z1)
        Z2 = A1 @ self.Wo + self.bo
        exp_scores = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] != self.branch_ind[0][0].shape[0]:
            raise ValueError(f"[!] Input dim {X.shape[1]} tidak cocok dengan model: {self.branch_ind[0][0].shape[0]}")
        probs = self.forward(X)
        idx = np.argmax(probs, axis=1)[0]
        return ['sell', 'hold', 'buy'][idx], probs[0, idx]

    def train(self, X, y, epochs=20, batch_size=128, verbose=True, save_path="/LucyRFX/hybrid_model_real.pkl"):
        m = X.shape[0]
        y_onehot = np.eye(3)[y]
        best_loss = np.inf

        for epoch in range(1, epochs+1):
            perm = np.random.permutation(m)
            X_shuf, y_shuf = X[perm], y_onehot[perm]
            epoch_loss = 0

            for i in range(0, m, batch_size):
                xb = X_shuf[i:i+batch_size]
                yb = y_shuf[i:i+batch_size]

                cont = self.forward_branch(xb, self.branch_ind)
                tex = self.forward_branch(xb, self.branch_tex)

                feats = np.hstack([cont, tex])
                Z1 = feats @ self.Wh + self.bh
                A1 = self.relu(Z1)
                Z2 = A1 @ self.Wo + self.bo
                exp_scores = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                loss = -np.sum(yb * np.log(probs + 1e-8)) / xb.shape[0]
                epoch_loss += loss * xb.shape[0]

                dZ2 = (probs - yb) / xb.shape[0]
                dWo = A1.T @ dZ2
                dbo = np.sum(dZ2, axis=0, keepdims=True)
                dA1 = dZ2 @ self.Wo.T
                dZ1 = dA1 * (Z1 > 0)
                dWh = feats.T @ dZ1
                dbh = np.sum(dZ1, axis=0, keepdims=True)

                self.Wo -= self.lr * dWo
                self.bo -= self.lr * dbo
                self.Wh -= self.lr * dWh
                self.bh -= self.lr * dbh

            epoch_loss /= m

            if epoch_loss < best_loss:
                print(f"\n[🧠] Epoch {epoch}: Loss improved ({best_loss:.6f} → {epoch_loss:.6f}). Added new neurons!")
                self.grow_new_neurons()

                ensure_folder_exists(save_path)
                with open(save_path, "wb") as f:
                    pickle.dump(self, f)
                print(f"[💾] Model updated and saved after neuron growth to {save_path}")

                best_loss = epoch_loss

            if verbose:
                print(f"[AI-HYBD] Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}", end='\r')

        print("\n[✔] Training complete")

    def grow_new_neurons(self, extra_neurons=8):
        old_Wh_shape = self.Wh.shape
        new_Wh = np.random.randn(old_Wh_shape[0], old_Wh_shape[1] + extra_neurons) * np.sqrt(2 / old_Wh_shape[0])
        new_Wh[:, :old_Wh_shape[1]] = self.Wh
        self.Wh = new_Wh

        self.bh = np.hstack([self.bh, np.zeros((1, extra_neurons))])

        new_Wo = np.random.randn(self.Wh.shape[1], self.Wo.shape[1]) * np.sqrt(2 / self.Wh.shape[1])
        new_Wo[:self.Wo.shape[0], :] = self.Wo
        self.Wo = new_Wo

        print(f"[AI-HYBD] Added {extra_neurons} new neurons! New hidden layer size: {self.Wh.shape[1]}")

    @classmethod
    def train_from_dataset(cls, dataset, epochs=5, batch_size=32, verbose=True):
        X = np.array([d['features'] for d in dataset])
        y = np.array([{'sell':0,'hold':1,'buy':2}[d['target']] for d in dataset])

        ensure_folder_exists('/LucyRFX/hybrid_model.pkl')
        if os.path.exists('/LucyRFX/hybrid_model.pkl'):
            model = load_hybrid_model()
        else:
            model = cls(input_dim=X.shape[1])

        model.train(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        save_hybrid_model(model)
        return model

def save_hybrid_model(model, filename='/LucyRFX/hybrid_model_real.pkl'):
    ensure_folder_exists(filename)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_hybrid_model(filename='/LucyRFX/hybrid_model_real.pkl'):
    ensure_folder_exists(filename)
    if not os.path.exists(filename):
        return None
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def ai_executor_hybrid(symbol, model):
    global disable_ai2
    news = fetch_economic_calendar()
    now = datetime.now()
    for e in news:
        event_time = datetime.fromtimestamp(e['timestamp'])
        if abs((event_time - now).total_seconds()) < 300:
            print(f"[AI-HYBD] Skipping {symbol}, news at {event_time}")
            return
    feat = extract_features(symbol).reshape(1,-1)
    action, conf = model.predict(feat)
    if action in ('buy','sell') and conf > 0.3:
        disable_ai2 = True
        buy_vol, sell_vol = read_market_volume(symbol)
        if action=='buy' and buy_vol < sell_vol:
            print(f"[AI-HYBD] Buy signal but sell volume ({sell_vol}) > buy volume ({buy_vol}), skipping.")
            return True
        if action=='sell' and sell_vol < buy_vol:
            print(f"[AI-HYBD] Sell signal but buy volume ({buy_vol}) > sell volume ({sell_vol}), skipping.")
            return True
        print("[AI-HYBD] Open position by: Artificial Intelligence-Hybrid Dual-Core")
        open_position(symbol, action)
    else:
        print(f"[AI-HYBD] Hold or low confidence ({conf:.2f}) for {symbol}")
        disable_ai2 = False
        return False

##########################################################################################################

class SuperHybridNN:
    def __init__(self, input_dim, hidden_dim=4096, learning_rate=0.001):
        self.lr = learning_rate

        self.deep_branch = DeepNN([input_dim, hidden_dim, 2048, 1])
        self.action_branch = ActionNN([input_dim, hidden_dim, 2048, 3])

        self.hybrid_branch = HybridNN(input_dim)
        self.hybrid_branch.branch_ind = self.hybrid_branch.init_layer([input_dim, 4096, 2048, 1])
        self.hybrid_branch.branch_tex = self.hybrid_branch.init_layer([input_dim, 4096, 2048, 1])

        self.final_W = np.random.randn(7, 3) * np.sqrt(2. / 7)
        self.final_b = np.zeros((1, 3))

    def forward(self, X):
        batch_size = X.shape[0]
        deep_out, _ = self.deep_branch.forward(X, training=False)
        action_probs, _ = self.action_branch.forward(X, training=False)
        hybrid_probs = self.hybrid_branch.forward(X)

        deep_out = deep_out.reshape(-1, 1)
        action_probs = action_probs.reshape(-1, 3)
        hybrid_probs = hybrid_probs.reshape(-1, 3)

        if deep_out.shape[0] != action_probs.shape[0] or deep_out.shape[0] != hybrid_probs.shape[0]:
            raise ValueError("Batch sizes do not match")

        combined = np.hstack([deep_out, action_probs, hybrid_probs])

        logits = combined @ self.final_W + self.final_b
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs, combined

    def predict(self, X):
        probs, _ = self.forward(X)
        idx = np.argmax(probs, axis=1)[0]
        return ['sell', 'hold', 'buy'][idx], probs[0, idx]

    def train(self, X, y, epochs=30, batch_size=128, patience=7, save_path="/LucyRFX/super_hybrid_model.pkl"):
        m = X.shape[0]
        y_onehot = np.eye(3)[y]

        best_loss = np.inf
        patience_counter = 0

        base_batch_size = 128
        base_lr = self.lr
        scale = batch_size / base_batch_size
        effective_lr = min(base_lr * scale, 0.005)

        lr_warmup_epochs = int(epochs * 0.1)
        min_lr = effective_lr * 0.2

        adam_m_W = np.zeros_like(self.final_W)
        adam_v_W = np.zeros_like(self.final_W)
        adam_m_b = np.zeros_like(self.final_b)
        adam_v_b = np.zeros_like(self.final_b)
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        t = 0

        print(f"[~] Effective Learning Rate adjusted: {effective_lr:.6f} (Batch Size: {batch_size})")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(m)
            X_shuf, y_shuf = X[perm], y_onehot[perm]
            epoch_loss = 0

            if epoch <= lr_warmup_epochs:
                curr_lr = min_lr + (effective_lr - min_lr) * (epoch / lr_warmup_epochs)
            else:
                decay_factor = 0.98 ** (epoch - lr_warmup_epochs)
                curr_lr = max(min_lr, effective_lr * decay_factor)

            for i in range(0, m, batch_size):
                xb = X_shuf[i:i+batch_size]
                yb = y_shuf[i:i+batch_size]

                deep_out, _ = self.deep_branch.forward(xb, training=True)
                action_probs, _ = self.action_branch.forward(xb, training=True)
                hybrid_probs = self.hybrid_branch.forward(xb)
                deep_out = (deep_out - np.mean(deep_out)) / (np.std(deep_out) + 1e-8)
                hybrid_probs = (hybrid_probs - np.mean(hybrid_probs)) / (np.std(hybrid_probs) + 1e-8)

                deep_out = deep_out.reshape(-1, 1)
                action_probs = action_probs.reshape(-1, 3)
                hybrid_probs = hybrid_probs.reshape(-1, 3)

                combined = np.hstack([deep_out, action_probs, hybrid_probs])

                logits = combined @ self.final_W + self.final_b
                exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-8)

                loss = -np.sum(yb * np.log(probs + 1e-8)) / xb.shape[0]
                epoch_loss += loss * xb.shape[0]

                dZ = (probs - yb) / xb.shape[0]
                dW = combined.T @ dZ
                db = np.sum(dZ, axis=0, keepdims=True)

                dW_norm = np.linalg.norm(dW)
                if dW_norm > 5.0:
                    dW = dW * (5.0 / dW_norm)
                db_norm = np.linalg.norm(db)
                if db_norm > 5.0:
                    db = db * (5.0 / db_norm)

                t += 1
                adam_m_W = beta1 * adam_m_W + (1 - beta1) * dW
                adam_v_W = beta2 * adam_v_W + (1 - beta2) * (dW ** 2)
                m_hat_W = adam_m_W / (1 - beta1 ** t)
                v_hat_W = adam_v_W / (1 - beta2 ** t)
                adam_m_b = beta1 * adam_m_b + (1 - beta1) * db
                adam_v_b = beta2 * adam_v_b + (1 - beta2) * (db ** 2)
                m_hat_b = adam_m_b / (1 - beta1 ** t)
                v_hat_b = adam_v_b / (1 - beta2 ** t)
                self.final_W -= curr_lr * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
                self.final_b -= curr_lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

            epoch_loss /= m
            print(f"[SuperHybridNN] Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f} - LR: {curr_lr:.6f}", end='\r')

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                self.save_model(save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        duration = time.time() - start_time
        print(f"\n[🏁] Training Completed in {duration:.2f} seconds.")

        return

    def save_model(self, filename="/LucyRFX/super_hybrid_model.pkl"):
        ensure_folder_exists(filename)
        model_data = {
            'final_W': self.final_W.tolist(),
            'final_b': self.final_b.tolist(),
        }

        deep_params = {k: v.tolist() for k, v in self.deep_branch.params.items()}
        action_params = {k: v.tolist() for k, v in self.action_branch.params.items()}
        hybrid_params = {
            'branch_ind': [(W.tolist(), b.tolist()) for W, b in self.hybrid_branch.branch_ind],
            'branch_tex': [(W.tolist(), b.tolist()) for W, b in self.hybrid_branch.branch_tex],
            'Wh': self.hybrid_branch.Wh.tolist(),
            'bh': self.hybrid_branch.bh.tolist(),
            'Wo': self.hybrid_branch.Wo.tolist(),
            'bo': self.hybrid_branch.bo.tolist()
        }

        model_data['deep_branch'] = deep_params
        model_data['action_branch'] = action_params
        model_data['hybrid_branch'] = hybrid_params

        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

    @staticmethod
    def load_model(filename="/LucyRFX/super_hybrid_model.pkl"):
        ensure_folder_exists(filename)
        if not os.path.exists(filename):
            print(f"[!] Model file {filename} tidak ditemukan.")
            return None
        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        if isinstance(model_data, dict):
            input_dim = 29
            model = SuperHybridNN(input_dim=input_dim)
            model.final_W = np.array(model_data['final_W'])
            model.final_b = np.array(model_data['final_b'])

            deep_params = model_data['deep_branch']
            for key in deep_params:
                model.deep_branch.params[key] = np.array(deep_params[key])

            action_params = model_data['action_branch']
            for key in action_params:
                model.action_branch.params[key] = np.array(action_params[key])

            hybrid_params = model_data['hybrid_branch']
            model.hybrid_branch.branch_ind = [(np.array(w), np.array(b)) for w, b in hybrid_params['branch_ind']]
            model.hybrid_branch.branch_tex = [(np.array(w), np.array(b)) for w, b in hybrid_params['branch_tex']]
            model.hybrid_branch.Wh = np.array(hybrid_params['Wh'])
            model.hybrid_branch.bh = np.array(hybrid_params['bh'])
            model.hybrid_branch.Wo = np.array(hybrid_params['Wo'])
            model.hybrid_branch.bo = np.array(hybrid_params['bo'])

            return model
        else:
            return model_data


##########################################################################################################

class AlchemistNN:
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=5, learning_rate=0.001):
        self.lr = learning_rate
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2. / dims[i])
            b = np.zeros((1, dims[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        A = X
        self.caches = []
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            self.caches.append((A, Z))
            A = self.relu(Z)
        Z_final = A @ self.weights[-1] + self.biases[-1]
        self.caches.append((A, Z_final))
        probs = self.softmax(Z_final)
        return probs

    def backward(self, X, Y_true, probs):
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        m = X.shape[0]
        dZ = probs.copy()
        dZ[range(m), Y_true] -= 1
        dZ /= m

        A_prev, _ = self.caches[-1]
        grads_W[-1] = A_prev.T @ dZ
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True)

        for l in range(len(self.weights)-2, -1, -1):
            A_prev, Z = self.caches[l]
            dA = dZ @ self.weights[l+1].T
            dZ = dA * self.relu_derivative(Z)
            grads_W[l] = A_prev.T @ dZ
            grads_b[l] = np.sum(dZ, axis=0, keepdims=True)

        return grads_W, grads_b

    def update_parameters(self, grads_W, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]

    def predict(self, X):
        probs = self.forward(X)
        idx = np.argmax(probs, axis=1)[0]
        actions = ['buy', 'sell', 'buy_limit', 'sell_limit', 'hold']
        return actions[idx], probs[0, idx]

    def train(self, X, Y, epochs=10, batch_size=64, verbose=True):
        for epoch in range(1, epochs+1):
            perm = np.random.permutation(len(X))
            X_shuf = X[perm]
            Y_shuf = Y[perm]
            loss_total = 0

            for i in range(0, len(X), batch_size):
                X_batch = X_shuf[i:i+batch_size]
                Y_batch = Y_shuf[i:i+batch_size]

                probs = self.forward(X_batch)
                loss = -np.sum(np.log(probs[range(len(Y_batch)), Y_batch] + 1e-8)) / len(Y_batch)
                loss_total += loss * len(Y_batch)

                grads_W, grads_b = self.backward(X_batch, Y_batch, probs)
                self.update_parameters(grads_W, grads_b)

            avg_loss = loss_total / len(X)
            if verbose:
                print(f"[AlchemistNN] Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")

    def save_model(self, path):
        model_data = {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.weights = [np.array(w) for w in data["weights"]]
        self.biases = [np.array(b) for b in data["biases"]]

def save_alchemist_dataset(dataset, path="/LucyRFX/alchemist_dataset.json"):
    try:
        with open(path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"[Dataset] ✅ Saved dataset to {path}")
    except Exception as e:
        print(f"[Dataset] ❌ Failed to save: {e}")

def load_alchemist_dataset(path="/LucyRFX/alchemist_dataset.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        X = np.array([d["features"] for d in data])
        Y = np.array([d["label"] for d in data])
        return X, Y
    except Exception as e:
        print(f"[Dataset] ❌ Failed to load dataset: {e}")
        return None, None

def extract_features_alchemist(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if rates is None or len(rates) < 20:
        return np.zeros((1, 30))

    highs = [r['high'] for r in rates]
    lows = [r['low'] for r in rates]
    closes = [r['close'] for r in rates]
    volumes = [r['tick_volume'] for r in rates]

    bos_flag = 1 if detect_break_of_structure(highs, lows) else 0
    fvg_count = len(find_fvg(highs, lows))
    ob_count = len(find_order_blocks(rates))
    candle_patterns = candlestick_signals(rates[-10:])
    candle_bullish = int(any(sig[0] == 'bull_engulf' for sig in candle_patterns))
    candle_hammer = int(any(sig[0] == 'hammer' for sig in candle_patterns))

    atr = get_atr_from_array(highs, lows, closes)
    rsi = get_rsi(closes)
    spread = rates[-1]['spread'] / 100.0
    vol_avg = np.mean(volumes[-20:]) / 10000.0

    features = [
        bos_flag,
        fvg_count / 5.0,
        ob_count / 5.0,
        candle_bullish,
        candle_hammer,
        atr,
        (rsi or 50) / 100.0,
        spread,
        vol_avg
    ]

    features += list(np.random.normal(0, 1, 21))

    return np.array(features).reshape(1, -1)

def rsi_ma_confirmation(symbol, timeframe):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    if rates is None or len(rates) < 50:
        return False, 50.0, "none"

    closes = [r['close'] for r in rates]
    rsi = get_rsi(closes)
    ema10 = get_ema(closes, 10)
    ema50 = get_ema(closes, 50)

    if rsi is None or ema10 is None or ema50 is None:
        return False, 50.0, "none"

    ma_cross = "golden" if ema10 > ema50 else "death" if ema10 < ema50 else "none"
    return True, rsi, ma_cross

def ai_executor_alchemist(symbol, model, mode='scalping'):
    news = fetch_economic_calendar()
    now = datetime.now()
    if any(abs((datetime.fromtimestamp(e['timestamp']) - now).total_seconds()) < 300 for e in news):
        print(f"[AlchemistNN] Trade skipped on {symbol} due to nearby news.")
        return

    feat = extract_features_alchemist(symbol)
    action, conf = model.predict(feat)
    print(f"[AlchemistNN] Prediction: {action.upper()} ({conf:.2f})")

    tf = mt5.TIMEFRAME_M15 if mode == 'scalping' else mt5.TIMEFRAME_H1
    valid, rsi, cross = rsi_ma_confirmation(symbol, tf)

    if not valid:
        print(f"[AlchemistNN] Data not valid for double confirmation.")
        return

    confirm = False
    if action == 'buy' and rsi < 40 and cross == 'golden':
        confirm = True
    elif action == 'sell' and rsi > 60 and cross == 'death':
        confirm = True
    elif action in ['buy_limit', 'sell_limit']:
        confirm = True
    else:
        print(f"[AlchemistNN] Double confirmation failed: RSI={rsi:.2f}, Cross={cross}")

    if not confirm:
        return

    if action == 'buy':
        open_position(symbol, 'buy')
    elif action == 'sell':
        open_position(symbol, 'sell')
    elif action == 'buy_limit':
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
        obs = find_order_blocks(rates, 'bull')
        if obs:
            place_pending_order(symbol, 'buy_limit', obs[-1][0])
    elif action == 'sell_limit':
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
        obs = find_order_blocks(rates, 'bear')
        if obs:
            place_pending_order(symbol, 'sell_limit', obs[-1][0])
    else:
        print(f"[AlchemistNN] HOLD signal. No trade executed.")

def log_trade_to_file(symbol, action, price, sl, tp, lot, mode, model="AlchemistNN", type="market", features=None, filename="trades.json"):
    trade_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "action": action,
        "price": round(price, 5),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "lot": round(lot, 2),
        "mode": mode,
        "model": model,
        "type": type,
        "features": features if features is not None else []
    }

    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(trade_data)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[LOGGER] ✅ Trade logged to {filename}")
    except Exception as e:
        print(f"[LOGGER] ❌ Failed to log trade: {e}")

def retrain_alchemist_from_logs(
    log_file="trades.json",
    dataset_file="alchemist_dataset.json",
    model_file="alchemist_model.json",
    epochs=10
):
    if not os.path.exists(log_file):
        print("[Retrain] No trade log found.")
        return

    with open(log_file, "r") as f:
        logs = json.load(f)

    dataset = []
    for trade in logs:
        features = trade.get("features")
        if not features or len(features) < 30:
            continue

        price = trade.get("price")
        sl = trade.get("sl")
        tp = trade.get("tp")
        action = trade.get("action")

        label = None
        if action in ['buy', 'buy_limit']:
            label = 1 if price < tp else 0 if price > sl else None
        elif action in ['sell', 'sell_limit']:
            label = 1 if price > tp else 0 if price < sl else None

        if label is None:
            continue

        dataset.append({
            "features": features,
            "label": label
        })

    if not dataset:
        print("[Retrain] No valid entries for training.")
        return

    with open(dataset_file, "w") as f:
        json.dump(dataset, f, indent=2)

    X = np.array([d["features"] for d in dataset])
    Y = np.array([d["label"] for d in dataset])

    print(f"[Retrain] 🧠 Training on {len(X)} samples...")
    model = AlchemistNN(input_dim=30, output_dim=5)
    model.train(X, Y, epochs=epochs)
    model.save_model(model_file)
    print(f"[Retrain] ✅ Model saved to {model_file}")


##########################################################################################################

def read_market_volume(symbol, duration_sec=60):
    end = datetime.now()
    start = end - timedelta(seconds=duration_sec)
    ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
    buy_vol = sell_vol = 0
    for t in ticks:
        mid = (t['bid'] + t['ask']) / 2
        last = t['last'] if not np.isnan(t['last']) else mid
        if last >= mid:
            buy_vol += t['volume']
        else:
            sell_vol += t['volume']
    return buy_vol, sell_vol

def log_trade_journal(symbol, entry_time, result, features, decision, profit, journal_file="/LucyRFX/trade_journal.json"):
    ensure_folder_exists(journal_file)
    try:
        if os.path.exists(journal_file):
            with open(journal_file, 'r') as f:
                journal = json.load(f)
        else:
            journal = []

        entry = {
            "symbol": symbol,
            "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "result": "profit" if result else "loss",
            "profit": profit,
            "decision": decision,
            "features": features,
            "confidence": 1.0 if result else 0.3
        }

        journal.append(entry)
        with open(journal_file, 'w') as f:
            json.dump(journal, f, indent=2)
        print(f"[📒] Trade logged ({'WIN' if result else 'LOSS'}): {symbol} | Profit: {profit}")
    except Exception as e:
        print(f"[!] Error writing journal: {e}")

def auto_close_positions():
    while True:
        positions = mt5.positions_get()
        if positions:
            for p in positions:
                tick = mt5.symbol_info_tick(p.symbol)
                price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
                # check TP
                if p.tp and ((p.type == mt5.POSITION_TYPE_BUY and price >= p.tp) or
                            (p.type == mt5.POSITION_TYPE_SELL and price <= p.tp)):
                    mt5.order_send({
                        'action': mt5.TRADE_ACTION_DEAL,
                        'symbol': p.symbol,
                        'volume': p.volume,
                        'type': mt5.ORDER_TYPE_SELL if p.type==mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                        'position': p.ticket,
                        'price': price,
                        'deviation': 10,
                        'magic': 234000
                    })
                    # setelah close posisi
                    profit = p.profit
                    result = profit > 0
                    features = extract_features(p.symbol).tolist()
                    decision = 'buy' if p.type == mt5.POSITION_TYPE_BUY else 'sell'
                    log_trade_journal(p.symbol, datetime.fromtimestamp(p.time), result, features, decision, profit)

                # check SL
                if p.sl and ((p.type == mt5.POSITION_TYPE_BUY and price <= p.sl) or
                            (p.type == mt5.POSITION_TYPE_SELL and price >= p.sl)):
                    mt5.order_send({
                        'action': mt5.TRADE_ACTION_DEAL,
                        'symbol': p.symbol,
                        'volume': p.volume,
                        'type': mt5.ORDER_TYPE_SELL if p.type==mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                        'position': p.ticket,
                        'price': price,
                        'deviation': 10,
                        'magic': 234000
                    })
                    profit = p.profit
                    result = profit > 0
                    features = extract_features(p.symbol).tolist()
                    decision = 'buy' if p.type == mt5.POSITION_TYPE_BUY else 'sell'
                    log_trade_journal(p.symbol, datetime.fromtimestamp(p.time), result, features, decision, profit)
            time.sleep(1)
        else:
            pass

def apply_trailing_stop(order_ticket, symbol, initial_profit=global_trailiing_set if global_trailiing_set else 0.50, drawdown_threshold=global_trailiing_usd if global_trailiing_usd else 0.50):
    global global_trailiing_usd, global_trailiing_set
    order_info = mt5.positions_get(ticket=order_ticket.ticket)
    if not order_info:
        print(f"Failed to retrieve order info for ticket {order_ticket}. Error: {mt5.last_error()}")
        return

    profit_peak = 0.0
    for order in order_info:
        current_profit = order.profit
        if current_profit > profit_peak:
            profit_peak = current_profit

        if profit_peak >= initial_profit and current_profit <= (profit_peak - drawdown_threshold):
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": order.volume,
                "type": mt5.ORDER_TYPE_SELL if order.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": order_ticket,
                "price": mt5.symbol_info_tick(symbol).bid,
                "deviation": 10,
            }
            mt5.order_send(close_request)

def get_stochastic(prices, highs, lows, period=14):
    if len(prices) < period or len(highs) < period or len(lows) < period:
        return None
    highest_high = max(highs[-period:])
    lowest_low = min(lows[-period:])
    if highest_high - lowest_low == 0:
        return 0
    return (prices[-1] - lowest_low) / (highest_high - lowest_low) * 100

def get_ema(prices, period):
    if not prices or len(prices) < period:
        return None
    k = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = price * k + ema * (1 - k)
    return ema

def get_ma(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def kill_process():
    exit(1)

def calculate_position_size(symbol, entry_price, sl_price, risk_percent=1):
    account_info = mt5.account_info()
    if account_info is None:
        return config['trade_rules'].get('lot_size', 0.01)

    balance = account_info.balance
    risk_amount = balance * (risk_percent / 100)
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    tick_value = symbol_info.trade_tick_value

    if entry_price > sl_price: #buy
        sl_points = (entry_price - sl_price) / point
    else:#sell
        sl_points = (sl_price - entry_price) / point

    pip_value = (tick_value * point) / symbol_info.trade_tick_size

    lot = risk_amount / (sl_points * pip_value)
    lot = max(lot, symbol_info.volume_min)
    lot = min(lot, symbol_info.volume_max)
    lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step

    return lot

def find_key_levels(symbol, timeframe=mt5.TIMEFRAME_H4, num_bars=100, pivot_window=3, atr_period=14):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) < (pivot_window * 2 + 1):
        return None

    highs = [r['high'] for r in rates]
    lows = [r['low'] for r in rates]
    closes = [r['close'] for r in rates]

    atr_value = calculate_atr(symbol, period=atr_period, timeframe=timeframe)
    atr_threshold = atr_value * 0.5 if atr_value is not None else 0

    swing_highs = []
    swing_lows = []

    for i in range(pivot_window, len(highs) - pivot_window):
        window_high = max(highs[i - pivot_window : i + pivot_window + 1])
        window_low = min(lows[i - pivot_window : i + pivot_window + 1])

        if highs[i] == window_high:
            if not swing_highs or abs(highs[i] - swing_highs[-1]) > atr_threshold:
                swing_highs.append(highs[i])
        if lows[i] == window_low:
            if not swing_lows or abs(lows[i] - swing_lows[-1]) > atr_threshold:
                swing_lows.append(lows[i])

    if swing_highs:
        last_swing_high = max(swing_highs[-min(3, len(swing_highs)):])
    else:
        last_swing_high = max(highs)
    if swing_lows:
        last_swing_low = min(swing_lows[-min(3, len(swing_lows)):])
    else:
        last_swing_low = min(lows)

    fib_levels = {
        '0.236': last_swing_high - (last_swing_high - last_swing_low) * 0.236,
        '0.382': last_swing_high - (last_swing_high - last_swing_low) * 0.382,
        '0.5':   last_swing_high - (last_swing_high - last_swing_low) * 0.5,
        '0.618': last_swing_high - (last_swing_high - last_swing_low) * 0.618,
        '0.786': last_swing_high - (last_swing_high - last_swing_low) * 0.786
    }
    fib_levels.update({
        '0.127': last_swing_high - (last_swing_high - last_swing_low) * 0.127,
        '0.75':  last_swing_high - (last_swing_high - last_swing_low) * 0.75
    })

    current_price = closes[-1]

    if current_price < 1:
        tick_size = 0.0001
    elif current_price < 10:
        tick_size = 0.01
    elif current_price < 100:
        tick_size = 0.1
    else:
        tick_size = 1.0

    lower_bound = current_price - (5 * tick_size)
    upper_bound = current_price + (5 * tick_size)
    psychological_levels = [round(level, 5) for level in frange(lower_bound, upper_bound, tick_size)]

    return {
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'fib_levels': fib_levels,
        'psychological': psychological_levels
    }

def frange(start, stop, step):
    levels = []
    while start < stop:
        levels.append(start)
        start += step
    return levels

def calculate_atr(symbol, period=14, timeframe=mt5.TIMEFRAME_H1):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period+1)

    if rates is None or len(rates) < period + 1:
        return 0

    true_ranges = []
    for i in range(1, len(rates)):
        high = rates[i]['high']
        low = rates[i]['low']
        prev_close = rates[i-1]['close']
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    atr = sum(true_ranges[:period]) / period
    return atr

def has_fast_market_movement(symbol):
    atr = calculate_atr(symbol, period=14, timeframe=mt5.TIMEFRAME_H4)
    if atr == 0:
        return False

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False
    current_price = tick.ask if tick.ask != 0 else tick.bid
    if current_price == 0:
        return False

    atr_percent = (atr / current_price) * 100
    threshold = config['monitoring'].get('fast_market_threshold', 1.0)

    return atr_percent >= threshold

def calculate_sl_tp(symbol, action, entry_price):
    info = mt5.symbol_info(symbol)
    point = info.point if info else 0.0001

    levels = find_key_levels(symbol, timeframe=mt5.TIMEFRAME_H4)
    atr_period = 14
    atr = calculate_atr(symbol, atr_period, timeframe=mt5.TIMEFRAME_H1)
    atr_multiplier = config['trade_rules'].get('atr_multiplier', 1.5)

    if action == 'buy':
        if levels and levels['swing_lows']:
            nearest_swing_low = min(levels['swing_lows'])
        else:
            nearest_swing_low = entry_price - 100 * point
        sl_candidate1 = nearest_swing_low - 3 * point
        sl_candidate2 = entry_price - atr_multiplier * atr
        sl = min(sl_candidate1, sl_candidate2)

        risk = entry_price - sl
        tp_candidate1 = entry_price + 2 * risk
        tp_candidate2 = levels['fib_levels'].get('0.618', entry_price + 2 * risk) if levels else entry_price + 2 * risk
        tp = max(tp_candidate1, tp_candidate2)
    else:
        if levels and levels['swing_highs']:
            nearest_swing_high = max(levels['swing_highs'])
        else:
            nearest_swing_high = entry_price + 100 * point
        sl_candidate1 = nearest_swing_high + 3 * point
        sl_candidate2 = entry_price + atr_multiplier * atr
        sl = max(sl_candidate1, sl_candidate2)

        risk = sl - entry_price
        tp_candidate1 = entry_price - 2 * risk
        tp_candidate2 = levels['fib_levels'].get('0.618', entry_price - 2 * risk) if levels else entry_price - 2 * risk
        tp = min(tp_candidate1, tp_candidate2)

    return sl, tp

def sniper_analysis(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 50)
    if rates is None or len(rates) < 50:
        return None

    close_prices = [r['close'] for r in rates]
    rsi_value = get_rsi(close_prices, period=7)
    if rsi_value is None:
        return None

    last_candle = rates[-1]
    prev_candle = rates[-2]

    sma_20 = sum(close_prices[-20:]) / 20

    avg_volume = sum([r['tick_volume'] for r in rates[-5:]]) / 5

    def body_ratio(candle):
        body = abs(candle['close'] - candle['open'])
        range_ = candle['high'] - candle['low']
        return body / range_ if range_ != 0 else 0

    body_ratio_last = body_ratio(last_candle)
    body_ratio_prev = body_ratio(prev_candle)

    if rsi_value < 40:
        if (last_candle['close'] > last_candle['open'] and
            prev_candle['close'] < prev_candle['open'] and
            last_candle['close'] > sma_20 and
            last_candle['tick_volume'] > avg_volume * 1.2 and
            body_ratio_last > 0.4):
            return 'buy'

    if rsi_value > 60:
        if (last_candle['close'] < last_candle['open'] and
            prev_candle['close'] > prev_candle['open'] and
            last_candle['close'] < sma_20 and
            last_candle['tick_volume'] > avg_volume * 1.2 and
            body_ratio_last > 0.4):
            return 'sell'

    return None

filling_mode_cache = {}

def detect_valid_filling_mode(symbol, order_type, lot):
    if symbol in filling_mode_cache:
        return filling_mode_cache[symbol]

    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if symbol_info is None or tick is None:
        return None

    test_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    test_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK]
    deviation = config['trade_rules'].get('deviation', 10)
    use_sl_tp = config['trade_rules'].get('use_sl_tp', False)
    sl_pips = config['trade_rules'].get('sl_pips', 50)
    tp_pips = config['trade_rules'].get('tp_pips', 100)
    action = 'buy' if order_type == mt5.ORDER_TYPE_BUY else 'sell'
    sl, tp = calculate_sl_tp(symbol, action, test_price)

    if use_sl_tp:
        if order_type == 'buy':
            sl = test_price - sl_pips * symbol_info.point
            tp = test_price + tp_pips * symbol_info.point
        else:
            sl = test_price + sl_pips * symbol_info.point
            tp = test_price - tp_pips * symbol_info.point

    for mode in test_modes:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": test_price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": 234000,
            "comment": f"Test filling",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mode
        }
        result = mt5.order_check(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE or result._asdict()['comment'] == 'Done':
            filling_mode_cache[symbol] = mode
            print(f"[FILLING] Valid (checked) filling mode for {symbol}: {mode}")
            return mode
        elif result._asdict()['comment'] == 'No money':
            print(f"[FILLING] No money, please refill your trading account")
            return False
        elif result._asdict()['comment'] == 'Unsupported filling mode':
            filling_mode_cache[symbol] = mode
            continue
        else:
            print(f"[FILLING] No valid (checked) filling mode for {symbol}")
            print(f"[FILLING] Comment: {result._asdict()['comment']}")
            return None

    fallback_mode = symbol_info.filling_mode
    filling_mode_cache[symbol] = fallback_mode
    return fallback_mode

def monitor_profit_and_close(interval=1.0):
    global set_tp
    if set_tp is not None:
        threshold_profit=float(set_tp)
        def monitor():
            while True:
                if set_tp is not None:
                    positions = mt5.positions_get()
                    if positions:
                        for pos in positions:
                            profit = pos.profit
                            if profit >= threshold_profit:
                                close_position_in_profit(pos)
                    time.sleep(interval)
                else:
                    break

        threading.Thread(target=monitor, daemon=True).start()
    else:
        pass

def close_position_in_profit(pos):
    symbol = pos.symbol
    ticket = pos.ticket
    lot = pos.volume
    price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": ticket,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "comment": "Auto close by profit monitor",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[✓] Posisi {ticket} ditutup (profit {pos.profit})")
    else:
        print(f"[X] Gagal menutup posisi {ticket}, kode: {result.retcode}")

def open_position(symbol, action):
    global recorded, recorded2, set_lot
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.ask == 0 or tick.bid == 0:
        return

    base_lot = config['trade_rules'].get('lot_size', 0.01)
    deviation = config['trade_rules'].get('deviation', 10)
    use_sl_tp = config['trade_rules'].get('use_sl_tp', True)
    sl_pips = config['trade_rules'].get('sl_pips', 50)
    tp_pips = config['trade_rules'].get('tp_pips', 100)

    price = tick.ask if action == 'buy' else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL

    sl, tp = calculate_sl_tp(symbol, action, price)

    if use_sl_tp:
        if action == 'buy':
            sl = price - sl_pips * symbol_info.point
            tp = price + tp_pips * symbol_info.point
        else:
            sl = price + sl_pips * symbol_info.point
            tp = price - tp_pips * symbol_info.point

    risk_percent = config['trade_rules'].get('risk_percent', 1)
    lot = calculate_position_size(symbol, price, sl, risk_percent)

    filling_mode = detect_valid_filling_mode(symbol, order_type, lot)
    if filling_mode is None:
        bot.send_message(user_id, f"❌ No valid filling mode for {symbol}, skipping order.")
        return
    elif filling_mode is False:
        return

    if set_lot is not None:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": set_lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": 234000,
            "comment": f"LucyNetwork-{action.upper()}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode
        }
    else:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": 234000,
            "comment": f"LucyNetwork-{action.upper()}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode
        }

    result = mt5.order_send(request)
    if result is None:
        return

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        msg = (
            f"[EXECUTE] ✅ Success! {action.upper()} {symbol}:\n"
            f"[EXECUTE] {action.upper()} {symbol} @ {price:.2f} (Lot={lot}, SL={sl:.4f}, TP={tp:.4f})"
        )
        bot.send_message(user_id, msg)
        print(msg)
        recorded = False
        recorded2 = False
    else:
        msg = (
            f"[EXECUTE] ❌ Failed! {action.upper()} {symbol}:\n"
            f"[EXECUTE] Retcode: {result.retcode}\n"
            f"[EXECUTE] Deskripsi: {mt5.last_error()}\n"
            f"[EXECUTE] Detail: {result._asdict()}"
        )
        print(msg)
        if 'No money' in msg:
            if not recorded:
                bot.send_message(user_id, "❌ Balance is not enough to open an order position")
                recorded = True
        elif 'Market closed' in msg:
            if not recorded2:
                bot.send_message(user_id, f"❌ Market is closed - {symbol}")
                recorded2 = True
        elif 'Requote' in msg:
            result = mt5.order_send(request)

def place_pending_order(symbol, order_type, price):
    global recorded, recorded2, set_lot
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return

    point = symbol_info.point
    deviation = config['trade_rules'].get('deviation', 10)
    use_sl_tp = config['trade_rules'].get('use_sl_tp', True)
    sl_pips = config['trade_rules'].get('sl_pips', 50)
    tp_pips = config['trade_rules'].get('tp_pips', 100)
    risk_percent = config['trade_rules'].get('risk_percent', 1)

    if order_type == 'buy_limit':
        sl = price - sl_pips * point
        tp = price + tp_pips * point
        mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
    elif order_type == 'sell_limit':
        sl = price + sl_pips * point
        tp = price - tp_pips * point
        mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
    else:
        print(f"[PENDING] ❌ Invalid order type: {order_type}")
        return

    lot = set_lot if set_lot is not None else calculate_position_size(symbol, price, sl, risk_percent)
    filling_mode = detect_valid_filling_mode(symbol, mt5_order_type, lot)
    if filling_mode is None:
        bot.send_message(user_id, f"❌ No valid filling mode for {symbol}, skipping pending order.")
        return
    elif filling_mode is False:
        return

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot,
        "type": mt5_order_type,
        "price": price,
        "sl": sl if use_sl_tp else 0.0,
        "tp": tp if use_sl_tp else 0.0,
        "deviation": deviation,
        "magic": 234001,
        "comment": f"LucyNetwork-PENDING-{order_type.upper()}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode
    }

    result = mt5.order_send(request)
    if result is None:
        return

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        msg = (
            f"[PENDING] ✅ Pending {order_type.upper()} set:\n"
            f"{symbol} @ {price:.5f} (Lot={lot}, SL={sl:.5f}, TP={tp:.5f})"
        )
        bot.send_message(user_id, msg)
        print(msg)
    else:
        msg = (
            f"[PENDING] ❌ Failed to set pending order:\n"
            f"Symbol: {symbol}, Retcode: {result.retcode}\n"
            f"Deskripsi: {mt5.last_error()}\n"
            f"Detail: {result._asdict()}"
        )
        print(msg)
        if 'No money' in msg:
            if not recorded:
                bot.send_message(user_id, "❌ Balance is not enough to set a pending order")
                recorded = True
        elif 'Market closed' in msg:
            if not recorded2:
                bot.send_message(user_id, f"❌ Market is closed - {symbol}")
                recorded2 = True

def check_and_trade(symbol):
    if not has_fast_market_movement(symbol):
        return

    rates_h4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)
    if rates_h4 is None:
        return
    close_prices_h4 = [r['close'] for r in rates_h4]

    rsi_value = get_rsi(close_prices_h4, period=14)
    if rsi_value is None:
        return

    rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
    if rates_h1 is None:
        return
    close_prices_h1 = [r['close'] for r in rates_h1]
    ema_h1 = get_ema(close_prices_h1, period=20)
    current_price_h1 = close_prices_h1[-1]
    low_thresh = config['monitoring'].get('momentum_rsi_threshold', 40)
    high_thresh = 100 - low_thresh

    if rsi_value < low_thresh and current_price_h1 > ema_h1:
        open_position(symbol, 'buy')
    elif rsi_value > high_thresh and current_price_h1 < ema_h1:
        open_position(symbol, 'sell')

    signal = sniper_analysis(symbol)
    if signal == 'buy':
        open_position(symbol, 'buy')
    elif signal == 'sell':
        open_position(symbol, 'sell')
    else:
        pass
    return True

def check_momentum(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M10, 0, 100)
    if rates is None:
        return False
    close_prices = [r['close'] for r in rates]
    rsi_value = get_rsi(close_prices, period=14)
    if rsi_value is None:
        return False
    return rsi_value < config['monitoring'].get('momentum_rsi_threshold', 14)

def connect_to_account():
    account = config['mt5']['accounts'][config['preferences']['active_account_index']]
    connected = mt5.initialize(
        path=config['mt5']['path'],
        login=account['login'],
        password=account['password'],
        server=account['server']
    )
    if not connected:
        error_msg = f"❌ Account connect to MetaTrader5 failed.\nError: {mt5.last_error()}"
        print(error_msg)
        bot.send_message(user_id, error_msg)
        exit()
    else:
        akun_nama = account.get('name', f"Akun {account['login']}")
        msg = f"✅ Account connect successfully to MetaTrader5: {akun_nama}"
        print(msg)
        bot.send_message(user_id, msg)

def starts(symbols):
    global tipe
    best_model = SuperHybridNN.load_model("/LucyRFX/lucy_model.pkl")

    for symbol in symbols:
        check_symbol_ready(symbol)
    feat = extract_features(symbol)
    feat = feat.reshape(1, -1)
    action, confidence = best_model.predict(feat)
    for s in symbols:
        if tipe == 0:
            symbol = s.name
            if 'Market closed' in symbol:
                print(f"[X] Market closed: {s}")
                continue
            if recorded:
                return
            if has_fast_market_movement(symbol):
                signal = sniper_analysis(symbol)
                if signal == 'buy':
                    print("Open position by: Sniper-Analytics")
                    open_position(symbol, 'buy')
                elif signal == 'sell':
                    print("Open position by: Sniper-Analytics")
                    open_position(symbol, 'sell')
                else:
                    print("[LUCYRFX] Open position by LucyRadarFX - AI Large Model" if ai_for_trade == 2 else "[AI-HK02] Open position by DeepNN - AI Large Model")
                    if mass:
                        for _ in range(repeating):
                            open_position(symbol, action) if ai_for_trade == 2 else ai_market_executor_with_model(symbol)
                    else:
                        open_position(symbol, action) if ai_for_trade == 2 else ai_market_executor_with_model(symbol)
            else:
                continue
        else:
            if mt5.symbols_get(s):
                symbol = s
            else:
                print(f"[X] No market named: {s}")
                continue
            if 'Market closed' in symbol:
                print(f"[X] Market closed: {s}")
                continue
            signal = sniper_analysis(symbol)
            if signal == 'buy':
                print("Open position by: Sniper-Analytics")
                open_position(symbol, 'buy')
            elif signal == 'sell':
                print("Open position by: Sniper-Analytics")
                open_position(symbol, 'sell')
            else:
                print("[LUCYRFX] Open position by LucyRadarFX - AI Large Model" if ai_for_trade == 2 else "[AI-HK02] Open position by DeepNN - AI Large Model")
                if mass:
                    for _ in range(repeating):
                        open_position(symbol, action) if ai_for_trade == 2 else ai_market_executor_with_model(symbol)
                else:
                    open_position(symbol, action) if ai_for_trade == 2 else ai_market_executor_with_model(symbol)
    return True

def aiinfo(model_types=["LucyNetwork", "HybridNN", "DeepNN"]):
    X, y = load_real_dataset()
    print("\n" + "="*60)
    print(f"[AI INFO] INPUT FEATURES : {X.shape[1]} | DATASET SIZE: {X.shape[0]}")
    print(f"[AI INFO] LABELS (Counts): SELL={np.sum(y==0)}, HOLD={np.sum(y==1)}, BUY={np.sum(y==2)}")

    for model_type in model_types:
        print("-" * 60)
        print(f"[MODEL] TYPE: {model_type}")

        if model_type == "DeepNN":
            model = DeepNN([X.shape[1], 512, 256, 64, 1])
            model.print_model_summary()

        elif model_type == "HybridNN":
            model = HybridNN(input_dim=X.shape[1])
            total_params = (
                sum(w.size + b.size for w, b in model.branch_ind) +
                sum(w.size + b.size for w, b in model.branch_tex) +
                np.prod(model.Wh.shape) + np.prod(model.bh.shape) +
                np.prod(model.Wo.shape) + np.prod(model.bo.shape)
            )
            total_neurons = model.Wh.shape[1]
            p1, p2 = format_ribuan(int(total_params))
            print(f"[AI INFO] Total Params   : {p1} ({p2})")
            print(f"[AI INFO] Hidden Neurons : {total_neurons}")

        elif model_type == "LucyNetwork":
            model = SuperHybridNN(input_dim=X.shape[1])
            total_params = (
                sum(v.size for v in model.deep_branch.params.values()) +
                sum(v.size for v in model.action_branch.params.values()) +
                sum(w.size + b.size for w, b in model.hybrid_branch.branch_ind) +
                sum(w.size + b.size for w, b in model.hybrid_branch.branch_tex) +
                np.prod(model.hybrid_branch.Wh.shape) + np.prod(model.hybrid_branch.bh.shape) +
                np.prod(model.hybrid_branch.Wo.shape) + np.prod(model.hybrid_branch.bo.shape) +
                np.prod(model.final_W.shape) + np.prod(model.final_b.shape)
            )
            p1, p2 = format_ribuan(int(total_params))
            print(f"[AI INFO] Total Params   : {p1} ({p2})")
            print(f"[AI INFO] Deep Branch    : {model.deep_branch.layer_dims}")
            print(f"[AI INFO] Action Branch  : {model.action_branch.layer_dims}")
            print(f"[AI INFO] Hybrid Branch  : {model.hybrid_branch.Wh.shape[1]} hidden neurons")
            print(f"[AI INFO] Final Output   : {model.final_W.shape[1]} classes")

    print("="*60 + "\n")

def train_super_hybrid(X, y, epochs=30, batch_size=128, patience=5, save_path="/LucyRFX/super_hybrid_model.pkl"):
    ensure_folder_exists(save_path)

    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            model = pickle.load(f)
        print("[🔄] Loaded existing SuperHybridNN model.")
    else:
        model = SuperHybridNN(input_dim=X.shape[1])
        print("[🚀] Created new SuperHybridNN model.")

    model.train(X, y, epochs=epochs, batch_size=batch_size, patience=patience, save_path=save_path)
    model.save_model("/LucyRFX/super_hybrid_model.pkl")

    return model

def main_index(symbols):
    repeat =  starts(symbols)
    if repeat:
        while True:
            if not recorded:
                ask = input("[REPEATS] Entry again? Y/N: ").lower()
                if ask == 'y':
                    repeat = starts(symbols)
                else:
                    break
            else:
                break

def cable_ai_train(resume_or_nah, Epochs, batch_Size, pick):
    X, y = load_real_dataset("/LucyRFX/dataset_real.json")
    if pick is None:
        if not exists('/LucyRFX//LucyRFX/lucy_model.pkl'):
            if not exists('/LucyRFX/hybrid_model_real.pkl'):
                model1 = HybridNN(input_dim=X.shape[1])
            else:
                model1 = load_hybrid_model('/LucyRFX/hybrid_model_real.pkl')
            input_dim = X.shape[1]
            model1 = HybridNN(input_dim=input_dim)
            model1.train(X, y, epochs=Epochs+500, batch_size=batch_Size)
            save_hybrid_model(model1)
            ##################################################################
            model2 = SuperHybridNN(input_dim=29)

            model2.train(X, y, epochs=Epochs, batch_size=batch_Size, patience=9, save_path="/LucyRFX/lucy_model.pkl")

            best_model = SuperHybridNN.load_model("/LucyRFX/lucy_model.pkl")

            symbols = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "BTCUSD"]
            for symbol in symbols:
                check_symbol_ready(symbol)
            feat = extract_features(symbol)
            feat = feat.reshape(1, -1)
            action, confidence = best_model.predict(feat)
            print(f"[+] Lucy: {action} | Confidence: {confidence:.4f}")
            ##################################################################
            if resume_or_nah != 'yes':
                train_ai_model_multifeature(epochs=Epochs, batch_size=batch_Size)
            else:
                train_ai_model_multifeature(epochs=Epochs, batch_size=batch_Size)
            return
    elif pick is not None:
        print("[⚡] Cable AI Train Initialized")
        model = SuperHybridNN.load_model("/LucyRFX/lucy_model.pkl")

        if pick not in ['deep', 'lucy', 'hybrid']:
            print(f"[!] Pilihan model tidak valid: {picks}")
            return

        X, y = load_real_dataset("/LucyRFX/dataset_real.json")
        print(f"[~] Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features")

        if pick == 'lucy':
            if model is None:
                print("[+] Membuat model baru Lucy (SuperHybridNN)")
                model = SuperHybridNN(input_dim=X.shape[1])
            else:
                print("[~] Melanjutkan model Lucy (SuperHybridNN) yang sudah ada")
            model.train(X, y, epochs=Epochs, batch_size=batch_Size, patience=9, save_path="/LucyRFX/lucy_model.pkl")
            best_model = SuperHybridNN.load_model("/LucyRFX/lucy_model.pkl")
            symbols = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "BTCUSD"]
            for symbol in symbols:
                check_symbol_ready(symbol)
            feat = extract_features(symbol)
            feat = feat.reshape(1, -1)
            action, confidence = best_model.predict(feat)
            print(f"[+] Lucy: {action} | Confidence: {confidence:.4f}")
        elif pick == 'hybrid':
            if not exists('/LucyRFX/hybrid_model_real.pkl'):
                model1 = HybridNN(input_dim=X.shape[1])
            else:
                model1 = load_hybrid_model('/LucyRFX/hybrid_model_real.pkl')
            input_dim = X.shape[1]
            model1 = HybridNN(input_dim=input_dim)
            model1.train(X, y, epochs=Epochs+500, batch_size=batch_Size)
            save_hybrid_model(model1)
        elif pick == 'deep':
            model = DeepNN(layer_dims=[X.shape[1], 512, 256, 64, 1])
            model.train(X, y, epochs=epochs, batch_size=batch_size)
            with open("/LucyRFX/deep_model.pkl", "wb") as f:
                pickle.dump(model, f)
            print("[💾] DeepNN model saved!")

        print("[✔] Cable AI Train Selesai")
    else:
        return

def monitor_positions():
    while True:
        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                entry_price = pos.price_open
                if trailing:
                    apply_trailing_stop(pos, entry_price)
                else:
                    pass
        time.sleep(1)

if __name__ == '__main__':
    if not exists('LucyRFX'):
        os.makedirs('LucyRFX')
    try:
        if exists('config.json'):
            connect_to_account()
        else:
            pass
        if not exists('/LucyRFX/dataset_real.json'):
            symbols = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "BTCUSD"]
            for symbol in symbols:
                check_symbol_ready(symbol)
            save_dataset(symbols)
        else:
            pass
        if not exists('/LucyRFX/ai_training_dataset_multifeature.json') or not exists('/LucyRFX/ai_model_multifeature_parameters_improved.json'):
            print('[!] AI Model not detected, training new AI model.. (please be patient)')
            cable_ai_train('nah', 15, 1000, None)
        else:
            pass
        threading.Thread(target=auto_close_positions, daemon=True).start()
        threading.Thread(target=monitor_positions, daemon=True).start()
        monitor_profit_and_close()
        while True:
            options = input('\n┌(SoloAIV2 Trading)-()\n┕━>').lower()
            if options == 'single execute':
                mass = False
                custom_market = input("[!] Would you like to use custom market? Y/N: ").lower()
                if custom_market == 'y':
                    tipe = 1
                    market_name = input("[~] Market list [e.g: EURUSD,XAUUSD,GBPJPY]: ").upper().split(',')
                    main_index(market_name)
                elif custom_market == 'n':
                    tipe = 0
                    market_name = mt5.symbols_get()
                    main_index(market_name)
                else:
                    pass
            if options == 'mass execute':
                mass = True
                tipe = 1
                market_name = input("[~] Market [e.g: BTCUSD]: ").upper().split(',') or None
                repeating = input("[~] Repeat for [e.g: 5]: ") or None
                if market_name and repeating:
                    repeating = int(repeating)
                    main_index(market_name)
                else:
                    pass
            elif options == 'train ai':
                print("[+] Epochs & Batch size are optional, skip if you don't wanna fill")
                epochs = int(input("[!] Epochs (default 15): ") or 15)
                batch_size = int(input("[!] Batch size (default 100000): ") or 100000)
                picks = str(input("[!] Model to train(deep, lucy, hybrid): ") or None).lower()
                cable_ai_train('nah', epochs, batch_size, picks)
                while True:
                    train_or_trade = input('\n[!] Would you like to train the AI again? Y/N: ').lower()
                    if train_or_trade == 'y':
                        cable_ai_train('yes', epochs, batch_size, picks)
                    else:
                        break
            elif options == 'buy license':
                print("[!] Purchase new License-ID at: https://lucynet.serveo.net/payments")
            elif options == 'expired':
                verify_license_secure()
            elif options == 'licenseinfo':
                licenses = LID()
                print(f"[+] License-ID: {licenses}")
                verify_license_secure()
            elif options == 'onlineuser':
                useronline = check_user()
                print(f"[+] Total user online today: {useronline}")
            elif options == 'aiinfo':
                aiinfo()
            elif options == 'disclaimer':
                print(disclaimer)
            elif options == 'help':
                help = """
                ╔════════════════════════════════════════════════════════════╗
                ║                   SOLO AI V2 - HELP MENU                   ║
                ╠════════════════════════════════════════════════════════════╣
                ║ COMMAND         │ DESCRIPTION                              ║
                ╟─────────────────┼──────────────────────────────────────────╢
                ║ Single Execute  │ Mass symbols scanning and auto-trade     ║
                ║ Mass Execute    │ Execute 1 symbols and execute rapidly    ║
                ║ Train AI        │ Train the AI model for better accuracy   ║
                ║ Expired         │ Show software license expiration date    ║
                ║ Buy License     │ Purchase or extend license key           ║
                ║ Help            │ Display this help menu                   ║
                ║ LicenseInfo     │ Show license status and Lic-ID           ║
                ║ OnlineUser      │ Show online user                         ║
                ║ AIInfo          │ Show AI parameter& neurons               ║
                ║ Disclaimer      │ Disclaimer about using this software     ║
                ║ SettingMT5      │ Set your custom lot, max tp, etc..       ║
                ║ AIExecutor      │ Choose AI as executor / trader           ║
                ║ Exit            │ Close this software                      ║
                ╚════════════════════════════════════════════════════════════╝
                TIP: Kill your emotion so you wouldn't stress when you got MC
                Note: Please use CTRL+C or type 'exit' if you wanna close this
                      software
                """
                print(help)
            elif options == 'settingmt5':
                help = """
                ╔════════════════════════════════════════════════════════════╗
                ║                  SOLO AI V2 - Setting Mt5                  ║
                ╠════════════════════════════════════════════════════════════╣
                ║ COMMAND         │ DESCRIPTION                              ║
                ╟─────────────────┼──────────────────────────────────────────╢
                ║ SetCustomTP     │ Set maximum TP in USD                    ║
                ║ SetCustomLOT    │ Set maximum LOT for each trades          ║
                ║ DisTrailing     │ Disable trailing stop(risk at your own)  ║
                ║ SetTrailing     │ Set trailing stop at pullback .. USD     ║
                ║ SetTrailSet     │ Set trailing stop every profit .. USD    ║
                ║ Exit            │ Close this software                      ║
                ╚════════════════════════════════════════════════════════════╝
                TIP: Kill your emotion so you wouldn't stress when you got MC
                Note: Please use CTRL+C or type 'exit' if you wanna close this
                      software, SetTrailing must be lower than SetTrailSet!
                """
                print(help)
            elif options == 'aiexecutor':
                print("[~] Available model: DeepNN, LucyNetwork")
                print(aiinfo(model_types=["LucyNetwork", "DeepNN"]))
                print("[=] LucyNetwotk   ; (Big Parameter, LLM, Data-Based Analyze) - (Good for swing/day)")
                print("[=] DeepNN        ; (Small Parameter, Enchanced, Advanced, MLM) - (Good for scalping)")
                choices = input("[~] DeepNN/LucyNetwork: ").lower()
                if choices:
                    if choices == 'deepnn':
                        ai_for_trade = 1
                        print(f"[+] Model set to: DeepNN")
                    elif choices == 'lucynetwork':
                        ai_for_trade = 2
                        print(f"[+] Model set to: LucyNetwork")
                else:
                    pass
            elif options == 'distrailing':
                trailing = False
                print("[!] SL+ / Trailing has been disabled, type 'enbtrailing' to enable.")
            elif options == 'enbtrailing':
                print('[~] Enabled.')
                trailing = True
            elif options == 'settrailing':
                usd_input = input("[~] Stop when down at(USD. skip to set default): ")
                if usd_input:
                    global_trailiing_usd = float(usd_input)
                    print(f"[!] SL+ / Trailing has been set at {global_trailiing_usd}")
                else:
                    global_trailiing_usd = 0.30
            elif options == 'settrailset':
                trailset_input = input("[~] Set trailing every profit(USD): ")
                if trailset_input:
                    global_trailiing_set = float(trailset_input)
                    print(f"[!] Trailing has been set every profit {global_trailiing_set} USD")
                else:
                    global_trailiing_set = 0.50
            elif options == 'setcustomtp':
                max_profit = input('[+] Max TP(USD): ') or None
                set_tp = max_profit
                if max_profit is not None:
                    print(f"[!] Profit has been set for: {set_tp}")

            elif options == 'setcustomlot':
                max_lot = input('[+] Set LOT: ') or None
                set_lot = float(max_lot) if max_lot is not None else None
                if max_lot is not None:
                    print(f"[!] Lot has been set for: {set_lot}")

            elif options == 'exit':
                print("\n[!] Please wait a while before exiting..")
                kill_process()
    except KeyboardInterrupt:
        print("\n[!] Please wait a while before exiting..")
        signal.signal(signal.SIGINT, lambda: kill_process())
