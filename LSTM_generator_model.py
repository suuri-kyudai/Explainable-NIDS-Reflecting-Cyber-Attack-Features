"""
v3における変更点
- Attentionに対応

v4における変更点
- マスクの仕組みの変更に対応
"""

import time
import csv
import collections
import tensorflow.keras as keras
import os
import warnings
# import gensim
import math
import sklearn
import matplotlib
import pandas as pd
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
# import lime
# import lime.lime_tabular
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import pprint

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras import backend as K
# from livelossplot import PlotLossesKeras
from plot_keras_history import show_history, plot_history
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import Callback
from collections import defaultdict

from LSTM_model import generate_simple_LSTM_model, generate_Hwang_LSTM_model, generate_Hwang_LSTM_model_v2,\
    generate_Hwang_LSTM_model_v4, generate_Hwang_LSTM_model_v4a, generate_Hwang_LSTM_model_v4f,\
    generate_Hwang_LSTM_model_v4f_FAPI_multi_packet, generate_Hwang_LSTM_model_v4f_FAPI_multi_packet_attention,\
    generate_Hwang_LSTM_model_v4f_FAPI_multi_packet_multi_attention,\
    generate_Hwang_LSTM_model_v5_FAPI_multi_packet,\
    generate_Hwang_LSTM_model_v5_FAPI_multi_packet_attention

# seed固定
seed = 1
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
tf.random.set_seed(seed)

warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)

WITH_ATTENTION = True # Attention Scoreを出力する場合はTrueにする
HPLEN = 700
n_packets = 2
DEBUG = False # 動作確認用
if DEBUG==True:
    SAVE_PATH = f"/mnt/fuji/kawanaka/results/test" # 動作確認用
elif WITH_ATTENTION == True:
    SAVE_PATH = f"/mnt/fuji/kawanaka/results/LSTM_multipackets_attention_{HPLEN}_np{n_packets}_v3_tuned"
else:
    SAVE_PATH = f"/mnt/fuji/kawanaka/results/LSTM_multipackets_{HPLEN}_np{n_packets}_v3_tuned"
os.makedirs(SAVE_PATH, exist_ok=True)

if DEBUG==True:
    EPOCHS = 1 # 動作確認用
else:
    EPOCHS = 3

TRAIN_PATH = f"/mnt/fuji/kawanaka/data/CSE-CIC-IDS2018_0222_field_{HPLEN}f_mask.csv"
TEST_PATH = f"/mnt/fuji/kawanaka/data/CSE-CIC-IDS2018_0223_field_{HPLEN}f_mask.csv"
VALIDATION_PATH = f"/mnt/fuji/kawanaka/data/CSE-CIC-IDS2018_0223_field_{HPLEN}f_mask.csv"

testtime_file = 0.0

# seed固定
seed = 1
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
tf.random.set_seed(seed)

'''
lstmの定義，実行プログラム
評価段階では混同行列の値を返す
'''

# GPU指定
gpu_id = 6 # GPUのポート番号
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)

def label_txt2num(label):
    """文字列形式のラベルを数値に変換

    Args:
        label (str): 文字列形式ラベル

    Returns:
        int: 数値形式ラベル
    """    
    if label == 'BENIGN':
        return 0
    elif label == 'BruteForce-Web':
        return 1
    elif label == 'BruteForce-XSS':
        return 2
    elif label == 'SQLInjection':
        return 3

def np_min_nonzero(a):
    return np.min([np.max(a) if v == 0 else v for v in a])

class masked_dataset():
    """データセットを格納するためのクラス

    Attributes:
        packets (np.ndarray[int]): 処理済みのパケット
        labels (np.ndarray[int]): ラベル（数値）
        labels_raw (np.ndarray[str]): ラベル（文字列）
        flow_ids (np.ndarray[int]): フローID
        labels_flow (np.ndarray[int]): フローIDごとのラベル（数値）
        labels_flow_raw (np.ndarray[str]): フローIDごとのラベル（文字列）
        length (int): パケット数
    """

    def __init__(self, data_path, n_samples, n_packets=3, binary=True):
        """データセットを読み込む

        Args:
            data_path (str): データセット（csvファイル）のパス
            n_samples (int): サンプル数
            n_packets (int, optional): 1サンプルに対するパケット数（デフォルトは3）
            binary (bool, optional): 二値分類フラグ（デフォルトはTrue）
        """        
        
        list_packets = []
        list_labels = []
        list_labels_raw = []
        list_flow_ids = []
        list_labels_flow = []
        list_labels_flow_raw = []
        self.length = n_samples
        
        with open(data_path, 'r') as f:
            next(f)  # 1行目（csvヘッダ）はとばす
            reader = csv.reader(f)  # float型で読み込み
            packets_of_flows = {} # フローごとに過去のパケットを保持
            
            input_len = HPLEN * n_packets + n_packets # 区切り・終端を含む

            for row in tqdm(reader, total=n_samples):  # 1行ずつ処理
                
                # ラベルの処理
                label_raw = row[0]
                label = label_txt2num(row[0])  #csvの1列目のラベルを入力
                if (label > 0) and (binary == True): # 二値分類の場合
                    label = 1
                list_labels.append(label)  # ラベルを追加
                list_labels_raw.append(label_raw)

                # フローIDの処理
                flow_id = int(row[1])
                list_flow_ids.append(flow_id)

                # パケットの処理
                packet = row[2:]  # csvのパケット部分のベクトルを入力
                packet = [int(s) for s in packet]  #strをintに変換
                if 0 in packet:
                    packet[packet.index(0):] = [] # 末尾のパディング部を一旦除去

                if flow_id not in packets_of_flows: # 新しいフローIDの場合
                    list_labels_flow.append(label) # フローのラベルを追加
                    list_labels_flow_raw.append(label_raw)
                    packets_of_flows[flow_id] = [0] * input_len # 入力ベクトルを初期化
                    for i in range(n_packets):
                        packets_of_flows[flow_id][i] = 0x10002
                else: # 既存のフローIDの場合
                    if list_labels_flow_raw[flow_id] == 'BENIGN' and label_raw != 'BENIGN':
                        list_labels_flow[flow_id] = label
                        list_labels_flow_raw[flow_id] = label_raw
                
                if 0 in packets_of_flows[flow_id]:
                    packets_of_flows[flow_id][packets_of_flows[flow_id].index(0):] = [] # パディングがある場合はそれを削除
                packets_of_flows[flow_id][:packets_of_flows[flow_id].index(0x10002)+1] = [] # 古いパケットを削除
                packets_of_flows[flow_id].extend(packet) # 入力ベクトルにパケットを追加
                packets_of_flows[flow_id].append(0x10002) # 終端を追加
                if len(packets_of_flows[flow_id]) < input_len:
                    packets_of_flows[flow_id].extend([0] * (input_len - len(packets_of_flows[flow_id])))# 不足分をパディング
                assert len(packets_of_flows[flow_id]) == input_len
                list_packets.append(deepcopy(packets_of_flows[flow_id]))  # パケットのリストに追加
        
        self.packets = np.array(list_packets)
        self.labels = np.array(list_labels)
        self.labels_raw = np.array(list_labels_raw)
        self.flow_ids = np.array(list_flow_ids)
        self.labels_flow = np.array(list_labels_flow)
        self.labels_flow_raw = np.array(list_labels_flow_raw)
    
    def data_tuple(self):
        """(packets, labels)のタプルを返す

        Returns:
            tuple(np.ndarray[int], np.ndarray[int]): パケットとラベルのndarrayのタプル
        """        
        return (self.packets, self.labels)
    
    def data_remask(self, mask):
        """マスクをさらに追加したパケットを返す

        Args:
            mask (np.ndarray[bool]]): マスクの有無を表す配列（Falseでマスクする）

        Returns:
            np.ndarray[int]: maskに従って処理を行ったパケット
        """        
        plist = []
        for i, packet in enumerate(self.packets):
            p_remask = []
            for j, field in enumerate(packet):
                if mask[i][j] == False and field > 0 and field < 0x10001: # マスクする場合（0と0x10002は変更しない）
                    p_remask.append(0x10001)
                else:
                    p_remask.append(field)
            plist.append(p_remask)
        return np.array(plist)


class F1Callback(Callback):    
    """f1score計算用のコールバック

    Attributes:
        f1s (list[float]): f1score記録用のリスト
    """

    def __init__(self):
        """記録用リストの初期化
        """        
        self.f1s = []

    def on_epoch_end(self, epoch, logs):
        """f1scoreの記録（各エポック終了後に実行）
        """        
        eps = np.finfo(np.float32).eps
        if WITH_ATTENTION == True:
            recall = logs["val_output_true_positives"] / (logs["val_output_possible_positives"] + eps)
            precision = logs["val_output_true_positives"] / (logs["val_output_predicted_positives"] + eps)
        else:
            recall = logs["val_true_positives"] / (logs["val_possible_positives"] + eps)
            precision = logs["val_true_positives"] / (logs["val_predicted_positives"] + eps)
        f1 = 2*precision*recall / (precision+recall+eps)
        print("f1_val (from log) =", f1)

        self.f1s.append(f1)

def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

def possible_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true, 0, 1)))

def predicted_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_pred, 0, 1)))

def get_embedding_matrix(model, word_index):
    """
    keras.layers.Embeddingのweights引数で指定するための重み行列作成
    model: gensim model
    num_word: modelのvocabularyに登録されている単語数
    emb_dim: 分散表現の次元
    word_index: gensim modelのvocabularyに登録されている単語名をkeyとし、token idをvalueとする辞書 ex) {'word A in gensim model vocab': integer token id}
    """
    # gensim modelの分散表現を格納するための変数を宣言
    embedding_matrix = np.zeros((max(list(word_index.values())) + 1, model.vector_size), dtype="float32")

    # 分散表現を順に行列に格納する
    for word, label in word_index.items():
        try:
            # gensimのvocabularyに登録している文字列をembedding layerに入力するone-hot vectorのインデックスに変換して、該当する重み行列の要素に分散表現を代入
            embedding_matrix[label] = model.wv[word]
        except KeyError:
            pass
    return embedding_matrix

def predict_lime(func):
    """LIME用model.predict（二値分類用）のラッパー関数

    model.predictの出力をLIMEで受け付ける形式に変換する

    Args:
        func (func): 予測で使用しているmodel.predict

    Returns:
        np.ndarray[list[list[float]]]: 予測クラスが0となる確率と，1となる確率のリスト
    """    
    def wrapper(*args,**kwargs):
        result = func(*args,**kwargs)
        result = result.flatten()
        ret = np.array([[1 - y, y] for y in result])
        print(ret)
        return ret
    return wrapper

def confusion_matrix(save_path, pred, dataset):
    """混同行列を求める

    Args:
        save_file (str): 混同行列の保存先のパス
        pred (np.ndarray[int]): 予測結果（0または1）
        dataset (masked_dataset): データセットのクラス
    """
    # 混同行列の初期化
    pred_dict = {
            "BENIGN":
                {"positive": 0,
                 "negative": 0,},
            "BruteForce-Web":
                {"positive": 0,
                 "negative": 0,},
            "BruteForce-XSS":
                {"positive": 0,
                 "negative": 0,},
            "SQLInjection":
                {"positive": 0,
                 "negative": 0,},
            }
    pred_dict_flow = {
            "BENIGN":
                {"positive": 0,
                 "negative": 0,},
            "BruteForce-Web":
                {"positive": 0,
                 "negative": 0,},
            "BruteForce-XSS":
                {"positive": 0,
                 "negative": 0,},
            "SQLInjection":
                {"positive": 0,
                 "negative": 0,},
            }

    # 実験結果の読み込みと保存ファイルの準備
    rfile = open(save_path, "w") 
    
    # 混同行列の作成処理
    flow_pred = [0] * dataset.labels_flow_raw.shape[0]
    print("Processing the prediction results...")
    for i in tqdm(range(dataset.labels_raw.shape[0])):
        p = pred[i] # 予測結果
        l = dataset.labels_raw[i] # ラベル
        s = dataset.flow_ids[i] # フロー番号
        sl = dataset.labels_flow_raw[s] # フローのラベル（攻撃パケットがある場合はそのラベル）
        if p == 0:
            pred_dict[l]["negative"] += 1
        elif p == 1:
            pred_dict[l]["positive"] += 1
        if p == 1 and flow_pred[s] == 0:
            flow_pred[s] = 1
    
    # フローごとの処理
    print("Processing the prediction results of flows...")
    for i in tqdm(range(len(flow_pred))):
        sp = flow_pred[i]     # ストリームごとの予測結果
        sl = dataset.labels_flow_raw[i]   # ストリームのラベル（攻撃パケットがある場合はそのラベル）
        if sp == 0:
            pred_dict_flow[sl]["negative"] += 1
        elif sp == 1:
            pred_dict_flow[sl]["positive"] += 1

    rfile.write("Packet\n")
    pprint.pprint(pred_dict, stream=rfile)

    num_posi = 0 # num of positive prediction
    num_nega = 0 # num of negative prediction
    num_posi_f = 0 # num of positive prediction (flow)
    num_nega_f = 0 # num of negative prediction (flow)
    for v in pred_dict.values():
        num_posi += v["positive"]
        num_nega += v["negative"]
    for v in pred_dict_flow.values():
        num_posi_f += v["positive"]
        num_nega_f += v["negative"]
    num_total = num_posi + num_nega
    num_total_f = num_posi_f + num_nega_f
    rfile.write("total: \n")
    rfile.write("\tpositive: {}\n".format(num_posi))
    rfile.write("\tnegative: {}\n".format(num_nega))
    rfile.write("\tsum: {}\n".format(num_total))
    rfile.write("\n")

    rfile.write("Stream\n")
    pprint.pprint(pred_dict_flow, stream=rfile)
    rfile.write("total (flow): \n")
    rfile.write("\tpositive (flow): {}\n".format(num_posi_f))
    rfile.write("\tnegative (flow): {}\n".format(num_nega_f))
    rfile.write("\tsum (flow): {}\n".format(num_total_f))


def _lstm():    
    # モデルの生成
    global testtime_file
    n_inputs = None  # 特徴ベクトルの次元数(固定長)

    # モデルの呼び出し
    if WITH_ATTENTION == True:
        # model = generate_Hwang_LSTM_model_v4f_FAPI_multi_packet_multi_attention(input_dim=n_inputs, n_packets=n_packets, output_units=1)
        # model = generate_Hwang_LSTM_model_v4f_FAPI_multi_packet_attention(input_dim=n_inputs, n_packets=n_packets, output_units=1)
        model = generate_Hwang_LSTM_model_v5_FAPI_multi_packet_attention(input_dim=n_inputs, n_packets=n_packets, output_units=1)
    else:
        model = generate_Hwang_LSTM_model_v5_FAPI_multi_packet(input_dim=n_inputs, n_packets=n_packets, output_units=1)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0008326022097793894)  # optimizer指定
    model.compile(optimizer=optimizer, loss="binary_crossentropy",\
                  metrics=['accuracy', true_positives, possible_positives, predicted_positives])  # モデルコンパイル
    model.summary(line_length=180)  # モデル表示
    print(model.outputs)

    f1cb = F1Callback()
    print("\n")

    # IPv6除去後 0222:84319, 0223:94201
    train_data_size = 84319  # 訓練データの全packet数：84319
    test_data_size = 229124  # テストデータの全packet数：229124
    test_data_size_na = 134923  # テストデータの除外するpacket数
    validation_data_size = 94201 # 検証データの全packet数：94201

    train_batch_size = 32  # 訓練データの入力batch_size
    validation_batch_size = train_batch_size  # 検証データの入力batch_size
    test_batch_size = 1 # testデータの入力batch_size，(test_data_size - test_data_size_na)の因数である必要がある

    # データセットの読み込み
    print("Loading training dataset...")
    dataset_train = masked_dataset(data_path=TRAIN_PATH, n_samples=train_data_size, n_packets=n_packets)
    print("Loading testing dataset...")
    dataset_test = masked_dataset(data_path=TEST_PATH, n_samples=validation_data_size, n_packets=n_packets)

    # モデルの学習
    print("fitting now ********************************************")
    fittime_start = time.time()  # 学習時間計測開始
    
    history = model.fit(dataset_train.packets, dataset_train.labels, epochs=EPOCHS, batch_size=train_batch_size,\
                            verbose=1, validation_data=dataset_test.data_tuple(), validation_batch_size=validation_batch_size,\
                                callbacks=[f1cb])

    fittime_end = time.time()  # 学習時間計測終了
    fittime = fittime_end - fittime_start

    # metricsのplot
    plot_history(history, path=os.path.join(SAVE_PATH, "acc_loss.png"))  #プロットの保存先指定
    plt.close()

    # f1-scoreのplot
    xlab = np.arange(EPOCHS) + 1
    plt.figure()
    plt.plot(xlab, np.array(f1cb.f1s), label="val_f1")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("f1 score")
    plt.savefig(os.path.join(SAVE_PATH, "f1.png"))  #プロットの保存先指定

    print("fittime:"+ str(fittime))


    # モデルのテスト
    print("\n")
    print("testing now ********************************************")
    testtime_start = time.time()  # テスト時間計測開始

    if WITH_ATTENTION == True:
        result, attention_score = model.predict(dataset_test.packets, verbose=1)
    else:
        result = model.predict(dataset_test.packets, verbose=1)
    
    testtime_end = time.time()  # テスト時間計測終了
    testtime = testtime_end - testtime_start - testtime_file # テスト時間 = 終了時刻 -　開始時刻 - ファイル読み込み時間

    print(np.shape(result))
    print(len(result))
    if WITH_ATTENTION == True:
        print(attention_score)
        print(np.shape(attention_score))
    np.savetxt(os.path.join(SAVE_PATH, "result_raw.csv"), result)  # 出力値（生の値）の保存先
    result_raw = deepcopy(result)

    print("**********************************")
    result = [1 if result[i] >= 0.5 else 0 for i in range(len(result))]  #0か1に分類
    np.savetxt(os.path.join(SAVE_PATH, "result_binary.csv"), result, fmt='%d')  # 出力値（二値）の保存先

    print("testtime:" + str(testtime))

    # モデルの評価
    print("\n")
    print("evaluating now ********************************************")
    TP = 0  # True Positive 真陽性
    FN = 0  # False Negative 偽陰性
    FP = 0  # False Positive 偽陽性
    TN = 0  # True Negative 真陽性
    assert len(dataset_test.labels) == len(result)  # 元のラベルと分類結果の個数が不一致であれば停止
    for i in range(len(result)):
        if dataset_test.labels[i] == 1 and result[i] == 1:
            TP += 1
        elif dataset_test.labels[i] == 1 and result[i] == 0:
            FN += 1
        elif dataset_test.labels[i] == 0 and result[i] == 1:
            FP += 1
        else:
            TN += 1

    print("\n")
    print("*******************************************************")
    print(f"TP: {TP}")
    print(f"FN: {FN}")
    print(f"FP: {FP}")
    print(f"TN: {TN}")
    print("\n")
    print(f"label 0: {TN + FP}")  # ラベルの個数の確認
    print(f"label 1: {TP + FN}")
    print("\n")
    print(f"fitting time = :{fittime}")
    print(f"testing time = :{testtime}")
    print("*******************************************************")

    if WITH_ATTENTION == True:

        print("\n")
        print("writing out attention score now ***********************")

        # Attention Score書き出し用pd.DataFrameの列名
        # at_columns = ["L3Type", "IPVersion", "IPHeaderLength", "TOS", "DatagramLength", "IPHeaderID", "IPFlag",\
        #                 "FlagmentOffset", "TTL", "ProtocolNumber", "IPHeaderChecksum", "SequenceNumber1",\
        #                     "SequenceNumber2", "ACKNumber1", "ACKNumber2", "TCPHeaderLength", "TCPFlag", "Window",\
        #                         "TCPChecksum", "UrgentPointer", "UDPHeaderLength", "UDPChecksum"] # 特徴量名リスト（field, header）
        # at_columns.extend([f"Payload_{i:03}" for i in range(HPLEN-22)]) # 特徴量名リストにペイロード部分(byte)を追加
        # attention_columns = []
        # for i in reversed(range(n_packets)):
        #     attention_columns.extend([f"{i}_{x}" for x in at_columns])
        at_index = [i for i in range(validation_data_size)]
        attention_columns = [f"f/b_{i}" for i in range(n_packets * HPLEN + n_packets)] # f/bはfield/byteのこと

        # pd.DataFrame作成と書き出し
        attention_score = attention_score.reshape((validation_data_size, n_packets * HPLEN + n_packets))
        attention_table_score = pd.DataFrame(attention_score, columns=attention_columns, index=at_index) # Attention Scoreのpd.DataFrameを作成
        attention_table_predict = pd.DataFrame(result_raw, columns=["Prediction"], index=at_index) # 予測結果をpd.DataFrameに変換
        attention_table_label = pd.DataFrame(np.array(dataset_test.labels_raw), columns=["Label"], index=at_index) # ラベルをpd.DataFrameに変換
        attention_table = pd.concat([attention_table_label, attention_table_predict, attention_table_score], axis=1)
        attention_table.to_csv(os.path.join(SAVE_PATH, "attention_score.zip"), compression='zip') # 容量が非常に大きくなるためzip圧縮

        # Attention Scoreが高い部分にマークを付ける
        print("marking parts with high attention score *****************")
        mark = []
        attention_score = np.array(attention_score)
        for attention_score_packet in tqdm(attention_score):
            score_max = np.max(attention_score_packet)
            score_min_nonzero = np_min_nonzero(attention_score_packet)
            score_border = score_min_nonzero + (score_max - score_min_nonzero) * 0.8
            mark_packet = [True if s >= score_border else False for s in attention_score_packet]
            mark.append(mark_packet)
        mark = np.array(mark)

        # Attention Scoreが高い部分のみを使用してテスト
        print("testing on data of high attention score *****************")
        result_h, _ = model.predict(dataset_test.data_remask(mark), verbose=1)
        np.savetxt(os.path.join(SAVE_PATH, "result_h.csv"), result_h)
        result_h = [1 if result_h[i] >= 0.5 else 0 for i in range(len(result_h))]
        np.savetxt(os.path.join(SAVE_PATH, "result_h_binary.csv"), result_h, fmt='%d')

        TP_h = 0  # True Positive 真陽性
        FN_h = 0  # False Negative 偽陰性
        FP_h = 0  # False Positive 偽陽性
        TN_h = 0  # True Negative 真陽性
        assert len(dataset_test.labels) == len(result_h)  # 元のラベルと分類結果の個数が不一致であれば停止
        for i in range(len(result_h)):
            if dataset_test.labels[i] == 1 and result_h[i] == 1:
                TP_h += 1
            elif dataset_test.labels[i] == 1 and result_h[i] == 0:
                FN_h += 1
            elif dataset_test.labels[i] == 0 and result_h[i] == 1:
                FP_h += 1
            else:
                TN_h += 1

        print("\n")
        print("*******************************************************")
        print(f"TP: {TP_h}")
        print(f"FN: {FN_h}")
        print(f"FP: {FP_h}")
        print(f"TN: {TN_h}")
        print("\n")
        print(f"label 0: {TN_h + FP_h}")  # ラベルの個数の確認
        print(f"label 1: {TP_h + FN_h}")
        print("*******************************************************")

        # Attention Scoreが低い部分のみを使用したテストも行う
        print("testing on data of low attention score ******************")
        result_l, _ = model.predict(dataset_test.data_remask(~mark), verbose=1)
        np.savetxt(os.path.join(SAVE_PATH, "result_l.csv"), result_l)
        result_l = [1 if result_l[i] >= 0.5 else 0 for i in range(len(result_l))]
        np.savetxt(os.path.join(SAVE_PATH, "result_l_binary.csv"), result_l, fmt='%d')

        TP_l = 0  # True Positive 真陽性
        FN_l = 0  # False Negative 偽陰性
        FP_l = 0  # False Positive 偽陽性
        TN_l = 0  # True Negative 真陽性
        assert len(dataset_test.labels) == len(result_l)  # 元のラベルと分類結果の個数が不一致であれば停止
        for i in range(len(result_l)):
            if dataset_test.labels[i] == 1 and result_l[i] == 1:
                TP_l += 1
            elif dataset_test.labels[i] == 1 and result_l[i] == 0:
                FN_l += 1
            elif dataset_test.labels[i] == 0 and result_l[i] == 1:
                FP_l += 1
            else:
                TN_l += 1

        print("\n")
        print("*******************************************************")
        print(f"TP: {TP_l}")
        print(f"FN: {FN_l}")
        print(f"FP: {FP_l}")
        print(f"TN: {TN_l}")
        print("\n")
        print(f"label 0: {TN_l + FP_l}")  # ラベルの個数の確認
        print(f"label 1: {TP_l + FN_l}")
        print("*******************************************************")

    #ログ書き出し
    with open(os.path.join(SAVE_PATH, "result_log.txt"), "w") as f:
        print("model summary *******************************************", file=f)
        model.summary(line_length=180, print_fn=lambda x: f.write(x + "\r\n"))
        print("\n", file=f)
        print("result shape ********************************************", file=f)
        print(f"  shape : {np.shape(result)}", file=f)
        print(f" length : {len(result)}", file=f)
        print("\n", file=f)
        print("evaluate ************************************************", file=f)
        print(f" result length : {len(result)}", file=f)
        print(f"  label length : {len(dataset_test.labels)}", file=f)
        print("\n", file=f)
        print("confusion matrix (binary label) *************************", file=f)
        print(f" TP : {TP}", file=f)
        print(f" FN : {FN}", file=f)
        print(f" FP : {FP}", file=f)
        print(f" TN : {TN}", file=f)
        print("\n", file=f)
        print(f" label 0 : {TN + FP}", file=f)
        print(f" label 1 : {TP + FN}", file=f)
        print("\n", file=f)
        print("time *****************************************************", file=f)
        print(f" fitting time : {fittime}", file=f)
        print(f" testing time : {testtime}", file=f)
        print("\n", file=f)
        if WITH_ATTENTION == True:
            print("confusion matrix (binary label, with high attention score) *", file=f)
            print(f" TP : {TP_h}", file=f)
            print(f" FN : {FN_h}", file=f)
            print(f" FP : {FP_h}", file=f)
            print(f" TN : {TN_h}", file=f)
            print("\n", file=f)
            print(f" label 0 : {TN_h + FP_h}", file=f)
            print(f" label 1 : {TP_h + FN_h}", file=f)
            print("\n", file=f)
            print("confusion matrix (binary label, with low attention score) *", file=f)
            print(f" TP : {TP_l}", file=f)
            print(f" FN : {FN_l}", file=f)
            print(f" FP : {FP_l}", file=f)
            print(f" TN : {TN_l}", file=f)
            print("\n", file=f)
            print(f" label 0 : {TN_l + FP_l}", file=f)
            print(f" label 1 : {TP_l + FN_l}", file=f)
            print("\n", file=f)
               
    confusion_matrix(os.path.join(SAVE_PATH, "confmat.txt"), result, dataset_test)
    if WITH_ATTENTION == True:
        confusion_matrix(os.path.join(SAVE_PATH, "confmat_h.txt"), result_h, dataset_test)
        confusion_matrix(os.path.join(SAVE_PATH, "confmat_l.txt"), result_l, dataset_test)


if __name__ == "__main__":
    _lstm()