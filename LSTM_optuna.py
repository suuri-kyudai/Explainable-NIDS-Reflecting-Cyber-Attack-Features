"""
v4における変更点
- LSTM_generator_model_v4.pyに対応
    - Attention, マスクへの対応など
"""

import os
import random
import tensorflow as tf
import json
import joblib
import csv
import gensim
import numpy as np
import tensorflow.keras as keras
import optuna
import datetime

from tensorflow.python.keras import backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Embedding, Flatten, Multiply, Attention, Flatten
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.text import Tokenizer
from optuna.trial import TrialState
from sklearn.metrics import accuracy_score, f1_score
from plot_keras_history import show_history, plot_history
from matplotlib import pyplot as plt
from copy import deepcopy
from tqdm import tqdm


# seed固定
seed = 1
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
tf.random.set_seed(seed)

"""
optunaで検証データのf1-scoreを最大にするハイパーパラメータを探索するプログラム
"""

WITH_ATTENTION = True # Attention Scoreを出力する場合はTrueにする
HPLEN = 700 # 入力の長さ

# GPU指定
gpu_id = 6  # GPUのポート番号
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


class MyModel(tf.keras.Model):
    def train_step(self, data):
        """Attention Scoreを出力に含むモデルを，その出力を無視して学習できるようにするためのカスタムトレーニングループ
        """
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred, _ = self(x, training=True) # 出力が2つあるが，1つは無視
            loss = self.compiled_loss(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


def create_model(input_dim=None, output_units=None, activation=None, n_layer=None, lstm_units=None,
                 dropout_rate=None) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(None, input_dim)))
    for i in range(n_layer):
        model.add(LSTM(lstm_units, return_sequences=True, time_major=False))
        model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, time_major=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_units, activation=activation))

    return model


def create_simple_model(input_dim=None, output_units=None, activation=None, lstm_units=None) -> Sequential:
    """
    単層LSTMの場合のoptuna探索用学習モデル
    Embedding Layer がないので使用しない

    input_dim: 出力ベクトルの次元
    output_units: 出力の数
    activation: 活性化関数
    lstm_units: LSTMのユニット数
    """

    model = Sequential()
    model.add(Input(shape=(None, input_dim)))
    model.add(LSTM(lstm_units, time_major=False))
    model.add(Dense(output_units, activation=activation))

    return model


def create_three_layer_model(input_dim=None, output_units=None, activation=None, lstm_units1=None, lstm_units2=None,
                             lstm_units3=None, dropout_rate1=None, dropout_rate2=None,
                             dropout_rate3=None) -> Sequential:
    """
    3層LSTMの場合のoptuna探索用学習モデル
    Embedding Layer がないので使用しない

    input_dim: 出力ベクトルの次元
    output_units: 出力の数
    activation: 活性化関数
    lstm_units1: LSTMのユニット数(1層目)
    lstm_units2: LSTMのユニット数(2層目)
    lstm_units3: LSTMのユニット数(3層目)
    dropout_rate1: LSTMのドロップアウト率(1層目)
    dropout_rate2: LSTMのドロップアウト率(2層目)
    dropout_rate3: LSTMのドロップアウト率(3層目)
    """
    model = Sequential()
    model.add(Input(shape=(None, input_dim)))
    model.add(LSTM(lstm_units1, return_sequences=True, time_major=False))
    model.add(Dropout(dropout_rate1))
    model.add(LSTM(lstm_units2, return_sequences=True, time_major=False))
    model.add(Dropout(dropout_rate2))
    model.add(LSTM(lstm_units3, time_major=False))
    model.add(Dropout(dropout_rate3))
    model.add(Dense(output_units, activation=activation))

    return model


def create_Hwang_three_layer_model_v2(output_units=None, activation=None, dropout_rate1=None, dropout_rate2=None,
                                      dropout_rate3=None) -> Sequential:
    """
    Hwang et al. のモデル
    Embedding Layer あり
    入力 ->　Embedding　-> 3層LSTM -> 出力レイヤー
    Dropoutを含めた3段階のLSTM(128, 64, 32) -> 分類
    """
    model = Sequential([
        # Input(shape=(input_dim)),
        Embedding(input_dim=65536, output_dim=64, input_length=29), # input_dim:単語の種類数，output_dim:出力ベクトルの次元，input_length:packetの単語数 29or54
        LSTM(units=128, return_sequences=True, time_major=False),  # LSTMを連結する場合はreturn_sequencesが必須
        Dropout(dropout_rate1),
        LSTM(units=64, return_sequences=True, time_major=False),
        Dropout(dropout_rate2),
        LSTM(units=32, return_sequences=False, time_major=False),
        Dropout(dropout_rate3),
        Dense(output_units, activation=activation)
    ])

    return model


def create_Hwang_three_layer_model_v3(output_units=None, activation=None, dropout_rate1=None, dropout_rate2=None,
                                      dropout_rate3=None) -> Sequential:
    """
    Hwang et al. のモデルの Embedding を任意の埋め込み行列(Word2Vec)で置き換え
    Dropoutを含めた3段階のLSTM(128, 64, 32) -> 分類
    """
    embeddings_model = gensim.models.KeyedVectors.load_word2vec_format(
        "/mnt/fuji/kashiwabara/CSE-CIC-IDS-2018/model/model_header_field_54_64_1")

    tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(tokenized_text_list)

    word_index = tokenizer.word_index
    num_words = len(word_index)

    embedding_matrix = np.zeros((num_words + 65536, 64))  # 埋め込みサイズのzero行列生成
    for word, i in word_index.items():  # embedding_matrix に任意の行列(Word2Vecで学習した行列)を入力
        if word in embeddings_model.index2word:
            embedding_matrix[i] = embeddings_model[word]

    num_words, w2v_size = embedding_matrix.shape

    model = Sequential([
        # Input(shape=(input_dim)),
        Embedding(num_words,  # 単語の種類数
                  w2v_size,  # 埋め込み行列のサイズ
                  weights=[embedding_matrix],  # 埋め込み行列の入力
                  input_length=29,  # 一つのpacket中の最大単語数，29 or 54
                  trainable=True),  # 固定or学習の指定，固定するとLSTMで学習はできなかった
        LSTM(units=128, return_sequences=True, time_major=False),  # LSTMを連結する場合はreturn_sequencesが必須
        Dropout(dropout_rate1),
        LSTM(units=64, return_sequences=True, time_major=False),
        Dropout(dropout_rate2),
        LSTM(units=32, return_sequences=False, time_major=False),
        Dropout(dropout_rate3),
        Dense(output_units, activation=activation)
    ])

    return model


def create_Hwang_three_layer_model_v4(output_units=None, activation=None, dropout_rate1=None, dropout_rate2=None,
                                      dropout_rate3=None) -> Sequential:
    """
    Hwang et al. のモデルにmask_zeroの要素を追加
    Embedding の mask_zero=True で，特徴ベクトルの0の系列を学習しないようにするためのコード
    embedding_imitializer で埋め込み行列の初期化
    Dropout を含めた3段階の LSTM(128, 64, 32) -> 分類
    """
    #max_seq_len = 54  # 一つのpacket中の最大単語数 29or54
    max_seq_len = 200
    feature = 64  # 出力ベクトルの特徴次元
    initializer = keras.initializers.Zeros()  # 埋め込み行列の初期値

    model = Sequential([
        # Input(shape=(input_dim)),
        Embedding(input_dim=256, output_dim=feature, input_length=max_seq_len, mask_zero=False,
                  embeddings_initializer=initializer),  # embedding_initializerで重みの初期化
        LSTM(units=128, return_sequences=True, time_major=False),  # LSTMを連結する場合はreturn_sequencesが必須
        Dropout(dropout_rate1),
        LSTM(units=64, return_sequences=True, time_major=False),
        Dropout(dropout_rate2),
        LSTM(units=32, return_sequences=False, time_major=False),
        Dropout(dropout_rate3),
        Dense(output_units, activation=activation)
    ])

    return model

def create_Hwang_three_layer_model_v4a(output_units=None, activation=None, dropout_rate1=None, dropout_rate2=None,
                                      dropout_rate3=None) -> Sequential:
    """
    create_Hwang_three_layer_model_v4の0x100 padding対応版
    """
    #max_seq_len = 54  # 一つのpacket中の最大単語数 29or54
    max_seq_len = 200
    feature = 64  # 出力ベクトルの特徴次元
    initializer = keras.initializers.Zeros()  # 埋め込み行列の初期値

    model = Sequential([
        # Input(shape=(input_dim)),
        Embedding(input_dim=257, output_dim=feature, input_length=max_seq_len, mask_zero=False,
                  embeddings_initializer=initializer),  # embedding_initializerで重みの初期化
        LSTM(units=128, return_sequences=True, time_major=False),  # LSTMを連結する場合はreturn_sequencesが必須
        Dropout(dropout_rate1),
        LSTM(units=64, return_sequences=True, time_major=False),
        Dropout(dropout_rate2),
        LSTM(units=32, return_sequences=False, time_major=False),
        Dropout(dropout_rate3),
        Dense(output_units, activation=activation)
    ])

    return model

def create_Hwang_three_layer_model_v4f(max_seq_len=200, output_units=None, activation=None, dropout_rate1=None, dropout_rate2=None,
                                      dropout_rate3=None) -> Sequential:
    """
    create_Hwang_three_layer_model_v4aをフィールド単位分割に変更
    """
    feature = 64  # 出力ベクトルの特徴次元
    initializer = keras.initializers.Zeros()  # 埋め込み行列の初期値

    model = Sequential([
        # Input(shape=(input_dim)),
        Embedding(input_dim=65537, output_dim=feature, input_length=max_seq_len, mask_zero=False,
                  embeddings_initializer=initializer),  # embedding_initializerで重みの初期化
        LSTM(units=128, return_sequences=True, time_major=False),  # LSTMを連結する場合はreturn_sequencesが必須
        Dropout(dropout_rate1),
        LSTM(units=64, return_sequences=True, time_major=False),
        Dropout(dropout_rate2),
        LSTM(units=32, return_sequences=False, time_major=False),
        Dropout(dropout_rate3),
        Dense(output_units, activation=activation)
    ])

    return model

def generate_Hwang_LSTM_model_v4f_FAPI(max_seq_len=200, output_units=None, activation=None, dropout_rate1=None, dropout_rate2=None, dropout_rate3=None):
    """
    Fanctional APIに変更
    """
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    inputs = Input(shape=(max_seq_len))
    x = Embedding(input_dim=65537, output_dim=feature, input_length=max_seq_len, mask_zero=False,
                  embeddings_initializer=initializer)(inputs)
    x = LSTM(units=128, return_sequences=True, time_major=False)(x)
    x = Dropout(dropout_rate1)(x)
    x = LSTM(units=64, return_sequences=True, time_major=False)(x)
    x = Dropout(dropout_rate2)(x)
    x = LSTM(units=32, return_sequences=False, time_major=False)(x)
    x = Dropout(dropout_rate3)(x)
    predictions = Dense(output_units, activation=activation)(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model

def generate_Hwang_LSTM_model_v4f_FAPI_multi_packet(max_seq_len=200, n_packets=None, output_units=None, activation=None, dropout_rate1=None, dropout_rate2=None, dropout_rate3=None):
    """
    Fanctional APIに変更
    複数パケットの入力に対応（shapeはパケット数*特徴量数のベクトル）
    """
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    inputs = Input(shape=(n_packets * max_seq_len, )) # [n_packets * max_seq_len]
    x = Embedding(input_dim=65537, output_dim=feature, input_length=max_seq_len, mask_zero=False,
                  embeddings_initializer=initializer)(inputs)
    x = LSTM(units=128, return_sequences=True, time_major=False)(x)
    x = Dropout(dropout_rate1)(x)
    x = LSTM(units=64, return_sequences=True, time_major=False)(x)
    x = Dropout(dropout_rate2)(x)
    x = LSTM(units=32, return_sequences=False, time_major=False)(x)
    x = Dropout(dropout_rate3)(x)
    predictions = Dense(output_units, activation=activation)(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model

def generate_Hwang_LSTM_model_v4f_Attention(max_seq_len=200, output_units=None, dropout_rate1=None, dropout_rate2=None, dropout_rate3=None):
    """
    create_Hwang_three_layer_model_v4f_FAPIにAttentionを追加
    """
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    inputs = Input(shape=(max_seq_len))
    x = Embedding(input_dim=65537, output_dim=feature, input_length=max_seq_len, mask_zero=False, embeddings_initializer=initializer)(inputs)
    x = LSTM(units=128, return_sequences=True, time_major=False)(x)
    x = Dropout(dropout_rate1)(x)
    x = LSTM(units=64, return_sequences=True, time_major=False)(x)
    x = Dropout(dropout_rate2)(x)
    x = LSTM(units=32, return_sequences=True, time_major=False)(x)
    x = Dropout(dropout_rate3)(x)
    x = Dense(output_units, activation=None)(x) # [batch, 200, 32] -> [batch, 200, 1]
    x = Flatten()(x) # [batch, 200, 1] -> [batch, 200]
    attention_score = Dense(max_seq_len, activation="relu")(x) # [batch, 200] -> [batch, 200]
    attention_score = Dense(max_seq_len, activation="sigmoid")(attention_score) # [batch, 200] -> [batch, 200]
    x = Multiply()([x, attention_score]) # [[batch, 200], [batch, 200]] -> [batch, 200]
    prediction = Dense(output_units, activation="sigmoid", name="pred")(x) # [batch, 200] -> [batch, 1]

    # model = MyModel(inputs=inputs, outputs=(prediction, attention_score))
    model = Model(inputs=inputs, outputs=prediction)
    return model

def generate_Hwang_LSTM_model_v5_FAPI_multi_packet(max_seq_len=None, n_packets=None, output_units=None, activation=None, dropout_rate1=None, dropout_rate2=None, dropout_rate3=None):
    """
    マスクに対応
    """
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    inputs = Input(shape=(n_packets * max_seq_len + n_packets, )) # [n_packets * max_seq_len + n_packets]
    x = Embedding(input_dim=65539, output_dim=feature, input_length=max_seq_len, mask_zero=True,
                  embeddings_initializer=initializer)(inputs)
    x = LSTM(units=128, return_sequences=True, time_major=False)(x)
    x = Dropout(dropout_rate1)(x)
    x = LSTM(units=64, return_sequences=True, time_major=False)(x)
    x = Dropout(dropout_rate2)(x)
    x = LSTM(units=32, return_sequences=False, time_major=False)(x)
    x = Dropout(dropout_rate3)(x)
    predictions = Dense(output_units, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model

def generate_Hwang_LSTM_model_v5_FAPI_multi_packet_attention(max_seq_len=None, n_packets=None, output_units=None, activation=None, dropout_rate1=None, dropout_rate2=None, dropout_rate3=None):
    """
    Attentionを改良
    マスクに対応
    """
    max_seq_len = HPLEN # 一つのpacket中の最大単語数
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    inputs = Input(shape=(n_packets * max_seq_len + n_packets, )) # [n_packets * max_seq_len + n_packets]
    x = Embedding(input_dim=65539, output_dim=feature, input_length=max_seq_len, mask_zero=True,
                  embeddings_initializer=initializer)(inputs) # [n_packets * max_seq_len + n_packets, 64]
    x = LSTM(units=128, return_sequences=True, time_major=False)(x) # [n_packets * max_seq_len + n_packets, 128]
    x = Dropout(dropout_rate1)(x) # [n_packets * max_seq_len + n_packets, 128]
    x = LSTM(units=64, return_sequences=True, time_major=False)(x) # [n_packets * max_seq_len + n_packets, 64]
    x = Dropout(dropout_rate2)(x) # [n_packets * max_seq_len + n_packets, 64]
    x, h, c = LSTM(units=32, return_sequences=True, return_state=True, time_major=False)(x) # x[n_packets * max_seq_len + n_packets, 32], h[32]
    x = Dropout(dropout_rate3)(x) # x[n_packets * max_seq_len + n_packets, 32]
    k = Dense(32, activation=None, name="Dense_key")(x) # k[n_packets * max_seq_len + n_packets, 32]
    v = Dense(32, activation=None, name="Dense_value")(x) # v[n_packets * max_seq_len + n_packets, 32]
    q = Dense(32, activation=None, name="Dense_query")(h) # q[32]  
    q = tf.expand_dims(q, axis=-2) # q[1, 32]  
    r, attention_score = Attention(use_scale=True)(inputs=[q, v, k], return_attention_scores=True) # r[1, 32], attention_score[1, n_packets * max_seq_len + n_packets]
    r = Flatten()(r) # r[32]
    attention_score = Flatten()(attention_score) # attention_score[n_packets * max_seq_len + n_packets]
    predictions = Dense(output_units, activation="sigmoid", name="output")(r) # predictions(スカラー値)

    # model = MyModel(inputs=inputs, outputs=(predictions, attention_score))
    model = Model(inputs=inputs, outputs=predictions)
    return model


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

def objective_wrapper(train_dataset, test_dataset, save_path, max_seq_len=HPLEN, loss='binary_crossentropy', fixed_epochs=None, n_packets=1):
    """最適化する関数

    Args:
        train_dataset (masked_dataset): 訓練データセット
        test_dataset (masked_dataset): テストデータセット
        save_path (str): 結果出力先のパス
        max_seq_len (int, optional): データの長さ（デフォルトはHPLEN）
        loss (str, optional): 損失関数（デフォルトは'binary_crossentropy'）
    """

    def objective(trial):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        """
        optunaのプログラム
        """

        # ハイパーパラメータの範囲を設定
        if fixed_epochs == None:
            epochs = trial.suggest_int('epochs', 1, 10, step=1, log=True)
        else:
            epochs = fixed_epochs # エポック数を固定する場合
        dropout_rate1 = trial.suggest_float('dropout_rate1', 0, 0.5)  # ドロップアウト
        dropout_rate2 = trial.suggest_float('dropout_rate2', 0, 0.5)  # ドロップアウト
        dropout_rate3 = trial.suggest_float('dropout_rate3', 0, 0.5)  # ドロップアウト
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)  # 最適化アルゴリズム
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])  # 訓練データの入力batch_size

        # その他パラメータの設定
        output_units = 1  # 出力数指定
        activation = "sigmoid"  # 活性化関数

        # モデル決定
        if WITH_ATTENTION == True:
            model = generate_Hwang_LSTM_model_v5_FAPI_multi_packet_attention(max_seq_len, n_packets, output_units, activation, dropout_rate1, dropout_rate2, dropout_rate3)
        else:
            model = generate_Hwang_LSTM_model_v5_FAPI_multi_packet(max_seq_len, n_packets, output_units, activation, dropout_rate1, dropout_rate2, dropout_rate3)

        # モデルの学習
        model.compile(optimizer=optimizer, loss=loss,
                      metrics=['accuracy', true_positives, possible_positives, predicted_positives])  #モデルコンパイル
        f1cb = F1Callback()
        history = model.fit(train_dataset.packets, train_dataset.labels, epochs=epochs,
                            batch_size=batch_size, verbose=1,
                            validation_data=test_dataset.data_tuple(), callbacks=[f1cb])

        # trialごとにplotを保存
        xlab = np.arange(epochs) + 1
        plt.figure()
        plt.plot(xlab, np.array(f1cb.f1s), label="val_f1")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("f1 score")
        plt.savefig(os.path.join(save_path, "tuning_f1_trial", "f1_trial_" + str(trial.number)))  # 学習曲線の保存先指定（暫定）
        plt.close()

        plot_history(history, path=os.path.join(
            save_path, "tuning_accloss_trial", "accloss_trial_" + str(trial.number)))  #学習曲線(f1-score)の指定

        # 最小値探索なので
        maxf1 = -np.amax(f1cb.f1s)
        maxf1_epoch = np.argmax(f1cb.f1s)
        trial.set_user_attr("maxf1_epoch", str(maxf1_epoch))
        return maxf1  # f1-scoreを最大化
    
    return objective

def main():
    # ここを必要に応じて変更（上にHPLENとWITH_ATTENTIONがあります）
    DEBUG = False # 動作確認時Trueにする
    N_PACKETS = 2

    if WITH_ATTENTION == True:
        STUDY_NAME_PREFIX = f"LSTM_multipackets_attention_{HPLEN}_np{N_PACKETS}_v3_tuned"
    else:
        STUDY_NAME_PREFIX = f"LSTM_multipackets_{HPLEN}_np{N_PACKETS}_v3_tuned"

    if DEBUG == True:
        SAVE_PATH = f"/mnt/fuji/kawanaka/results/test"
        STUDY_NAME = f"{STUDY_NAME_PREFIX}_test"
    else:
        SAVE_PATH = f"/mnt/fuji/kawanaka/results/{STUDY_NAME_PREFIX}"
        STUDY_NAME = f"{STUDY_NAME_PREFIX}"
    
    TRAIN_PATH = f"/mnt/fuji/kawanaka/data/CSE-CIC-IDS2018_0222_field_{HPLEN}f_mask.csv"
    TEST_PATH = f"/mnt/fuji/kawanaka/data/CSE-CIC-IDS2018_0223_field_{HPLEN}f_mask.csv"
    train_data_size = 84319  # 訓練データの全packet数：84319
    test_data_size = 94201 # 検証データの全packet数：94201

    # データセットの読み込み
    print("Loading training dataset...")
    dataset_train = masked_dataset(data_path=TRAIN_PATH, n_samples=train_data_size, n_packets=N_PACKETS)
    print("Loading testing dataset...")
    dataset_test = masked_dataset(data_path=TEST_PATH, n_samples=test_data_size, n_packets=N_PACKETS)


    # チューニング結果の出力先フォルダを作成
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, "tuning_f1_trial"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, "tuning_accloss_trial"), exist_ok=True)
    db_path = os.path.join(SAVE_PATH, f"{STUDY_NAME}.sqlite")

    study = optuna.create_study(study_name=STUDY_NAME, storage=f"sqlite:///{db_path}", load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=1))
    if DEBUG == True:
        study.optimize(objective_wrapper(dataset_train, dataset_test, SAVE_PATH,
                                         max_seq_len=HPLEN, loss='binary_crossentropy',
                                         n_packets=N_PACKETS, fixed_epochs=3), n_trials=3)  # trial：探索回数の指定（動作確認用）
    else:
        study.optimize(objective_wrapper(dataset_train, dataset_test, SAVE_PATH,
                                         max_seq_len=HPLEN, loss='binary_crossentropy',
                                         n_packets=N_PACKETS, fixed_epochs=10), n_trials=30)  # trial：探索回数の指定
    
    print('best_params')
    print(study.best_params)
    print('best_value : ' + str(abs(study.best_value)))
    print('best trial : ' + str(study.best_trial.number))
    print(' epochs (best trial) : ' + str(study.best_trial.user_attrs))

    print('\n --- sorted --- \n')
    sorted_best_params = sorted(study.best_params.items(), key=lambda x: x[0])
    for i, k in sorted_best_params:
        print(i + ' : ' + str(k))

    #ログ書き出し
    with open(os.path.join(SAVE_PATH, "tuning_result.txt"), "w") as f:
        print("study time **********************************************", file=f)
        all_trials = study.trials
        for study_data in all_trials:
            stime = study_data.datetime_complete - study_data.datetime_start
            print(" study {0:02} : {1}".format(study_data.number, stime.total_seconds()), file=f)
        print("\n", file=f)
        print("f1 score **********************************************", file=f)
        all_trials = study.trials
        for study_data in all_trials:
            s_f1 = study_data.value
            print(" study {0:02} : {1}".format(study_data.number, s_f1), file=f)
        print("\n", file=f)
        print("param ***************************************************", file=f)
        print(' best_params', file=f)
        print(study.best_params, file=f)
        print(' best_value : ' + str(abs(study.best_value)), file=f)
        print(' best trial : ' + str(study.best_trial.number), file=f)
        print(' epochs (best trial) : ' + str(study.best_trial.user_attrs), file=f)

        print('\n --- sorted --- \n', file=f)
        for i, k in sorted_best_params:
            print(i + ' : ' + str(k), file=f)

if __name__ == '__main__':
    main()
