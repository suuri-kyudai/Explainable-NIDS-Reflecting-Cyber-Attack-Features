import json
import os
import pandas as pd
# import gensim
import joblib
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Embedding, Masking, Multiply, Reshape, Attention, Flatten

HPLEN = 200
# seed固定
seed = 1
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
tf.random.set_seed(seed)


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
        
class Attention_old(tf.keras.Model):
    def __init__(self, units, return_sequences=False):
        super(Attention_old, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.return_sequences = return_sequences

    def call(self, seq, hidden, mask=None): # seq[n_packets * max_seq_len, units], hidden[units]
        hidden_with_time_axis = tf.expand_dims(hidden, 1) # hidden_with_time_axis[1, units]
        if mask == None:
            raise Exception
            score = tf.nn.tanh(self.W1(seq) + self.W2(hidden_with_time_axis)) # score[n_packets * max_seq_len, units]
        else:
            broadcast_float_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
            score = tf.nn.tanh(self.W1(seq) + self.W2(hidden_with_time_axis)) * broadcast_float_mask # score[n_packets * max_seq_len, units] マスク適用
        attention_weights = tf.nn.softmax(self.V(score), axis=1) # attention_weights[n_packets * max_seq_len, 1]
        context_vector = attention_weights * seq # context_vector[n_packets * max_seq_len, units]
        if self.return_sequences == False:
            context_vector = tf.reduce_sum(context_vector, axis=1) # context_vector[units]

        return context_vector, attention_weights


"""
モデルの生成例
"""

def generate_simple_LSTM_model(lstm_units=None, output_units=1, output_activation=None, input_dim = None) -> Sequential:
    """
    単層LSTM
    入力 -> LSTM -> 出力レイヤー
    入力は(batch, sequence_length, data_dim)のshapeを持つ必要がある．

    lstm_units: LSTMの出力ユニット数
    output_units: 出力層のユニット数
    output_activation: 出力層の活性化関数
    input_dim: 入力次元
    """

    model = Sequential()
    model.add(Input(shape=(None, input_dim)))
    model.add(LSTM(lstm_units, time_major = False))  # time_major: 入力のshape指定
    model.add(Dense(output_units, activation=output_activation))

    return model


def generate_Hwang_LSTM_model(input_dim=None, output_units=1) -> Sequential:
    """
    Hwang et al. のモデルの誤り例（Embedding層がない）
    入力 -> 3層LSTM -> 出力レイヤー
    Dropoutを含めた3段階のLSTM(128, 64, 32) -> 分類
    """
    model = Sequential([
        Input(shape=(None, input_dim)),
        LSTM(units=128, return_sequences=True, time_major=False),  # LSTMを連結する場合はreturn_sequencesが必須
        Dropout(0.2),
        LSTM(units=64, return_sequences=True, time_major=False),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False, time_major=False),
        Dropout(0.2),
        Dense(output_units, activation="sigmoid")
    ])

    return model

def generate_Hwang_LSTM_model_v2(input_dim=None, output_units=1) -> Sequential:
    """
    Hwang et al. のモデル
    Embedding Layer あり
    入力 ->　Embedding　-> 3層LSTM -> 出力レイヤー
    Dropoutを含めた3段階のLSTM(128, 64, 32) -> 分類
    """
    model = Sequential([
        #Input(shape=(input_dim)),
        Embedding(input_dim=256, output_dim=64, input_length=None),
        LSTM(units=128, return_sequences=True, time_major=False),  # LSTMを連結する場合はreturn_sequencesが必須
        Dropout(0.2742069589897837),
        LSTM(units=64, return_sequences=True, time_major=False),
        Dropout(0.10284708907646053),
        LSTM(units=32, return_sequences=False, time_major=False),
        Dropout(0.04893787990528288),
        Dense(output_units, activation="sigmoid")
    ])

    return model

# def generate_Hwang_LSTM_model_v3(input_dim=None, output_units=1) -> Sequential:
#     """
#     Hwang et al. のモデルの Embedding を任意の埋め込み行列(Word2Vec)で置き換え
#     """
#     embeddings_model = gensim.models.KeyedVectors.load_word2vec_format("/mnt/fuji/kashiwabara/CSE-CIC-IDS-2018/model/model_header_field_54_64_1")  #事前に学習したWord2Vecモデル

#     tokenizer = Tokenizer()
#     word_index = tokenizer.word_index
#     num_words = len(word_index)
#     embedding_matrix = np.zeros((num_words + 65536, 64))  # 埋め込みサイズのzero行列生成
#     for word, i in word_index.items():  # embedding_matrix に任意の行列(Word2Vecで学習した行列)を入力
#         if word in embeddings_model.index2word:
#             embedding_matrix[i] = embeddings_model[word]

#     num_words, w2v_size = embedding_matrix.shape
#     #print(num_words,w2v_size)

#     model = Sequential([
#         #Input(shape=(input_dim)),
#         Embedding(num_words,  # 単語の種類数
#                   w2v_size,  # 埋め込み行列のサイズ
#                   weights=[embedding_matrix],  # 埋め込み行列の入力
#                   input_length=29,  # 一つのpacket中の最大単語数，29or54
#                   trainable=True),  # 固定or学習の指定，固定するとLSTMで学習はできなかった
#         LSTM(units=128, return_sequences=True, time_major=False),  # LSTMを連結する場合はreturn_sequencesが必須
#         Dropout(0.49926861148316853),
#         LSTM(units=64, return_sequences=True, time_major=False),
#         Dropout(0.01648918343064293),
#         LSTM(units=32, return_sequences=False, time_major=False),
#         Dropout(0.2778711209272472),
#         Dense(output_units, activation="sigmoid")
#     ])

#     return model

def generate_Hwang_LSTM_model_v4(input_dim=None, output_units=1) -> Sequential:
    """
    Hwang et al. のモデルにmask_zeroの要素を追加
    Embedding の mask_zero=True で，特徴ベクトルの0の系列を学習しないようにするためのコード
    embedding_imitializer で埋め込み行列の初期化
    Dropout を含めた3段階の LSTM(128, 64, 32) -> 分類
    """
    #max_seq_len = 54  # 一つのpacket中の最大単語数 29or54
    max_seq_len = 200
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    model = Sequential([
        #Input(shape=(input_dim)),
        Embedding(input_dim=256, output_dim=feature, input_length=max_seq_len, mask_zero=False, embeddings_initializer=initializer), # embedding_initializerで重みの初期化
        LSTM(units=128, return_sequences=True, time_major=False),  # LSTMを連結する場合はreturn_sequencesが必須
        Dropout(0.4207849216080977),
        LSTM(units=64, return_sequences=True, time_major=False),
        Dropout(0.4169442091900613),
        LSTM(units=32, return_sequences=False, time_major=False),
        Dropout(0.16329489824872537),
        Dense(output_units, activation="sigmoid")
    ])

    return model

def generate_Hwang_LSTM_model_v4a(input_dim=None, output_units=1) -> Sequential:
    """
    create_Hwang_three_layer_model_v4の0x100 padding対応版
    """
    #max_seq_len = 54  # 一つのpacket中の最大単語数 29or54
    max_seq_len = HPLEN
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    model = Sequential([
        #Input(shape=(input_dim)),
        Embedding(input_dim=257, output_dim=feature, input_length=max_seq_len, mask_zero=False, embeddings_initializer=initializer), # embedding_initializerで重みの初期化
        LSTM(units=128, return_sequences=True, time_major=False),  # LSTMを連結する場合はreturn_sequencesが必須
        Dropout(0.44138043929182386),
        LSTM(units=64, return_sequences=True, time_major=False),
        Dropout(0.02197135755988886),
        LSTM(units=32, return_sequences=False, time_major=False),
        Dropout(0.49994567111065097),
        Dense(output_units, activation="sigmoid")
    ])

    return model

def generate_Hwang_LSTM_model_v4f(input_dim=None, output_units=1) -> Sequential:
    """
    create_Hwang_three_layer_model_v4aをフィールド単位分割に変更
    """
    #max_seq_len = 54  # 一つのpacket中の最大単語数 29or54
    max_seq_len = HPLEN
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    model = Sequential([
        #Input(shape=(input_dim)),
        Embedding(input_dim=65537, output_dim=feature, input_length=max_seq_len, mask_zero=False, embeddings_initializer=initializer), # embedding_initializerで重みの初期化
        LSTM(units=128, return_sequences=True, time_major=False),  # LSTMを連結する場合はreturn_sequencesが必須
        Dropout(0.436308087040522),
        LSTM(units=64, return_sequences=True, time_major=False),
        Dropout(0.3182714810133205),
        LSTM(units=32, return_sequences=False, time_major=False),
        Dropout(0.3624123417647748),
        Dense(output_units, activation="sigmoid")
    ])

    return model

def generate_Hwang_LSTM_model_v4f_FAPI(input_dim=65537, output_units=1):
    """
    Fanctional APIに変更
    """
    #max_seq_len = 54  # 一つのpacket中の最大単語数 29or54
    max_seq_len = HPLEN
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    inputs = Input(shape=(max_seq_len))
    x = Embedding(input_dim=65537, output_dim=feature, input_length=max_seq_len, mask_zero=False, embeddings_initializer=initializer)(inputs)
    x = LSTM(units=128, return_sequences=True, time_major=False)(x)
    x = Dropout(0.436308087040522)(x)
    x = LSTM(units=64, return_sequences=True, time_major=False)(x)
    x = Dropout(0.3182714810133205)(x)
    x = LSTM(units=32, return_sequences=False, time_major=False)(x)
    x = Dropout(0.3624123417647748)(x)
    predictions = Dense(output_units, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model

def generate_Hwang_LSTM_model_v4f_FAPI_multi_packet(input_dim=65537, n_packets=3, output_units=1):
    """
    Fanctional APIに変更
    複数パケットの入力に対応（shapeはパケット数*特徴量数のベクトル）
    """
    max_seq_len = HPLEN # 一つのpacket中の最大単語数
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    inputs = Input(shape=(n_packets * max_seq_len, )) # [n_packets * max_seq_len]
    x = Embedding(input_dim=65537, output_dim=feature, input_length=max_seq_len, mask_zero=True,
                  embeddings_initializer=initializer)(inputs)
    x = LSTM(units=128, return_sequences=True, time_major=False)(x)
    x = Dropout(0.35993359690312204)(x)
    x = LSTM(units=64, return_sequences=True, time_major=False)(x)
    x = Dropout(0.13027954206505651)(x)
    x = LSTM(units=32, return_sequences=False, time_major=False)(x)
    x = Dropout(0.17406097386537459)(x)
    predictions = Dense(output_units, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model

def generate_Hwang_LSTM_model_v4f_FAPI_multi_packet_attention(input_dim=65537, n_packets=3, output_units=1):
    """
    Fanctional APIに変更
    複数パケットの入力に対応（shapeはパケット数*特徴量数のベクトル）
    Attentionを追加
    """
    max_seq_len = HPLEN # 一つのpacket中の最大単語数
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    inputs = Input(shape=(n_packets * max_seq_len, )) # [n_packets * max_seq_len]
    x = Embedding(input_dim=65537, output_dim=feature, input_length=max_seq_len, mask_zero=True,
                  embeddings_initializer=initializer)(inputs) # [n_packets * max_seq_len, 64]
    x = LSTM(units=128, return_sequences=True, time_major=False)(x) # [n_packets * max_seq_len, 128]
    x = Dropout(0.35993359690312204)(x) # [n_packets * max_seq_len, 128]
    x = LSTM(units=64, return_sequences=True, time_major=False)(x) # [n_packets * max_seq_len, 64]
    x = Dropout(0.13027954206505651)(x) # [n_packets * max_seq_len, 64]
    x, h, c = LSTM(units=32, return_sequences=True, return_state=True, time_major=False)(x) # x[n_packets * max_seq_len, 32]
    x = Dropout(0.17406097386537459)(x) # [n_packets * max_seq_len, 32]
    x, attention_score = Attention_old(32)(x, h) # x[32]
    predictions = Dense(output_units, activation="sigmoid", name="output")(x) # スカラー値

    model = MyModel(inputs=inputs, outputs=(predictions, attention_score))
    return model


def generate_Hwang_LSTM_model_v4f_FAPI_multi_packet_multi_attention(input_dim=65537, n_packets=3, output_units=1):
    """
    Fanctional APIに変更
    複数パケットの入力に対応（shapeはパケット数*特徴量数のベクトル）
    Attentionを追加（複数回適用）
    """
    max_seq_len = HPLEN # 一つのpacket中の最大単語数
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    inputs = Input(shape=(n_packets * max_seq_len, )) # [n_packets * max_seq_len]
    x = Embedding(input_dim=65537, output_dim=feature, input_length=max_seq_len, mask_zero=False,
                  embeddings_initializer=initializer)(x) # [n_packets * max_seq_len, 64]
    x, h1, c1 = LSTM(units=128, return_sequences=True, return_state=True, time_major=False)(x) # [n_packets * max_seq_len, 128]
    # x = Dropout(0.35993359690312204)(x) # [n_packets * max_seq_len, 128]
    x, attention_score_1 = Attention(128, return_sequences=True)(x, h1) # x[128]
    x, h2, c2 = LSTM(units=64, return_sequences=True, return_state=True, time_major=False)(x) # [n_packets * max_seq_len, 64]
    # x = Dropout(0.13027954206505651)(x) # [n_packets * max_seq_len, 64]
    x, attention_score_2 = Attention(64, return_sequences=True)(x, h2) # x[64]
    x, h3, c3 = LSTM(units=32, return_sequences=True, return_state=True, time_major=False)(x) # x[n_packets * max_seq_len, 32]
    # x = Dropout(0.17406097386537459)(x) # [n_packets * max_seq_len, 32]
    x, attention_score_3 = Attention(32, return_sequences=False)(x, h3) # x[32]
    predictions = Dense(output_units, activation="sigmoid", name="output")(x) # スカラー値
    attention_score = tf.keras.layers.Average()([attention_score_1, attention_score_2, attention_score_3])

    model = MyModel(inputs=inputs, outputs=(predictions, attention_score))
    return model

def generate_Hwang_LSTM_model_v5_FAPI_multi_packet(input_dim=65539, n_packets=3, output_units=1):
    """
    マスクに対応
    """
    max_seq_len = HPLEN # 一つのpacket中の最大単語数
    feature = 64  # 出力ベクトルの特徴次元
    initializer = tf.keras.initializers.Zeros()  # 埋め込み行列の初期値

    inputs = Input(shape=(n_packets * max_seq_len + n_packets, )) # [n_packets * max_seq_len + n_packets]
    x = Embedding(input_dim=65539, output_dim=feature, input_length=max_seq_len, mask_zero=True,
                  embeddings_initializer=initializer)(inputs)
    x = LSTM(units=128, return_sequences=True, time_major=False)(x)
    x = Dropout(0.15974237059873464)(x)
    x = LSTM(units=64, return_sequences=True, time_major=False)(x)
    x = Dropout(0.2982036895055062)(x)
    x = LSTM(units=32, return_sequences=False, time_major=False)(x)
    x = Dropout(0.04505619185169181)(x)
    predictions = Dense(output_units, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def generate_Hwang_LSTM_model_v5_FAPI_multi_packet_attention(input_dim=65539, n_packets=3, output_units=1):
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
    x = Dropout(0.46534934662915567)(x) # [n_packets * max_seq_len + n_packets, 128]
    x = LSTM(units=64, return_sequences=True, time_major=False)(x) # [n_packets * max_seq_len + n_packets, 64]
    x = Dropout(0.2564957097855458)(x) # [n_packets * max_seq_len + n_packets, 64]
    x, h, c = LSTM(units=32, return_sequences=True, return_state=True, time_major=False)(x) # x[n_packets * max_seq_len + n_packets, 32], h[32]
    x = Dropout(0.2391251496745483)(x) # x[n_packets * max_seq_len + n_packets, 32]
    k = Dense(32, activation=None, name="Dense_key")(x) # k[n_packets * max_seq_len + n_packets, 32]
    v = Dense(32, activation=None, name="Dense_value")(x) # v[n_packets * max_seq_len + n_packets, 32]
    q = Dense(32, activation=None, name="Dense_query")(h) # q[32]  
    q = tf.expand_dims(q, axis=-2) # q[1, 32]  
    r, attention_score = Attention(use_scale=True)(inputs=[q, v, k], return_attention_scores=True) # r[1, 32], attention_score[1, n_packets * max_seq_len + n_packets]
    r = Flatten()(r) # r[32]
    attention_score = Flatten()(attention_score) # attention_score[n_packets * max_seq_len + n_packets]
    predictions = Dense(output_units, activation="sigmoid", name="output")(r) # predictions(スカラー値)

    model = MyModel(inputs=inputs, outputs=(predictions, attention_score))
    return model
