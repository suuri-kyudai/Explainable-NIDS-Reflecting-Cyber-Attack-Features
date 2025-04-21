"""Attention Scoreを可視化するためのプログラム

Note:
    htmlファイルを出力するため, 確認にはブラウザ等のhtmlファイルを閲覧できるソフトウェアが必要です.

v2における変更点
    - マスクの変更に伴うAttention Scoreの変更に対応
    - 閾値判断機能を追加

v3における変更点
    - 定量的評価に対応
    - 攻撃の特性と関連する文字列のマーク色を緑色に変更
"""

import csv
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from pprint import pprint


HPLEN = 700 # ヘッダとペイロードを合わせた長さ
N_PACKETS = 1
N_SAMPLES = 94201 # テスト用データセット内のパケット数
TEST_PATH = f"/mnt/fuji/kawanaka/data/CSE-CIC-IDS2018_0223_field_{HPLEN}f_mask.csv" # テストデータのcsvファイルのパス

DEBUG_DATA = False
if DEBUG_DATA:
    RESULT_DIR = f"/mnt/fuji/kawanaka/results/test"
else:
    RESULT_DIR = f"/mnt/fuji/kawanaka/results/LSTM_multipackets_attention_{HPLEN}_np{N_PACKETS}_v3_tuned" # 実験結果が保存されているディレクトリ

SCORE_FILE = "attention_score.zip" # Attention Scoreのzipファイル名
OUT_FEATURE_SCORE_FILE = "attention_score_feature.csv"
OUT_FILE_PREFIX = "attention" # 可視化結果の出力htmlファイル名のプレフィックス

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
        flow_packet_ends (np.ndarray[np.ndarray[int]]): パケットの終端位置
    """

    def __init__(self, data_path, n_samples, n_packets=3, binary=True):
        """データセットを読み込む

        Args:
            data_path (str): データセット（csvファイル）のパス
            n_samples (int): 進捗表示用サンプル数
            n_packets (int, optional): 1サンプルに対するパケット数（デフォルトは3）
            binary (bool, optional): 二値分類フラグ（デフォルトはTrue）
        """        
        
        list_packets    = []
        list_labels     = []
        list_labels_raw = []
        list_flow_packet_ends  = []

        with open(data_path, 'r') as f:
            next(f)  # 1行目（csvヘッダ）はとばす
            reader = csv.reader(f)  # float型で読み込み
            packets_of_flows = {} # フローごとに過去のパケットを保持
            
            input_len = HPLEN * n_packets + n_packets # 区切り・終端を含む

            for row in tqdm(reader, total=n_samples):  # 1行ずつ処理
                packet_ends = []

                # ラベルの処理
                label_raw = row[0]
                label = label_txt2num(row[0])  #csvの1列目のラベルを入力
                if (label > 0) and (binary == True): # 二値分類の場合
                    label = 1
                list_labels.append(label)  # ラベルを追加
                list_labels_raw.append(label_raw)

                # パケットの処理
                packet = row[2:]  # csvのパケット部分のベクトルを入力
                packet = [int(s) for s in packet]  #strをintに変換
                if 0 in packet:
                    packet[packet.index(0):] = [] # 末尾のパディング部を一旦除去

                flow_id = int(row[1])
                if flow_id not in packets_of_flows: # 新しいフロー番号の場合
                    packets_of_flows[flow_id] = [0] * input_len # 入力ベクトルを初期化
                    for i in range(n_packets):
                        packets_of_flows[flow_id][i] = 0x10002
                
                if 0 in packets_of_flows[flow_id]:
                    packets_of_flows[flow_id][packets_of_flows[flow_id].index(0):] = [] # パディングがある場合はそれを削除
                packets_of_flows[flow_id][:packets_of_flows[flow_id].index(0x10002)+1] = [] # 古いパケットを削除
                packets_of_flows[flow_id].extend(packet) # 入力ベクトルにパケットを追加
                packets_of_flows[flow_id].append(0x10002) # 終端を追加
                if len(packets_of_flows[flow_id]) < input_len:
                    packets_of_flows[flow_id].extend([0] * (input_len - len(packets_of_flows[flow_id])))# 不足分をパディング
                assert len(packets_of_flows[flow_id]) == input_len
                list_packets.append(deepcopy(packets_of_flows[flow_id]))  # パケットのリストに追加
            
        list_flow_packet_ends = [[n for n, v in enumerate(list_packets[i]) if v == 0x10002]
                                   for i in range(len(list_packets))]
        self.packets    = np.array(list_packets)
        self.labels     = np.array(list_labels)
        self.labels_raw = np.array(list_labels_raw)
        self.flow_packet_ends  = np.array(list_flow_packet_ends)
    
    def data_tuple(self):
        """(packets, labels)のタプルを返す

        Returns:
            tuple(np.ndarray[int], np.ndarray[int]): パケットとラベルのndarrayのタプル
        """        
        return (self.packets, self.labels)

def highlight(word, attn, col_palette=None, conditions=None, feature_str=False):
    """文字列をAttention Scoreでハイライトするhtml文を作成

    Args:
        word (str): 文字列
        attn (float): Attention Score
        col_palette (list(str), optional): カラーパレット（色コードの文字列を指定）指定しない場合はScoreに応じた濃さの赤色
        conditions (list(function), optional): カラーパレットに対応するAttention Scoreの条件関数\
         引数は1つ(float)で、attnが代入される。\
         戻り値はboolであること（Trueの場合にカラーパレットで同じindexの色を選択する）。
        feature_str (boor, optional): 特性に関連する文字列かを指定（該当する場合はマーク色を変更）\
         col_paletteを指定した場合は無視される。（デフォルトはFalse）

    Returns:
        str: 文字列をハイライトして表示するためのhtml文

    Note:
        参考：https://qiita.com/m__k/items/e312ddcf9a3d0ea64d72#attention%E3%81%AE%E5%8F%AF%E8%A6%96%E5%8C%96 
        col_paletteを指定した場合はconditionsも指定すること
    """
    
    if col_palette == None and conditions == None and feature_str == True: # Attention Scoreから色の濃さを設定（攻撃特性文字列の場合）
        html_color = '#%02X%02X%02X' % (int(255*(1 - attn)), 255, int(255*(1 - attn)))
    elif col_palette == None and conditions == None and feature_str == False: # Attention Scoreから赤色の濃さを設定
        html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    else: # カラーパレットと条件関数から色を設定
        html_color_set = False
        for color, condition in zip(col_palette, conditions):
            if condition(attn) == True:
                html_color = color
                html_color_set = True
                break
        if html_color_set == False: # どの条件にも当てはまらなかった場合
            html_color = '#FFFFFF'
        
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def write_multi(txt, *args):
    """複数のファイルに指定した文字列を書き込む

    Args:
        txt (str): 共通して書き込む文字
        args (tuple(TextIOWrapper)): 書き込み対象のファイル
    """
    for f in args:
        f.write(txt)

def addtext_multi(txt, *args):
    """複数の文字列に指定した文字列を連結

    Args:
        txt (str): 共通して連結する文字
        args (tuple(str)): 連結先の文字列（複数指定可能）

    Returns:
        tuple(str): 連結後の文字列のタプル
    """    
    r = []
    for t in args:
        r.append(t + txt)
    return tuple(r)

def print_withfile(txt, *args):
    """printをターミナルとファイルの両方に行う

    Args:
        txt (str): 共通してprintする文字
        args (tuple(TextIOWrapper)): print(書き込み)対象のファイル
    """
    print(txt)
    for f in args:
        print(txt, file=f)

def get_feature_place(feature_str_list, packet_ends):
    """入力におけるパケット別に攻撃の特性に関連する部分の有無・パディングの状態を取得する

    Args:
        feature_str_list (list(int)): 攻撃の特性に関連する部分のインデックスのリスト
        packet_ends (list(int)): パケットの終端位置のリスト

    Returns:
        list(int): 取得した状態（0:False, 1:True, -1:Padding）
    """
    feature_place = [0] * N_PACKETS
    feature_length = len(feature_str_list)
    if feature_length == 0: # 攻撃の特性と関連する部分が含まれていない場合
        if N_PACKETS > 1:
            # パケットがパディングされているかを確認
            for i in range(N_PACKETS):
                if i == 0:
                    if packet_ends[i] == 0:
                        feature_place[i] = -1
                else:
                    if packet_ends[i]- packet_ends[i - 1] == 1:
                        feature_place[i] = -1
    elif N_PACKETS > 1:
        # 攻撃の特性と関連する部分がどのパケットに含まれているかを確認
        for i in range(feature_length):
            for j in range(N_PACKETS):
                if j == 0:
                    if feature_str_list[i] < packet_ends[j]:
                        feature_place[j] = 1
                else:
                    if packet_ends[j - 1] < feature_str_list[i] < packet_ends[j]:
                        feature_place[j] = 1
        
        # パケットがパディングされているかを確認
        for i in range(N_PACKETS):
            if i == 0:
                if packet_ends[i] == 0:
                    feature_place[i] = -1
            else:
                if packet_ends[i]- packet_ends[i - 1] == 1:
                    feature_place[i] = -1
    else: # N_PACKETS == 1 and len(feature_str_dict) > 0
        feature_place = [1]
    return feature_place

def make_html_p(idx, packet, attention_scores, packet_ends):
    """Attention Scoreの可視化html文字列を作成

    Args:
        idx (str): パケットのインデックス
        packet (list(int)): パケットの値（1ずつずれた値）
        attention_scores (pd.DataFrame): Attention ScoreのDataFrame
        packet_ends (list(int)): パケットの終端位置のリスト

    Returns:
        tuple(str): html文字列のタプル
    """    

    label = attention_scores.at[idx, 'Label']
    p = attention_scores.at[idx, 'Prediction']
    prediction = "ATTACK" if p >= 0.5 else 'BENIGN'
    html_txt = f"<h2>Index: {idx}</h2>\n"
    html_txt += f"<p>Label: {label}, Prediction: {prediction}</p>\n"
    scores = attention_scores.values.tolist()
    columns = attention_scores.columns.to_list()
    columns = [x[x.find('_')+1:] for x in columns]
    columns = columns[3:]
    scores = scores[0][3:]

    # Attention scoreに関する統計情報の作成
    scores = np.array(scores)
    _ , score_ranks = np.unique(scores, return_inverse=True)
    score_ranks = np.array(score_ranks)
    score_ranks = np.array([x/score_ranks.max() for x in score_ranks])
    scores_nonzero = scores[scores != 0]
    score_max = np.max(scores)
    score_min = np.min(scores)
    score_min_nonzero = np.min(scores_nonzero)
    score_sum = np.sum(scores)
    score_mean = np.mean(scores_nonzero)
    score_var = np.var(scores_nonzero)
    score_feature_sum = 0
    no_feature = False

    # 最大値の90%以上, 80%以上, 70%以上のカラーパレット作成
    scores_90percent = score_min_nonzero + (score_max - score_min_nonzero) * 0.9
    condition_90percent = lambda x: x >= scores_90percent
    color_90percent = '#FF0000'
    scores_80percent = score_min_nonzero + (score_max - score_min_nonzero) * 0.8
    condition_80percent = lambda x: x >= scores_80percent
    color_80percent = '#00FF00'
    scores_70percent = score_min_nonzero + (score_max - score_min_nonzero) * 0.7
    condition_70percent = lambda x: x >= scores_70percent
    color_70percent = '#69E3FF'
    color_palette = [color_90percent, color_80percent, color_70percent]
    conditions = [condition_90percent, condition_80percent, condition_70percent]

    # 統計情報の書き込み
    html_txt += f"<p>max: {score_max}, min: {score_min} (nonzero: {score_min_nonzero}), sum: {score_sum}, mean: {score_mean}, var: {score_var}</p>\n"
    html_txt_relative = deepcopy(html_txt)
    html_txt_ranking = deepcopy(html_txt)
    html_txt_percentile = deepcopy(html_txt)
    html_txt, html_txt_relative, html_txt_ranking = addtext_multi(
        f"<p>90% score: {scores_90percent}, 80% score: {scores_80percent}, 70% score: {scores_70percent}</p>\n",
          html_txt, html_txt_relative, html_txt_ranking)
    html_txt_percentile += f"<p>90% score: {scores_90percent} (color: <font color={color_90percent}>{color_90percent}</font>), 80% score: {scores_80percent} (color: <font color={color_80percent}>{color_80percent}</font>), 70% score: {scores_70percent} (color: <font color={color_70percent}>{color_70percent}</font>)</p>\n"

    at_columns = ["L3Type", "IPVersion", "IPHeaderLength", "TOS", "DatagramLength", "IPHeaderID", "IPFlag",\
                    "FlagmentOffset", "TTL", "ProtocolNumber", "IPHeaderChecksum", "SequenceNumber1",\
                        "SequenceNumber2", "ACKNumber1", "ACKNumber2", "TCPHeaderLength", "TCPFlag", "Window",\
                            "TCPChecksum", "UrgentPointer", "UDPHeaderLength", "UDPChecksum"] # 特徴量名リスト（field, header）
    packet_number = N_PACKETS - 1 # 現在のパケット番号
    change_packet = True # 新しいパケットになったことを表すフラグ
    fetch_packet_part = 0 # パケットで読み込んでいる部分

    feature_str_list = []
    if label != 'BENIGN': # 攻撃のパケットの場合，特性となる文字列を記録
        length_mask = np.count_nonzero(packet == 0)
        payload_string = ''.join(map(chr, [v - 1 if v > 0 else 0 for v in packet])) # 1ずつずれているので元に戻す（パディングは0のまま）
        feature_end = False
        s = 0
        while feature_end == False:
            if label == 'BruteForce-Web':
                f_match_u = payload_string.find('username=', s)
                f_match_p = payload_string.find('password=', s)
                if f_match_u >= 0 and (f_match_u < f_match_p or f_match_p < 0):
                    feature_str_list.extend(list(range(f_match_u, payload_string.find('&', f_match_u))))
                    s = payload_string.find('&', f_match_u) + 1
                elif f_match_p >= 0 and (f_match_p < f_match_u or f_match_u < 0):
                    feature_str_list.extend(list(range(f_match_p, payload_string.find('&', f_match_p))))
                    s = payload_string.find('&', f_match_p) + 1
                else:
                    feature_end = True
            elif label == 'BruteForce-XSS':
                f_match_s = payload_string.find('%3Cscript%3E', s)
                f_match_e = payload_string.find('%3C%2Fscript%3E', s)
                if f_match_s >= 0 and (f_match_s < f_match_e or f_match_e < 0):
                    feature_str_list.extend(list(range(f_match_s, f_match_s+12)))
                    s = f_match_s + 12
                elif f_match_e >= 0 and (f_match_e < f_match_s or f_match_s < 0):
                    feature_str_list.extend(list(range(f_match_e, f_match_e+15)))
                    s = f_match_e + 15
                else:
                    feature_end = True
            elif label == 'SQLInjection':
                f_match_id = payload_string.find('id=', s)
                f_match_ie = payload_string.find('&', f_match_id)
                f_match_h = payload_string.find('%23', f_match_id)
                f_match_q = payload_string.find('%27', f_match_id)
                if f_match_id >= 0 and f_match_ie >= 0 and ((f_match_h >= 0 and f_match_h < f_match_ie) or (f_match_q >= 0 and f_match_q < f_match_ie)):
                    feature_str_list.extend(list(range(f_match_id, f_match_ie)))
                    s = f_match_ie + 1
                else:
                    feature_end = True
            else:
                print(label)
                raise Exception("Unknown label error!")
            
    feature_length = len(feature_str_list)
    no_feature = (feature_length == 0)
    feature_place = get_feature_place(feature_str_list, packet_ends)

    for i in range(N_PACKETS * HPLEN + N_PACKETS):
        if change_packet == True: # 新しいパケットになったとき
            if packet_number < 0: # もうパケットがない場合は終了
                break
            feature_place_txt = ['Padding', 'No feature', 'With feature'][feature_place[N_PACKETS - packet_number - 1] + 1]
            html_txt, html_txt_relative, html_txt_ranking, html_txt_percentile = addtext_multi(
                f"<h3>Packet {packet_number} ({feature_place_txt})</h3>\n", html_txt, html_txt_relative, html_txt_ranking, html_txt_percentile)
            change_packet = False

        score = scores[i]
        score_rank = score_ranks[i]
        value = packet[i] - 1

        feature = False
        if no_feature == False: # 攻撃の特性に関連する文字列がある場合
            feature = True if i in feature_str_list else False
        if feature: # 攻撃の特性に関連する文字の場合
            score_feature_sum += score

        if value == 0x10001: # 終端の場合
            html_txt += f"<br>\n{highlight('(End of packet)', score, feature_str=feature)}"
            html_txt_relative += f"<br>\n{highlight('(End of packet)', score/score_max, feature_str=feature)}"
            html_txt_ranking += f"<br>\n{highlight('(End of packet)', score_rank, feature_str=feature)}"
            html_txt_percentile += f"<br>\n{highlight('(End of packet)', score, col_palette=color_palette, conditions=conditions)}"
            fetch_packet_part = 0
            packet_number -= 1
            change_packet = True
            continue

        if fetch_packet_part < 22: # ヘッダの場合
            if value == 0x10000: # パディングの場合
                html_txt += f"{at_columns[fetch_packet_part]}: {highlight('(Padding)', score, feature_str=feature)}"
                html_txt_relative += f"{at_columns[fetch_packet_part]}: {highlight('(Padding)', score/score_max, feature_str=feature)}"
                html_txt_ranking += f"{at_columns[fetch_packet_part]}: {highlight('(Padding)', score_rank, feature_str=feature)}"
                html_txt_percentile += f"{at_columns[fetch_packet_part]}: {highlight('(Padding)', score, col_palette=color_palette, conditions=conditions)}"
            else:
                html_txt += f"{at_columns[fetch_packet_part]}: {highlight(value, score, feature_str=feature)}"
                html_txt_relative += f"{at_columns[fetch_packet_part]}: {highlight(value, score/score_max, feature_str=feature)}"
                html_txt_ranking += f"{at_columns[fetch_packet_part]}: {highlight(value, score_rank, feature_str=feature)}"
                html_txt_percentile += f"{at_columns[fetch_packet_part]}: {highlight(value, score, col_palette=color_palette, conditions=conditions)}"
            if fetch_packet_part < 21: # ヘッダの最後のフィールド以外
                html_txt, html_txt_relative, html_txt_ranking, html_txt_percentile = addtext_multi(
                    ", ", html_txt, html_txt_relative, html_txt_ranking, html_txt_percentile)
        else: # ペイロードの場合
            if fetch_packet_part == 22:
                html_txt, html_txt_relative, html_txt_ranking, html_txt_percentile = addtext_multi(
                    "<br><br>\n", html_txt, html_txt_relative, html_txt_ranking, html_txt_percentile)
            html_txt += highlight(chr(value), score, feature_str=feature)
            html_txt_relative += highlight(chr(value), score/score_max, feature_str=feature)
            html_txt_ranking += highlight(chr(value), score_rank, feature_str=feature)
            html_txt_percentile += highlight(chr(value), score, col_palette=color_palette, conditions=conditions)
        
        fetch_packet_part += 1

    if no_feature:
        html_txt, html_txt_relative, html_txt_ranking = addtext_multi(
            "<br>[There is no string related to the features of the attack.]\n", html_txt, html_txt_relative, html_txt_ranking)     
    else:
        html_txt, html_txt_relative, html_txt_ranking = addtext_multi(
            f"<br>Number of characters of the attack features: {feature_length}\n", html_txt, html_txt_relative, html_txt_ranking) 
        feature_rate = feature_length / (HPLEN * N_PACKETS + N_PACKETS - length_mask)
        html_txt, html_txt_relative, html_txt_ranking = addtext_multi(
            f"<br> Characters of the attack features (rate): {feature_rate}\n", html_txt, html_txt_relative, html_txt_ranking)     
        score_rate = score_feature_sum / score_sum
        html_txt, html_txt_relative, html_txt_ranking = addtext_multi(
            f"<br>Attention score of the attack features (rate): {score_rate}\n", html_txt, html_txt_relative, html_txt_ranking)     

    html_txt, html_txt_relative, html_txt_ranking, html_txt_percentile = addtext_multi(
        "<hr>\n", html_txt, html_txt_relative, html_txt_ranking, html_txt_percentile)
    return html_txt, html_txt_relative, html_txt_ranking, html_txt_percentile


def visualize_attention_score():
    """メインとなる関数

    Raises:
        Exception: 未知のラベルがある場合に発生
    """    
    print("Loading packet data...")
    test_dataset = masked_dataset(data_path=TEST_PATH, n_samples=N_SAMPLES, n_packets=N_PACKETS) # パケットデータを読み込み
    print("Loading attention score...")
    attention_scores = pd.read_csv(os.path.join(RESULT_DIR, SCORE_FILE)) # Attention Scoreの読み込み

    os.makedirs(os.path.join(RESULT_DIR, 'attention_plot'), exist_ok=True)
    with open(os.path.join(RESULT_DIR, f"{OUT_FILE_PREFIX}.html"), 'w') as f:
        with open(os.path.join(RESULT_DIR, f"{OUT_FILE_PREFIX}_relative.html"), 'w') as f_relative:
            with open(os.path.join(RESULT_DIR, f"{OUT_FILE_PREFIX}_ranking.html"), 'w') as f_ranking:
                with open(os.path.join(RESULT_DIR, f"{OUT_FILE_PREFIX}_percentile.html"), 'w') as f_percentile:
                    write_multi("<!DOCTYPE html>\n<html>\n<head>\n", f, f_relative, f_ranking, f_percentile)

                    f.write("<title>Attention Score</title>\n")
                    f_relative.write("<title>Attention Score (Relative)</title>\n")
                    f_ranking.write("<title>Attention Score (Ranking)</title>\n")
                    f_percentile.write("<title>Attention Score (Percentile)</title>\n")
                    
                    write_multi("</head>\n<body>\n", f, f_relative, f_ranking, f_percentile)
                    write_multi('<div style="overflow-wrap: anywhere;">\n', f, f_relative, f_ranking, f_percentile)

                    f.write("<h1>Attention Score</h1>\n")
                    f_relative.write("<h1>Attention Score (Relative)</h1>\n")
                    f_ranking.write("<h1>Attention Score (Ranking)</h1>\n")
                    f_percentile.write("<h1>Attention Score (Percentile)</h1>\n")

                    write_multi(f"<p>Test dataset: {TEST_PATH}</p>\n", f, f_relative, f_ranking, f_percentile)
                    write_multi(f"<p>Direction: {RESULT_DIR}</p>\n<hr>\n", f, f_relative, f_ranking, f_percentile)

                    list_labels = attention_scores['Label'].unique().tolist()
                    df_idx = []
                    print("Extracting indexes...")
                    for l in tqdm(list_labels): # ラベル・予測結果ごとにインデックスを抽出
                        df = attention_scores.query(f"Label == '{l}'")
                        df_p = df.query('Prediction < 0.5')    # BENIGN
                        df_s = df_p.sample(n=min(len(df_p), 15)).sort_index() # サンプリング
                        df_idx.extend(df_s.index.tolist()) # インデックスを追加
                        df_p = df.query('Prediction >= 0.5')   # ATTACK
                        df_s = df_p.sample(n=min(len(df_p), 15)).sort_index() # サンプリング
                        df_idx.extend(df_s.index.tolist()) # インデックスを追加

                    print("Processing attention score...")
                    score_feature_rate_list_dict = {'BruteForce-Web_ATTACK':[], 'BruteForce-Web_BENIGN':[],
                                                    'BruteForce-XSS_ATTACK':[], 'BruteForce-XSS_BENIGN':[],
                                                    'SQLInjection_ATTACK':[], 'SQLInjection_BENIGN':[]}
                    part_feature_rate_list_dict = {'BruteForce-Web_ATTACK':[], 'BruteForce-Web_BENIGN':[],
                                                   'BruteForce-XSS_ATTACK':[], 'BruteForce-XSS_BENIGN':[],
                                                   'SQLInjection_ATTACK':[], 'SQLInjection_BENIGN':[]}
                    n_packets_list_dict = {'BENIGN_ATTACK':[], 'BENIGN_BENIGN':[],
                                           'BruteForce-Web_ATTACK':[], 'BruteForce-Web_BENIGN':[],
                                           'BruteForce-XSS_ATTACK':[], 'BruteForce-XSS_BENIGN':[],
                                           'SQLInjection_ATTACK':[], 'SQLInjection_BENIGN':[]}

                    feature_place_dict = {'Label':[], 'Prediction':[]}
                    for i in reversed(range(N_PACKETS)):
                        feature_place_dict[f'packet_{i}'] = []
                    
                    # 入力ごとの処理
                    for i in tqdm(range(N_SAMPLES)):
                        attention_scores_packet = attention_scores.loc[[i]]
                        label = attention_scores_packet.at[i, 'Label']
                        feature_place_dict['Label'].append(label)
                        scores = attention_scores_packet.values.tolist()[0][3:]
                        prediction = 'ATTACK' if attention_scores_packet.at[i, 'Prediction'] >= 0.5 else 'BENIGN'
                        feature_place_dict['Prediction'].append(prediction)
                        feature_place = [0] * N_PACKETS # 0:False, 1:True, -1:Padding

                        # 特徴量と関連する文字列の確認（ラベルがBENIGNのものを除く）
                        no_feature = False # 特徴量と関連する文字列がない場合はTrueとなる
                        feature_str_list = [] # 特徴量と関連する文字列と対応するindexを記録するリスト
                        if label != 'BENIGN': # 攻撃のパケットの場合，特性となる文字列を記録
                            length_mask = np.count_nonzero(test_dataset.packets[i] == 0)
                            payload_string = ''.join(map(chr, [v - 1 if v > 0 else 0 for v in test_dataset.packets[i]])) # 1ずつずれているので元に戻す（パディングは0のまま）
                            feature_end = False
                            s = 0
                            while feature_end == False:
                                if label == 'BruteForce-Web':
                                    # 'username='と'password='を検知
                                    f_match_u = payload_string.find('username=', s)
                                    f_match_p = payload_string.find('password=', s)
                                    if f_match_u >= 0 and (f_match_u < f_match_p or f_match_p < 0):
                                        feature_str_list.extend(list(range(f_match_u, payload_string.find('&', f_match_u))))
                                        s = payload_string.find('&', f_match_u) + 1
                                    elif f_match_p >= 0 and (f_match_p < f_match_u or f_match_u < 0):
                                        feature_str_list.extend(list(range(f_match_p, payload_string.find('&', f_match_p))))
                                        s = payload_string.find('&', f_match_p) + 1
                                    else:
                                        feature_end = True
                                elif label == 'BruteForce-XSS':
                                    # '<script>'と'</script>'を検知
                                    f_match_s = payload_string.find('%3Cscript%3E', s)
                                    f_match_e = payload_string.find('%3C%2Fscript%3E', s)
                                    if f_match_s >= 0 and (f_match_s < f_match_e or f_match_e < 0):
                                        feature_str_list.extend(list(range(f_match_s, f_match_s+12)))
                                        s = f_match_s + 12
                                    elif f_match_e >= 0 and (f_match_e < f_match_s or f_match_s < 0):
                                        feature_str_list.extend(list(range(f_match_e, f_match_e+15)))
                                        s = f_match_e + 15
                                    else:
                                        feature_end = True
                                elif label == 'SQLInjection':
                                    #'id='から'&'までの間に'#'またはシングルクォーテーションがあることを検知
                                    f_match_id = payload_string.find('id=', s)
                                    f_match_ie = payload_string.find('&', f_match_id)
                                    f_match_h = payload_string.find('%23', f_match_id)
                                    f_match_q = payload_string.find('%27', f_match_id)
                                    if f_match_id >= 0 and f_match_ie >= 0 and ((f_match_h >= 0 and f_match_h < f_match_ie) or (f_match_q >= 0 and f_match_q < f_match_ie)):
                                        feature_str_list.extend(list(range(f_match_id, f_match_ie)))
                                        s = f_match_ie + 1
                                    else:
                                        feature_end = True
                                else: # 未知のラベルの場合例外を発生させる
                                    print(label)
                                    raise Exception("Unknown label error!")
                        
                        lp = f"{label}_{prediction}"
                        no_feature = (len(feature_str_list) == 0)
                        feature_place = get_feature_place(feature_str_list, test_dataset.flow_packet_ends[i])
                        for j in range(N_PACKETS):
                            feature_place_dict[f'packet_{j}'].append(feature_place[N_PACKETS - j - 1])
                        assert feature_place.count(-1) < N_PACKETS
                        n_packets_list_dict[lp].append(N_PACKETS - feature_place.count(-1)) # パディングでない数

                        # 特徴量と関連する文字列のAttention scoreの処理
                        if no_feature == False:
                            feature_scores = [v if j in feature_str_list else 0 for j, v in enumerate(scores)]
                            score_sum = np.sum(np.array(scores))
                            feature_score_sum = np.sum(np.array(feature_scores))
                            score_feature_rate_list_dict[lp].append(feature_score_sum / score_sum)
                            part_feature_rate_list_dict[lp].append(len(feature_str_list) / (HPLEN * N_PACKETS + N_PACKETS - length_mask))

                    for i in tqdm(df_idx):
                        t, t_relative, t_ranking, t_percentile = make_html_p(i, test_dataset.packets[i], attention_scores.loc[[i]], test_dataset.flow_packet_ends[i])
                        f.write(t)
                        f_relative.write(t_relative)
                        f_ranking.write(t_ranking)
                        f_percentile.write(t_percentile)

                    write_multi("</div>\n</body>\n</html>\n", f, f_relative, f_ranking, f_percentile)

    score_feature_rate_average_dict = {'BruteForce-Web_ATTACK':0, 'BruteForce-Web_BENIGN':0,
                                       'BruteForce-XSS_ATTACK':0, 'BruteForce-XSS_BENIGN':0,
                                       'SQLInjection_ATTACK':0, 'SQLInjection_BENIGN':0}
    count_with_feature_dict = {'BruteForce-Web_ATTACK':0, 'BruteForce-Web_BENIGN':0,
                               'BruteForce-XSS_ATTACK':0, 'BruteForce-XSS_BENIGN':0,
                               'SQLInjection_ATTACK':0, 'SQLInjection_BENIGN':0}
    diff_feature_average_dict = {'BruteForce-Web_ATTACK':0, 'BruteForce-Web_BENIGN':0,
                                 'BruteForce-XSS_ATTACK':0, 'BruteForce-XSS_BENIGN':0,
                                 'SQLInjection_ATTACK':0, 'SQLInjection_BENIGN':0}
    diff_feature_positive_average_dict = {'BruteForce-Web_ATTACK':0, 'BruteForce-Web_BENIGN':0,
                                          'BruteForce-XSS_ATTACK':0, 'BruteForce-XSS_BENIGN':0,
                                          'SQLInjection_ATTACK':0, 'SQLInjection_BENIGN':0}
    diff_feature_negative_average_dict = {'BruteForce-Web_ATTACK':0, 'BruteForce-Web_BENIGN':0,
                                          'BruteForce-XSS_ATTACK':0, 'BruteForce-XSS_BENIGN':0,
                                          'SQLInjection_ATTACK':0, 'SQLInjection_BENIGN':0}
    diff_feature_positive_count_dict = {'BruteForce-Web_ATTACK':0, 'BruteForce-Web_BENIGN':0,
                                        'BruteForce-XSS_ATTACK':0, 'BruteForce-XSS_BENIGN':0,
                                        'SQLInjection_ATTACK':0, 'SQLInjection_BENIGN':0}
    diff_feature_negative_count_dict = {'BruteForce-Web_ATTACK':0, 'BruteForce-Web_BENIGN':0,
                                        'BruteForce-XSS_ATTACK':0, 'BruteForce-XSS_BENIGN':0,
                                        'SQLInjection_ATTACK':0, 'SQLInjection_BENIGN':0}
    n_packets_average_dict = {'BENIGN_ATTACK':0, 'BENIGN_BENIGN':0,
                              'BruteForce-Web_ATTACK':0, 'BruteForce-Web_BENIGN':0,
                              'BruteForce-XSS_ATTACK':0, 'BruteForce-XSS_BENIGN':0,
                              'SQLInjection_ATTACK':0, 'SQLInjection_BENIGN':0}
    n_packets_count_dict = {'BENIGN_ATTACK':[], 'BENIGN_BENIGN':[],
                            'BruteForce-Web_ATTACK':[], 'BruteForce-Web_BENIGN':[],
                            'BruteForce-XSS_ATTACK':[], 'BruteForce-XSS_BENIGN':[],
                            'SQLInjection_ATTACK':[], 'SQLInjection_BENIGN':[]}
    fig_all = plt.figure()
    ax_all = fig_all.add_subplot(1,1,1)
    ax_all.set_title('Attention score rate of part of attack features')
    ax_all.set_xlabel('Part of attack features (Rate)')
    ax_all.set_ylabel('Attention score of attack features (Rate)')
    ax_all.set_xlim(0, 1)
    ax_all.set_ylim(0, 1)
    ax_all.grid(True)
    ax_all.set_aspect('equal')
    ax_all.plot([0, 1], [0, 1], color='gray')
    sc_colors = ['b', 'g', 'r', 'c', 'm', 'y']
    sc_color = 0

    for k in score_feature_rate_list_dict.keys():
        c = len(score_feature_rate_list_dict[k])
        if c > 0:
            score_feature_rate_average_dict[k] = np.sum(np.array(score_feature_rate_list_dict[k])) / c
            count_with_feature_dict[k] = c
            diff_feature = np.array(score_feature_rate_list_dict[k]) - np.array(part_feature_rate_list_dict[k])
            diff_feature_average_dict[k] = np.sum(diff_feature) / c
            diff_feature_positive_count_dict[k] = diff_feature[diff_feature > 0].shape[0]
            if diff_feature_positive_count_dict[k] > 0:
                diff_feature_positive_average_dict[k] = np.sum(diff_feature[diff_feature > 0]) / diff_feature_positive_count_dict[k]
            diff_feature_negative_count_dict[k] = diff_feature[diff_feature < 0].shape[0]
            if diff_feature_negative_count_dict[k] > 0:
                diff_feature_negative_average_dict[k] = np.sum(diff_feature[diff_feature < 0]) / diff_feature_negative_count_dict[k]
            data_table = pd.DataFrame({'score_rate':score_feature_rate_list_dict[k], 'feature_rate': part_feature_rate_list_dict[k]})
            data_table.to_csv(os.path.join(RESULT_DIR, 'attention_plot', f'{k}.csv'))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            # print(score_feature_rate_list_dict[k])
            assert c == len(part_feature_rate_list_dict[k])
            ax.scatter(part_feature_rate_list_dict[k], np.array(score_feature_rate_list_dict[k]), s=10)
            ax_all.scatter(part_feature_rate_list_dict[k], np.array(score_feature_rate_list_dict[k]), s=10, color=sc_colors[sc_color], label=k)
            ax.set_title(k)
            ax.set_xlabel('Part of attack features (Rate)')
            ax.set_ylabel('Attention score of attack features (Rate)')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.set_aspect('equal')
            ax.plot([0, 1], [0, 1], color='gray')
            fig.savefig(os.path.join(RESULT_DIR, 'attention_plot', f'attention_{k}.png'))
            plt.close(fig)
        sc_color += 1
    
    ax_all.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0,))
    # fig_all.tight_layout() 
    fig_all.savefig(os.path.join(RESULT_DIR, 'attention_plot', f'attention_all.png'), bbox_inches='tight')
    plt.close(fig_all)

    if N_PACKETS > 1:
        for k in n_packets_list_dict.keys():
            c = len(n_packets_list_dict[k])
            if c > 0:
                n_packets_average_dict[k] = np.sum(np.array(n_packets_list_dict[k])) / c
                n_packets_count_dict[k] = [n_packets_list_dict[k].count(i+1) for i in range(N_PACKETS)]

    feature_place_df = pd.DataFrame(feature_place_dict)
    feature_place_df.to_csv(os.path.join(RESULT_DIR, 'feature_place.csv'))

    output_dict = {'Real Packets in Input Blocks (Average)': n_packets_average_dict,
                   'Input Blocks with Attack Features': count_with_feature_dict,
                   'Attack Feature Attention Percentage (Average)': score_feature_rate_average_dict,
                   'Attention-Input Percentage Difference (Average)': diff_feature_average_dict,
                   'Attention-Input Percentage Difference (Average for Positive)': diff_feature_positive_average_dict,
                   'Attention-Input Percentage Difference (Number of Positive)': diff_feature_positive_count_dict,
                   'Attention-Input Percentage Difference (Average for Negative)': diff_feature_negative_average_dict,
                   'Attention-Input Percentage Difference (Number of Negative)': diff_feature_negative_count_dict}
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(os.path.join(RESULT_DIR, OUT_FEATURE_SCORE_FILE))

    # with open(os.path.join(RESULT_DIR, OUT_FEATURE_SCORE_FILE), 'w') as f:
        # print_withfile("BENIGN_ATTACK", f)
        # if N_PACKETS > 1:
        #     print_withfile(f" n_packets(avg) : {n_packets_average_dict['BENIGN_ATTACK']}", f)
        #     print_withfile(f"       n_packets: {n_packets_count_dict['BENIGN_ATTACK']}", f)
        # print_withfile("BENIGN_BENIGN", f)
        # if N_PACKETS > 1:
        #     print_withfile(f" n_packets(avg) : {n_packets_average_dict['BENIGN_BENIGN']}", f)
        #     print_withfile(f"       n_packets: {n_packets_count_dict['BENIGN_BENIGN']}", f)
        # for k in score_feature_rate_average_dict.keys():
        #     print_withfile(f"{k}", f)
        #     print_withfile(f"          count : {count_with_feature_dict[k]}", f)
        #     print_withfile(f"        average : {score_feature_rate_average_dict[k]}", f)
        #     print_withfile(f"      diff(avg) : {diff_feature_average_dict[k]}", f)
        #     if N_PACKETS > 1:
        #         print_withfile(f" n_packets(avg) : {n_packets_average_dict[k]}", f)
        #         print_withfile(f"       n_packets: {n_packets_count_dict[k]}", f)


if __name__ == "__main__":
    visualize_attention_score()