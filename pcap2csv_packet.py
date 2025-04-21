"""
ペイロードがあるパケットのみにおいて, csvのheader+payloadと, ラベルを生成するプログラム
header+payloadは, 長さを指定することができる
指定した長さに満たないパケットは，その長さまで"00"を付与する

v3における変更点
- ペイロードの無いパケット/TCPでもUDPでもないパケットをスキップ
- 多値ラベルに変更
- 処理を修正

v4における変更点
- 空白を表す文字を0から0x100に変更
    - Embedding Layerのinput_dimの変更を忘れずに

v5における変更点
- IPアドレス, MACアドレス, ポート番号をパディングするのではなく, 削除するように変更
    - TCP/UDPどちらも24バイト短くなり, その分はペイロードとする

v5fにおける変更点
- フィールド単位の分割に変更
    - これに合わせて, パディングを0x10000に変更
- 長さをヘッダとペイロードの長さの合計に合わせる機能を追加
    - 注意：最大長を指定した場合，巨大なcsvファイルが生成されます

v6における変更点
- 各フィールドを別々の要素にするように変更

v7における変更点
- TCP/UDPストリーム番号を付与するように変更（識別に必要なIPアドレスおよびポート番号を削除するため）
- IPv6パケットを除去するように変更（攻撃パケットには含まれない）
- ラベルを文字列形式に変更

v8における変更点
- マスクに対応するため, パディングを0にし, それ以外は値を1ずつずらすように変更
    - 入力時 0x10002:パディング, 0x10001:パケットの区切り, 0x10000:TCP/UDPのうち存在しない方

Input:
    merged.pcap
    labels.csv
Output:
    csv
    The csv file's header+payload is "Label,Packet".
"""
from scapy.all import PcapReader
import csv
import time
import os
from tqdm import tqdm

def loadlabel_v2(labelfile):
    """文字列形式のラベルを数値形式に変換（one-hot encodeはまだ）

    Args:
        labelfile (string): ラベルのcsvファイルのパス

    Raises:
        Exception: 利用できないラベルの時に発生

    Returns:
        list: ラベルのリスト（例：[0,3,1,2,1,...]）
    
    Note:
        他のデータセットを使用する場合，ラベルの種類を適宜追加すること
    """    
    labels = []
    with open(labelfile) as f:
        csvreader = csv.reader(f)
        for l in csvreader:
            if l == ['BENIGN']:
                labels.append(['BENIGN'])
            elif l == ['BruteForce-Web']:
                labels.append(['BruteForce-Web'])
            elif l == ['BruteForce-XSS']:
                labels.append(['BruteForce-XSS'])
            elif l == ['SQLInjection']:
                labels.append(['SQLInjection'])
            else:
                raise Exception("Unknown label error!")
    return labels

def payload_length(pcapfile):
    """pcapファイルに含まれるパケットのペイロードの最大長を求める

    Args:
        pcapfile (str): pcapファイルのパス

    Returns:
        int: ペイロードの最大長
    """    
    payload_max_len = 0
    packets = PcapReader(pcapfile)
    n_packets = len(packets.read_all())
    for packet in tqdm(PcapReader(pcapfile), total=n_packets):  # read pcap packet by packet
        if "TCP" in packet:
            p = packet["TCP"].payload  # TCPのペイロードを抜き出し
            p2 = "TCP"
        elif "UDP" in packet:  # UDPのペイロードを抜き出し
            p = packet["UDP"].payload
            p2 = "UDP"
        #elif "ICMP" in packet:  # ICMPのペイロードを抜き出し
            #p = packet["ICMP"].payload
        #elif "ARP" in packet:  # ARPのペイロードを抜き出し
            #p = packet["ARP"].payload
        else:
            p = "others"

        # when there's no payload, it's replaced by "nan"
        if isinstance(p, scapy.packet.NoPayload):
            # ペイロードが無いパケットはこの時点でスキップする
            pass
        elif isinstance(p, scapy.packet.Padding):
            # ペイロードが無いパケットはこの時点でスキップする
            pass
        elif p == "others":
            # TCP/UDPで無いパケットはこの時点でスキップする
            pass
        elif packet.haslayer('IPv6'):
            # v7で変更：IPv6のパケットはこの時点でスキップする
            pass
        elif p2 == "TCP":  # TCPの場合
            # ペイロードの処理（バイト単位）
            payload = bytes(p)
            payload_max_len = max(payload_max_len, len(list(payload)))
        else:  # UDPの場合
            payload = bytes(p)
            payload_max_len = max(payload_max_len, len(list(payload)))

    return payload_max_len

def pcap2csv(pcapfile, labelfile, outputfile, header_fields, with_payload=True, header_payload_length=200, log_file=None):
    """パケットを処理してcsvファイルに変換

    pcapファイルのパケットと，それに対応するラベルデータを処理し，csvファイルとして出力する

    Args:
        pcapfile (str): pcapファイルのパス
        labelfile (str): ラベルデータが格納されたファイル（csv形式）のパス
        outputfile (str): 出力ファイル（csv形式）のパス
        header_fields (list(str)): ヘッダのフィールドの名前
        with_payload (bool): ペイロードを含めるか（デフォルトはTrue）
        header_payload_length (int, optional): 使用する長さ（バイト単位）（デフォルトは200）
            注: with_payload == False の場合はこの値は使用しない
        log_file (str, optional): ログファイルを出力する場合，出力先ファイル（txt形式）のパス
    """    
    label_process_time_start = time.time()
    labels = loadlabel_v2(labelfile)
    n_header_fields = len(header_fields)
    field_names = ['Label', 'Stream Number']
    field_names.extend(header_fields)
    if header_payload_length > n_header_fields:
        field_names.extend([f"Payload_{i:03}" for i in range(header_payload_length-n_header_fields)]) # 特徴量名リストにペイロード部分(byte)を追加
    label_process_time_end = time.time()
    label_process_time = label_process_time_end - label_process_time_start
    stream_numbers = []
    count = 0
    count_nopayload = 0
    count_IPv6 = 0

    # パケットを処理してcsvファイルに書き込み
    packet_process_time_start = time.time()
    with open(outputfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(field_names)  # CSV header
        j = 0
        packets = PcapReader(pcapfile)
        n_packets = len(packets.read_all())
        for packet in tqdm(PcapReader(pcapfile), total=n_packets):  # read pcap packet by packet
            if "TCP" in packet:
                p = packet["TCP"].payload  # TCPのペイロードを抜き出し
                p2 = "TCP"
            elif "UDP" in packet:  # UDPのペイロードを抜き出し
                p = packet["UDP"].payload
                p2 = "UDP"
            #elif "ICMP" in packet:  # ICMPのペイロードを抜き出し
                #p = packet["ICMP"].payload
            #elif "ARP" in packet:  # ARPのペイロードを抜き出し
                #p = packet["ARP"].payload
            else:
                p = "others"

            # when there's no payload, it's replaced by "nan"
            if isinstance(p, scapy.packet.NoPayload):
                # ペイロードが無いパケットはこの時点でスキップする
                # payload = "NA"
                # payload_ = labels[j]
                # payload_.append(payload)  # [Label,Payload]
                # writer.writerow(payload_)
                count_nopayload += 1
            elif isinstance(p, scapy.packet.Padding):
                # ペイロードが無いパケットはこの時点でスキップする
                # payload = "NA"
                # payload_ = labels[j]
                # payload_.append(payload)  # [Label,Payload]
                # writer.writerow(payload_)
                count_nopayload += 1
            elif p == "others":
                # TCP/UDPで無いパケットはこの時点でスキップする
                # payload = "NA"
                # payload_ = labels[j]
                # payload_.append(payload)  # [Label,Payload]
                # writer.writerow(payload_)
                count_nopayload += 1
            elif packet.haslayer('IPv6'):
                # v7で変更：IPv6のパケットはこの時点でスキップする
                count_IPv6 += 1
            elif p2 == "TCP":  # TCPの場合
                if "IP" not in packet:
                    print(packet.layers())
                    raise Exception
                # v7で変更：TCPストリーム番号を付与
                ip_src = packet['IP'].src # 送信元IPアドレス
                ip_dst = packet['IP'].dst # 宛先IPアドレス
                port_src = packet['TCP'].sport  # 送信元ポート番号
                port_dst = packet['TCP'].dport  # 宛先ポート番号
                if ip_src > ip_dst:
                    ip_src, ip_dst = ip_dst, ip_src
                    port_src, port_dst = port_dst, port_src
                stream_info = {'ip_1':ip_src, 'ip_2':ip_dst, 'port_1':port_src, 'port_2':port_dst,
                                'proto':'TCP'}
                if stream_info not in stream_numbers:
                    stream_numbers.append(stream_info)
                    stream_number = len(stream_numbers) - 1
                else:
                    stream_number = stream_numbers.index(stream_info)

                # バイト単位に分割
                raw_data_bytes = bytes(packet)
                raw_data_bytes = raw_data_bytes[:54]  # 先頭54バイトのみ抜き出し raw_data[0:54]
                raw_data_bytes = list(raw_data_bytes)

                # v5で変更：一部の情報のmask→削除に変更
                raw_data_bytes[26:38] = [] # IP addressと port番号を削除
                raw_data_bytes[0:12] = [] # MAC addressを削除

                # バイト単位からフィールド単位に変換
                raw_data = []
                raw_data.append((raw_data_bytes[0]<<8)+raw_data_bytes[1]) # タイプ
                raw_data.append(raw_data_bytes[2]>>4) # IPバージョン
                raw_data.append(raw_data_bytes[2]&0x0f) # IPヘッダ長
                raw_data.append(raw_data_bytes[3]) # TOS
                raw_data.append((raw_data_bytes[4]<<8)+raw_data_bytes[5]) # データグラム長
                raw_data.append((raw_data_bytes[6]<<8)+raw_data_bytes[7]) # ID
                raw_data.append(raw_data_bytes[8]>>5) # IPフラグ
                raw_data.append(((raw_data_bytes[8]&0b00011111)<<8)+raw_data_bytes[9]) # フラグメントオフセット
                raw_data.append(raw_data_bytes[10]) # TTL
                raw_data.append(raw_data_bytes[11]) # プロトコル番号
                raw_data.append((raw_data_bytes[12]<<8)+raw_data_bytes[13]) # IPヘッダチェックサム

                raw_data.append((raw_data_bytes[14]<<8)+raw_data_bytes[15]) # シーケンス番号1
                raw_data.append((raw_data_bytes[16]<<8)+raw_data_bytes[17]) # シーケンス番号2
                raw_data.append((raw_data_bytes[18]<<8)+raw_data_bytes[19]) # ACK番号1
                raw_data.append((raw_data_bytes[20]<<8)+raw_data_bytes[21]) # ACK番号2
                raw_data.append(raw_data_bytes[22]>>4) # ヘッダ長
                raw_data.append(((raw_data_bytes[22]&0b00001111)<<8)+raw_data_bytes[23]) # TCPフラグ
                raw_data.append((raw_data_bytes[24]<<8)+raw_data_bytes[25]) # ウインドウ
                raw_data.append((raw_data_bytes[26]<<8)+raw_data_bytes[27]) # TCPチェックサム
                raw_data.append((raw_data_bytes[28]<<8)+raw_data_bytes[29]) # 緊急ポインタ

                raw_data.append(0x10000) # 長さ
                raw_data.append(0x10000) # UDPチェックサム

                if with_payload == True:
                    # ペイロードの処理（バイト単位）
                    payload = bytes(p)
                    payload = payload[0:header_payload_length - n_header_fields]  # cut payload in  header_payload_length - len(header)
                    payload = list(payload)

                    # v5fで変更：空白を示す値を0x100から0x10000に変更
                    raw_data.extend(payload)  # headerとpayloadを連結
                    raw_data_zero = [0x10002]*header_payload_length  # 0x10000の任意の長さのbyte列を生成
                    raw_data_zero[0:len(raw_data)] = raw_data  # 0x10000で埋めたバイト列を上書き
                    raw_data = raw_data_zero  # raw_data_zero を raw_dataとする

                if with_payload == True and len(raw_data) != header_payload_length:
                    print(len(raw_data))
                raw_data = [(x+1) % 0x10003 for x in raw_data]
                raw_data_ = labels[j]
                raw_data_.extend([stream_number])
                raw_data_.extend(raw_data)
                writer.writerow(raw_data_)
                count += 1
            else:  # UDPの場合
                if "IP" not in packet:
                    print(packet.layers())
                    raise Exception
                # v7で変更：UDPストリーム番号を付与
                ip_src = packet['IP'].src # 送信元IPアドレス
                ip_dst = packet['IP'].dst # 宛先IPアドレス
                port_src = packet['UDP'].sport  # 送信元ポート番号
                port_dst = packet['UDP'].dport  # 宛先ポート番号
                if ip_src > ip_dst:
                    ip_src, ip_dst = ip_dst, ip_src
                    port_src, port_dst = port_dst, port_src
                stream_info = {'ip_1':ip_src, 'ip_2':ip_dst, 'port_1':port_src, 'port_2':port_dst,
                                'proto':'UDP'}
                if stream_info not in stream_numbers:
                    stream_numbers.append(stream_info)
                    stream_number = len(stream_numbers) - 1
                else:
                    stream_number = stream_numbers.index(stream_info)
                
                # バイト単位に分割
                raw_data_bytes = bytes(packet)
                raw_data_bytes = raw_data_bytes[:42]  # 先頭42バイトのみ抜き出し
                raw_data_bytes = list(raw_data_bytes)

                # v5で変更：一部の情報のmask→削除に変更
                raw_data_bytes[26:38] = [] # IP addressと port番号を削除
                raw_data_bytes[0:12] = [] # MAC addressを削除

                # バイト単位からフィールド単位に変換
                raw_data = []
                raw_data.append((raw_data_bytes[0]<<8)+raw_data_bytes[1]) # タイプ
                raw_data.append(raw_data_bytes[2]>>4) # IPバージョン
                raw_data.append(raw_data_bytes[2]&0x0f) # IPヘッダ長
                raw_data.append(raw_data_bytes[3]) # TOS
                raw_data.append((raw_data_bytes[4]<<8)+raw_data_bytes[5]) # データグラム長
                raw_data.append((raw_data_bytes[6]<<8)+raw_data_bytes[7]) # ID
                raw_data.append(raw_data_bytes[8]>>5) # IPフラグ
                raw_data.append(((raw_data_bytes[8]&0b00011111)<<8)+raw_data_bytes[9]) # フラグメントオフセット
                raw_data.append(raw_data_bytes[10]) # TTL
                raw_data.append(raw_data_bytes[11]) # プロトコル番号
                raw_data.append((raw_data_bytes[12]<<8)+raw_data_bytes[13]) # IPヘッダチェックサム

                raw_data.append(0x10000) # シーケンス番号1
                raw_data.append(0x10000) # シーケンス番号2
                raw_data.append(0x10000) # ACK番号1
                raw_data.append(0x10000) # ACK番号2
                raw_data.append(0x10000) # ヘッダ長
                raw_data.append(0x10000) # TCPフラグ
                raw_data.append(0x10000) # ウインドウ
                raw_data.append(0x10000) # TCPチェックサム
                raw_data.append(0x10000) # 緊急ポインタ

                raw_data.append((raw_data_bytes[14]<<8)+raw_data_bytes[15]) # 長さ
                raw_data.append((raw_data_bytes[16]<<8)+raw_data_bytes[17]) # UDPチェックサム

                if with_payload == True:
                    # ペイロードの処理（バイト単位）
                    payload = bytes(p)
                    payload = payload[0:header_payload_length - n_header_fields]  # cut payload in  header_payload_length - len(header)
                    payload = list(payload)

                    # v5fで変更：空白を示す値を0x100から0x10000に変更
                    raw_data.extend(payload)  # headerとpayloadを連結
                    raw_data_zero = [0x10002]*header_payload_length  # 0x10000の任意の長さのbyte列を生成
                    raw_data_zero[0:len(raw_data)] = raw_data
                    raw_data = raw_data_zero  # 0x10000で埋めたバイト列を上書き

                if with_payload == True and len(raw_data) != header_payload_length:
                    print(len(raw_data))
                raw_data = [(x+1) % 0x10003 for x in raw_data]
                raw_data_ = labels[j]
                raw_data_.extend([stream_number])
                raw_data_.extend(raw_data)
                writer.writerow(raw_data_)
                count += 1

            j += 1

    # ログの出力
    packet_process_time_end = time.time()
    packet_process_time = packet_process_time_end - packet_process_time_start
    print(f"                  pcap : {pcapfile}")
    print(f"                packet : {outputfile}")
    print(f"         # all packets : {j}")
    print(f"  # no payload packets : {count_nopayload}")
    print(f" # IPv6 packets w/ pl. : {count_IPv6}")
    print(f"# packets with payload : {count}")
    print(f"   packet process time : {packet_process_time}")
    print(f"    label process time : {label_process_time}")
    print()
    if log_file != None:
        with open(log_file, mode='a') as logf:
            logf.write(f"                  pcap : {pcapfile}\n")
            logf.write(f"                packet : {outputfile}\n")
            logf.write(f"         # all packets : {j}\n")
            logf.write(f"  # no payload packets : {count_nopayload}\n")
            logf.write(f" # IPv6 packets w/ pl. : {count_IPv6}\n")
            logf.write(f"# packets with payload : {count}\n")
            logf.write(f"   packet process time : {packet_process_time}\n")
            logf.write(f"    label process time : {label_process_time}\n")
            logf.write("\n")


def main():
    # ここを必要に応じて変更
    PCAP_PATH = "/mnt/fuji/kawanaka/data"
    pcapfiles = ["CSE-CIC-IDS2018_0222.pcap", "CSE-CIC-IDS2018_0223.pcap"] # 
    labelfiles = ["true_label_multiclass_1.csv", "true_label_multiclass_2.csv"] # 読み込むラベルファイル
    header_payload_length = 200 # 使用するheader+payloadの長さを指定（Noneを指定した場合，最大長に合わせる）
    OUTPUT_PATH = "/mnt/fuji/kawanaka/data" # 出力ファイルを保存するフォルダ
    # outputfiles = [f"CSE-CIC-IDS2018_0222_field_{header_payload_length}.csv", f"CSE-CIC-IDS2018_0223_field_{header_payload_length}.csv"] # 出力ファイル名
    # LOG_PATH = "/mnt/fuji/kawanaka/results/pcap2csv_log/field_{header_payload_length}.txt" # ログの保存先
    outputfiles = [f"CSE-CIC-IDS2018_0222_field_{header_payload_length}f_mask.csv", f"CSE-CIC-IDS2018_0223_field_{header_payload_length}f_mask.csv"] # 出力ファイル名
    LOG_PATH = "/mnt/fuji/kawanaka/results/pcap2csv_log/field_{header_payload_length}f_mask.txt" # ログの保存先
    with_payload = True
    header_field_names = ["L3Type", "IPVersion", "IPHeaderLength", "TOS", "DatagramLength", "IPHeaderID", "IPFlag", "FlagmentOffset", "TTL", "ProtocolNumber", "IPHeaderChecksum",
                          "SequenceNumber1", "SequenceNumber2", "ACKNumber1", "ACKNumber2", "TCPHeaderLength", "TCPFlag", "Window", "TCPChecksum", "UrgentPointer",
                          "UDPHeaderLength", "UDPChecksum"] # 特徴量名リスト（field, header）

    # ペイロードの長さを調べる
    if header_payload_length == None:
        payload_len = [0, 0]
        for i in range(len(pcapfiles)):
            payload_len[i] = payload_length(os.path.join(PCAP_PATH, pcapfiles[i]))
        header_payload_length = max(payload_len) + len(header_field_names) # ヘッダの長さを足す

    # ログファイル作成
    with open(LOG_PATH, mode='w') as logf:
        logf.write("")
    
    # パケットの処理・csvファイルへの出力
    time_start = time.time()
    for i in range(len(pcapfiles)):
        pcap2csv(os.path.join(PCAP_PATH, pcapfiles[i]), os.path.join(PCAP_PATH, labelfiles[i]), os.path.join(OUTPUT_PATH, outputfiles[i]), header_field_names, with_payload=with_payload, header_payload_length=header_payload_length, log_file=LOG_PATH)
    time_end = time.time()
    process_time = time_end - time_start

    # ログの出力
    print(f"    whole process time : {process_time}")
    with open(LOG_PATH, mode='a') as logf:
        logf.write(f"    whole process time : {process_time}\n")


if __name__ == "__main__":
    main()    # make payload file
