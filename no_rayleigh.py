#軟判定
from tkinter import LEFT
from turtle import color
import numpy as np
import csv
import operator
import matplotlib.pyplot as plt
import math
import random
from numpy.random import *
from scipy import special

S_REG = 3  # レジスタ数(前後半共通)
LENGTH = 259  # 符号長
TEST = 10  # テスト回数
OUT_BITS = 2  # 後半組は3
OUT_LEN = LENGTH * OUT_BITS #777
K = S_REG + 1  # 拘束長は4(前後半共通)
STATE_NUM = 8  # 後半組は16


def awgn(SNRdB, size):
    No = OUT_BITS * 1 * 10 ** (-SNRdB / 10)
    noise = np.random.normal(0, np.sqrt(No / 2), size) + 1j * np.random.normal(
        0, np.sqrt(No / 2), size
    )
    return noise


def hamming(s1, s2):
    # ハミング距離計算
    return sum(map(operator.xor, s1, s2))
  
def distance(s1,s2,i,j):
    np.place(s1, s1== 0, -1)
    return  np.dot(s1,s2,h_channel[i][j])


def convolutional_encoder(data, state):
    # 状態と入力から，次の状態を返す
    # 状態遷移図の規則h性を活用
    state = (2 * state + data) % 8
    return state


def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))


# 初期化
tdata =  np.zeros((TEST, LENGTH), dtype=int)
rdata =  np.zeros((TEST, LENGTH), dtype=int)
tcode =  np.zeros((TEST, OUT_LEN), dtype=int)
receive =  np.zeros((TEST, OUT_LEN), dtype=float)
# TODO レイリーチャネルの生成
h_channel = np.random.rayleigh(scale = 1,size = (TEST,OUT_LEN))
h_nocode = np.random.rayleigh(scale = 1,size = (TEST,LENGTH))
state = 0
snr_list = []
ber_list = []
nocode_ber_list = []
nocode_rayleigh_ber_list = []
p_b_list = []





    



# 各時間，各状態において，ハミング距離を記録する
# metric[状態][時刻]
metric = -10000 * np.ones((STATE_NUM, LENGTH + 1), dtype=float)
metric[0][0] = 0
##各時間(260)，各状態(8)へのパス(2;どの状態からどの入力)を記録する
# path[状態][時刻][[前状態,入力]]
path = np.zeros((STATE_NUM, LENGTH, 2), dtype=int)
transmit = receive = np.zeros((TEST, OUT_LEN))
nocode_transmit = nocode_receive = nocode_demo = nocode_rayleigh_recieve = nocode_rayleith_demo= np.zeros((TEST, LENGTH))
# 状態と入力が決まると，出力が決まる3次元配列
# output[状態][入力][出力]
output = np.zeros((STATE_NUM, 2, OUT_BITS), dtype=int)
output[0, 0] = [0, 0]
output[0, 1] = [1, 1]
output[1, 0] = [1, 1]
output[1, 1] = [0, 0]
output[2, 0] = [0, 1]
output[2, 1] = [1, 0]
output[3, 0] = [1, 0]
output[3, 1] = [0, 1]
output[4, 0] = [1, 1]
output[4, 1] = [0, 0]
output[5, 0] = [0, 0]
output[5, 1] = [1, 1]
output[6, 0] = [1, 0]
output[6, 1] = [0, 1]
output[7, 0] = [0, 1]
output[7, 1] = [1, 0]

hugou = np.zeros((STATE_NUM, 2, OUT_BITS), dtype=int)
hugou[0, 0] = [-1, -1]
hugou[0, 1] = [1, 1]
hugou[1, 0] = [1, 1]
hugou[1, 1] = [-1, -1]
hugou[2, 0] = [-1, 1]
hugou[2, 1] = [1, -1]
hugou[3, 0] = [1, -1]
hugou[3, 1] = [-1, 1]
hugou[4, 0] = [1, 1]
hugou[4, 1] = [-1, -1]
hugou[5, 0] = [-1, -1]
hugou[5, 1] = [1, 1]
hugou[6, 0] = [1, -1]
hugou[6, 1] = [-1, 1]
hugou[7, 0] = [-1, 1]
hugou[7, 1] = [1, -1]




array = [["SNR", "BER", "NOCODE_BER", "p_k"]]
file_path = "./test.csv"  # CSVの書き込みpath．任意で変えて．

# tdata: 符号化前の送信データ transmission
# tcode: 符号化後の送信データ
# rdata: 復号化前の受信データ receive
# rcode: 復号化後の受信データ
# transmit: 送信信号
# receive: 受信信号


if __name__ == "__no_rayleigh__":
    # 表示
    print("# SNR BER:")
    
    

    # 伝送シミュレーション
    for SNRdB in np.arange(0, 6.25, 0.25):
        # 送信データの生成
        tdata = np.random.randint(0, 2, (TEST, LENGTH - S_REG))  # 送信データをランダムのバイナリで生成
        rdata = np.zeros((TEST, LENGTH), dtype=float)

        # 終端ビット系列の付加h
        end = np.zeros((TEST, S_REG), dtype=int)
        tdata = np.append(tdata, end, axis=1)

        # 畳み込み符号化
        for i in range(TEST):
            
            
            
            
            for k in range(OUT_LEN):
                test = (randn(1, 1) + 1j * randn(1, 1)) * 1 / np.sqrt(2)
                
                
            for j in range(LENGTH):
                if j == 0:
                    state = 0
                else:
                    tcode[i][2 * j], tcode[i][2 * j + 1] = output[state][tdata[i][j]]
                    state = convolutional_encoder(tdata[i][j], state)

        # BPSK変調
        transmit[tcode == 0] = -1
        transmit[tcode == 1] = 1
        nocode_transmit[tdata == 0] = -1
        nocode_transmit[tdata == 1] = 1

        # 伝送
        receive = transmit + awgn(SNRdB, (TEST, OUT_LEN))
        nocode_receive = nocode_transmit + awgn(SNRdB, (TEST, LENGTH))
        nocode_rayleith = h_nocode * nocode_transmit+ awgn(SNRdB,(TEST,LENGTH))

        # BPSK復調
        #rcode[receive < 0] = 0

        nocode_demo[nocode_receive < 0] = 0
        nocode_demo[nocode_receive >= 0] = 1
        nocode_rayleith_demo[nocode_rayleith < 0] = 0
        nocode_rayleith_demo[nocode_rayleith >= 0] = 1
        print(np.count_nonzero(nocode_demo != tdata))

        # ビタビ復号
        for i in range(TEST):
            for j in range(LENGTH):
                # TODO メトリックの書き換え
                r_pair = [0] * OUT_BITS

                r_pair = np.append(receive[i][2 * j], receive[i][2 * j + 1])
                #h_pair = np.append(h_channel[i][2*j], h_channel[i][2 * j + 1])

                if j == 0:
                    metric[0][0] = 0
                    metric[0][1] = metric[0][0] + np.sum(hugou[0][0]*r_pair)
                    metric[1][1] = metric[0][0] + np.sum(hugou[0][1]*r_pair)

                    # metric[0][2] = metric[0][1] + distance(hugou[0][0], r_pair)
                    # metric[1][2] = metric[0][1] + distance(hugou[0][1], r_pair)
                    # metric[2][2] = metric[1][1] + distance(hugou[1][0], r_pair)
                    # metric[3][2] = metric[1][1] + distance(hugou[1][1], r_pair)

                    # metric[0][3] = metric[0][2] + distance(hugou[0][0], r_pair)
                    # metric[1][3] = metric[0][2] + distance(hugou[0][1], r_pair)
                    # metric[2][3] = metric[1][2] + distance(hugou[1][0], r_pair)
                    # metric[3][3] = metric[1][2] + distance(hugou[1][1], r_pair)
                    # metric[4][3] = metric[2][2] + distance(hugou[2][0], r_pair)
                    # metric[5][3] = metric[2][2] + distance(hugou[2][1], r_pair)
                    # metric[6][3] = metric[3][3] + distance(hugou[3][0], r_pair)
                    # metric[7][3] = metric[3][3] + distance(hugou[3][1], r_pair)

                else:

                    # 8状態においてハミング距離更新かつパスの記録

                    # template

                    # if (metric[状態a][j-1]+distance(output[状態a][入力a],r_pair)) <(metric[状態b][j-1] + distance(output[状態b][入力b],r_pair)):
                    #     metric[2][j] = 前者
                    #     path[2][j]  = [状態a,入力a]

                    # else:
                    #     metric[2][j] =後者
                    #     path[2][j]  =[状態b,入力b]

                    # 状態0
                    # 左辺の方がパスメトリック小さい場合
                    if (metric[0][j - 1] + np.sum(hugou[0][0] * r_pair)) > (
                        metric[4][j - 1] + np.sum(hugou[4][0] * r_pair)
                    ):
                        metric[0][j] = metric[0][j - 1] + np.sum(
                            hugou[0][0] * r_pair
                        )  # ハミング距離更新．(状態0時刻jのハミング距離を求める)
                        path[0][j] = [0, 0]  # パスの記録(状態0からの入力0)

                    # 右辺の方がパスメトリック小さい場合
                    else:
                        metric[0][j] = metric[4][j - 1] + np.sum(
                            hugou[4][0]* r_pair
                        )  # ハミング距離更新
                        path[0][j] = [4, 0]  # パスの記録，(状態4からの入力0)

                    # 状態1
                    if (metric[0][j - 1] + np.sum(hugou[0][1] * r_pair)) > (
                        metric[4][j - 1] + np.sum(hugou[4][1] * r_pair)
                    ):  # 状態1時刻jのハミング距離を求める
                        metric[1][j] = metric[0][j - 1] + np.sum(hugou[0][1] * r_pair)
                        path[1][j] = [0, 1]  # 状態1に来るパスは，状態0からの入力1

                    else:
                        metric[1][j] = metric[4][j - 1] + np.sum(
                            hugou[4][1] * r_pair
                        )  # 状態1時刻jのハミング距離を求める
                        path[1][j] = [4, 1]  # 状態1に来るパスは，状態4からの入力1

                    ##状態2

                    if (metric[1][j - 1] + np.sum(hugou[1][0] * r_pair)) > (
                        metric[5][j - 1] + np.sum(hugou[5][0] * r_pair)
                    ):
                        metric[2][j] = metric[1][j - 1] + np.sum(hugou[1][0] * r_pair)
                        path[2][j] = [1, 0]

                    else:
                        metric[2][j] = metric[5][j - 1] + np.sum(hugou[5][0] * r_pair)
                        path[2][j] = [5, 0]

                    ##状態3

                    if (metric[1][j - 1] + np.sum(hugou[1][1] * r_pair)) > (
                        metric[5][j - 1] + np.sum(hugou[5][1] * r_pair)
                    ):
                        metric[3][j] = metric[1][j - 1] + np.sum(hugou[1][1] * r_pair)
                        path[3][j] = [1, 1]

                    else:
                        metric[3][j] = metric[5][j - 1] + np.sum(hugou[5][1] * r_pair)
                        path[3][j] = [5, 1]

                    ##状態4

                    if (metric[2][j - 1] + np.sum(hugou[2][0] * r_pair)) > (
                        metric[6][j - 1] + np.sum(hugou[6][0] * r_pair)
                    ):
                        metric[4][j] = metric[2][j - 1] + np.sum(hugou[2][0] * r_pair)
                        path[4][j] = [2, 0]

                    else:
                        metric[4][j] = metric[6][j - 1] + np.sum(hugou[6][0] * r_pair)
                        path[4][j] = [6, 0]

                    ##状態5

                    if (metric[2][j - 1] + np.sum(hugou[2][1] * r_pair)) > (
                        metric[6][j - 1] + np.sum(hugou[6][1] * r_pair)
                    ):
                        metric[5][j] = metric[2][j - 1] + np.sum(hugou[2][1] * r_pair)
                        path[5][j] = [2, 1]

                    else:
                        metric[5][j] = metric[6][j - 1] + np.sum(hugou[6][1] * r_pair)
                        path[5][j] = [6, 1]

                    ##状態6

                    if (metric[3][j - 1] + np.sum(hugou[3][0] * r_pair)) > (
                        metric[7][j - 1] + np.sum(hugou[7][0] * r_pair)
                    ):
                        metric[6][j] = metric[3][j - 1] + np.sum(hugou[3][0] * r_pair)
                        path[6][j] = [3, 0]

                    else:
                        metric[6][j] = metric[7][j - 1] + np.sum(hugou[7][0] * r_pair)
                        path[6][j] = [7, 0]

                    ##状態7

                    if (metric[3][j - 1] + np.sum(hugou[3][1] * r_pair)) > (
                        metric[7][j - 1] + np.sum(hugou[7][1] * r_pair)
                    ):
                        metric[7][j] = metric[3][j - 1] + np.sum(hugou[3][1] * r_pair)
                        path[7][j] = [3, 1]

                    else:
                        metric[7][j] = metric[7][j - 1] + np.sum(hugou[7][1] * r_pair)
                        path[7][j] = [7, 1]

                # path[0][0] = [0, 0]
                # path[1][0] = [0, 1]
                # path[0][1] = [0, 0]
                # path[1][1] = [0, 1]
                # path[2][1] = [1, 0]
                # path[3][1] = [1, 1]

                # path[0][2] = [0,0]
                # path[1][2] = [0,1]
                # path[2][2] = [1,0]
                # path[3][2] = [1,1]
                # path[4][2] = [2,0]
                # path[5][2] = [2,1]
                # path[6][2] = [3,0]
                # path[7][2] = [3,1]

            # 復号系列を求める
            for t in reversed(range(LENGTH)):
                if t == LENGTH - 1:
                    rdata[i][t] = path[0][t][1]
                    prev = path[0][t][0]
                elif t == 0:
                    rdata[i][t] = tdata[i][t]
                else:
                    # 復元データはpath
                    rdata[i][t] = path[prev][t][1]  # 状態prevの時刻t+1に向かってくるパスの入力
                    prev = path[prev][t][0]  # 状態prevの時刻tに向かってくるパスの状態

        # 誤り回数計算
        ok = np.count_nonzero(rdata == tdata)
        error = rdata.size - ok

        nocode_ok = np.count_nonzero(nocode_demo == tdata)
        nocode_rayleigh_ok =np.count_nonzero(nocode_rayleith_demo == tdata)
        nocode_error = rdata.size - nocode_ok
        nocode_rayleith_error = rdata.size - nocode_rayleigh_ok

        # BER計算
        BER = error / (ok + error)
        NOCODE_BER = nocode_error / (nocode_ok + nocode_error)
        NOCODE_RAYLEIGH_BER = nocode_rayleith_error / (nocode_rayleigh_ok + nocode_rayleith_error)

        #TODO 軟判定理論上界
        # 硬判定理論上界計算
        p_k = [0] * 14  # 後半組は18にする

        p = 1 / 2 * special.erfc(np.sqrt(1 / 2 * 10 ** (SNRdB / 10)))

        for k in range(6, 14):  # 後半組はrange(10,18)
            p_k[k] = 1 / 2 * special.erfc(np.sqrt(k* (1 / 2) * 10 ** (SNRdB / 10)))
        p_b = 0

        p_b = (
            2 * p_k[6]
            + 7 * p_k[7]
            + 18 * p_k[8]
            + 49 * p_k[9]
            + 130 * p_k[10]
            + 333 * p_k[11]
            + 836 * p_k[12]
            + 2069 * p_k[13]
        )
        # 後半組は以下のように書き換え
        # p_b = 6 * p_k[10] + 0 * p_k[11] + 6 * p_k[12] + 0 * p_k[13] + 58 * p_k[14] + 0 * p_k[15] + 118 * p_k[16] + 0 * p_k[17]
        if p_b >= 1 / 2:
            p_b = 1 / 2
        p_b_list.append(p_b)

        snr_list.append(SNRdB)
        ber_list.append(BER)
        nocode_ber_list.append(NOCODE_BER)
        nocode_rayleigh_ber_list.append(NOCODE_RAYLEIGH_BER)

        # 結果表示
        print(
            "SNR: {0:.2f}, BER: {1:.4e}, NOCODE_BER:{2:.4e},NOCOXE_RAYLEIGH_BER:{3:.4e}, UPPER_BOUND:{3:.4e}".format(
                SNRdB, BER, NOCODE_BER, NOCODE_RAYLEIGH_BER, p_b
            )
        )
        # print('NOCODE_BER:{1:.4e}'.format(*NOCODE_BER))
        # print(rdata)
        # CSV書き込み．コメントアウト解除すれば書き込める
        array.append([SNRdB, BER, NOCODE_BER,NOCODE_RAYLEIGH_BER])
        # array.append([tdata,rdata])
        with open(file_path, "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(array)


fig = plt.figure()
plt.plot(snr_list, ber_list, label="simulation(soft) with rayleigh", color="blue")
plt.plot(snr_list, nocode_ber_list, label="without coding ", color="red")
plt.plot(snr_list, nocode_rayleigh_ber_list, label="rayleigh without coding", color="black")
plt.plot(snr_list, p_b_list, label="upper bound", color="green")


ax = plt.gca()
ax.set_yscale("log")
ax.legend(loc=0)
fig.tight_layout()
plt.xlabel("E_b/N_0")
plt.ylabel("BER")
fig.savefig("img.png")
plt.show()