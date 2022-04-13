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
TEST = 100  # テスト回数
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

# 初期化
tdata =  np.zeros((TEST, LENGTH), dtype=int)
rdata =  np.zeros((TEST, LENGTH), dtype=int)
tcode =  np.zeros((TEST, OUT_LEN), dtype=int)
# TODO レイリーチャネルの生成
h_channel = np.random.rayleigh(scale = 1,size = (TEST,OUT_LEN))
h_nocode  = np.random.rayleigh(scale = 1,size = (TEST,LENGTH))
state = 0
snr_list = []
ber_list = []
nocode_ber_list = []
nocode_rayleigh_ber_list = []
p_b_list = []

path = np.zeros((STATE_NUM, LENGTH, 2), dtype=int)
transmit = receive = np.zeros((TEST, OUT_LEN))
nocode_transmit = nocode_receive = nocode_demo = nocode_rayleigh_recieve = nocode_rayleith_demo= np.zeros((TEST, LENGTH))
array = [["SNR", "BER", "NOCODE_BER", "p_k"]]

if __name__ == "__main__":
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
        
        # BER計算
        nocode_ok = np.count_nonzero(nocode_demo == tdata)
        nocode_rayleigh_ok =np.count_nonzero(nocode_rayleith_demo == tdata)
        nocode_error = rdata.size - nocode_ok
        nocode_rayleith_error = rdata.size - nocode_rayleigh_ok
        NOCODE_BER = nocode_error / (nocode_ok + nocode_error)
        NOCODE_RAYLEIGH_BER = nocode_rayleith_error / (nocode_rayleigh_ok + nocode_rayleith_error)

        #TODO 軟判定理論上界
        snr_list.append(SNRdB)
        nocode_ber_list.append(NOCODE_BER)
        nocode_rayleigh_ber_list.append(NOCODE_RAYLEIGH_BER)

        # 結果表示
        print(
            "SNR: {0:.2f}, NOCODE_BER:{1:.4e},NOCODE_RAYLEIGH_BER:{2:.4e}".format(
                SNRdB, NOCODE_BER, NOCODE_RAYLEIGH_BER
            )
        )
fig = plt.figure()
plt.plot(snr_list, nocode_ber_list, label="without coding ", color="red")
plt.plot(snr_list, nocode_rayleigh_ber_list, label="rayleigh without coding", color="black")
ax = plt.gca()
ax.set_yscale("log")
ax.legend(loc=0)
fig.tight_layout()
plt.xlabel("E_b/N_0")
plt.ylabel("BER")
fig.savefig("img.png")
plt.show()