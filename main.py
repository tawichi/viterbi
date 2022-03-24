import numpy as np
import csv
import random,operator,math

S_REG = 3 # レジスタ数
LENGTH = 259 # 符号長
TEST = 100000 # テスト回数
OUT = 2
OUT_LEN = LENGTH*OUT
K = S_REG + 1#拘束長

# 初期化
tdata = rdata = np.zeros((TEST, LENGTH), dtype=np.int)
tcode = rcode = np.zeros((TEST, OUT_LEN), dtype=np.int)
state = 0
code = [0,0]
dist0, dist1, dist2, dist3, dist4, dist5, dist6, dist7 = 0

transmit = receive = np.zeros((TEST, OUT_LEN))#LT3ではなく1/Rではないでしょうか？
array = [['SNR', 'BER']]
path = './test.csv'  # CSVの書き込みpath．任意で変えて．


# tdata: 符号化前の送信データ transmission
# tcode: 符号化後の送信データ
# rdata: 復号化前の受信データ recieve
# rcode: 復号化後の受信データ
# transmit: 送信信号
# receive: 受信信号

def awgn(SNRdB, size):
    #awgnを実装する
    No = OUT * 1* 10**(-SNRdB/10)
    noise = np.random.normal(0, np.sqrt(No / 2), size) + 1j * np.random.normal(0, np.sqrt(No / 2), size)
    return noise
# compute hamming distance of two bit sequences
def hamming(s1,s2):
    return sum(map(operator.xor,s1,s2))

# 整数の全ビットをまとめてxorする
def xorbits(n):
    result = 0
    while n > 0:
        result ^= (n & 1)
        n >>= 1
    return result

def expected_parity(from_state,to_state,k,glist):
    # x[n] comes from to_state
    # x[n-1] ... x[n-k-1] comes from from_state
    x = ((to_state >> (S_REG-1)) << (S_REG)) + from_state
    return [xorbits(g & x) for g in glist]

def convolutional_encoder(data,state):
  
    if state == 0 and data == 0:
        code=[0,0]
        state = 0
          
    if state == 0 and data == 1:
        code = [1,1]
        state = 1
        
    if state == 1 and data == 0:
        code = [1,1]
        state = 2
        
    if state == 1 and data == 1:
        code = [0,0]
        state = 3
    
    if state == 2 and data == 0:
        code = [0,1]
        state = 4
        
    if state == 2 and data == 1:
        code = [1,0]
        state = 5
    
    if state == 3 and data == 0:
        code = [1,0]
        state = 6
        
    if state == 3 and data == 1:
        code = [0,1]
        state = 7
        
    if state == 4 and data == 0:
        code = [1,1]
        state = 0
        
    if state == 4 and data == 1:
        code = [0,0]
        state = 1
    
    if state == 5 and data == 0:
        code = [0,0]
        state = 2
        
    if state == 5 and data == 1:
        code = [1,1]
        state = 3
    
    if state == 6 and data == 0:
        code = [1,0]
        state = 4
        
    if state == 6 and data == 1:
        code = [0,1]
        state = 5
    
    if state == 7 and data == 0:
        code = [0,1]
        state = 6
        
    if state == 7 and data == 1:
        code = [1,0]
        state = 7
    
    return code,state;

# def get_prev_state(data,state):
    
#     if state == 0 and data == 0:
#         return 0
          
#     if state == 0 and data == 1:
#         return 1
        
#     if state == 1 and data == 0:
#         return 2
        
#     if state == 1 and data == 1:
#         return 3
    
#     if state == 2 and data == 0:
#         return 4
        
#     if state == 2 and data == 1:
#         return 5
    
#     if state == 3 and data == 0:
#         return 6
        
#     if state == 3 and data == 1:
#         return 7
        
#     if state == 4 and data == 0:
#         return 0
        
#     if state == 4 and data == 1:
#         return 1
    
#     if state == 5 and data == 0:
#         return 2
        
#     if state == 5 and data == 1:
#         return 3
    
#     if state == 6 and data == 0:
#         return 4
        
#     if state == 6 and data == 1:
#         return 5
    
#     if state == 7 and data == 0:
#         return 6
        
#     if state == 7 and data == 1:
#         return 7
    
    
    
    
    


if __name__ == '__main__':
    # 表示
    print('# SNR BER:')

    # 伝送シミュレーション
    for SNRdB in np.arange(0, 6.25, 0.25):
        # 送信データの生成
        tdata = np.random.randint(0, 2, (TEST, LENGTH - S_REG))#送信データをランダムのバイナリで生成

        # 終端ビット系列の付加
        end = np.zeros((TEST, S_REG), dtype=np.int)
        tdata= np.append(tdata,end,axis=1)
        

        # 畳み込み符号化
        for i in range(TEST):
            for j in range(LENGTH):
                if j == 0:
                    state =0
                else:
                    code,state = convolutional_encoder(tdata[i][j],state)
                
                tcode[i][2*j] = code[0]
                tcode[i][2*j+1] = code[1]

        # BPSK変調
        transmit[tcode == 0] = -1
        transmit[tcode == 1] = 1

        # 伝送
        receive = transmit + awgn(SNRdB, (TEST, OUT_LEN))

        # BPSK復調
        rcode[receive < 0] = 0
        rcode[receive >= 0] = 1

        # ビタビ復号
        for i in range(TEST):
            for j in range(LENGTH):
                
    
                code,state = convolutional_encoder(tdata[i][j],state)
                

        # 誤り回数計算
        ok = np.count_nonzero(rdata == tdata)
        error = rdata.size - ok

        # BER計算
        BER = error / (ok + error)

        # 結果表示
        print('SNR: {0:.2f}, BER: {1:.4e}'.format(SNRdB, BER))

        # CSV書き込み．コメントアウト解除すれば書き込める
        array.append([SNRdB, BER])
        with open(path, 'w') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(array)