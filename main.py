import numpy as np
import csv
import operator
import matplotlib.pyplot as plt

S_REG = 3 #レジスタ数
LENGTH = 259 #符号長
TEST = 100000 #テスト回数
OUT_BITS = 2
OUT_LEN = LENGTH*OUT_BITS
K = S_REG + 1 #拘束長
STATE_NUM = 8

def awgn(SNRdB, size):
    No = OUT_BITS * 1* 10**(-SNRdB/10)
    noise = np.random.normal(0, np.sqrt(No / 2), size) + 1j * np.random.normal(0, np.sqrt(No / 2), size)
    return noise


def hamming(s1,s2):   
    # ハミング距離計算
    return sum(map(operator.xor,s1,s2))


def convolutional_encoder(data,state):
    # 状態と入力から，次の状態を返す
    # 状態遷移図の規則性を活用
    state = (2 * state + data) % 8
    return state;

# 初期化
tdata = rdata  = np.zeros((TEST, LENGTH), dtype=int)
tcode = rcode = np.zeros((TEST, OUT_LEN), dtype=int)
state = 0
snr_list = []
ber_list = []


#状態と入力が決まると，出力が決まる3次元配列
#output[状態][入力][出力]
output= np.zeros((STATE_NUM,2,OUT_BITS),dtype=int)
output[0,0] = [0,0]
output[0,1] = [1,1]
output[1,0] = [1,1]
output[1,1] = [0,0]
output[2,0] = [0,1]
output[2,1] = [1,0]
output[3,0] = [1,0]
output[3,1] = [0,1]
output[4,0] = [1,1]
output[4,1] = [0,0]
output[5,0] = [0,0]
output[5,1] = [1,1]
output[6,0] = [1,0]
output[6,1] = [0,1]
output[7,0] = [0,1]
output[7,1] = [1,0]


#各時間，各状態において，ハミング距離を記録する
#h[状態][時刻]
h = np.zeros((STATE_NUM,LENGTH + 1) ,dtype=int)

##各時間(260)，各状態(8)へのパス(2;どの状態からどの入力)を記録する
#path[状態][時刻][[前状態,入力]]
path = np.zeros((STATE_NUM,LENGTH,2),dtype=int)

transmit = receive = np.zeros((TEST, OUT_LEN))
array = [['SNR', 'BER']]
file_path = './test.csv'  # CSVの書き込みpath．任意で変えて．


# tdata: 符号化前の送信データ transmission
# tcode: 符号化後の送信データ
# rdata: 復号化前の受信データ recieve
# rcode: 復号化後の受信データ
# transmit: 送信信号
# receive: 受信信号




if __name__ == '__main__':
    # 表示
    print('# SNR BER:')

    # 伝送シミュレーション
    for SNRdB in np.arange(0, 6.25, 0.25):
        # 送信データの生成
        tdata = np.random.randint(0, 2, (TEST, LENGTH - S_REG))#送信データをランダムのバイナリで生成
        rdata = np.zeros((TEST, LENGTH), dtype=int)

        # 終端ビット系列の付加
        end = np.zeros((TEST, S_REG), dtype=int)
        tdata= np.append(tdata,end,axis=1)
        

        # 畳み込み符号化
        for i in range(TEST):
            for j in range(LENGTH):
                if j == 0:
                    state =0
                else:
                    tcode[i][2*j], tcode[i][2*j+1] = output[state][tdata[i][j]]
                    state = convolutional_encoder(tdata[i][j],state)
                    

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
                r_pair = [0]*OUT_BITS
                    
                r_pair  = np.append(rcode[i][2*j],rcode[i][2*j+1]) 
                
    
                if(j == 0):
                    h[0][1] = h[0][0] + hamming(output[0][0],r_pair)
                    h[1][1] = h[0][0] + hamming(output[0][1],r_pair)
                    path[0][0] = [0,0]
                    path[1][0] = [0,1]
                    
                if(j == 1): 
                    h[0][2] = h[0][1] + hamming(output[0][0],r_pair)
                    h[1][2] = h[0][1] + hamming(output[0][1],r_pair)
                    h[2][2] = h[1][1] + hamming(output[1][0],r_pair)
                    h[3][2] = h[1][1] * hamming(output[1][1],r_pair)
                    path[0][j] = [0,0]
                    path[1][j] = [0,1]
                    path[2][j] = [1,0]
                    path[3][j] = [1,1]

                    
                
                else:
                                
                    #8状態においてハミング距離更新かつパスの記録
                    
                    #template

                    # if (h[状態a][j-1]+hamming(output[状態a][入力a],r_pair)) <(h[状態b][j-1] + hamming(output[状態b][入力b],r_pair)):
                    #     h[2][j] = 前者
                    #     path[2][j]  = [状態a,入力a]
                        
                    # else:
                    #     h[2][j] =後者
                    #     path[2][j]  =[状態b,入力b]
                    
                    #状態0
                    #左辺の方がパスメトリック小さい場合
                    if (h[0][j-1] + hamming(output[0][0],r_pair)) <  (h[4][j-1] + hamming(output[4][0],r_pair)):
                        h[0][j] = h[0][j-1] + hamming(output[0][0],r_pair)#ハミング距離更新．(状態0時刻jのハミング距離を求める)
                        path[0][j] = [0,0] # パスの記録(状態0からの入力0)
                        
                    #右辺の方がパスメトリック小さい場合   
                    else:
                        h[0][j] = (h[4][j-1] + hamming(output[4][0],r_pair))#ハミング距離更新
                        path[0][j] = [4,0]#パスの記録，(状態4からの入力0)
                    
                    #状態1
                    if (h[0][j-1] + hamming(output[0][1],r_pair)) <  (h[4][j-1] + hamming(output[4][1],r_pair)):#状態1時刻jのハミング距離を求める
                        h[1][j] = h[0][j-1] + hamming(output[0][1],r_pair)
                        path[1][j] = [0,1] #状態1に来るパスは，状態0からの入力1
                        
                    else:
                        h[1][j] =(h[4][j-1] + hamming(output[4][1],r_pair))#状態1時刻jのハミング距離を求める
                        path[1][j] = [4,1]#状態1に来るパスは，状態4からの入力1
                    
                    ##状態2
                    
                    if (h[1][j-1]+hamming(output[1][0],r_pair)) <(h[5][j-1] + hamming(output[5][0],r_pair)):
                        h[2][j] = h[1][j-1]+hamming(output[1][0],r_pair)
                        path[2][j] = [1,0]
                        
                    else:
                        h[2][j] = h[5][j-1] + hamming(output[5][0],r_pair)
                        path[2][j] = [5,0]
                    
                    
                    ##状態3
                    
                    if (h[1][j-1]+hamming(output[1][1],r_pair)) <(h[5][j-1] + hamming(output[5][1],r_pair)):
                        h[3][j] = h[1][j-1]+hamming(output[1][1],r_pair)
                        path[3][j] = [1,1]
                        
                    else:
                        h[3][j] = h[5][j-1] + hamming(output[5][1],r_pair)
                        path[3][j] = [5,1]
                    
                    ##状態4
                    
                    if (h[2][j-1]+hamming(output[2][0],r_pair)) <(h[6][j-1] + hamming(output[6][0],r_pair)):
                        h[4][j] = h[2][j-1]+hamming(output[2][0],r_pair)
                        path[4][j] = [2,0]
                        
                    else:
                        h[4][j] = h[6][j-1] + hamming(output[6][0],r_pair)
                        path[4][j] = [6,0]
                    
                     ##状態5
                    
                    if (h[2][j-1]+hamming(output[2][1],r_pair)) <(h[6][j-1] + hamming(output[6][1],r_pair)):
                        h[5][j] = h[2][j-1]+hamming(output[2][1],r_pair)
                        path[5][j]  = [2,1]
                        
                    else:
                        h[5][j] =h[6][j-1] + hamming(output[6][1],r_pair)
                        path[5][j]  =[6,1]
                    
                    
                    ##状態6
                    
                    if (h[3][j-1]+hamming(output[3][0],r_pair)) <(h[7][j-1] + hamming(output[7][0],r_pair)):
                        h[6][j] = h[3][j-1]+hamming(output[3][0],r_pair)
                        path[6][j]  = [3,0]
                        
                    else:
                        h[6][j] =h[7][j-1] + hamming(output[7][0],r_pair)
                        path[6][j]  =[7,0]
                        
                    ##状態7
                    
                    if (h[3][j-1]+hamming(output[3][1],r_pair)) <(h[7][j-1] + hamming(output[7][1],r_pair)):
                        h[7][j] = h[3][j-1]+hamming(output[3][1],r_pair)
                        path[7][j]  = [3,1]
                        
                    else:
                        h[7][j] =h[7][j-1] + hamming(output[7][1],r_pair)
                        path[7][j]  =[7,1]
            
            #復号系列を求める    
            for t in reversed(range(LENGTH)):
                if(t == LENGTH - 1):
                    rdata[i][t] = path[0][t][1]
                    prev = path[0][t][0]
                else:
                    #復元データはpath
                    rdata[i][t] = path[prev][t][1]# 状態prevの時刻tに向かってくるパスの入力
                    prev = path[prev][t][0]# 状態prevの時刻tに向かってくるパスの状態
                                       
                    
                    
        # 誤り回数計算
        ok = np.count_nonzero(rdata == tdata)
        error = rdata.size - ok

        # BER計算
        BER = error / (ok + error)
        
        snr_list.append(SNRdB)
        ber_list.append(BER)

        # 結果表示
        print('SNR: {0:.2f}, BER: {1:.4e}'.format(SNRdB, BER))
        #print(rdata)
        # CSV書き込み．コメントアウト解除すれば書き込める
        array.append([SNRdB, BER])
        #array.append([tdata,rdata])
        with open(file_path, 'w') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(array)
fig = plt.figure()
plt.plot(snr_list, ber_list)

ax = plt.gca()
ax.set_yscale('log') 
plt.xlabel('E_b/N_0')
plt.ylabel('BER')
fig.savefig("img.png")
plt.show()