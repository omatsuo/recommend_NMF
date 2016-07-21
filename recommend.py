#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
実行プログラム本体
python recommend.py で実行
"""

import numpy as np
import sys

## 関数 : ユーザ数、アイテム数データの読み込み(u.info)
def load_u_info():
    info_file = open('ml-100k/u.info', 'rb')
    info_lines = info_file.readlines()   # infoはspace区切り
    user_num = int(info_lines[0].split(' ')[0])  # user 数
    item_num = int(info_lines[1].split(' ')[0])  # item(映画) 数
    info_file.close()
    return user_num, item_num


## 関数 : ユーザxアイテム評価データの読み込み(u.data系)
def load_u_data(UxI_data, user_num, item_num, batch='u.data'):
    data_file = open('ml-100k/'+batch, 'rb')
    for line in data_file: # 各行のデータを UxI_data の各要素に代入
        line = line.strip()
        user_id, item_id, rate, tt = line.split('\t') # dataはtab区切り
        # IDは1から始まるので注意
        UxI_data[int(user_id)-1][int(item_id)-1] = float(rate)
    data_file.close()
    return UxI_data


## 関数 : アイテム(映画)データの読み込み(u.item,u.genre)
def load_u_item():
    # ジャンルデータ読み込み,19種
    # unknown | Action | Adventure | Animation | Children's | Comedy |
    # Crime | Documentary | Drama | Fantasy | Film-Noir | Horror |
    # Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western |
    genre_data = []
    genre_file = open('ml-100k/u.genre', 'rb')
    for line in genre_file: # 各行のgenreを genre_data に追加していく
        line = line.strip()
        genre_data.append(line.split('|')[0]) # genreは|区切り
    genre_file.close()

    # 各映画データ読み込み
    item_data = []
    item_file = open('ml-100k/u.item', 'rb')

    for line in item_file: # 各行のデータitemを item_data に追加していく
        line = line.strip()
        item_tmp = line.split('|') # itemは|区切り

        item = item_tmp[:5] # ジャンル以外
        item[0] = int(item_tmp[0]) # IDをint化,1から始まるので注意
        item_genre = [True if label=='1' else False for label in item_tmp[5:]] # 映画ジャンル情報

        # movie id, title, release date, video release date, IMDb URL, genres
        item.append(item_genre)
        item_data.append(item)
    item_file.close()

    return item_data, genre_data[:19]


## 関数 : ジャンル情報を文字列に変換
def genre2string(item, genre_data):
    string = 'ganre -> '
    for i,TorF in enumerate(item[5]):
        if TorF:
            string = string + genre_data[i] + ', '
    return string[:-2]


## 関数 : 平均二乗誤差
def mean_squared_error(a, b):
    diffs = (a - b).flatten()  # 差分を1次元配列に
    return diffs.dot(diffs) / len(diffs)  # 二乗誤差(=差分の内積)の平均


## 関数 : 非負値行列因子分解
# X = TV となるような T,V の獲得
def trainNMF(x, f_dim=10, max_ite=100, testloss=False):
    # X : ユーザxアイテム (t_dim*v_dim),
    # T : ユーザx特徴量 (t_dim*f_dim)
    # V : 特徴量xアイテム (f_dim*v_dim)
    x = np.asarray(x)

    # 次元数定義
    t_dim = x.shape[0]
    v_dim = x.shape[1]

    # ユーザ行列 T, アイテム行列 V をランダム値で初期化
    t = np.random.rand(t_dim, f_dim)
    v = np.random.rand(f_dim, v_dim)

    # 学習ループ : loss閾値以下 or max_ite回
    for i in range(max_ite):
        tv = t.dot(v)

        # 平均二乗誤差loss
        loss = mean_squared_error(x, tv)
        if testloss:   # loss表示onなら
            if i % 10 == 0: print loss # 10回に1回表示

        # lossが閾値以下なら終了
        if loss < 0.01:
            break

        # ユーザ行列 T を更新
	t = t * x.dot(v.T) / t.dot(v).dot(v.T)

        # アイテム行列 V を更新
	v = v * t.T.dot(x) / t.T.dot(t.dot(v))

    return t, v, loss


## main 関数
if __name__ == '__main__':
    print(' Loading now...')
    # データ読み込み
    user_num, item_num = load_u_info() # ユーザ数, 映画数, ユーザx映画
    UxI_data = np.zeros((user_num, item_num), dtype=np.float) #ユーザx映画データ
    UxI_data = load_u_data(UxI_data, user_num, item_num) # UxIデータ追加
    item_data, genre_data = load_u_item()  # 映画データ、ジャンルデータ

    # レコメンドするユーザのIDを取得
    rec_id = int( raw_input('Please input user ID (1~%d) > '%(user_num)) )
    if (rec_id < 1) or (rec_id > user_num): # 1~user_num の整数でなければ終了
        print 'the ID is invalid.'
        sys.exit()
    print('Wait a moment.')
    print(' Processing now...')

    # NMF で行列分解
    # 計算時間と精度を考え、ひとまず f_dim = , iteration = 100
    user_mat, item_mat, loss = trainNMF(UxI_data, f_dim=100, max_ite=100)

    # 類似度の算出
    user_sim = user_mat.dot(user_mat.T) # ユーザx特徴量の内積(対称行列)

    # ユーザ類似度による重み付きスコアを算出
    UxI_score = user_sim.dot(UxI_data)  # 単純に類似度とデータの内積

    # まだ見ていない映画で,スコアが高い順に5位まで推薦
    ranking = sorted(enumerate(UxI_score[rec_id].tolist()), 
                     key=lambda x:x[1], reverse=True)
    count = 0  # ループ内順位カウント
    for item_id, score in ranking: # item_id:映画ID-1, score:スコア
        if UxI_data[rec_id][item_id] == 0:  # まだ見ていなければ推薦
            count += 1
            print # 改行
            print '%d: %s' % (count,item_data[item_id][1]) # 映画名
            print '   URL -> %s' % (item_data[item_id][4]) # URL
            #print '   score -> %f' % (score) # スコア
            print '   %s'%(genre2string(item_data[item_id],genre_data)) # ジャンル
        if count >= 5: break  # 5つ表示したら終了
    print  # 改行

