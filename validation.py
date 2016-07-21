#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
recommend.pyの NMF ハイパーパラメータ「f_dim」の調整用
python validation.py で実行
"""

import numpy as np
import time
from recommend import *

## main 関数
if __name__ == '__main__':
    print('Loading now...')
    # データ読み込み
    user_num, item_num = load_u_info() # ユーザ数, 映画数, ユーザx映画
    UxI_data = np.zeros((user_num, item_num), dtype=np.float) #ユーザx映画データ
    UxI_data = load_u_data(UxI_data, user_num, item_num) # UxIデータ追加


    # f_dim = 50~1000 を 50 ずつずらしてテスト(20回)

    # iteration 100 回 (最終結果のみ出力)
    for i in range(50,1050,50):
        start = time.time()
        umx, imx, loss = trainNMF(UxI_data, f_dim=i, max_ite=100)
        end = time.time()
        print '%4d\'s MSE : %f, time : %f' % (i, loss, end-start)

    # iteration 1000 回 (途中経過も出力)
    for i in range(50,1050,50):
        start = time.time()
        umx, imx, loss = trainNMF(UxI_data, f_dim=i, max_ite=1000,testloss=True)
        end = time.time()
        print '%4d\'s MSE : %f, time : %f' % (i, loss, end-start)
