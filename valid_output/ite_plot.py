#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ite100_noloss.txt をグラフ表示
python ite_plot.py ite100_noloss.txt で実行
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

loss_list = []
time_list = []

input_file = open(sys.argv[1], 'rb')
input_file.readline() # 1行読み捨て
for line in input_file:
    line = line.strip()
    f_dim, loss, time = line.split(' : ')
    time = float(time)
    loss = float(loss.split(',')[0])
    time_list.append(time)
    loss_list.append(loss)
    
input_file.close()
plt.plot(np.array(time_list), np.array(loss_list), 'o')
plt.show()
