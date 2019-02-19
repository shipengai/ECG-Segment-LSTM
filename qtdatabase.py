# -*- coding:utf-8 -*-
"""
将QT数据库按照如下过程处理： 2个signal分别当做2个feature
标签对应关系： 0：正常未标注 1：P波 2：PQ段 3：QR段 4：RS段 5：ST段
过滤不参与本次实验的record --> 读取原始信号与标注 --> 将标注转换成标签值(0-5,共6类)
--> 过滤非正常未标注片段 --> 固定长度片段分割 --> 构成训练和验证数据库

有一个问题需要后续优化的是：
去掉未正常标注的片段后如何衔接？
"""
import os
import math
import wfdb
import pickle
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


train_percentage = 0.7  # 训练数据比例
features = 2  # 特征数目，用一条导联就是1，用两条导联就是2
Seqlength = 1300  # num+2*overlap
qtdbpath = './qtdb/'  # 数据路径
ann_suffix = 'q1c'  # 标注文件的后缀
qtdb_pickle_save = './qtdb_pkl/'  # 经过处理后保存数据路径

if not os.path.exists(qtdb_pickle_save):
    os.mkdir(qtdb_pickle_save)

# 下面几个文件没有P波，不参与本次实验
exclude = set()
exclude.update(["sel35", "sel36", "sel37", "sel50",
                "sel102", "sel104", "sel221",
                "sel232", "sel310"])
# 过滤不参与本次实验的record
datafiles = [x[:-4] for x in os.listdir(qtdbpath) if x[-4:] == '.dat']
record_names = list(set(datafiles)-exclude)


def remove_seq_gaps(x, y):
    """
    去掉非正常标注的片段， 如何衔接需要优化
    :param x:
    :param y:
    :return:
    """
    window = 150
    c = 0
    include = []
    print("filterering.")
    print("before shape x,y", x.shape, y.shape)
    for i in range(y.shape[0]):
        # 将连续未标注的超过150个点的整个未标注的去掉
        if 0 < c < window and y[i] != 0:
            for t in reversed(range(c)):
                include.append(i-t-1)
            c = 0
            include.append(i)
        elif c >= window and y[i] != 0:
            include.append(i)
            c = 0
        elif y[i] == 0:
            c += 1
        else:
            include.append(i)
    x, y = x[include, :], y[include]
    print(" after shape x,y", x.shape, y.shape)
    return x, y


def splitseg(signal, label, num, overlap):
    """
    创建LSTM训练和验证使用的片段，长度为num+2*overlap
    :param signal:
    :param label:
    :param num:
    :param overlap:
    :return:
    """
    length = signal.shape[0]
    num_seg = math.ceil(length / num)  # 计算可以得到多少个数据片段, 向上取整可能不是很合适，原因见下面的shape检查
    upper = num_seg * num  # math.ceil(8.5)=9
    print("splitting on", num, "with overlap of ", overlap, "total datapoints:", signal.shape[0], "; upper:", upper)
    xx = np.empty((num_seg, num + 2 * overlap, signal.shape[1]))  # 训练数据
    yy = np.empty((num_seg, num + 2 * overlap, ))  # 标签
    # 第一个片段取前num+overlap个 然后再在前面补overlap个零
    # 最后一个片段取后面num+overlap个 然后再在后面补overlap个零
    for i in range(num_seg):
        if i == 0:
            tmp = np.zeros((num+2*overlap, signal.shape[1]))
            tmp[overlap:, :] = signal[:num+overlap, :]
        elif i == num_seg-1:
            tmp = np.zeros((num+2*overlap, signal.shape[1]))
            tmp[:num+overlap, :] = signal[-(num+overlap):, :]
        else:
            # shape 检查，如果小于(num+2overlap,),则后面补零,这种情况会出现在7089扩充到8000
            tmp = np.zeros((num + 2 * overlap, signal.shape[1]))
            signal_i = signal[i*num-overlap: ((i+1)*num+overlap), :]
            tmp[:signal_i.shape[0]] = signal_i
        xx[i] = tmp

    for i in range(num_seg):
        if i == 0:
            tmp = np.zeros((num+2*overlap, ))
            tmp[overlap:] = label[:num+overlap]
        elif i == num_seg-1:
            tmp = np.zeros((num+2*overlap, ))
            tmp[:num+overlap] = label[-(num+overlap):]
        else:
            # shape 检查，如果小于(num+2overlap,),则后面补零,这种情况会出现在7089扩充到8000
            tmp = np.zeros((num + 2 * overlap, ))
            label_i = label[i*num-overlap: ((i+1)*num+overlap)]
            tmp[:label_i.shape[0]] = label_i
        yy[i] = tmp
    return xx, yy


def plotecg(x, y, start, end):
    x = x[start:end, 0]  # 只取第一条信号
    y = y[start:end]
    cmap = ['k', 'r', 'g', 'b', 'c', 'y']
    start = end = 0
    for i in range(len(y)-1):
        if y[i] != y[i+1]:
            end = i
            plt.plot(np.arange(start, end), x[start:end], cmap[int(y[i])])
            start = i+1
    plt.show()


x = np.zeros((1, Seqlength, features))
y = np.zeros((1, Seqlength,))

for record_name in record_names:

    # 先读标注文件，再根据标注文件的长度来读record
    annotation = wfdb.rdann(qtdbpath+record_name, extension=ann_suffix)
    start = annotation.sample[0]
    end = annotation.sample[-1]
    print('record {} start,end: {}, {}'.format(record_name, start, end))
    record, _ = wfdb.rdsamp(qtdbpath+record_name, sampfrom=start, sampto=end+1)
    # 两个信号都当做特征，所以每一个采样点2个特征
    signal = record

    Ann = list(zip(annotation.sample, annotation.symbol))
    poses = []
    for i in range(len(Ann)):
        ann = Ann[i]
        # 先找到P波,根据p波查找整个波形
        if ann[1] == 'p':
            pstart = pend = qpos = rpos = spos = tpos = tend = 0
            # 确定p波的起始和结束位置
            if Ann[i - 1][1] == '(':
                pstart = Ann[i - 1][0]
            if Ann[i + 1][1] == ')':
                pend = Ann[i + 1][0]
            # p波紧随其后的就是QRS， 确定QRS波的位置
            if Ann[i + 3][1] == 'N':
                rpos = Ann[i + 3][0]
                if Ann[i + 2][1] == '(':
                    qpos = Ann[i + 2][0]
                if Ann[i + 4][1] == ')':
                    spos = Ann[i + 4][0]
                # 确认t波，因为有的没有‘(’,分情况讨论：
                if Ann[i + 6][1] == 't':
                    tpos = Ann[i + 6][0]
                    if Ann[i + 7] == ')':
                        tend = Ann[i + 7][0]
                elif Ann[i + 5][1] == 't':
                    tpos = Ann[i + 5][0]
                    if Ann[i + 6][1] == ')':
                        tend = Ann[i + 6][0]
                else:
                    print("can't find t wave")
            poses.append((pstart - start, pend - start, qpos - start,
                          rpos - start, spos - start, tpos - start, tend - start))
    label = np.zeros((end - start + 1))
    for pose in poses:
        (pstart, pend, qpos, rpos, spos, tpos, tend) = pose
        label[pstart: pend + 1] = 1
        label[pend+1: qpos] = 2
        label[qpos: rpos] = 3
        label[rpos: spos + 1] = 4
        label[spos + 1: tend + 1] = 5

    # 将过滤前后的信号与标注图画出来
    # plotecg(signal, label, 0, len(label))
    signal, label = remove_seq_gaps(signal, label)
    # plotecg(signal, label, 0, len(label))

    xx, yy = splitseg(signal, label, 1000, 150)
    x = np.vstack((x, xx))
    y = np.vstack((y, yy))

# 将初始化的第一个sample去掉, 然后将片段打乱
x, y = x[1:], y[1:]
plotecg(x[0], y[0], 0, len(y[0]))
# 将x进行zscore归一化
for i in range(x.shape[0]):
    x[i] = st.zscore(x[i])
# 画归一化后的信号图像
plotecg(x[0], y[0], 0, len(y[0]))

assert len(x) == len(y)
p = np.random.permutation(range(len(x)))
x, y = x[p], y[p]

# 划分训练集和验证集，然后保存下
nums = len(x)
train_len = int(math.ceil(nums*train_percentage))
x_train, y_train = x[:train_len], y[:train_len]
x_val, y_val = x[train_len:], y[train_len:]

print('训练集共有{}个片段，验证集共有{}个片段'.format(train_len, nums-train_len))

with open(qtdb_pickle_save+'train_data.pkl', 'wb') as f:
    pickle.dump((x_train, y_train), f)

with open(qtdb_pickle_save+'val_data.pkl', 'wb') as f:
    pickle.dump((x_val, y_val), f)
