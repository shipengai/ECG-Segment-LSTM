# -*- coding:utf-8 -*-
"""
本实验基于类似class-oriented 所以效果要好很多
基于record-oriented未做，改一下数据集划分便可以。
"""
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

TRAIN = False  # 训练标志
CONTINUE_TRAIN = False  # 接着上次某一次训练结果继续训练
TEST = False  # 测试标志 设置成True时候，需要指定加载哪个模型
PAPER_TEST = True  # 得到写paper使用的测试指标
SAVE_TEST_FIG = True
EPOCHS = 100
BATCH_SIZE = 32
Seqlength = 300
NUM_SEGS_CLASS = 4

qtdb_pkl = './qtdb_pkl/'  # 数据预处理后的路径，便于调试网络
save_path = './ckpt/'  # 保存模型的路径


if not os.path.exists(save_path):
    os.mkdir(save_path)


class ECGDataset(Dataset):
    """ecg dataset.
       返回字典：{'signal':  ,'label': }
    """
    def __init__(self, qtdb_pkl, data):
        """
        :param qtdb_pkl: 数据库存放路径
        :param data: 训练集和验证集数据
        """
        pkl = os.path.join(qtdb_pkl, data)
        with open(pkl, 'rb') as f:
            x, y = pickle.load(f)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.x[idx]).float()
        label = torch.from_numpy(self.y[idx]).float()
        sample = {'signal': signal, 'label': label}
        return sample


class SegModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.features = torch.nn.Sequential(
            # torch.nn.Linear(in_features=input_size, out_features=hidden_size),
            torch.nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_size, 2*hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),

            torch.nn.Linear(2 * hidden_size, 2 * hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
        )
        self.output = torch.nn.Linear(2*hidden_size, out_size)

    def forward(self, x):
        """
        :param x: shape(batch, seq_len, input_size)
        :return:
        """
        batch, seq_len, nums_fea = x.size()
        features, _ = self.features(x)
        output = self.classifier(features)
        output = self.output(output.view(batch * seq_len, -1))
        return output


def train(net, data_loader, epochs):
    for step in range(epochs):
        net.train()
        for i, samples_batch in enumerate(data_loader):
            total = 0.0
            correct = 0.0

            output = net(samples_batch['signal'])
            target = samples_batch['label'].contiguous().view(-1).long()
            loss = criterion(output, target)

            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            if (i+1) % 20 == 0:
                print("EPOCHS:{},Iter:{},Loss:{:.4f},Acc:{:.4f}".format(step, i+1, loss.item(), correct/total))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 每2个epoch,保存一次模型
        if (step+1) % 2 == 0:
            torch.save(net, save_path+'epoch_{}.ckpt'.format(step))
        test(ecg_train_dl, 'train', step)
        test(ecg_val_dl, 'val', step)


def test(data_loader, str1, step):
    with torch.no_grad():
        right = 0.0
        total = 0.0
        net.eval()
        for sample in data_loader:
            output = net(sample['signal'])
            _, predicted = torch.max(output.data, 1)
            label = sample['label'].contiguous().view(-1).long()
            total += label.size(0)
            right += (predicted == label).sum().item()
        print("epoch:{},{} ACC: {:.4f}".format(step, str1, right / total))


def restore_net(ckpt):
    # load models
    with open(ckpt, 'rb') as f:
        net = torch.load(f)
    return net


def get_charateristic(y):
    p_end = QRS_onset = QRS_end = T_mid = 0
    for i, val in enumerate(y):
        if val == 1 and y[i+1] == 0:
            p_end = i
        if val == 2 and y[i-1] == 0:
            QRS_onset = i
        if val == 0 and y[i-1] == 2:
            QRS_end = i-1
        if val == 3 and y[i-1] == 0:
            T_mid = i
    return p_end, QRS_onset, QRS_end, T_mid


def point_equal(label, predict, tolerte):
    if predict <= label + tolerte*250 and predict >= label-tolerte*250:
        return True
    else:
        return False


def right_point(label_tuple, predict_tuple, tolerte):
    n = np.array([0, 0, 0, 0])
    for i,(x,x_p) in enumerate(zip(label_tuple,predict_tuple)):
        if point_equal(x, x_p, tolerte):
            n[i] = 1
    return n


def plotlabel(y, bias):
    cmap = ['k', 'r', 'g', 'b', 'c', 'y']
    start = end = 0
    for i in range(len(y) - 1):
        if y[i] != y[i + 1]:
            end = i
            plt.plot(np.arange(start, end), y[start:end] - bias, cmap[int(y[i])])
            start = i + 1
        if i == len(y) - 2:
            end = len(y) - 1
            plt.plot(np.arange(start, end), y[start:end] - bias, cmap[int(y[i])])


if __name__ == '__main__':

    # loading data
    ecg_train_db = ECGDataset(qtdb_pkl, 'train_data.pkl')
    ecg_train_dl = DataLoader(ecg_train_db, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=1)

    ecg_val_db = ECGDataset(qtdb_pkl, 'val_data.pkl')
    ecg_val_dl = DataLoader(ecg_val_db, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=1)

    if TRAIN:
        if CONTINUE_TRAIN:
            # continue training
            net = restore_net(save_path + 'epoch_102.ckpt')
        else:
            # model
            net = SegModel(input_size=2, hidden_size=32, num_layers=2, out_size=NUM_SEGS_CLASS)

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()

        train(net, ecg_train_dl, EPOCHS)

    if TEST:
        # vis
        net = restore_net(save_path+'epoch_99.ckpt')
        net.eval()
        # test(ecg_val_dl, 'val', 4)
        for i, idx in enumerate([20, 60,
                                 160, 280]):
            sample = ecg_val_db[idx]
            signal = sample['signal'].numpy()
            label = sample['label'].numpy()
            # plotecg(signal, label, 0, 1300)
            output = net(sample['signal'].unsqueeze(0))
            _, predict = torch.max(output, 1)
            # 将predict 和 label画出来
            predict = predict.numpy()
            x = np.arange(len(predict))
            plt.subplot(2, 2, i+1)
            plt.plot(x, signal[:, 0])
            plotlabel(label, 0.2)
            plotlabel(predict, 0.4)
        plt.show()

    if PAPER_TEST:
        net = restore_net(save_path + 'epoch_99.ckpt')
        net.eval()
        print('waiting several minutes')
        right_point_num = np.array([0, 0, 0, 0])
        if SAVE_TEST_FIG:
            with PdfPages('test.pdf') as pdf:
                for i in range(len(ecg_val_db)):
                    sample = ecg_val_db[i]
                    signal = sample['signal'].numpy()
                    label = sample['label'].numpy()
                    # 得到预测结果
                    output = net(sample['signal'].unsqueeze(0))
                    _, predict = torch.max(output, 1)
                    predict = predict.numpy()

                    x = np.arange(Seqlength)
                    plt.plot(x, signal[:, 0])
                    plotlabel(label, 0.2)
                    plotlabel(predict, 0.4)
                    pdf.savefig()
                    plt.close()

                    # 得到p-end, QRS onset end , T-middle
                    p_end, QRS_onset, QRS_end, T_mid = get_charateristic(label)
                    p_end_p, QRS_onset_p, QRS_end_p, T_mid_p = get_charateristic(predict)
                    right_point_num += right_point((p_end, QRS_onset, QRS_end, T_mid),
                                                   (p_end_p, QRS_onset_p, QRS_end_p, T_mid_p), 0.1)
        else:
            for i in range(len(ecg_val_db)):
                sample = ecg_val_db[i]
                signal = sample['signal'].numpy()
                label = sample['label'].numpy()
                # 得到预测结果
                output = net(sample['signal'].unsqueeze(0))
                _, predict = torch.max(output, 1)
                predict = predict.numpy()
                # 得到p-end, QRS onset end , T-middle
                p_end, QRS_onset, QRS_end, T_mid = get_charateristic(label)
                p_end_p, QRS_onset_p, QRS_end_p, T_mid_p = get_charateristic(predict)
                right_point_num += right_point((p_end, QRS_onset, QRS_end, T_mid),
                                               (p_end_p, QRS_onset_p, QRS_end_p, T_mid_p), 0.025)
        print(right_point_num/len(ecg_train_db))
