# ECG-Segment-LSTM
环境：python3 + pytorch1.0.0

ENV:python3 + pytorch1.0.0

依赖的包：wfdb、pickle、numpy、scipy、matplotlib

Independent Package:wfdb、pickle、numpy、scipy、matplotlib

使用数据库：https://physionet.org/physiobank/database/qtdb/

Used database Url:https://physionet.org/physiobank/database/qtdb/

参考链接：https://github.com/niekverw/Deep-Learning-Based-ECG-Annotator

Ref Rep:https://github.com/niekverw/Deep-Learning-Based-ECG-Annotator

本工程通过LSTM模型实现了对ECG信号的波形分割，共6个波形：背景、P波、PQ段、QR段、RS段、ST段，分别对应标签0~5

This project achieved ECG signal wave segmentation,by using LSTM net.There are six waves:background/P Segment/PQ Seg/QR Seg/RS Seg/ST Seg,label is 0~6.

## Getting Start
* 下载qt数据集至qtdb文件夹， 应该包含.hed .q1c等后缀格式文件

  download qtdb,including .hed .q1c files

* 运行`python qtdatabase.py` 会在qtdb_pkl文件下生成train_data.pkl和val_data.pkl

  run `python qtdatabase.py`, it will generate train_data.pkl and val_data.pkl in folder "qtdb_pkl"

* 运行`python model_lstm.py` 训练LSTM模型，模型存储在ckpt文件下

  run `python model_lstm.py` to training LSTM Net, its result will be in folder "ckpt"

阅读源代码和注释

Please to read the source code and annotations

## Output

下图为预测结果与标签对比，对于一组红色的线，上面那条为label，下面那条为predict val

The figure shows the predict and label, for example, a couple of red lines, the upper is label, the lower is predict

![](./result/result.png)
