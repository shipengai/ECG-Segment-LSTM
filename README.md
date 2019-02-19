# ECG-Segment-LSTM
数据库：https://physionet.org/physiobank/database/qtdb/
参考链接：https://github.com/niekverw/Deep-Learning-Based-ECG-Annotator
本工程通过LSTM模型实现了对ECG信号的波形分割，
共6个波形：背景、P波、PQ段、QR段、RS段、ST段，分别对应标签0~5
## Getting Start
* 下载qt数据集之qtdb文件夹， 应该包含.hed .q1c等后缀格式文件
* 运行python qtdatabase.py 会在qtdb_pkl文件下生成train_data.pkl和val_data.pkl
* 运行python model_lstm.py 训练LSTM模型，模型存储在ckpt文件下
