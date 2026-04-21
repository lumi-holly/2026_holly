import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.utils import parametrizations
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ==================== 早停机制（防止过拟合）====================
class EarlyStopping:
    def __init__(self, patience=5, save_path='best_model.pth'):
        self.patience = patience
        self.save_path = save_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ==================== 数据标准化类 ====================
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / (std + 10e-9)

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


# ==================== 数据集定义 ====================
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


# ==================== 滑动窗口划分数据 ====================
def create_inout_sequences(input_data, tw, pre_len, config):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - pre_len + 1):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + pre_len]
        if config.feature == 'MS':
            train_label = train_label[:, -1:]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# ==================== 创建数据加载器 ====================
def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df = pd.read_csv(config.data_path)
    pre_len = config.pre_len
    train_window = config.window_size

    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    cols_data = df.columns[1:]
    df_data = df[cols_data]
    true_data = df_data.values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(true_data)

    train_data = true_data[:int(0.6 * len(true_data))]
    valid_data = true_data[int(0.6 * len(true_data)):int(0.8 * len(true_data))]
    test_data = true_data[int(0.8 * len(true_data)):]

    print("训练集尺寸:", len(train_data), "验证集尺寸:", len(valid_data), "测试集尺寸:", len(test_data))

    train_data_normalized = scaler.transform(train_data)
    valid_data_normalized = scaler.transform(valid_data)
    test_data_normalized = scaler.transform(test_data)

    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)

    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, config)

    train_dataset = TimeSeriesDataset(train_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    print("训练集滑窗数:", len(train_inout_seq), "批次:", len(train_loader))
    print("验证集滑窗数:", len(valid_inout_seq), "批次:", len(valid_loader))
    print("测试集滑窗数:", len(test_inout_seq), "批次:", len(test_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<")

    return train_loader, valid_loader, test_loader, scaler


# ==================== 图1：训练损失曲线（和指导手册一致）====================
def plot_loss_data(data):
    plt.figure(figsize=(6, 4))
    plt.plot(data, marker='o', linewidth=1, color='#1f77b4', label='Loss')
    plt.title("loss results Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=False)


# ==================== 误差计算函数 ====================
def calculate_mre(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8))


def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ==================== TCN模型（修复API警告）====================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = parametrizations.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            name='weight', dim=0
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = parametrizations.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            name='weight', dim=0
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, outputs, pre_len, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.pre_len = pre_len
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        return x[:, -self.pre_len:, :]


# ==================== 训练函数（加入早停+梯度裁剪）====================
def train(model, args, scaler, device, train_loader, valid_loader):
    start_time = time.time()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs
    model.train()
    results_loss = []

    early_stopping = EarlyStopping(patience=5, save_path='best_model.pth')

    for i in tqdm(range(epochs)):
        losss = []
        for seq, label in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, label)
            single_loss.backward()
            # 【核心优化4】梯度裁剪，从根本上杜绝梯度爆炸出NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losss.append(single_loss.detach().cpu().numpy())

        avg_loss = sum(losss) / len(losss)
        tqdm.write(f"\t Epoch {i + 1}/{epochs}, Train Loss: {avg_loss:.6f}")
        results_loss.append(avg_loss)

        # 验证集评估+早停判断
        model.eval()
        val_losss = []
        with torch.no_grad():
            for seq, label in valid_loader:
                pred = model(seq)
                val_loss = loss_function(pred, label)
                val_losss.append(val_loss.detach().cpu().numpy())
        avg_val_loss = sum(val_losss) / len(val_losss)
        tqdm.write(f"\t Valid Loss: {avg_val_loss:.6f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"早停触发！第{i + 1}轮验证损失无改善，停止训练")
            break

        model.train()
        time.sleep(0.1)

    # 加载最优模型
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    plot_loss_data(results_loss)
    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.2f}分钟<<<<<<<<<<<<<<<<<<")


# ==================== 测试函数（和指导手册一致）====================
def test(model, args, test_loader, scaler):
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    model.eval()
    results = []
    labels = []
    with torch.no_grad():
        for seq, label in test_loader:
            pred = model(seq)
            pred = pred[:, 0, :].detach().cpu().numpy()
            label = label[:, 0, :].detach().cpu().numpy()
            for i in range(len(pred)):
                results.append(pred[i][-1])
                labels.append(label[i][-1])

    y_true = np.array(labels)
    y_pred = np.array(results)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    smape = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8)) * 100
    mre = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8))

    print("\n========== 测试集评估结果 ==========")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"sMAPE: {smape:.2f}%")
    print(f"MRE: {mre:.6f}")

    # 图2：测试集预测
    plt.figure(figsize=(6, 4))
    plt.plot(labels, label='true', linewidth=1, color='#1f77b4')
    plt.plot(results, label='pred', linewidth=1, alpha=0.8, color='#ff7f0e')
    plt.title("TCN")
    plt.xlabel("time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=False)


# ==================== 训练集拟合检查（和指导手册一致）====================
def inspect_model_fit(model, args, train_loader, scaler):
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    model.eval()
    results = []
    labels = []
    with torch.no_grad():
        for seq, label in train_loader:
            pred = model(seq)[:, 0, :].detach().cpu().numpy()
            label = label[:, 0, :].detach().cpu().numpy()
            for i in range(len(pred)):
                results.append(pred[i][-1])
                labels.append(label[i][-1])

    # 图3：训练集拟合
    plt.figure(figsize=(8, 5))
    plt.plot(labels, label='History', linewidth=1, color='#1f77b4')
    plt.plot(results, label='Prediction', linewidth=1, alpha=0.8, color='#ff7f0e')
    plt.title("inspect model fit state")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ==================== 主函数入口（已优化4个核心参数）====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='TCN', help="模型名称")
    # 【核心优化1】窗口大小改为24（对应1天数据，符合电力负荷日周期）
    parser.add_argument('-window_size', type=int, default=24, help="时间窗口大小")
    parser.add_argument('-pre_len', type=int, default=1, help="预测未来多少个点")
    parser.add_argument('-data_path', type=str, default='etth1.csv', help="数据集路径")
    parser.add_argument('-target', type=str, default='OT', help="预测目标列名")
    parser.add_argument('-input_size', type=int, default=7, help="输入特征数")
    parser.add_argument('-feature', type=str, default='MS', help="M/S/MS")
    # 【核心优化2】模型层数改为3层，降低复杂度防止过拟合
    parser.add_argument('-model_dim', type=list, default=[32, 64, 128], help="TCN各层通道数")
    parser.add_argument('-lr', type=float, default=0.001, help="学习率")
    # 【核心优化3】Dropout改为0.1，增强正则化
    parser.add_argument('-drop_out', type=float, default=0.1, help="Dropout概率")
    parser.add_argument('-epochs', type=int, default=20, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=25, help="批次大小")
    parser.add_argument('-kernel_sizes', type=int, default=3, help="卷积核大小")
    parser.add_argument('-use_gpu', type=bool, default=False)
    parser.add_argument('-device', type=int, default=0)
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-inspect_fit', type=bool, default=True)

    args = parser.parse_args()

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print("使用设备:", device)

    train_loader, valid_loader, test_loader, scaler = create_dataloader(args, device)

    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size

    try:
        model = TemporalConvNet(args.input_size, args.output_size, args.pre_len,
                                args.model_dim, args.kernel_sizes, args.drop_out).to(device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<")
    except Exception as e:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>初始化{args.model}模型失败: {e}<<<<<<<<<<<<<<<<<<<<<<<<<")
        exit()

    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, scaler, device, train_loader, valid_loader)

    if args.test:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<")
        test(model, args, test_loader, scaler)

    if args.inspect_fit:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>检验{args.model}模型拟合情况<<<<<<<<<<<<<<<<<<<<<<<<<")
        inspect_model_fit(model, args, train_loader, scaler)