import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ==================== 1. 数据标准化类 ====================
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        """计算数据的均值和标准差"""
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        """标准化数据：(x - mean) / std"""
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / (std + 10e-9)  # 加小值防止除零

    def inverse_transform(self, data):
        """逆标准化：还原数据"""
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


# ==================== 2. 数据集定义 ====================
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


# ==================== 3. 滑动窗口划分数据 ====================
def create_inout_sequences(input_data, tw, pre_len, config):
    """
    input_data: 输入数据
    tw: 时间窗口大小（用过去多少个点预测）
    pre_len: 预测未来多少个点
    """
    inout_seq = []
    L = len(input_data)
    # 修复：确保有足够的数据进行滑窗
    for i in range(L - tw - pre_len + 1):
        train_seq = input_data[i:i + tw]  # 过去tw个点
        train_label = input_data[i + tw:i + tw + pre_len]  # 未来pre_len个点
        # 如果是MS（多变量预测单变量），只取最后一列（OT）作为标签
        if config.feature == 'MS':
            train_label = train_label[:, -1:]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# ==================== 4. 创建数据加载器 ====================
def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # 读取数据
    df = pd.read_csv(config.data_path)
    pre_len = config.pre_len
    train_window = config.window_size

    # 将预测目标列（OT）移到最后
    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    # 去掉第一列时间，只取数值列
    cols_data = df.columns[1:]
    df_data = df[cols_data]
    true_data = df_data.values.astype(np.float32)

    # 标准化
    scaler = StandardScaler()
    scaler.fit(true_data)

    # 划分数据集：60%训练，20%验证，20%测试
    train_data = true_data[:int(0.6 * len(true_data))]
    valid_data = true_data[int(0.6 * len(true_data)):int(0.8 * len(true_data))]
    test_data = true_data[int(0.8 * len(true_data)):]

    print("训练集尺寸:", len(train_data), "验证集尺寸:", len(valid_data), "测试集尺寸:", len(test_data))

    # 标准化转换
    train_data_normalized = scaler.transform(train_data)
    valid_data_normalized = scaler.transform(valid_data)
    test_data_normalized = scaler.transform(test_data)

    # 转Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)

    # 生成滑窗序列
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, config)

    # 创建Dataset和DataLoader
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


# ==================== 5. 损失曲线绘制 ====================
def plot_loss_data(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o', linewidth=2, color='#1f77b4')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.show()


# ==================== 6. 误差计算函数 ====================
def calculate_mre(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8))


def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ==================== 7. TCN模型核心组件 ====================
class Chomp1d(nn.Module):
    """裁剪层：保证因果卷积（未来信息不会泄露到过去）"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN残差块"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一层因果卷积
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层因果卷积
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 残差连接的下采样（如果输入输出维度不同）
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
    """完整TCN网络"""

    def __init__(self, num_inputs, outputs, pre_len, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.pre_len = pre_len
        num_levels = len(num_channels)

        # 堆叠多个TCN块，膨胀系数指数增长：1, 2, 4, 8...
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], outputs)  # 最后一层线性层输出预测值

    def forward(self, x):
        # 输入形状：(batch, window_size, feature) -> 转置为 (batch, feature, window_size)
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)  # 转置回来
        x = self.linear(x)
        return x[:, -self.pre_len:, :]  # 只取最后pre_len个时间步作为预测输出


# ==================== 8. 训练函数 ====================
def train(model, args, scaler, device, train_loader, valid_loader):
    start_time = time.time()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 学习率调度器：验证损失5轮不下降则减半
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    epochs = args.epochs
    model.train()
    results_loss = []

    for i in tqdm(range(epochs)):
        losss = []
        # 训练一个epoch
        for seq, label in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, label)
            single_loss.backward()
            optimizer.step()
            losss.append(single_loss.detach().cpu().numpy())

        avg_loss = sum(losss) / len(losss)
        tqdm.write(f"\t Epoch {i + 1}/{epochs}, Train Loss: {avg_loss:.6f}")
        results_loss.append(avg_loss)
        torch.save(model.state_dict(), 'save_model.pth')  # 保存模型

        # 验证集评估
        valid_loss = valid(model, args, scaler, valid_loader)
        scheduler.step(valid_loss)  # 根据验证损失调整学习率
        tqdm.write(f"\t Valid Loss: {valid_loss:.6f}, Adjusted LR: {scheduler.optimizer.param_groups[0]['lr']:.8f}")
        time.sleep(0.1)

    plot_loss_data(results_loss)
    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.2f}分钟<<<<<<<<<<<<<<<<<<")


# ==================== 9. 验证函数 ====================
def valid(model, args, scaler, valid_loader):
    model.load_state_dict(torch.load('save_model.pth'))
    model.eval()
    losss_mse = []
    with torch.no_grad():
        for seq, label in valid_loader:
            pred = model(seq)
            mse = calculate_mse(pred.detach().cpu().numpy(), label.detach().cpu().numpy())
            losss_mse.append(mse)
    return sum(losss_mse) / len(losss_mse)


# ==================== 10. 测试函数 ====================
def test(model, args, test_loader, scaler):
    model.load_state_dict(torch.load('save_model.pth'))
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

    # 计算各项评估指标
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

    # 绘制预测结果对比图
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True Value', linewidth=1.5, color='#2ca02c')
    plt.plot(results, label='Predicted Value', linewidth=1.5, alpha=0.8, color='#ff7f0e')
    plt.title("TCN Time Series Prediction (Test Set)")
    plt.xlabel("Time Step")
    plt.ylabel("OT Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ==================== 11. 训练集拟合检查 ====================
def inspect_model_fit(model, args, train_loader, scaler):
    model.load_state_dict(torch.load('save_model.pth'))
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

    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True History', linewidth=1.5, color='#1f77b4')
    plt.plot(results, label='Model Prediction', linewidth=1.5, alpha=0.8, color='#d62728')
    plt.title("Model Fit on Training Set")
    plt.xlabel("Time Step")
    plt.ylabel("OT Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ==================== 主函数入口 ====================
if __name__ == '__main__':
    # 超参数配置
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='TCN', help="模型名称")
    parser.add_argument('-window_size', type=int, default=10, help="时间窗口大小（用过去多少个点预测）")
    parser.add_argument('-pre_len', type=int, default=1, help="预测未来多少个点")
    parser.add_argument('-data_path', type=str, default='etth1.csv', help="数据集路径")
    parser.add_argument('-target', type=str, default='OT', help="预测目标列名")
    parser.add_argument('-input_size', type=int, default=7, help="输入特征数（不含时间列）")
    parser.add_argument('-feature', type=str, default='MS',
                        help="M:多变量预测多变量, S:单变量预测单变量, MS:多变量预测单变量")
    parser.add_argument('-model_dim', type=list, default=[32, 64, 128, 256], help="TCN各层通道数（列表长度=层数）")
    parser.add_argument('-lr', type=float, default=0.001, help="学习率")
    parser.add_argument('-drop_out', type=float, default=0.05, help="Dropout概率")
    parser.add_argument('-epochs', type=int, default=15, help="训练轮次（已改为15轮，更快出结果）")
    parser.add_argument('-batch_size', type=int, default=25, help="批次大小")
    parser.add_argument('-kernel_sizes', type=int, default=3, help="卷积核大小")
    parser.add_argument('-use_gpu', type=bool, default=False)
    parser.add_argument('-device', type=int, default=0)
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-inspect_fit', type=bool, default=True)

    args = parser.parse_args()

    # 选择设备（CPU/GPU）
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print("使用设备:", device)

    # 创建数据加载器
    train_loader, valid_loader, test_loader, scaler = create_dataloader(args, device)

    # 设置输出维度
    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size

    # 初始化模型
    try:
        model = TemporalConvNet(args.input_size, args.output_size, args.pre_len,
                                args.model_dim, args.kernel_sizes, args.drop_out).to(device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<")
    except Exception as e:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>初始化{args.model}模型失败: {e}<<<<<<<<<<<<<<<<<<<<<<<<<")
        exit()

    # 开始训练
    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, scaler, device, train_loader, valid_loader)

    # 开始测试
    if args.test:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<")
        test(model, args, test_loader, scaler)

    # 检查训练集拟合情况
    if args.inspect_fit:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>检验{args.model}模型拟合情况<<<<<<<<<<<<<<<<<<<<<<<<<")
        inspect_model_fit(model, args, train_loader, scaler)