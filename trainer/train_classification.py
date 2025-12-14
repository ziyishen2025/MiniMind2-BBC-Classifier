# 训练命令python train_classification.py --epochs 5 --batch_size 32 --learning_rate 2e-5 --data_path "../dataset/bbc-news-data.csv" --num_labels 5 --from_weight "full_sft" --device "cuda:0" --log_interval 10
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, random_split, DistributedSampler
from model.model_minimind import MiniMindConfig
from model.model_classifier import Classification
from data.bbc_dataset import BBCNewsDataset
from typing import Tuple
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    init_distributed_mode,
    setup_seed,
    init_model,
    load_sft_weights_to_classification,
)

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    loss_fct = nn.CrossEntropyLoss()
    start_time = time.time()
    for step, (X, Y) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(X, labels=Y)

            loss = res.loss

            if hasattr(res, "aux_loss"):
                loss += res.aux_loss

            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f"Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )

            if wandb:
                wandb.log(
                    {"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                )

        del X, Y, res, loss


def evaluate(
    epoch: int, model, val_loader: DataLoader, args
) -> Tuple[float, float, list[int], list[int]]:
    # 将模型设置为评估模式
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_predictions = []  # 新增：用于收集所有预测结果
    all_true_labels = []  # 新增：用于收集所有真实标签

    # 确保不计算梯度，减少内存消耗，加速计算
    with torch.no_grad():

        # 迭代验证集数据加载器
        for step, (X, Y) in enumerate(val_loader):

            # 将数据移动到设备上 (X 是 input_ids, Y 是 labels)
            X = X.to(args.device)
            Y = Y.to(args.device)

            # 1. 前向传播
            # 传入 labels=Y 让 MiniMindForSequenceClassification 模型自动计算 loss
            res = model(X, labels=Y)

            # 2. 累加损失 (res.loss 是模型自动计算的标量)
            loss = res.loss
            total_loss += loss.item() * Y.size(0)  # 乘以 batch size，确保对齐总样本

            # 3. 计算准确率
            # 找出 logits 中值最大的索引作为预测类别
            # res.logits 的形状是 (batch_size, num_labels)
            predictions = torch.argmax(res.logits, dim=-1)

            all_predictions.extend(predictions.cpu().tolist())
            all_true_labels.extend(Y.cpu().tolist())

            # 比较预测和真实标签，并累加正确的数量
            # 注意：如果使用了 DDP，这里计算的准确率是当前进程的，但在评估阶段 DDP 不是主要的同步问题
            correct_predictions += (predictions == Y).sum().item()
            total_samples += Y.size(0)  # 累加当前批次的样本数

    # 计算平均指标
    avg_loss = total_loss / total_samples
    avg_accuracy = correct_predictions / total_samples

    # 重新将模型设置回训练模式
    model.train()

    return avg_loss, avg_accuracy, all_true_labels, all_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Classification")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument(
        "--save_weight", default="classfication", type=str, help="保存权重的前缀名"
    )
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="初始学习率")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len", default=512, type=int, help="训练的最大截断长度"
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/bbc-news-data.csv",
        help="训练数据路径",
    )
    parser.add_argument(
        "--from_weight",
        default="pretrain",
        type=str,
        help="基于哪个权重训练，为none则不基于任何权重训练",
    )
    parser.add_argument(
        "--num_labels", default=5, type=int, help="序列分类任务的类别数"
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="MiniMind-Classification",
        help="wandb项目名",
    )
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        num_labels=args.num_labels,
    )

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_run_name = f"MiniMind-Classfication-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 5. 定义模型、数据、优化器 ==========
    _, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    model = Classification(lm_config).to(args.device)

    if args.from_weight:
        if is_main_process():
            Logger(f"加载预训练权重 {args.from_weight} 到分类模型...")

        model = load_sft_weights_to_classification(model, args.from_weight)

        if is_main_process():
            Logger("预训练核心权重加载完成。分类头 (self.score) 保持随机初始化。")
    else:
        if is_main_process():
            Logger("警告：未指定或找不到预训练权重，模型将从头开始随机初始化。")

    full_ds = BBCNewsDataset(args.data_path, 512)

    total_size = len(full_ds)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    Logger(
        f"数据集大小: 总计 {total_size} 样本. 训练: {train_size}, 验证: {val_size}, 测试: {test_size}"
    )

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # 固定种子确保可复现
    )

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    best_val_loss = float("inf")
    best_model_path = os.path.join(args.save_dir, "best_model.pth")

    # ========== 8. 开始训练 (简化版) ==========
    start_epoch, start_step = 0, 0

    for epoch in range(args.epochs):

        # 1. 设置 DistributedSampler 的 epoch (如果使用了 DDP)
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # 2. 创建 DataLoader (使用上面创建的 train_loader 即可，无需重复创建)
        # 这里直接使用 train_loader，不需要重复定义一个新的 loader = DataLoader(...)
        # 保持原代码中的 loader 变量名，并指向 train_loader
        loader = train_loader

        # 3. 开始训练当前 epoch
        train_epoch(
            epoch=epoch, loader=loader, iters=len(loader), start_step=0, wandb=wandb
        )

        # 4. 评估验证集性能 (仅在主进程中执行)
        if is_main_process():

            # 调用我们之前编写的评估函数
            val_loss, val_acc, _, _ = evaluate(epoch, model, val_loader, args)

            Logger(f"Epoch {epoch+1} Avg Loss={val_loss:.6f}, Avg Acc={val_acc:.4f}")

            # 5. 检查并保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # 获取状态字典 (考虑 DDP 模式)
                # 确保 args.save_dir 目录存在
                os.makedirs(args.save_dir, exist_ok=True)

                # 优先保存 model.module 的 state_dict (如果使用了 DDP)
                state_dict = (
                    model.module.state_dict()
                    if dist.is_initialized()
                    else model.state_dict()
                )

                torch.save(state_dict, best_model_path)
                Logger(
                    f"保存新的最佳模型：验证集 Loss {val_loss:.6f}，路径：{best_model_path}"
                )
            else:
                Logger(
                    f"当前验证集 Loss {val_loss:.6f} 未超越最佳 Loss {best_val_loss:.6f}"
                )

    # ========== 9. 最终模型权重保存 (在所有 Epoch 结束后) ==========

    if is_main_process():
        # 定义最终模型的保存路径
        final_weight_path = os.path.join(
            args.save_dir,
            (
                args.save_weight
                if args.save_weight.endswith(".pth")
                else "classification_final.pth"
            ),
        )

        # 获取状态字典
        state_dict = (
            model.module.state_dict() if dist.is_initialized() else model.state_dict()
        )

        # 执行保存操作
        Logger(f"\n训练完成，正在保存最终模型权重到: {final_weight_path}")
        torch.save(state_dict, final_weight_path)

        if os.path.exists(best_model_path):
            Logger(f"加载最佳模型权重进行测试：{best_model_path}")

            # 实例化一个新的模型或直接加载到当前模型中
            # 确保加载到当前模型中，如果模型是 DDP 封装的，可能需要特殊的处理

            # 考虑到 DDP 封装，这里加载 state_dict 的方式需要根据实际情况调整
            best_state_dict = torch.load(best_model_path, map_location=args.device)

            # 如果模型被 DDP 封装 (model.module)，我们需要加载到 model.module 中
            if dist.is_initialized():
                model.module.load_state_dict(best_state_dict)
            else:
                model.load_state_dict(best_state_dict)

        else:
            Logger("警告：未找到最佳模型，将使用最后一次训练的模型权重进行测试。")
            # 如果没有找到最佳模型，则使用循环结束后留在 model 中的最终权重。

        test_loss, test_acc, true_labels, predictions = evaluate(
            epoch=-1,  # 传入 -1 表示这不是常规 Epoch 评估
            model=model,
            val_loader=test_loader,
            args=args,
        )

        f1 = f1_score(true_labels, predictions, average="macro")
        cm = confusion_matrix(true_labels, predictions)

        Logger(f"\n测试集性能：")
        Logger(f"Loss: {test_loss:.6f}")
        Logger(f"Accuracy: {test_acc:.4f}")
        Logger(f"Macro F1-Score: {f1:.4f}")

        Logger("\n混淆矩阵")

        # class_names = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']

        # 打印矩阵
        Logger(f"\n{cm}")

        # 提示用户进一步分析混淆矩阵
        Logger(
            "\n混淆矩阵对角线上的值是正确分类的数量。非对角线的值是错误分类的数量。(i, j)元素表示标签i被分类为j的数量"
        )
