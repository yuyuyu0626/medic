import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from safetensors.torch import load_file
from tqdm import tqdm

# 1. 路径与环境配置（确保无论从哪个工作目录运行脚本，也能正确找到模型与数据）
#    - SCRIPT_DIR: 当前脚本所在目录
#    - REPO_ROOT: 项目根目录（上一级目录）
#    - sys.path: 将项目根目录加入导入路径，方便导入 pretrain_model 包
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, REPO_ROOT)

from pretrain_model.moondream import MoondreamModel
from pretrain_model.config import MoondreamConfig
from transformers import AutoTokenizer


# 2. 构建原生 LoRA 权重模块 (手动实现矩阵分解)
#
# LoRA 原理：
#   对于每个线性层 W, 通过低秩分解 W + BA 来实现可训练的“增量”权重。
#   其中 A 的维度为 (r, in_features)，B 的维度为 (out_features, r)，r 为 LoRA rank。
#
# 本模块直接构造一个与原模型结构匹配的 LoRA 参数集合，并通过 get_lora_dict 生成给模型使用的结构。
class LoraModule(nn.Module):
    def __init__(self, n_layers, dim, ff_dim, qkv_dim, r=16):
        super().__init__()
        self.r = r
        self.n_layers = n_layers
        self.params = nn.ParameterDict()

        # 这里的参数命名要与预训练模型内部名称“对齐”，以便能够正确 injected 到 text_decoder 的 lora 参数结构中。
        for i in range(n_layers):
            # Attention 层的 LoRA 参数：qkv 和 proj 分别对应 self-attention 的两部分线性变换
            self.params[f"layer_{i}_attn_qkv_A"] = nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_attn_qkv_B"] = nn.Parameter(torch.zeros(qkv_dim, r))
            self.params[f"layer_{i}_attn_proj_A"] = nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_attn_proj_B"] = nn.Parameter(torch.zeros(dim, r))

            # MLP 层的 LoRA 参数：对应 transformer block 里的前向网络 fc1 / fc2
            self.params[f"layer_{i}_mlp_fc1_A"] = nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_mlp_fc1_B"] = nn.Parameter(torch.zeros(ff_dim, r))
            self.params[f"layer_{i}_mlp_fc2_A"] = nn.Parameter(torch.randn(r, ff_dim) * 0.01)
            self.params[f"layer_{i}_mlp_fc2_B"] = nn.Parameter(torch.zeros(dim, r))

    def get_lora_dict(self):
        """构造符合源码 text_decoder 期待的递归字典结构。

        Moondream 的 text_decoder 接收一个 `lora` 参数，它是一个多层嵌套的 dict：
          lora["text"]["blocks"][i]["attn"]["qkv"]["A"] 等

        该函数将当前 Module 中的 Parameter 组织成这种结构，以便训练 / 推理时直接传给 text_decoder。
        """
        lora_structure = {"text": {"blocks": {}}}
        for i in range(self.n_layers):
            lora_structure["text"]["blocks"][str(i)] = {
                "attn": {
                    "qkv": {"A": self.params[f"layer_{i}_attn_qkv_A"], "B": self.params[f"layer_{i}_attn_qkv_B"]},
                    "proj": {"A": self.params[f"layer_{i}_attn_proj_A"], "B": self.params[f"layer_{i}_attn_proj_B"]},
                },
                "mlp": {
                    "fc1": {"A": self.params[f"layer_{i}_mlp_fc1_A"], "B": self.params[f"layer_{i}_mlp_fc1_B"]},
                    "fc2": {"A": self.params[f"layer_{i}_mlp_fc2_A"], "B": self.params[f"layer_{i}_mlp_fc2_B"]},
                },
            }
        return lora_structure


# 3. 针对医学 VQA 的自定义数据集 (保持 PIL 格式以适配模型编码器)
#
# 该 Dataset 读取 CSV 并按“行”返回一个训练样本：
#   - image: PIL Image (延迟到模型内部做预处理)
#   - prompt: 文本 prompt，用于与图像特征一起输入 decoder
#   - target_id: 目标标签的 token id（用于计算交叉熵 loss）
class MedicalVQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_dir,
        tokenizer,
        max_samples=None,
        shuffle=True,
    ):
        # 读取 CSV 并选择子集，方便快速实验与调参
        df = pd.read_csv(csv_path)
        if shuffle:
            # 随机打乱，有助于训练稳定
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        if max_samples is not None:
            # 只取前 N 条，加速调试
            df = df.head(max_samples)
        self.data = df
        self.img_dir = img_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 逐条读取数据，构建模型所需输入
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Figure_path"])

        # 1) 图像：保持 PIL 格式，模型内部负责预处理 + 特征提取
        image = Image.open(img_path).convert("RGB")

        # 2) 目标标签：答案通常是单个字母 (A/B/C/D)
        target_char = str(row["Answer"]).strip().upper()
        target_ids = self.tokenizer.encode(target_char, add_special_tokens=False)
        target_id = target_ids[0] if len(target_ids) > 0 else self.tokenizer.unk_token_id

        # 3) Prompt 构造：插入问题和选项信息，末尾保留 Answer: 让模型生成答案
        prompt = (
            f"Question: {row['Question']}\n"
            f"Choices: A: {row['Choice A']}, B: {row['Choice B']}, C: {row['Choice C']}, D: {row['Choice D']}\n"
            "Answer:"
        )

        return image, prompt, target_id


def build_mask(seq_len: int, visual_token_count: int = 730, device=None):
    """构建 Moondream 的混合 Attention Mask。

    Moondream 结构：
      - 前 730 个 token（BOS + 图像 patch）使用全向注意力（bidirectional attention），以便图像特征之间互相有信息流动
      - 后续文本 token 使用因果自回归注意力（causal attention），保证 decoder 生成时只能看到前面的 token

    Args:
        seq_len: 当前输入 token 数（BOS + 图像 + prompt）
        visual_token_count: 需要全向注意力的前置 token 数，默认为 730
        device: mask 所在的设备
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    mask[:visual_token_count, :visual_token_count] = True
    for j in range(visual_token_count, seq_len):
        mask[j, : j + 1] = True
    return mask


def evaluate(
    model,
    lora_dict,
    tokenizer,
    config,
    dataset,
    device,
    max_examples=None,
    verbose=False,
):
    """在给定数据集上评估 LoRA 适配器的准确率和平均损失。

    说明：
      - 该函数用于验证集/开发集评估，可在每个 epoch 结束后调用
      - 由于训练过程中使用了 LoRA，在 eval 时同样需要把 LoRA 注入到 text_decoder 中
    """
    from pretrain_model.text import text_encoder, text_decoder, lm_head

    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for idx in range(len(dataset)):
            if max_examples is not None and total >= max_examples:
                break

            # 1）取数据
            image, prompt, target_id = dataset[idx]

            # 2）图像编码：注意保持与训练一致的处理方式
            img_emb = model._run_vision_encoder(image)
            if img_emb.dim() == 2:
                img_emb = img_emb.unsqueeze(0)

            # 3）文本编码（Prompt）
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_toks = torch.tensor([prompt_ids], device=device)
            prompt_emb = text_encoder(prompt_toks, model.text)

            # 4）BOS + 图像 + Prompt 拼接
            bos_id = torch.tensor([[config.tokenizer.bos_id]], device=device)
            bos_emb = text_encoder(bos_id, model.text)
            inputs_embeds = torch.cat([bos_emb, img_emb, prompt_emb], dim=1)

            # 5）构造 attention mask（与训练一致）
            seq_len = inputs_embeds.size(1)
            mask = build_mask(seq_len, device=device)
            pos_ids = torch.arange(seq_len, device=device)

            # 6）前向计算（注入 LoRA）
            hidden = text_decoder(
                inputs_embeds,
                model.text,
                mask,
                pos_ids,
                config.text,
                lora=lora_dict,
            )

            logits = lm_head(hidden, model.text)

            # 7）损失/准确率计算（只比较最后一个 token）
            loss = F.cross_entropy(logits, torch.tensor([target_id], device=device))
            total_loss += loss.item()

            # --- 引入约束解码 (Constrained Decoding) ---
            valid_chars = ['A', 'B', 'C', 'D']
            # 动态获取当前 tokenizer 中 ABCD 的 token_id
            valid_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in valid_chars]
            
            # 只在 A/B/C/D 四个维度中取局部最大值
            target_logits = logits[0, valid_ids]
            best_local_idx = torch.argmax(target_logits).item()
            pred_id = valid_ids[best_local_idx]

            if pred_id == target_id:
                correct += 1

            total += 1

            if verbose and total % 50 == 0:
                print(f"  eval {total}/{len(dataset)}  acc={(correct/total):.3f}")

    return {
        "accuracy": correct / max(total, 1),
        "loss": total_loss / max(total, 1),
        "samples": total,
    }


def train(args):
    # 训练入口，支持命令行参数传递（见 parse_args）
    # device: 自动选择 CUDA/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 模型初始化（基座模型 + LoRA 模块）
    config = MoondreamConfig()
    model = MoondreamModel(config).to(device)

    print("[1/5] 加载基座模型权重...")
    weights_path = os.path.join(SCRIPT_DIR, "pretrain_model", "model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)

    # 2) 冻结基座模型参数，只训练 LoRA 参数（降低显存与计算）
    for param in model.parameters():
        param.requires_grad = False

    # 3) 关闭 KV Cache：避免在训练过程中因缓存状态导致行为差异
    for block in model.text.blocks:
        block.kv_cache = None

    # 4) 构造 LoRA 模块并设置优化器（只更新 LoRA 参数）
    qkv_dim = int(config.text.dim * (1 + 2 * config.text.n_kv_heads / config.text.n_heads))
    lora_module = LoraModule(
        config.text.n_layers, config.text.dim, config.text.ff_dim, qkv_dim
    ).to(device=device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(lora_module.parameters(), lr=args.lr)

    # 5) Tokenizer 加载：与基座模型保持一致
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(SCRIPT_DIR, "pretrain_model"), local_files_only=True
    )

    # 6) 构建训练/验证集
    train_ds = MedicalVQADataset(
        os.path.join(REPO_ROOT, args.train_csv),
        os.path.join(REPO_ROOT, args.img_dir),
        tokenizer,
        max_samples=args.max_train_samples,
        shuffle=True,
    )
    val_ds = None
    if args.val_csv:
        val_ds = MedicalVQADataset(
            os.path.join(REPO_ROOT, args.val_csv),
            os.path.join(REPO_ROOT, args.img_dir),
            tokenizer,
            max_samples=args.max_val_samples,
            shuffle=False,
        )

    print(f"[2/5] 训练集 {len(train_ds)} 样本", end="")
    if val_ds is not None:
        print(f"，验证集 {len(val_ds)} 样本")
    else:
        print("")

    best_acc = 0.0
    history = []

    from pretrain_model.text import text_encoder, text_decoder, lm_head

    print("[3/5] 开始训练 Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        # 改进进度条：显示当前 loss、平均 loss、lr，并用更直观的 ‘step’ 单位。
        pbar = tqdm(
            train_ds,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="step",
            mininterval=0.5,
            leave=False,
        )

        # 训练循环：每个 Step 处理一个样本（可扩展为 batch）
        for i, (image, prompt, target_id) in enumerate(pbar, start=1):
            # 1) 梯度清零 & 准备计算
            optimizer.zero_grad()

            # 2) 视觉编码（不计算梯度，节省显存）
            #    该步骤仅提取基础的视觉特征，后续与文本一起输入 decoder。
            with torch.no_grad():
                img_emb = model._run_vision_encoder(image)
                if img_emb.dim() == 2:
                    # 视觉特征有时返回 [seq, dim]，需要增加 batch 维度
                    img_emb = img_emb.unsqueeze(0)

            # 3) 文本编码：将 prompt 转为 embedding
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_toks = torch.tensor([prompt_ids], device=device)
            prompt_emb = text_encoder(prompt_toks, model.text)

            # 4) 构造输入序列：BOS + 图像特征 + prompt 文本特征
            bos_id = torch.tensor([[config.tokenizer.bos_id]], device=device)
            bos_emb = text_encoder(bos_id, model.text)
            inputs_embeds = torch.cat([bos_emb, img_emb, prompt_emb], dim=1)

            # 5) 构造 Attention mask 与 position ids（与训练 / 推理保持一致）
            seq_len = inputs_embeds.size(1)
            mask = build_mask(seq_len, device=device)
            pos_ids = torch.arange(seq_len, device=device)

            # 6) 前向传播：将 LoRA 注入到 text_decoder
            hidden = text_decoder(
                inputs_embeds,
                model.text,
                mask,
                pos_ids,
                config.text,
                lora=lora_module.get_lora_dict(),
            )

            # 7) 输出 logits + 计算 loss（只取最后一个 token）
            logits = lm_head(hidden, model.text)
            loss = F.cross_entropy(logits, torch.tensor([target_id], device=device))
            loss.backward()
            optimizer.step()

            # 8) 统计并展示训练进度
            total_loss += loss.item()
            avg_loss = total_loss / i
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{avg_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        train_loss = total_loss / len(train_ds)

        eval_stats = None
        if val_ds is not None:
            eval_stats = evaluate(
                model,
                lora_module.get_lora_dict(),
                tokenizer,
                config,
                val_ds,
                device,
                max_examples=args.max_val_samples,
                verbose=False,
            )

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            **({"val_loss": eval_stats["loss"], "val_acc": eval_stats["accuracy"]} if eval_stats else {}),
        }
        history.append(epoch_info)

        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            + (f"| val_loss={eval_stats['loss']:.4f} val_acc={eval_stats['accuracy']:.3f}" if eval_stats else "")
        )

        # 每个 epoch 结束后，如果当前模型在验证集上表现更好，则保存一份“最优” LoRA 权重。
        # 这样可避免最后一个 epoch 不一定是最优模型的情况。
        if eval_stats and eval_stats["accuracy"] > best_acc:
            best_acc = eval_stats["accuracy"]
            save_path = os.path.join(args.save_dir, f"best_lora_epoch{epoch}.pt")
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(lora_module.state_dict(), save_path)
            print(f"  ✅ 保存最佳 LoRA 适配器: {save_path} (val_acc={best_acc:.3f})")

    # 最终保存：无论是否提升，都会将当前 LoRA 权重保存为通用文件名
    print("[4/5] 开始保存 Loop...")
    final_path = os.path.join(args.save_dir, "medical_lora_adapter.pt")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(lora_module.state_dict(), final_path)
    print(f"训练结束，LoRA 权重已保存: {final_path}")

    # 输出训练、验证历史，便于后续分析与记录
    print("训练历史:")
    for row in history:
        print(row)
    
    print("[5/5] 谢谢老师...")


def parse_args():
    parser = argparse.ArgumentParser(description="Moondream Medical VQA LoRA 训练脚本")
    parser.add_argument("--train-csv", type=str, default="PMC_VQA/test_2.csv", help="训练集 CSV 路径")
    parser.add_argument("--val-csv", type=str, default="PMC_VQA/test_2.csv", help="验证集 CSV 路径（可与训练集相同，或置空不做验证）")
    parser.add_argument("--img-dir", type=str, default="PMC_VQA/images_2/figures", help="图片目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--max-train-samples", type=int, default=2000, help="最多训练样本数")
    parser.add_argument("--max-val-samples", type=int, default=2000, help="最多验证样本数")
    parser.add_argument("--save-dir", type=str, default=os.path.join(SCRIPT_DIR, "result"), help="LoRA 权重保存目录")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
