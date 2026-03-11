import argparse
import os
import sys

import torch
import pandas as pd
from PIL import Image
from safetensors.torch import load_file

# ============================
# 评估脚本说明：
# - 用于加载训练好的 LoRA 适配器（medical_lora_adapter.pt），并在 PMC_VQA 测试集上做推理
# - 只输出 A/B/C/D 四个选项中的一个（与训练时的任务一致）
# - 结果以 CSV 形式保存，可用于进一步分析（例如准确率、混淆矩阵等）
#
# 使用示例：
#   python test/test.py --csv PMC_VQA/test_2.csv --img-dir PMC_VQA/images_2/figures --lora-path result/medical_lora_adapter.pt
# ============================

# 1. 路径与环境配置（确保无论从哪个目录运行此脚本，都能找到模型与数据）
#
# - SCRIPT_DIR: 当前脚本所在目录
# - REPO_ROOT: 项目根目录（作为相对路径计算的基准）
# - sys.path.insert: 将 REPO_ROOT 加入导入路径，便于直接 import pretrain_model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, REPO_ROOT)

from pretrain_model.moondream import MoondreamModel
from pretrain_model.config import MoondreamConfig
from transformers import AutoTokenizer


class LoraModule(torch.nn.Module):
    """LoRA adapter 参数容器（与 trainer.py 保持一致）。

    本脚本只用来加载/保存 LoRA 适配器权重，因此结构需与训练时一致。

    结构说明：
      - 每个 transformer layer 里，LoRA 修改了 attention 的 qkv/proj 线性层与 mlp 的两个线性层
      - 这里采用 A/B 两个矩阵表示低秩分解：W_delta ≈ B @ A
      - 最终通过 get_lora_dict() 生成符合 pretrain_model.text_decoder 期待格式的字典
    """

    def __init__(self, n_layers, dim, ff_dim, qkv_dim, r=16):
        super().__init__()
        self.r = r
        self.n_layers = n_layers
        # 使用 ParameterDict 方便统一保存/加载权重，同时保持与 torch.nn.Module 兼容
        self.params = torch.nn.ParameterDict()

        # 初始化 LoRA 权重为小的随机值，以便后续训练稳定收敛
        for i in range(n_layers):
            self.params[f"layer_{i}_attn_qkv_A"] = torch.nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_attn_qkv_B"] = torch.nn.Parameter(torch.zeros(qkv_dim, r))
            self.params[f"layer_{i}_attn_proj_A"] = torch.nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_attn_proj_B"] = torch.nn.Parameter(torch.zeros(dim, r))
            self.params[f"layer_{i}_mlp_fc1_A"] = torch.nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_mlp_fc1_B"] = torch.nn.Parameter(torch.zeros(ff_dim, r))
            self.params[f"layer_{i}_mlp_fc2_A"] = torch.nn.Parameter(torch.randn(r, ff_dim) * 0.01)
            self.params[f"layer_{i}_mlp_fc2_B"] = torch.nn.Parameter(torch.zeros(dim, r))

    def get_lora_dict(self):
        """将内部 ParameterDict 转换为 TextDecoder 所需的 LoRA 字典结构。

        返回的 dict 结构如下：
          {
            "text": {
              "blocks": {
                "0": {"attn": {...}, "mlp": {...}},
                "1": {...},
                ...
              }
            }
          }
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


def load_model_and_lora(args, device):
    """加载基座 Moondream 模型 + LoRA 适配器权重，并返回推理所需的各类对象。

    Args:
        args: 命令行参数（包含 lora_path）
        device: 运行设备（'cuda' 或 'cpu'）

    Returns:
        model: 基座 MoondreamModel（已加载权重、设置为 eval 模式）
        lora_dict: 适用于 pretrain_model.text_decoder 的 LoRA 字典结构
        tokenizer: 用于将文本转换为 token id 的 tokenizer
        config: MoondreamConfig 实例
    """

    # 1) 加载模型配置和基座模型（权重尚未加载）
    config = MoondreamConfig()
    model = MoondreamModel(config).to(device)

    # 2) 加载预训练的基座权重
    # strict=False 允许模型结构与权重文件之间存在不完全匹配（例如新增/缺失键），
    # 因为我们只关心加载大部分的模型权重，用于 inference。
    weights_path = os.path.join(SCRIPT_DIR, "pretrain_model", "model.safetensors")
    model.load_state_dict(load_file(weights_path), strict=False)

    # 3) 清理推理时可能残留的缓存状态（如 kv_cache），保证评估过程不受之前运行影响。
    for block in model.text.blocks:
        block.kv_cache = None
    model.eval()

    # 4) 构建与训练中一致的 LoRA 模块结构，并加载权重
    #    qkv_dim 计算方式需与模型内部的 qkv 映射一致
    qkv_dim = int(config.text.dim * (1 + 2 * config.text.n_kv_heads / config.text.n_heads))
    lora_module = LoraModule(config.text.n_layers, config.text.dim, config.text.ff_dim, qkv_dim).to(
        device=device, dtype=torch.bfloat16
    )

    lora_ckpt = args.lora_path or os.path.join(SCRIPT_DIR, "result", "medical_lora_adapter.pt")
    lora_module.load_state_dict(torch.load(lora_ckpt, map_location=device))

    # 5) 加载 tokenizer（用于将 prompt 从文本转为 token id）
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(SCRIPT_DIR, "pretrain_model"), local_files_only=True)

    return model, lora_module.get_lora_dict(), tokenizer, config


def predict_one(model, lora_dict, tokenizer, config, device, image, prompt):
    """对单条样本进行推理，输出预测的选项（A/B/C/D）。

    过程说明：
      1) 图像先经过视觉编码器得到图像嵌入（img_emb）
      2) 文本 prompt（包含问题 + 选项）被 tokenizer 编码并经过 text_encoder 得到文本嵌入
      3) 将 BOS token 嵌入 + 图像嵌入 + 文本嵌入拼接到一起作为 decoder 的输入
      4) 构造 causal attention mask（前 730 token 允许双向注意力，之后为因果注意力）
      5) 调用 text_decoder 得到隐层表示，再用 lm_head 输出 logits
      6) 从 logits 中挑选对应选项 A/B/C/D 的 token logits，并选出最大值对应的选项
    """
    # 仅在此函数内导入模块，避免在不需要时诱发循环依赖
    from pretrain_model.text import text_encoder, text_decoder, lm_head

    # 1) 图像 -> 图像嵌入（batch=1）
    img_emb = model._run_vision_encoder(image)
    if img_emb.dim() == 2:
        # 视觉编码器可能返回 [seq_len, dim]，这里统一变成 [1, seq_len, dim]
        img_emb = img_emb.unsqueeze(0)

    # 2) 文本 prompt -> token ids -> 文本嵌入
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_toks = torch.tensor([prompt_ids], device=device)
    prompt_emb = text_encoder(prompt_toks, model.text)

    # 3) 在 decoder 输入序列前加上 BOS token 的嵌入（与训练时输入保持一致）
    bos_id = torch.tensor([[config.tokenizer.bos_id]], device=device)
    bos_emb = text_encoder(bos_id, model.text)
    inputs_embeds = torch.cat([bos_emb, img_emb, prompt_emb], dim=1)

    # 4) 构造 attention mask：
    #    - 前 730 个 token 允许全局双向 attention（用于图像 + prompt 的 cross-attention）
    #    - 之后 token 采用因果注意力（生成式推理只看前文）
    seq_len = inputs_embeds.size(1)
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    mask[:730, :730] = True
    for j in range(730, seq_len):
        mask[j, : j + 1] = True

    pos_ids = torch.arange(seq_len, device=device)

    # 5) 运行 decoder，并得出每个位置的 logits
    hidden = text_decoder(inputs_embeds, model.text, mask, pos_ids, config.text, lora=lora_dict)
    logits = lm_head(hidden, model.text)

    # 6) 仅比较 A/B/C/D 4 个 token 的 logits，然后选最大值对应的预测字符
    valid_chars = ["A", "B", "C", "D"]
    valid_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in valid_chars]
    target_logits = logits[0, valid_ids]
    best_idx = torch.argmax(target_logits).item()
    return valid_chars[best_idx]


def run_eval(args):
    """主评估入口。

    该函数负责：
      1) 初始化模型和 tokenizer
      2) 读取测试集 CSV、逐行遍历样本
      3) 调用 predict_one 得到预测结果
      4) 统计准确率，并将每条预测结果写入输出 CSV
    """

    # 选择运行设备（优先使用 GPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型和 LoRA 权重
    model, lora_dict, tokenizer, config = load_model_and_lora(args, device)

    # 读取测试数据
    df = pd.read_csv(os.path.join(REPO_ROOT, args.csv))
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    img_dir = os.path.join(REPO_ROOT, args.img_dir)

    results = []
    correct = 0

    # 遍历每条样本并推理
    for idx, row in df.iterrows():
        image_path = os.path.join(img_dir, row["Figure_path"])
        if not os.path.exists(image_path):
            # 如果图片文件缺失，记录错误并跳过
            results.append({"Figure_path": row["Figure_path"], "error": "IMAGE_NOT_FOUND"})
            continue

        image = Image.open(image_path).convert("RGB")

        # 构造 prompt：训练时使用同样的格式
        prompt = (
            f"Question: {row['Question']}\n"
            f"Choices: A: {row['Choice A']}, B: {row['Choice B']}, C: {row['Choice C']}, D: {row['Choice D']}\n"
            "Answer:"
        )

        # 预测并统计准确率
        pred = predict_one(model, lora_dict, tokenizer, config, device, image, prompt)
        actual = str(row.get("Answer", "")).strip().upper()
        is_correct = (pred == actual)
        correct += 1 if is_correct else 0

        results.append(
            {
                "Figure_path": row["Figure_path"],
                "Question": row["Question"],
                "Actual": actual,
                "Predicted": pred,
                "Is_Correct": is_correct,
            }
        )

        # 可选的中间结果输出（方便观察评估进度）
        if args.verbose and (idx + 1) % 50 == 0:
            acc = correct / (idx + 1)
            print(f"[{idx+1}/{len(df)}] Acc={acc:.3%}")

    # 将评估结果写入 CSV（用于后续分析）
    output_df = pd.DataFrame(results)
    output_csv = args.output_csv or os.path.join(REPO_ROOT, "pmc_vqa_results.csv")
    output_df.to_csv(output_csv, index=False)

    acc = correct / len(df) if len(df) > 0 else 0.0
    print(f"完成评估: 总样本={len(df)}, 准确率={acc:.3%}, 结果已保存至 {output_csv}")


def parse_args():
    """命令行参数解析函数。

    运行示例：
      python test/test.py --csv PMC_VQA/test_2.csv --img-dir PMC_VQA/images_2/figures --lora-path result/medical_lora_adapter.pt
    """
    parser = argparse.ArgumentParser(description="Moondream Medical VQA 评估脚本")
    parser.add_argument("--csv", type=str, default="PMC_VQA/test_2.csv", help="待评估 CSV 路径")
    parser.add_argument("--img-dir", type=str, default="PMC_VQA/images_2/figures", help="图片目录")
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA 权重文件路径")
    parser.add_argument("--max-samples", type=int, default=200, help="最多评估样本数")
    parser.add_argument("--output-csv", type=str, default=None, help="输出结果 CSV 路径")
    parser.add_argument("--verbose", action="store_true", help="打印评估过程中的中间准确率")
    return parser.parse_args()


if __name__ == "__main__":
    # 当脚本作为主程序运行时，解析命令行参数并执行评估
    run_eval(parse_args())
