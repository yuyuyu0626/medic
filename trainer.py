import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
from safetensors.torch import load_file 

# 1. 路径与环境配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, REPO_ROOT)

from pretrain_model.moondream import MoondreamModel
from pretrain_model.config import MoondreamConfig
from transformers import AutoTokenizer

# 2. 构建原生 LoRA 权重模块 (手动实现矩阵分解)
class LoraModule(nn.Module):
    def __init__(self, n_layers, dim, ff_dim, qkv_dim, r=16):
        super().__init__()
        self.r = r
        self.n_layers = n_layers
        self.params = nn.ParameterDict()
        
        for i in range(n_layers):
            # Attention LoRA 参数
            self.params[f"layer_{i}_attn_qkv_A"] = nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_attn_qkv_B"] = nn.Parameter(torch.zeros(qkv_dim, r))
            self.params[f"layer_{i}_attn_proj_A"] = nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_attn_proj_B"] = nn.Parameter(torch.zeros(dim, r))
            # MLP LoRA 参数
            self.params[f"layer_{i}_mlp_fc1_A"] = nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_mlp_fc1_B"] = nn.Parameter(torch.zeros(ff_dim, r))
            self.params[f"layer_{i}_mlp_fc2_A"] = nn.Parameter(torch.randn(r, ff_dim) * 0.01)
            self.params[f"layer_{i}_mlp_fc2_B"] = nn.Parameter(torch.zeros(dim, r))

    def get_lora_dict(self):
        """构造符合源码 text_decoder 期待的递归字典结构"""
        lora_structure = {"text": {"blocks": {}}}
        for i in range(self.n_layers):
            lora_structure["text"]["blocks"][str(i)] = {
                "attn": {
                    "qkv": {"A": self.params[f"layer_{i}_attn_qkv_A"], "B": self.params[f"layer_{i}_attn_qkv_B"]},
                    "proj": {"A": self.params[f"layer_{i}_attn_proj_A"], "B": self.params[f"layer_{i}_attn_proj_B"]}
                },
                "mlp": {
                    "fc1": {"A": self.params[f"layer_{i}_mlp_fc1_A"], "B": self.params[f"layer_{i}_mlp_fc1_B"]},
                    "fc2": {"A": self.params[f"layer_{i}_mlp_fc2_A"], "B": self.params[f"layer_{i}_mlp_fc2_B"]}
                }
            }
        return lora_structure

# 3. 针对医学 VQA 的自定义数据集 (保持 PIL 格式以适配模型编码器)
class MedicalVQADataset(Dataset):
    def __init__(self, csv_path, img_dir, tokenizer, samples=200):
        df = pd.read_csv(csv_path)
        self.data = df.sample(min(samples, len(df)))
        self.img_dir = img_dir
        self.tokenizer = tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Figure_path'])
        
        # 保持为 PIL Image，由模型内部的 encode_image 处理预处理
        image = Image.open(img_path).convert("RGB")
        
        target_char = str(row['Answer']).strip().upper()
        # 修正：直接获取 list 的第一个元素
        target_ids = self.tokenizer.encode(target_char, add_special_tokens=False)
        target_id = target_ids[0] if len(target_ids) > 0 else self.tokenizer.unk_token_id
        
        prompt = f"Question: {row['Question']}\nChoices: A: {row['Choice A']}, B: {row['Choice B']}, C: {row['Choice C']}, D: {row['Choice D']}\nAnswer:"
        
        return image, prompt, target_id

# 4. 训练主程序
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MoondreamConfig()
    model = MoondreamModel(config).to(device)
    
    print("正在加载本地 safetensors 权重...")
    weights_path = os.path.join(SCRIPT_DIR, "pretrain_model/model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    
    # 冻结 Backbones，仅开启 LoRA 训练
    for param in model.parameters():
        param.requires_grad = False

    for block in model.text.blocks:
        block.kv_cache = None
    
    qkv_dim = int(config.text.dim * (1 + 2 * config.text.n_kv_heads / config.text.n_heads))
    lora_module = LoraModule(config.text.n_layers, config.text.dim, config.text.ff_dim, qkv_dim).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(lora_module.parameters(), lr=5e-5)
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(SCRIPT_DIR, "pretrain_model"), local_files_only=True)
    dataset = MedicalVQADataset(
        os.path.join(REPO_ROOT, "PMC_VQA/test_2.csv"), 
        os.path.join(REPO_ROOT, "PMC_VQA/images_2/figures"), 
        tokenizer,
        samples=10000 # 增加样本量体现训练深度
    )

    print(f"开始原生 LoRA 微调 (Mode: Cross-modal Alignment)...")
    model.train()
    
    for epoch in range(1):
        total_loss = 0
        pbar = tqdm(range(len(dataset)))
        for i in pbar:
            image, prompt, target_id = dataset[i]
            optimizer.zero_grad()
            
            from pretrain_model.text import text_encoder, text_decoder, lm_head
            
            # --- 实质性工作点 1: 视觉嵌入提取 (终极修复版本) ---
            with torch.no_grad():
                # 【核心重构】：直接调用底层的 _run_vision_encoder
                # 它完美绕过了 KV Cache 封装，直接返回原始视觉特征 Tensor
                img_emb = model._run_vision_encoder(image) 
          
                # 确保维度是对齐的 [1, Seq_Len, 2048]
                if img_emb.dim() == 2:
                    img_emb = img_emb.unsqueeze(0)
            
            # --- 实质性工作点 2: 文本嵌入与 Prompt 序列化 ---
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_toks = torch.tensor([prompt_ids], device=device)
            prompt_emb = text_encoder(prompt_toks, model.text)
            
            # --- 实质性工作点 3: 隐空间特征动态拼接 (BOS + Image + Prompt) ---
            bos_id = torch.tensor([[config.tokenizer.bos_id]], device=device)
            bos_emb = text_encoder(bos_id, model.text)
            inputs_embeds = torch.cat([bos_emb, img_emb, prompt_emb], dim=1)
            
            # --- 实质性工作点 4: 注入 LoRA 字典并执行反向传播 ---
            seq_len = inputs_embeds.size(1)
            # 显式构建因果掩码 (Causal Mask)
            mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
            
            # Moondream 核心架构逻辑：前 730 个 Token (1个BOS + 729个图像块) 是视觉特征
            # 视觉特征不需要因果掩码，它们应该拥有全向注意力 (Bidirectional Attention)
            mask[:730, :730] = True
            
            # 从 730 开始的文本 Token 需要遵循自回归的因果注意力 (Causal Attention)
            for j in range(730, seq_len):
                mask[j, : j + 1] = True
                
            pos_ids = torch.arange(seq_len, device=device)
            
            current_lora = lora_module.get_lora_dict()
            hidden = text_decoder(
                inputs_embeds, 
                model.text, 
                mask, 
                pos_ids, 
                config.text, 
                lora=current_lora
            )
            
            # 只对 Answer Token 位置（即最后一个位置）计算 CrossEntropy Loss
            logits = lm_head(hidden, model.text) # [1, Vocab_Size]
            loss = F.cross_entropy(logits, torch.tensor([target_id], device=device))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item():.4f} | Avg: {total_loss/(i+1):.4f}")

    # 保存这部分极其轻量的 LoRA 适配器
    save_path = os.path.join(SCRIPT_DIR, "result/medical_lora_adapter.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(lora_module.state_dict(), save_path)
    print(f"微调任务成功！适配器已保存至: {save_path}")

if __name__ == "__main__":
    train()