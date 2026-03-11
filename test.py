import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
from safetensors.torch import load_file

# 1. 环境准备
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, REPO_ROOT)

from pretrain_model.moondream import MoondreamModel
from pretrain_model.config import MoondreamConfig
from pretrain_model.text import text_encoder, text_decoder, lm_head
from transformers import AutoTokenizer

# 2. 同样的 LoRA 结构定义 (用于加载权重)
class LoraModule(nn.Module):
    def __init__(self, n_layers, dim, ff_dim, qkv_dim, r=16):
        super().__init__()
        self.r = r
        self.n_layers = n_layers
        self.params = nn.ParameterDict()
        for i in range(n_layers):
            self.params[f"layer_{i}_attn_qkv_A"] = nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_attn_qkv_B"] = nn.Parameter(torch.zeros(qkv_dim, r))
            self.params[f"layer_{i}_attn_proj_A"] = nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_attn_proj_B"] = nn.Parameter(torch.zeros(dim, r))
            self.params[f"layer_{i}_mlp_fc1_A"] = nn.Parameter(torch.randn(r, dim) * 0.01)
            self.params[f"layer_{i}_mlp_fc1_B"] = nn.Parameter(torch.zeros(ff_dim, r))
            self.params[f"layer_{i}_mlp_fc2_A"] = nn.Parameter(torch.randn(r, ff_dim) * 0.01)
            self.params[f"layer_{i}_mlp_fc2_B"] = nn.Parameter(torch.zeros(dim, r))

    def get_lora_dict(self):
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

# 3. 单步推理逻辑 (极速版)
def predict_with_lora(model, lora_dict, tokenizer, config, device, image, prompt):
    with torch.no_grad():
        # 1. 视觉与文本编码
        img_emb = model._run_vision_encoder(image)
        if img_emb.dim() == 2: img_emb = img_emb.unsqueeze(0)
        
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_toks = torch.tensor([prompt_ids], device=device)
        prompt_emb = text_encoder(prompt_toks, model.text)
        
        # 2. 序列拼接
        bos_id = torch.tensor([[config.tokenizer.bos_id]], device=device)
        bos_emb = text_encoder(bos_id, model.text)
        inputs_embeds = torch.cat([bos_emb, img_emb, prompt_emb], dim=1)
        
        # 3. 掩码与前向传播
        seq_len = inputs_embeds.size(1)
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        mask[:730, :730] = True 
        for j in range(730, seq_len): mask[j, : j + 1] = True
        pos_ids = torch.arange(seq_len, device=device)
        
        hidden = text_decoder(inputs_embeds, model.text, mask, pos_ids, config.text, lora=lora_dict)
        
        # 此时的 logits 是 [1, 50000+] 大小的向量，包含了所有词的概率得分
        logits = lm_head(hidden, model.text) 
        
        # === 核心实质性工作量：Constrained Decoding ===
        valid_chars = ['A', 'B', 'C', 'D']
        # 获取这四个字母在词表中的具体 Token ID
        valid_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in valid_chars]
        
        # 我们只从 logits 中抽出 A, B, C, D 这四个维度的得分
        target_logits = logits[0, valid_ids]
        
        # 在这四个得分中选出最大的那个
        best_idx = torch.argmax(target_logits).item()
        
        return valid_chars[best_idx]

# 4. 主执行代码
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MoondreamConfig()
    
    print("1. 正在装载基座模型...")
    model = MoondreamModel(config).to(device)
    
    weights_path = os.path.join(SCRIPT_DIR, "pretrain_model/model.safetensors")
    model.load_state_dict(load_file(weights_path), strict=False)
    
    for block in model.text.blocks: 
        block.kv_cache = None # 关闭推理缓存
    model.eval()
    
    print("2. 正在挂载 LoRA 适配器...")
    qkv_dim = int(config.text.dim * (1 + 2 * config.text.n_kv_heads / config.text.n_heads))
    lora_module = LoraModule(config.text.n_layers, config.text.dim, config.text.ff_dim, qkv_dim).to(device=device, dtype=torch.bfloat16)
    
    # 【修复路径问题：使用绝对路径加载】
    lora_path = os.path.join(SCRIPT_DIR, "result/medical_lora_adapter.pt")
    lora_module.load_state_dict(torch.load(lora_path))
    lora_dict = lora_module.get_lora_dict()
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(SCRIPT_DIR, "pretrain_model"), local_files_only=True)
    
    print("3. 加载测试数据...")
    df = pd.read_csv(os.path.join(REPO_ROOT, "PMC_VQA/test_2.csv")).sample(5)
    img_dir = os.path.join(REPO_ROOT, "PMC_VQA/images_2/figures")
    
    print("\n" + "="*50)
    for _, row in df.iterrows():
        try:
            image = Image.open(os.path.join(img_dir, row['Figure_path'])).convert("RGB")
            prompt = f"Question: {row['Question']}\nChoices: A: {row['Choice A']}, B: {row['Choice B']}, C: {row['Choice C']}, D: {row['Choice D']}\nAnswer:"
            
            # 使用带 LoRA 的模型预测
            prediction = predict_with_lora(model, lora_dict, tokenizer, config, device, image, prompt)
            
            print(f"问题: {row['Question'][:50]}...")
            print(f"真实答案: {row['Answer']} | 模型预测: {prediction.strip().upper()}")
            print("-" * 50)
        except Exception as e:
            print(f"测试出错: {e}")