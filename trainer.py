import argparse # 导入命令行参数解析模块
import os # 导入操作系统接口模块，用于路径处理等
import sys # 导入系统特定参数和函数模块，用于修改环境变量

import torch # 导入PyTorch深度学习框架核心库
import torch.nn as nn # 导入PyTorch神经网络模块
import torch.nn.functional as F # 导入PyTorch神经网络函数式接口
from torch.utils.data import Dataset # 导入PyTorch数据集基类，用于构建自定义数据集
from PIL import Image # 导入Python图像处理库，用于读取和处理图像
import pandas as pd # 导入数据分析库pandas，用于读取CSV标签数据
from safetensors.torch import load_file #! load_file: 用于安全快速地加载safetensors格式的模型权重文件
from tqdm import tqdm # 导入进度条库，用于在循环中显示进度

# 1. 路径与环境配置（确保无论从哪个工作目录运行脚本，也能正确找到模型与数据）
#    - SCRIPT_DIR: 当前脚本所在目录
#    - REPO_ROOT: 项目根目录（上一级目录）
#    - sys.path: 将项目根目录加入导入路径，方便导入 pretrain_model 包
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # SCRIPT_DIR表示: 当前脚本的绝对路径目录; 维度为: 字符串格式标量
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir)) # REPO_ROOT表示: 项目根目录绝对路径; 维度为: 字符串格式标量
sys.path.insert(0, REPO_ROOT) # 将项目根目录插入到系统模块搜索路径的首位，确保优先导入本地包

from pretrain_model.moondream import MoondreamModel #! MoondreamModel: 预训练多模态模型基类; 提供视觉编码与文本解码功能
from pretrain_model.config import MoondreamConfig #! MoondreamConfig: 预训练模型配置类; 包含模型结构与维度参数
from transformers import AutoTokenizer #! AutoTokenizer: 拥抱脸分词器工厂类; 用于加载预训练模型的文本分词规则


# 2. 构建原生 LoRA 权重模块 (手动实现矩阵分解)
#
# LoRA 原理：
#   对于每个线性层 W, 通过低秩分解 W + BA 来实现可训练的“增量”权重。
#   其中 A 的维度为 (r, in_features)，B 的维度为 (out_features, r)，r 为 LoRA rank。
#
# 本模块直接构造一个与原模型结构匹配的 LoRA 参数集合，并通过 get_lora_dict 生成给模型使用的结构。
class LoraModule(nn.Module):
    """
    1) *summary*：构建用于低秩自适应微调（LoRA）的参数模块，模拟预训练模型的内部层结构。
    2) 参数分析 (Args)：
        n_layers (int): Transformer的层数，无默认值。
        dim (int): 模型隐藏层特征维度，无默认值。
        ff_dim (int): 前馈神经网络层的中间特征维度，无默认值。
        qkv_dim (int): 注意力机制中Q、K、V投影后的拼接总维度，无默认值。
        r (int, optional): LoRA的秩，决定新增参数量，默认值为16。
    3) 返回值 (Returns)：该类为nn.Module的子类，实例化后返回LoRA模块对象，包含所有可训练的LoRA参数。
    4) 变量维度分析：
        self.params: nn.ParameterDict对象，包含多个形状为 (r, in_features) 或 (out_features, r) 的Parameter张量。
    5) 举例：若dim=2048, r=16，则某层的attn_proj_A张量维度为 (16, 2048)。
    """
    def __init__(self, n_layers, dim, ff_dim, qkv_dim, r=16):
        super().__init__() # 调用父类初始化方法，初始化PyTorch模块基础组件
        self.r = r # self.r表示: LoRA的低秩数值; 维度为: 标量整数; 决定微调参数瓶颈层大小
        self.n_layers = n_layers # self.n_layers表示: 目标模型的Transformer层数; 维度为: 标量整数
        self.params = nn.ParameterDict() # self.params表示: 存储LoRA参数的字典; 维度为: 字典结构; 键为参数名，值为可求导张量

        # 这里的参数命名要与预训练模型内部名称“对齐”，以便能够正确 injected 到 text_decoder 的 lora 参数结构中。
        # 循环构建每一层的LoRA参数
        for i in range(n_layers):
            # Attention 层的 LoRA 参数：qkv 和 proj 分别对应 self-attention 的两部分线性变换
            # *作用: A矩阵采用正态分布初始化，B矩阵全零初始化，确保初始状态下BA相乘为0，不破坏预训练基座行为
            self.params[f"layer_{i}_attn_qkv_A"] = nn.Parameter(torch.randn(r, dim) * 0.01) # 维度为: (r, dim); 表示注意力QKV输入的降维矩阵A
            self.params[f"layer_{i}_attn_qkv_B"] = nn.Parameter(torch.zeros(qkv_dim, r)) # 维度为: (qkv_dim, r); 表示注意力QKV输出的升维矩阵B
            self.params[f"layer_{i}_attn_proj_A"] = nn.Parameter(torch.randn(r, dim) * 0.01) # 维度为: (r, dim); 表示注意力输出投影的降维矩阵A
            self.params[f"layer_{i}_attn_proj_B"] = nn.Parameter(torch.zeros(dim, r)) # 维度为: (dim, r); 表示注意力输出投影的升维矩阵B

            # MLP 层的 LoRA 参数：对应 transformer block 里的前向网络 fc1 / fc2
            self.params[f"layer_{i}_mlp_fc1_A"] = nn.Parameter(torch.randn(r, dim) * 0.01) # 维度为: (r, dim); 表示前馈网络第一层的降维矩阵A
            self.params[f"layer_{i}_mlp_fc1_B"] = nn.Parameter(torch.zeros(ff_dim, r)) # 维度为: (ff_dim, r); 表示前馈网络第一层的升维矩阵B
            self.params[f"layer_{i}_mlp_fc2_A"] = nn.Parameter(torch.randn(r, ff_dim) * 0.01) # 维度为: (r, ff_dim); 表示前馈网络第二层的降维矩阵A
            self.params[f"layer_{i}_mlp_fc2_B"] = nn.Parameter(torch.zeros(dim, r)) # 维度为: (dim, r); 表示前馈网络第二层的升维矩阵B
        # 循环结束，核心量self.params更新完成，包含了n_layers层所需要的全部A、B可训练矩阵字典

    def get_lora_dict(self):
        """
        1) *summary*：构造符合源码 text_decoder 期待的递归字典结构，以便动态注入模型。
        2) 参数分析 (Args)：无参数。
        3) 返回值 (Returns)：返回类型为嵌套字典 dict，键值层层映射到 text_decoder 的具体算子。
        4) 变量维度分析：返回的字典深层叶子节点为 (r, in) 或 (out, r) 的张量参数。
        5) 举例：lora_structure["text"]["blocks"]["0"]["attn"]["qkv"]["A"] 对应第0层qkv的A矩阵张量。
        """
        # 构造符合源码 text_decoder 期待的递归字典结构。
        #
        # Moondream 的 text_decoder 接收一个 `lora` 参数，它是一个多层嵌套的 dict：
        #   lora["text"]["blocks"][i]["attn"]["qkv"]["A"] 等
        #
        # 该函数将当前 Module 中的 Parameter 组织成这种结构，以便训练 / 推理时直接传给 text_decoder。
        lora_structure = {"text": {"blocks": {}}} # lora_structure表示: 要注入基座的嵌套配置字典; 维度为: 字典结构
        for i in range(self.n_layers): # 遍历所有层进行字典嵌套赋值
            lora_structure["text"]["blocks"][str(i)] = {
                "attn": { # 注意力子模块字典
                    "qkv": {"A": self.params[f"layer_{i}_attn_qkv_A"], "B": self.params[f"layer_{i}_attn_qkv_B"]}, # QKV张量绑定
                    "proj": {"A": self.params[f"layer_{i}_attn_proj_A"], "B": self.params[f"layer_{i}_attn_proj_B"]}, # Proj张量绑定
                },
                "mlp": { # 前馈网络子模块字典
                    "fc1": {"A": self.params[f"layer_{i}_mlp_fc1_A"], "B": self.params[f"layer_{i}_mlp_fc1_B"]}, # FC1张量绑定
                    "fc2": {"A": self.params[f"layer_{i}_mlp_fc2_A"], "B": self.params[f"layer_{i}_mlp_fc2_B"]}, # FC2张量绑定
                },
            }
        return lora_structure # 返回构建好的嵌套字典供模型解码器调用


# 3. 针对医学 VQA 的自定义数据集 (保持 PIL 格式以适配模型编码器)
#
# 该 Dataset 读取 CSV 并按“行”返回一个训练样本：
#   - image: PIL Image (延迟到模型内部做预处理)
#   - prompt: 文本 prompt，用于与图像特征一起输入 decoder
#   - target_id: 目标标签的 token id（用于计算交叉熵 loss）
class MedicalVQADataset(Dataset):
    """
    1) *summary*：用于医学视觉问答（VQA）任务的PyTorch数据集封装类。
    2) 参数分析 (Args)：
        csv_path (str): 包含数据集元信息的CSV文件路径，无默认值。
        img_dir (str): 图像存储的根目录，无默认值。
        tokenizer (PreTrainedTokenizer): 用于文本编码的预训练分词器，无默认值。
        max_samples (int, optional): 截取最大样本数以便调试，默认值为None（使用全量）。
        shuffle (bool, optional): 是否在初始化时打乱数据，默认值为True。
    3) 返回值 (Returns)：可迭代的Dataset对象，单次获取返回(image, prompt, target_id)元组。
    4) 变量维度分析：
        self.data: DataFrame结构，包含数据表信息。
    5) 举例：初始化传入max_samples=100，则数据集只包含100条有效样本。
    """
    def __init__(
        self,
        csv_path,
        img_dir,
        tokenizer,
        max_samples=None,
        shuffle=True,
    ):
        # 读取 CSV 并选择子集，方便快速实验与调参
        df = pd.read_csv(csv_path) # df表示: 从CSV读取的完整数据表格; 维度为: (N, 变量数); DataFrame类型
        if shuffle: # 如果开启打乱
            # 随机打乱，有助于训练稳定
            df = df.sample(frac=1, random_state=42).reset_index(drop=True) # 重建打乱后的索引，维度保持为(N, 变量数)
        if max_samples is not None: # 如果指定了最大样本限制
            # 只取前 N 条，加速调试
            df = df.head(max_samples) # 截断数据，维度变为(max_samples, 变量数)
        self.data = df # self.data表示: 最终使用的样本信息表; 维度为: DataFrame结构
        self.img_dir = img_dir # self.img_dir表示: 图像读取根目录; 维度为: 字符串标量
        self.tokenizer = tokenizer # self.tokenizer表示: 文本分词器对象; 维度为: Tokenizer类

    def __len__(self):
        """
        1) *summary*：返回数据集当前包含的样本总数量。
        2) 参数分析 (Args)：无参数。
        3) 返回值 (Returns)：int类型，样本数量。
        4) 变量维度分析：返回值为标量。
        5) 举例：若数据集有2000条样本，则返回2000。
        """
        return len(self.data) # 返回数据表的行数，即样本总数

    def __getitem__(self, idx):
        """
        1) *summary*：根据索引提取单条(图,文,标签)多模态样本数据。
        2) 参数分析 (Args)：
            idx (int): 数据集索引值，无默认值。
        3) 返回值 (Returns)：返回包含 (image, prompt, target_id) 的元组。image为PIL对象；prompt为字符串；target_id为整型标量。
        4) 变量维度分析：image (W, H, C)，prompt标量字符，target_id标量。
        5) 举例：返回 (PILImage, "Question: ... Answer:", 312)。
        """
        # 逐条读取数据，构建模型所需输入
        row = self.data.iloc[idx] # row表示: 第idx行的数据字典视图; 维度为: Series结构; 包含问题、选项、图片名等
        img_path = os.path.join(self.img_dir, row["Figure_path"]) # 拼接出图像文件的绝对路径

        # 1) 图像：保持 PIL 格式，模型内部负责预处理 + 特征提取
        # *作用: 读取并转为RGB三通道，防止灰度图引起的维度不匹配
        image = Image.open(img_path).convert("RGB") # image表示: PIL格式彩色图; 维度为: (W, H, 3)

        # 2) 目标标签：答案通常是单个字母 (A/B/C/D)
        target_char = str(row["Answer"]).strip().upper() # 提取正确答案字符，去除空格并转大写
        # 使用tokenizer编码该字母，不加首尾特殊符，得到列表
        target_ids = self.tokenizer.encode(target_char, add_special_tokens=False) # target_ids表示: 编码后的ID列表; 维度为: 列表，长度通常为1
        # 获取第一项，兜底机制若失败则用未知标记ID
        target_id = target_ids[0] if len(target_ids) > 0 else self.tokenizer.unk_token_id # target_id表示: 目标标签的TokenID; 维度为: 标量整型

        # 3) Prompt 构造：插入问题和选项信息，末尾保留 Answer: 让模型生成答案
        # *作用: 将结构化的选择题拼接为模型习惯阅读的自然段落
        prompt = ( # prompt表示: 完整的多选项问题提示词文本; 维度为: 字符串标量
            f"Question: {row['Question']}\n"
            f"Choices: A: {row['Choice A']}, B: {row['Choice B']}, C: {row['Choice C']}, D: {row['Choice D']}\n"
            "Answer:"
        )

        return image, prompt, target_id # 返回供单步迭代的三个核心模态信息


def build_mask(seq_len: int, visual_token_count: int = 730, device=None):
    """
    1) *summary*：构建 Moondream 的混合 Attention Mask（图像全向，文本因果）。
    2) 参数分析 (Args)：
        seq_len (int): 当前输入的总Token数量（BOS + 图像 + 文本），无默认值。
        visual_token_count (int, optional): 需要全向注意力的前置Token数，默认值为730。
        device (torch.device, optional): Mask张量分配的目标设备，默认值为None。
    3) 返回值 (Returns)：布尔型张量 mask，True表示允许注意力访问。
    4) 变量维度分析：返回张量维度为 (seq_len, seq_len)。
    5) 举例：seq_len=800，visual_token_count=730。返回 (800, 800) 的布尔矩阵，左上角(730x730)全为True，右下部分为下三角True。

    """
    # 构建 Moondream 的混合 Attention Mask。
    #
    # Moondream 结构：
    #   - 前 730 个 token（BOS + 图像 patch）使用全向注意力（bidirectional attention），以便图像特征之间互相有信息流动
    #   - 后续文本 token 使用因果自回归注意力（causal attention），保证 decoder 生成时只能看到前面的 token
    #
    # Args:
    #     seq_len: 当前输入 token 数（BOS + 图像 + prompt）
    #     visual_token_count: 需要全向注意力的前置 token 数，默认为 730
    #     device: mask 所在的设备
    # *作用: 初始化全False矩阵，作为因果掩码骨架
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool) # mask表示: 注意力可见性矩阵; 维度为: (seq_len, seq_len); mask[i, j]表示查询i能否看到键j
    # *作用: 图像部分解禁因果限制，设为全向双向可见
    mask[:visual_token_count, :visual_token_count] = True # 切片操作，图像区域全量赋予可见权
    # 循环遍历文本部分的Token
    for j in range(visual_token_count, seq_len):
        # *作用: 文本部分使用严格下三角(因果掩码)，防止“看到未来”
        mask[j, : j + 1] = True # 第j行直到主对角线j位置均设为可见
    # 循环结束，混合注意力的掩码构建完毕，维度保持(seq_len, seq_len)
    return mask # 返回构造好的布尔型掩码张量


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
    """
    1) *summary*：在给定数据集上使用约束解码机制，评估包含LoRA适配器的模型准确率与损失。
    2) 参数分析 (Args)：
        model (MoondreamModel): 待评估的多模态基座模型对象。
        lora_dict (dict): 嵌套字典格式的LoRA权重。
        tokenizer (PreTrainedTokenizer): 文本分词器。
        config (MoondreamConfig): 模型配置项对象。
        dataset (MedicalVQADataset): 用于评估的数据集实例。
        device (str 或 torch.device): 运行设备标识（如 'cuda' 或 'cpu'）。
        max_examples (int, optional): 最大评估样本数，默认None评估全部。
        verbose (bool, optional): 是否打印详细日志，默认False。
    3) 返回值 (Returns)：返回包含'accuracy', 'loss', 'samples'键的字典。
    4) 变量维度分析：accuracy、loss皆为浮点标量，samples为整型标量。
    5) 举例：返回 {"accuracy": 0.85, "loss": 0.23, "samples": 500}。
    """
    # 在给定数据集上评估 LoRA 适配器的准确率和平均损失。
    #
    # 说明：
    #   - 该函数用于验证集/开发集评估，可在每个 epoch 结束后调用
    #   - 由于训练过程中使用了 LoRA，在 eval 时同样需要把 LoRA 注入到 text_decoder 中
    #! 依赖内部模块预加载模型特定功能
    from pretrain_model.text import text_encoder, text_decoder, lm_head # 局部导入解码器的三个独立处理管线环节

    model.eval() # 将模型设为验证模式，关闭Dropout/BatchNorm行为
    correct = 0 # correct表示: 预测正确的样本计数器; 维度为: 标量整型
    total = 0 # total表示: 当前已评估样本计数器; 维度为: 标量整型
    total_loss = 0.0 # total_loss表示: 累计的交叉熵损失和; 维度为: 标量浮点型

    # #! 性能瓶颈与优化方向: 关闭自动求导机制，极大降低推理显存占用与计算开销
    with torch.no_grad():
        # 遍历数据集进行评估
        for idx in range(len(dataset)):
            # 评估截断机制，达到max_examples提前退出
            if max_examples is not None and total >= max_examples:
                break

            # 1）取数据
            image, prompt, target_id = dataset[idx] # 解析单一样本的三个核心要素

            # 2）图像编码：注意保持与训练一致的处理方式
            # *作用: 将PIL图像提维为视觉特征矩阵
            img_emb = model._run_vision_encoder(image) # img_emb表示: 视觉特征嵌入; 维度为: (729, dim) 或 (1, 729, dim)
            if img_emb.dim() == 2: # 若未包含batch维
                # 张量形状扩展操作：增加首个维度作为Batch，(729, dim) -> (1, 729, dim)
                img_emb = img_emb.unsqueeze(0) # 补全batch维，保持后续拼接统一

            # 3）文本编码（Prompt）
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) # 将字符串编码为ID列表
            prompt_toks = torch.tensor([prompt_ids], device=device) # prompt_toks表示: 提示词ID张量; 维度为: (1, 文本token数)
            prompt_emb = text_encoder(prompt_toks, model.text) # prompt_emb表示: 提示词特征嵌入; 维度为: (1, 文本token数, dim)

            # 4）BOS + 图像 + Prompt 拼接
            bos_id = torch.tensor([[config.tokenizer.bos_id]], device=device) # bos_id表示: 起始符ID; 维度为: (1, 1)
            bos_emb = text_encoder(bos_id, model.text) # bos_emb表示: 起始符特征; 维度为: (1, 1, dim)
            # 张量拼接操作：沿序列长度维度串联三大块特征，(1,1,dim) + (1,729,dim) + (1,L,dim) -> (1, 1+729+L, dim)
            inputs_embeds = torch.cat([bos_emb, img_emb, prompt_emb], dim=1) # inputs_embeds表示: 完整多模态序列; 维度为: (1, seq_len, dim)

            # 5）构造 attention mask（与训练一致）
            seq_len = inputs_embeds.size(1) # 获取当前序列的总长度标量
            mask = build_mask(seq_len, device=device) # 构造混合注意力可见掩码; 维度为: (seq_len, seq_len)
            pos_ids = torch.arange(seq_len, device=device) # pos_ids表示: 位置编码索引; 维度为: (seq_len,)

            # 6）前向计算（注入 LoRA）
            # *作用: 传入构建的lora_dict来使用微调权重影响解码器的前向传播计算
            hidden = text_decoder( # hidden表示: 解码器隐状态输出; 维度为: (1, seq_len, dim)
                inputs_embeds,
                model.text,
                mask,
                pos_ids,
                config.text,
                lora=lora_dict, # 此处注入字典格式的LoRA矩阵
            )

            # 通过线性头映射为词表大小的预测分布逻辑值
            logits = lm_head(hidden, model.text) # logits表示: 词表分布对数几率; 维度为: (1, seq_len, vocab_size)

            # 7）损失/准确率计算（只比较最后一个 token）
            # *作用: 自回归任务仅对序列最后一个Token进行下文预测；取最后一步的logits和真实答案计算差异
            # #! 风险提示：默认F.cross_entropy包含了softmax过程，传入原始logits防止梯度不稳
            loss = F.cross_entropy(logits[:, -1, :], torch.tensor([target_id], device=device)) # loss表示: 交叉熵损失标量; 注意切片logits[:, -1, :]选取最后时间步
            total_loss += loss.item() # 将张量转为python标量并累加

            # --- 引入约束解码 (Constrained Decoding) ---
            # *逻辑意图: 防止生成非预期的字母或符号，人为限制输出的搜索空间仅为A/B/C/D
            # 约束解码的含义：将解码的可选维度位数在四，也就是ABCD四个选项上进行限制
            # 不管别的概率多高，我们只将ABCD四个词中的最大概率作为最终预测结果
            # 说法：我们获得了很多token,每一个tokens有一个logits值，表示这个token的概率，我们先查询ABCD四个字母在词典中的ID，通过这个ID去截取这四个值的logits
            valid_chars = ['A', 'B', 'C', 'D'] # 合法的候选答案字符
            # 动态获取当前 tokenizer 中 ABCD 的 token_id
            valid_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in valid_chars] # valid_ids表示: 候选字符对应的4个ID列表
            
            # 只在 A/B/C/D 四个维度中取局部最大值
            # 张量切片：从最后一步(0处因为只有batch=1)中抽离出对应ABCD的4个输出概率
            target_logits = logits[0, -1, valid_ids] # 修正: 对单条样本序列最后一步的有效字符截取; 维度为: (4,)
            best_local_idx = torch.argmax(target_logits).item() # 取得4个概率中最大项的局部索引值，0到3之间
            pred_id = valid_ids[best_local_idx] # pred_id表示: 最终选定的预测Token ID; 维度为: 标量整型

            # 评估统计
            if pred_id == target_id:
                correct += 1 # 预测正确时计数加一

            total += 1 # 更新总评估计数

            # 输出每50步的进度监控
            if verbose and total % 50 == 0:
                print(f"  eval {total}/{len(dataset)}  acc={(correct/total):.3f}")
        # 遍历数据集循环结束，核心量correct为正确个数，total为评估总数，total_loss为累计损失标量

    return { # 返回包含综合指标的字典
        "accuracy": correct / max(total, 1), # 总体准确率，安全分母兜底防止除零错误
        "loss": total_loss / max(total, 1), # 平均测试集损失
        "samples": total, # 本次参与评估的总样本数
    }


def train(args):
    """
    1) *summary*：构建基座与LoRA模块，执行多轮次Medical VQA数据微调训练的顶层调度逻辑。
    2) 参数分析 (Args)：
        args (Namespace): 通过argparse解析的命令行参数集合，包含epochs、lr、文件路径等配置。
    3) 返回值 (Returns)：函数无明确返回值(None)。执行完毕会将LoRA权重保存到磁盘。
    4) 变量维度分析：不涉及特定单一输出张量，作用于全局模型权重更新。
    5) 举例：train(parse_args()) 将依据参数执行训练并将 best_lora_epochX.pt 保存在指定目录下。
    """
    # 训练入口，支持命令行参数传递（见 parse_args）
    # device: 自动选择 CUDA/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu" # device表示: 运算硬件标识; 标量字符串

    # 1) 模型初始化（基座模型 + LoRA 模块）
    config = MoondreamConfig() # config表示: 基座模型配置字典及超参; MoondreamConfig对象
    model = MoondreamModel(config).to(device) # model表示: 将配置实例化的完整模型; 并部署到计算设备

    print("[1/5] 加载基座模型权重...")
    weights_path = os.path.join(SCRIPT_DIR, "pretrain_model", "model.safetensors") # 基座权重绝对路径
    state_dict = load_file(weights_path) # state_dict表示: safetensor格式加载的模型权重字典
    # 加载状态字典，strict=False防止非严格匹配报错（比如部分不用的层）
    model.load_state_dict(state_dict, strict=False)

    # 2) 冻结基座模型参数，只训练 LoRA 参数（降低显存与计算）
    # *作用: 参数高效微调(PEFT)核心步骤，禁止所有基座权重传导梯度
    for param in model.parameters():
        param.requires_grad = False # 关闭张量的求导跟踪标志

    # 3) 关闭 KV Cache：避免在训练过程中因缓存状态导致行为差异
    # 为什么会有KV cache：因为在测试过程中，模型的词汇是一个一个生成的，所以每生成一个词，都会把之前的计算结果（键值对）缓存起来
    # 以便下一个词的生成可以直接利用这些缓存结果，而不需要重新计算整个序列的注意力。这种机制极大地提升了生成效率。
    # 而这里为什么要关闭KV cache：因为在训练过程中，我们通常是一次性输入整个序列（prompt + 图像特征）
    # 并且使用teacher-forcing的方式来计算损失。此时，模型的每个时间步都能直接访问到完整的输入序列，因此不需要也不适合使用KV cache。
    # *逻辑意图: 自回归训练使用teacher-forcing并行前向计算，且不同样本长度不一致，缓存机制不仅无用且容易引发状态泄露
    for block in model.text.blocks:
        block.kv_cache = None # 清理KV缓存引用

    # 4) 构造 LoRA 模块并设置优化器（只更新 LoRA 参数）
    # 计算带有KV的多头注意力拼接总维度，适配模型非标准结构
    # 这里主要是在适配整体的QKV拼接维度，因为Moondream的注意力机制将Q、K、V三个矩阵拼接在一起进行线性变换
    # 所以需要计算出这个拼接后的总维度；从而正确初始化LoRA的低秩矩阵维度以匹配模型结构
    # 低秩分解能降低参数量的理解：原始模型参数矩阵维度为 (in_features, out_features)；
    # 使用LoRA后变为 A (r, in_features) 和 B (out_features, r)
    # 总参数量从 in*out 下降到 r*(in + out)，当 r << min(in, out) 时，参数量大幅减少。
    qkv_dim = int(config.text.dim * (1 + 2 * config.text.n_kv_heads / config.text.n_heads)) # qkv_dim表示: Q+K+V通道拼装宽度; 标量整型
    # 实例化自定义的LoRA管理模块，并显式指定半精度bfloat16节省显存，推送至设备
    lora_module = LoraModule( # lora_module表示: 持有待训练低秩矩阵的模型实例
        config.text.n_layers, config.text.dim, config.text.ff_dim, qkv_dim
    ).to(device=device, dtype=torch.bfloat16)

    # 初始化自适应权重衰减优化器AdamW，且只传入lora_module的参数，确保更新孤立
    optimizer = torch.optim.AdamW(lora_module.parameters(), lr=args.lr) # optimizer表示: 优化器调度对象; 参数表长度等于所有LoRA矩阵数量

    # 5) Tokenizer 加载：与基座模型保持一致
    #一致性： 文本必须转换成数字才能喂给模型。如果 Tokenizer 加载错了，数字和词义的映射就乱了
    #local_files_only=True 参数确保从本地加载预训练模型的分词器，避免在线下载失败或版本不匹配问题
    tokenizer = AutoTokenizer.from_pretrained( # tokenizer表示: 从本地加载的预置分词器; Tokenizer类实例
        os.path.join(SCRIPT_DIR, "pretrain_model"), local_files_only=True
    )

    # 6) 构建训练/验证集
    train_ds = MedicalVQADataset( # train_ds表示: 训练数据集实例; Dataset继承类
        os.path.join(REPO_ROOT, args.train_csv),
        os.path.join(REPO_ROOT, args.img_dir),
        tokenizer,
        max_samples=args.max_train_samples, # 使用命令行设定的最大数据截断
        shuffle=True, # 启用数据随机打乱避免过拟合顺序
    )
    val_ds = None # 验证集预留对象引用
    if args.val_csv: # 如果传参启用了验证集CSV
        val_ds = MedicalVQADataset( # val_ds表示: 验证数据集实例; Dataset继承类
            os.path.join(REPO_ROOT, args.val_csv),
            os.path.join(REPO_ROOT, args.img_dir),
            tokenizer,
            max_samples=args.max_val_samples,
            shuffle=False, # 验证流程无需打乱，保证多次对比公平
        )

    print(f"[2/5] 训练集 {len(train_ds)} 样本", end="")
    if val_ds is not None:
        print(f"，验证集 {len(val_ds)} 样本")
    else:
        print("")

    best_acc = 0.0 # best_acc表示: 跨Epoch跟踪的最优验证准确率; 浮点型标量
    history = [] # history表示: 记录训练各轮统计的列表; 用于后续保存图表或总结

    from pretrain_model.text import text_encoder, text_decoder, lm_head # 按需局部导入解码关键算子

    print("[3/5] 开始训练 Loop...")
    # 进入Epoch外层大循环
    for epoch in range(1, args.epochs + 1):
        model.train() # 恢复模型的训练特性标识（例如Dropout层等）
        total_loss = 0.0 # 每一轮的起始损失累加器清零

        # 改进进度条：显示当前 loss、平均 loss、lr，并用更直观的 ‘step’ 单位。
        pbar = tqdm( # pbar表示: tqdm封装的训练数据集包装器; 用于交互控制台打印与计时
            train_ds,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="step",
            mininterval=0.5,
            leave=False, # 完成当前Epoch进度条后是否保留
        )

        # 训练循环：每个 Step 处理一个样本（可扩展为 batch）
        for i, (image, prompt, target_id) in enumerate(pbar, start=1):
            # 1) 梯度清零 & 准备计算
            # *逻辑意图: PyTorch中梯度默认累加，因此每个Step必须显式清空旧的累加值
            optimizer.zero_grad() # 清除所有参与更新的LoRA参数残余梯度

            # 2) 视觉编码（不计算梯度，节省显存）
            #    该步骤仅提取基础的视觉特征，后续与文本一起输入 decoder。
            # #! 性能优化: 冻结视觉塔本身计算流图，避免OOM
            with torch.no_grad():
                img_emb = model._run_vision_encoder(image) # img_emb表示: 从模型抽取的原始视觉嵌入; 维度可能为 (729, dim)
                if img_emb.dim() == 2:
                    # 视觉特征有时返回 [seq, dim]，需要增加 batch 维度
                    img_emb = img_emb.unsqueeze(0) # 提维扩充，(729, dim) -> (1, 729, dim)

            # 3) 文本编码：将 prompt 转为 embedding
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) # 取文本编码序列
            prompt_toks = torch.tensor([prompt_ids], device=device) # 将序列转化为批次设备张量 (1, L)
            prompt_emb = text_encoder(prompt_toks, model.text) # prompt_emb表示: 文本部分的深度特征嵌入; 维度为: (1, L, dim)

            # 4) 构造输入序列：BOS + 图像特征 + prompt 文本特征
            bos_id = torch.tensor([[config.tokenizer.bos_id]], device=device) # 初始化预设起始符 (1, 1)
            bos_emb = text_encoder(bos_id, model.text) # bos_emb表示: 起始符特征; 维度为: (1, 1, dim)
            # 序列维度级联，组合最终包含长上下文的输入特征
            inputs_embeds = torch.cat([bos_emb, img_emb, prompt_emb], dim=1) # 维度整合操作: (1, 1+729+L, dim)

            # 5) 构造 Attention mask 与 position ids（与训练 / 推理保持一致）
            # 增加注意力掩码，图像部分（730个token）全向可见，因为一个图像的每一个像素都是同时存在的；
            # 文本部分使用因果掩码，只能看到之前生成过的内容；
            # 位置编码则是简单的递增索引
            seq_len = inputs_embeds.size(1) # 获取组装完毕的输入Token真实长度
            mask = build_mask(seq_len, device=device) # mask表示: 因果双向混合注意力表; 维度为: (seq_len, seq_len)
            pos_ids = torch.arange(seq_len, device=device) # pos_ids表示: 绝对位置偏置; 维度为: (seq_len,)

            # 6) 前向传播：将 LoRA 注入到 text_decoder
            # *逻辑意图: 借助lora_dict在运算中进行低秩增量 W_new = W_old + BA 计算
            # 在解码器进行矩阵乘法 $xW$ 时，我们偷偷加上了微调的分支 $x(BA)$。
            # 意义在于保证原始模型参数不动，模型只通过更新 $A$ 和 $B$ 这两个小矩阵来学会“如何看医学影像并回答问题”。
            hidden = text_decoder( # hidden表示: 解码后深层表示输出; 维度为: (1, seq_len, dim)
                inputs_embeds,
                model.text,
                mask,
                pos_ids,
                config.text,
                lora=lora_module.get_lora_dict(), # 从LoRA模块获取最新的(A,B)参数词典并提供运算
            )

            # 7) 输出 logits + 计算 loss（只取最后一个 token）
            # --- 引入约束解码 (Constrained Decoding) ---
            # *逻辑意图: 防止生成非预期的字母或符号，人为限制输出的搜索空间仅为A/B/C/D
            # 约束解码的含义：将解码的可选维度位数在四，也就是ABCD四个选项上进行限制
            # 不管别的概率多高，我们只将ABCD四个词中的最大概率作为最终预测结果
            # 说法：我们获得了很多token,每一个tokens有一个logits值，表示这个token的概率，我们先查询ABCD四个字母在词典中的ID，通过这个ID去截取这四个值的logits
            logits = lm_head(hidden, model.text) # logits表示: 全序列各位置的预测词汇空间向量; 维度为: (1, seq_len, vocab_size)
            # *逻辑意图: 切片最后时间步 logits[:,-1,:] 计算标签分布偏差，并自动内部Softmax
            loss = F.cross_entropy(logits[:, -1, :], torch.tensor([target_id], device=device)) # loss表示: 单步误差损失标量图节点
            loss.backward() # #* 反向传播: 从Loss回溯计算图中所有A与B矩阵上的梯度
            optimizer.step() # #* 优化推进: 应用AdamW动量更新各LoRA张量数据值

            # 8) 统计并展示训练进度
            total_loss += loss.item() # 将PyTorch图节点计算解挂为纯数值标量，累加汇总
            avg_loss = total_loss / i # 动态求历史平均损失
            # 在tqdm进度条上附加字典格式的简易面板信息显示
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}", # 瞬时损失数值
                    "avg": f"{avg_loss:.4f}", # 平滑损失数值
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}", # 提取调度器记录的实时学习率
                }
            )
        # 单样本训练子循环结束，核心量lora_module参数完成了一个Epoch的所有迭代更新，total_loss记录了全样本误差之和

        train_loss = total_loss / len(train_ds) # 计算当前轮的严格整体平均Loss

        eval_stats = None # 验证结果统计字典初始化为空
        if val_ds is not None: # 若包含合规验证集
            # 穿插验证流程，评估当前LoRA权重的泛化表现
            # 相比与train过程，基本类似，只是没有更新参数的过程
            eval_stats = evaluate(
                model,
                lora_module.get_lora_dict(), # 注入当下验证时的LoRA分布状态
                tokenizer,
                config,
                val_ds, # 使用预初始化的验证子集
                device,
                max_examples=args.max_val_samples, # 最多评估指定的验证样本数量
                verbose=False, # 关闭验证集内部明细打印
            )

        # 构建囊括了验证信息(如有)与训练基础Loss的纪实字典
        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            # 利用字典解包语法，在存在验证状态时动态嵌入评价键值对
            **({"val_loss": eval_stats["loss"], "val_acc": eval_stats["accuracy"]} if eval_stats else {}),
        }
        history.append(epoch_info) # 计入全程历史阵列

        # 将综合结论汇总打印输出至控制台
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            + (f"| val_loss={eval_stats['loss']:.4f} val_acc={eval_stats['accuracy']:.3f}" if eval_stats else "")
        )

        # 每个 epoch 结束后，如果当前模型在验证集上表现更好，则保存一份“最优” LoRA 权重。
        # 这样可避免最后一个 epoch 不一定是最优模型的情况。
        # *逻辑意图: 在验证集导向下实施Early-Stopping变种操作，固化最佳局部参数
        if eval_stats and eval_stats["accuracy"] > best_acc:
            best_acc = eval_stats["accuracy"] # 用更高指标覆写历史记录门槛
            save_path = os.path.join(args.save_dir, f"best_lora_epoch{epoch}.pt") # 构造当前最佳权重的唯一保存文件名
            os.makedirs(args.save_dir, exist_ok=True) # 确保存档的父层级目标目录畅通存在
            torch.save(lora_module.state_dict(), save_path) # 将最佳状态字典序列化写入本地文件系统
            print(f"  ✅ 保存最佳 LoRA 适配器: {save_path} (val_acc={best_acc:.3f})")
    # Epoch大循环收尾完成，最终best_acc定格最优解，模型历史全部汇聚于history变量中

    # 最终保存：无论是否提升，都会将当前 LoRA 权重保存为通用文件名
    print("[4/5] 开始保存 Loop...")
    final_path = os.path.join(args.save_dir, "medical_lora_adapter.pt") # 构造普适训练尾盘保存路径
    os.makedirs(args.save_dir, exist_ok=True) # 文件夹二次容错性断言创建
    torch.save(lora_module.state_dict(), final_path) # 把最后一个Step的LoRA权重的确切张量序列导出磁盘
    print(f"训练结束，LoRA 权重已保存: {final_path}")

    # 输出训练、验证历史，便于后续分析与记录
    print("训练历史:")
    for row in history: # 顺次遍历各层Epoch纪要打印日志
        print(row)
    
    print("[5/5] 谢谢老师...") # 礼貌谢幕


def parse_args():
    """
    1) *summary*：解析启动程序时控制台传递的所有命令行配置参数。
    2) 参数分析 (Args)：无参数接收，依赖系统传入 sys.argv。
    3) 返回值 (Returns)：返回 argparse.Namespace 对象，包含提取好的所有运行时配置属性。
    4) 变量维度分析：返回对象中存储的皆为标量数据（如数字、字符串）。
    5) 举例：若运行命令包含 --lr 0.001，则返回对象args具备args.lr = 0.001。
    """
    # 实例化参数解析器对象，附带使用说明文本
    parser = argparse.ArgumentParser(description="Moondream Medical VQA LoRA 训练脚本")
    # 添加预设和描述项：训练数据CSV映射
    parser.add_argument("--train-csv", type=str, default="PMC_VQA/test_2.csv", help="训练集 CSV 路径")
    # 添加预设和描述项：验证数据CSV映射
    parser.add_argument("--val-csv", type=str, default="PMC_VQA/test_2.csv", help="验证集 CSV 路径（可与训练集相同，或置空不做验证）")
    # 添加预设和描述项：图库文件读取源头
    parser.add_argument("--img-dir", type=str, default="PMC_VQA/images_2/figures", help="图片目录")
    # 添加预设和描述项：控制训练遍历全量数据集的总轮次
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    # 添加预设和描述项：优化器学习步长比例因子
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    # 添加预设和描述项：用于粗略调试防止训练过久的训练截断阈值
    parser.add_argument("--max-train-samples", type=int, default=2000, help="最多训练样本数")
    # 添加预设和描述项：验证同理对应的调试上限阈值
    parser.add_argument("--max-val-samples", type=int, default=2000, help="最多验证样本数")
    # 添加预设和描述项：结果输出模型状态保存目标地
    parser.add_argument("--save-dir", type=str, default=os.path.join(SCRIPT_DIR, "result"), help="LoRA 权重保存目录")
    return parser.parse_args() # 解析命名空间并打包返回


# 判断当前脚本是否作为主程序被直接运行
if __name__ == "__main__":
    args = parse_args() # 拦截参数集合，存入args控制上下文
    train(args) # 进入主流程调度模块运行全套微调架构