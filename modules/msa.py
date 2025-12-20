import math
import torch
import torch.nn as nn

import torch
import torch.nn as nn

# 3. 前馈神经网络 (Position-wise Feed Forward)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class MSA(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.d = d
        self.n = n

        self.w_q = nn.Linear(d, d)
        self.w_k = nn.Linear(d, d)
        self.w_v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

        self.scale = (self.d//n) ** -0.5
		
    def forward(self, q, k, v, mask=None):
        """
        x: [b, l, d]
        return: [b, l, d]
        """
        b, l = q.shape[0], q.shape[1]
        q = self.w_q(q).reshape(b, l, self.n, self.d//self.n).transpose(1, 2) # [b, n, l, d/n]
        k = self.w_k(k).reshape(b, l, self.n, self.d//self.n).transpose(1, 2) # [b, n, l, d/n]
        v = self.w_v(v).reshape(b, l, self.n, self.d//self.n).transpose(1, 2) # [b, n, l, d/n]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [b, n, l, l]
        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, l, self.d)# [b, l, d]
        out = self.out(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MSA(d_model, n_head)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Self Attention + Residual + Norm
        # Add & Norm 的标准顺序是 x + dropout(sublayer(norm(x))) (Pre-Norm)
        # 或者 norm(x + dropout(sublayer(x))) (Post-Norm, 原论文写法)
        # 这里使用原论文写法: Post-Norm
        
        _x = x
        x = self.self_attn(x, x, x, mask)
        x = self.norm1(_x + self.dropout(x))
        
        # 2. FFN + Residual + Norm
        _x = x
        x = self.ffn(x)
        x = self.norm2(_x + self.dropout(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MSA(d_model, n_head)
        self.cross_attn = MSA(d_model, n_head)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 1. Masked Self Attention (只看过去)
        _x = x
        x = self.self_attn(x, x, x, tgt_mask) # Q=K=V=x
        x = self.norm1(_x + self.dropout(x))

        # 2. Cross Attention (Q来自解码器, K,V来自编码器)
        _x = x
        x = self.cross_attn(x, enc_output, enc_output, src_mask) # Q=x, K=V=enc_output
        x = self.norm2(_x + self.dropout(x))

        # 3. FFN
        _x = x
        x = self.ffn(x)
        x = self.norm3(_x + self.dropout(x))
        return x


# 1. 位置编码 (这是 Transformer 不可或缺的部分，否则它不知道词序)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建一个常量矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [max_len, d_model] -> [1, max_len, d_model] 方便广播
        pe = pe.unsqueeze(0)
        
        # register_buffer 告诉 PyTorch 这是一个状态，但不是需要更新梯度的参数
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # 加上位置编码
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_head=8, num_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        
        # Embedding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Encoder & Decoder Stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # Final Output
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: [batch, src_len]
        tgt: [batch, tgt_len]
        """
        
        # 1. Embeddings + Positional Encoding
        src = self.dropout(self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt = self.dropout(self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))

        # 2. Encoder Forward
        for layer in self.encoder_layers:
            src = layer(src, src_mask) # src 变成了 enc_output

        enc_output = src 

        # 3. Decoder Forward
        for layer in self.decoder_layers:
            tgt = layer(tgt, enc_output, src_mask, tgt_mask)

        # 4. Final Projection
        output = self.fc_out(tgt)
        return output
    

def make_pad_mask(q, k, pad_idx=0):
    # q: [batch, q_len], k: [batch, k_len]
    len_q, len_k = q.size(1), k.size(1)
    
    # [batch, 1, 1, k_len] 
    # 这里我们只关心 key 也就是输入部分的 padding
    k = k.ne(pad_idx).unsqueeze(1).unsqueeze(2)
    # 广播后形状: [batch, 1, len_q, len_k]
    return k.repeat(1, 1, len_q, 1)

def make_subsequent_mask(seq_len):
    # 生成下三角矩阵，上三角（未来）全是 0
    mask = torch.tril(torch.ones(seq_len, seq_len)).type(torch.ByteTensor)
    return mask # [seq_len, seq_len]

if __name__ == "__main__":
    # 参数设置
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 512
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model)

    # 模拟输入数据 (Batch=2, Len=5)
    src = torch.tensor([[1, 5, 6, 4, 3], [1, 5, 0, 0, 0]]) # 0 是 padding
    tgt = torch.tensor([[1, 7, 4, 3, 2], [1, 7, 4, 0, 0]])

    # 生成 Mask
    # Src Mask: 只要把 src 里的 padding 遮住
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # [2, 1, 1, 5]
    
    # Tgt Mask: 需要同时遮住 padding 和 未来的词
    tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    tgt_sub_mask = make_subsequent_mask(tgt.size(1))
    tgt_mask = tgt_pad_mask & tgt_sub_mask # 逻辑与运算

    # 前向传播
    out = model(src, tgt, src_mask, tgt_mask)
    
    print("Output shape:", out.shape) # 应该是 [2, 5, 100]
    
# if __name__ == "__main__":
#     model = MSA(512, 8)
#     x = torch.randn(2, 10, 512)
#     out = model(x, x, x)
#     print(out.shape)