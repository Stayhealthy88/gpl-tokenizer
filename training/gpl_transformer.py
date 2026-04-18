"""
GPL Transformer 모델
=====================
GPL 토큰 시퀀스를 생성하는 Decoder-only Transformer.

아키텍처:
    GPLEmbedding → N × TransformerDecoderLayer → LM Head

Micro 모델 (1-5M params):
    d_model=128, n_heads=4, n_layers=4, d_ff=512
    → ~2.5M params (임베딩 포함)

학습 목표: Next-token prediction (causal LM)
    P(token_t | token_1, ..., token_{t-1})
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embedding.gpl_embedding import GPLEmbedding


@dataclass
class GPLTransformerConfig:
    """모델 설정."""
    # 임베딩
    d_model: int = 128
    d_type: int = 16
    d_coord: int = 32

    # Transformer
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 128

    # HMN 초기화
    use_hmn_init: bool = True

    def param_estimate(self, vocab_size: int) -> int:
        """파라미터 수 추정."""
        emb = vocab_size * self.d_model  # token embedding
        emb += 7 * self.d_type           # type embedding
        emb += self.d_coord * 6 + 6      # coord encoder MLP
        proj = (self.d_model + self.d_type + self.d_coord) * self.d_model  # projection

        # Transformer layer × n_layers
        attn = 4 * self.d_model * self.d_model  # Q, K, V, O
        ff = 2 * self.d_model * self.d_ff       # FFN
        ln = 2 * self.d_model                   # LayerNorm ×2
        layer = attn + ff + ln
        total_tf = layer * self.n_layers

        # LM Head
        head = vocab_size * self.d_model

        return emb + proj + total_tf + head


class GPLTransformer(nn.Module):
    """
    GPL 토큰 시퀀스 생성용 Decoder-only Transformer.

    구조:
        GPLEmbedding → Causal Transformer Decoder → Linear → logits

    사용법:
        config = GPLTransformerConfig()
        model = GPLTransformer(vocab, config)
        logits = model(input_ids)  # (B, L, vocab_size)
    """

    def __init__(self, vocab: 'GPLVocabulary', config: GPLTransformerConfig):
        super().__init__()
        self.config = config
        self.vocab = vocab

        # 1. GPLEmbedding (v0.4)
        self.embedding = GPLEmbedding(
            vocab=vocab,
            d_model=config.d_model,
            d_type=config.d_type,
            d_coord=config.d_coord,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            use_hmn_init=config.use_hmn_init,
        )

        # 2. Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.n_layers,
        )

        # 3. LM Head
        self.lm_head = nn.Linear(config.d_model, vocab.vocab_size, bias=False)

        # Weight tying: LM head = embedding weight transpose
        # (임베딩 가중치 공유로 파라미터 절감)
        self.lm_head.weight = self.embedding.token_embedding.weight

        # 4. Causal mask 캐시
        self._register_causal_mask(config.max_seq_len)

        # 파라미터 초기화
        self._init_weights()

    def _register_causal_mask(self, max_len: int):
        """인과적 어텐션 마스크 사전 생성."""
        mask = torch.triu(
            torch.ones(max_len, max_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer('causal_mask', mask)

    def _init_weights(self):
        """LM Head 제외 가중치 초기화 (임베딩은 HMN이 처리)."""
        for name, param in self.named_parameters():
            if 'embedding' in name:
                continue  # GPLEmbedding은 HMN이 처리
            if 'lm_head' in name:
                continue  # weight tying
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) GPL 토큰 ID
            attention_mask: (batch, seq_len) 패딩 마스크 (1=valid, 0=pad)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, L = input_ids.shape

        # 1. 임베딩
        x = self.embedding(input_ids)  # (B, L, d_model)

        # 2. Causal mask
        causal = self.causal_mask[:L, :L]

        # 3. Key padding mask
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True = 무시
        else:
            key_padding_mask = None

        # 4. Transformer decoder (self-attention only, no cross-attention)
        # decoder requires memory input — use self as memory for decoder-only
        memory = torch.zeros(B, 1, self.config.d_model, device=x.device)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=key_padding_mask,
        )

        # 5. LM Head
        logits = self.lm_head(x)  # (B, L, vocab_size)

        return logits

    def compute_loss(self,
                     input_ids: torch.Tensor,
                     target_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None,
                     ) -> Dict[str, torch.Tensor]:
        """
        학습용 손실 계산.

        Args:
            input_ids: (B, L) 입력 토큰
            target_ids: (B, L) 목표 토큰 (input_ids를 1칸 shift)
            attention_mask: (B, L) 패딩 마스크
        Returns:
            {"loss": scalar, "logits": (B, L, V), "accuracy": scalar}
        """
        logits = self.forward(input_ids, attention_mask)

        # Cross-entropy loss (PAD 무시)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=0,  # PAD = 0
        )

        # 정확도 계산 (PAD 제외)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = target_ids != 0
            correct = ((preds == target_ids) & mask).sum()
            total = mask.sum()
            accuracy = correct.float() / total.float() if total > 0 else torch.tensor(0.0)

        return {
            "loss": loss,
            "logits": logits,
            "accuracy": accuracy,
        }

    def count_parameters(self) -> Dict[str, int]:
        """파라미터 수 집계."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for n, p in self.named_parameters() if 'embedding' in n)
        decoder_params = sum(p.numel() for n, p in self.named_parameters() if 'decoder' in n)
        return {
            "total": total,
            "trainable": trainable,
            "embedding": emb_params,
            "decoder": decoder_params,
        }

    @torch.no_grad()
    def generate(self,
                 prompt: torch.Tensor,
                 max_len: int = 64,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 eos_id: int = 2,
                 ) -> torch.Tensor:
        """
        토큰 시퀀스 생성 (autoregressive).

        Args:
            prompt: (1, prompt_len) 시작 토큰
            max_len: 최대 생성 길이
            temperature: 샘플링 온도
            top_k: top-k 필터링
            top_p: nucleus sampling threshold
            eos_id: 종료 토큰 ID
        Returns:
            (1, generated_len) 생성된 토큰 시퀀스
        """
        self.eval()
        generated = prompt.clone()

        for _ in range(max_len):
            # 최대 길이 제한
            if generated.size(1) >= self.config.max_seq_len:
                break

            logits = self.forward(generated)
            next_logits = logits[:, -1, :] / temperature

            # Top-k 필터링
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                min_topk = topk_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < min_topk] = float('-inf')

            # Nucleus (top-p) 필터링
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumsum - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = float('-inf')
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_id:
                break

        return generated
