"""
GPL 임베딩 레이어
=================
GPL 토큰 시퀀스를 밀집 벡터 시퀀스로 변환하는 PyTorch 모듈.

v0.4 핵심:
    일반적인 nn.Embedding과 달리, GPL의 기하학적 구조를 활용:
    1. HMN 초기화로 공간적 관계를 초기 가중치에 반영
    2. 토큰 유형 임베딩으로 명령어/좌표/속성을 구분
    3. 좌표 구조 인코딩으로 쿼드트리 위치 정보를 추가
    4. 사인파 위치 인코딩으로 시퀀스 내 순서 반영

출력: d_model 차원 벡터 시퀀스 → Transformer/LLM 입력으로 직접 사용 가능
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn

from .hmn_init import HMNInitializer


# 토큰 ID → 유형 인덱스 매핑 함수
def _token_type_id(token_id: int) -> int:
    """토큰 ID에서 유형 인덱스를 결정."""
    if token_id < 10:
        return 0   # special
    elif 10 <= token_id < 20:
        return 1   # command
    elif 20 <= token_id < 30:
        return 2   # composite
    elif 30 <= token_id < 34:
        return 3   # continuity
    elif 40 <= token_id < 56:
        return 4   # curvature
    elif 60 <= token_id < 80:
        return 5   # spatial
    else:
        return 6   # coord (100+)


# 토큰 유형 수
N_TOKEN_TYPES = 7


class PositionalEncoding(nn.Module):
    """
    고정 사인파 위치 인코딩.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CoordStructureEncoder(nn.Module):
    """
    좌표 토큰의 쿼드트리 구조를 인코딩.

    각 좌표 토큰 (level, qx, qy) → (level_norm, nx, ny) → d_coord 차원 벡터.
    비좌표 토큰에는 영벡터 출력.

    이를 통해 모델이 좌표의 공간적 의미를 직접 인식 가능.
    """

    def __init__(self, d_coord: int, max_coord_level: int = 6):
        super().__init__()
        self.d_coord = d_coord
        self.max_level = max_coord_level

        # (level_norm, nx, ny) → d_coord
        self.encoder = nn.Sequential(
            nn.Linear(3, d_coord),
            nn.GELU(),
            nn.Linear(d_coord, d_coord),
        )

        # 좌표 정보 룩업 테이블 구축
        self._build_coord_lut()

    def _build_coord_lut(self):
        """좌표 토큰 ID → (level_norm, nx, ny) 룩업 테이블."""
        from ..tokenizer.vocabulary import COORD_TOKEN_BASE
        max_id = COORD_TOKEN_BASE
        for level in range(self.max_level + 1):
            max_id += (2 ** level) ** 2

        # (max_id, 3) 테이블: [level_norm, nx, ny]
        lut = torch.zeros(max_id, 3)

        current_id = COORD_TOKEN_BASE
        for level in range(self.max_level + 1):
            grid_size = 2 ** level
            level_norm = level / max(self.max_level, 1)

            for qy in range(grid_size):
                for qx in range(grid_size):
                    nx = (qx + 0.5) / grid_size
                    ny = (qy + 0.5) / grid_size
                    lut[current_id] = torch.tensor([level_norm, nx, ny])
                    current_id += 1

        self.register_buffer('coord_lut', lut)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) — GPL 토큰 ID
        Returns:
            (batch, seq_len, d_coord) — 좌표 구조 인코딩
        """
        batch_size, seq_len = token_ids.shape

        # 유효 좌표 토큰 마스크
        from ..tokenizer.vocabulary import COORD_TOKEN_BASE
        is_coord = token_ids >= COORD_TOKEN_BASE

        # 좌표 정보 조회 (범위 밖은 0으로 클램프)
        safe_ids = token_ids.clamp(0, self.coord_lut.size(0) - 1)
        coord_features = self.coord_lut[safe_ids]  # (batch, seq, 3)

        # 비좌표 토큰은 0으로 마스킹
        coord_features = coord_features * is_coord.unsqueeze(-1).float()

        # MLP 인코딩
        encoded = self.encoder(coord_features)  # (batch, seq, d_coord)

        return encoded


class GPLEmbedding(nn.Module):
    """
    GPL 토큰 시퀀스를 밀집 벡터로 변환하는 임베딩 레이어.

    구성요소:
        1. token_embedding: HMN 초기화된 메인 임베딩 (vocab_size, d_model)
        2. type_embedding: 토큰 유형별 임베딩 (7, d_type)
        3. coord_encoder: 좌표 구조 인코딩 (좌표 토큰만)
        4. positional_encoding: 시퀀스 위치 인코딩
        5. projection: 모든 요소 결합 → d_model

    사용법:
        vocab = GPLVocabulary(max_coord_level=6)
        emb = GPLEmbedding(vocab, d_model=256)
        token_ids = torch.tensor([[1, 10, 150, 31, 42, 2]])
        output = emb(token_ids)  # (1, 6, 256)
    """

    def __init__(self,
                 vocab: 'GPLVocabulary',
                 d_model: int = 256,
                 d_type: int = 32,
                 d_coord: int = 64,
                 max_seq_len: int = 512,
                 dropout: float = 0.1,
                 use_hmn_init: bool = True):
        """
        Args:
            vocab: GPLVocabulary 인스턴스
            d_model: 최종 임베딩 차원
            d_type: 토큰 유형 임베딩 차원
            d_coord: 좌표 구조 인코딩 차원
            max_seq_len: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
            use_hmn_init: HMN 초기화 사용 여부
        """
        super().__init__()

        self.vocab = vocab
        self.d_model = d_model
        self.d_type = d_type
        self.d_coord = d_coord

        # 1. 메인 토큰 임베딩
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab.vocab_size,
            embedding_dim=d_model,
            padding_idx=0,  # PAD = 0
        )

        # 2. 토큰 유형 임베딩
        self.type_embedding = nn.Embedding(
            num_embeddings=N_TOKEN_TYPES,
            embedding_dim=d_type,
        )

        # 3. 좌표 구조 인코딩
        self.coord_encoder = CoordStructureEncoder(
            d_coord=d_coord,
            max_coord_level=vocab.max_coord_level,
        )

        # 4. 위치 인코딩
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # 5. 결합 프로젝션: (d_model + d_type + d_coord) → d_model
        self.projection = nn.Linear(d_model + d_type + d_coord, d_model)

        # 6. 레이어 정규화 + 드롭아웃
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 7. 토큰 유형 매핑 테이블 사전 구축
        type_ids = torch.zeros(vocab.vocab_size, dtype=torch.long)
        for tid in range(vocab.vocab_size):
            type_ids[tid] = _token_type_id(tid)
        self.register_buffer('type_id_lut', type_ids)

        # 8. HMN 초기화
        if use_hmn_init:
            initializer = HMNInitializer(
                max_coord_level=vocab.max_coord_level,
                d_model=d_model,
            )
            initializer.initialize(self.token_embedding.weight, vocab)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        GPL 토큰 시퀀스를 밀집 벡터 시퀀스로 변환.

        Args:
            token_ids: (batch_size, seq_len) — GPL 토큰 ID 텐서
        Returns:
            (batch_size, seq_len, d_model) — 임베딩 벡터 시퀀스
        """
        # 범위 클램핑 (안전)
        safe_ids = token_ids.clamp(0, self.vocab.vocab_size - 1)

        # 1. 메인 토큰 임베딩
        tok_emb = self.token_embedding(safe_ids)  # (B, L, d_model)

        # 2. 토큰 유형 임베딩
        type_ids = self.type_id_lut[safe_ids]  # (B, L)
        typ_emb = self.type_embedding(type_ids)  # (B, L, d_type)

        # 3. 좌표 구조 인코딩
        coord_emb = self.coord_encoder(safe_ids)  # (B, L, d_coord)

        # 4. 결합 + 프로젝션
        combined = torch.cat([tok_emb, typ_emb, coord_emb], dim=-1)
        projected = self.projection(combined)  # (B, L, d_model)

        # 5. 위치 인코딩 + 정규화
        output = self.positional_encoding(projected)
        output = self.layer_norm(output)

        return output

    def get_embedding_stats(self) -> Dict[str, float]:
        """임베딩 통계 정보 반환."""
        w = self.token_embedding.weight.detach()
        return {
            "vocab_size": w.size(0),
            "d_model": self.d_model,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "emb_mean_norm": w.norm(dim=-1).mean().item(),
            "emb_std_norm": w.norm(dim=-1).std().item(),
            "coord_emb_mean_norm": w[100:].norm(dim=-1).mean().item(),
            "special_emb_mean_norm": w[:5].norm(dim=-1).mean().item(),
        }
