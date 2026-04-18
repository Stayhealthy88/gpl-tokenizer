"""
HMN (Hierarchical Multi-scale Normalization) 초기화기
=====================================================
GPL 어휘의 기하학적 구조를 임베딩 초기화에 반영.

핵심 원리:
    랜덤 초기화 대신, 토큰의 의미적·공간적 관계를 초기 가중치에 인코딩.
    - 좌표 토큰: 쿼드트리 위치 기반 — 가까운 좌표 → 유사한 초기 벡터
    - 명령어 토큰: 의미 그룹 기반 — 유사 명령어 → 유사한 초기 벡터
    - 연속성/곡률: 순서 기반 — 스칼라 속성의 순서 보존

이론적 근거:
    1. 수렴 가속: 초기 상태가 최종 해에 더 가까움 → 적은 에폭으로 수렴
    2. 기하학적 귀납 편향: 공간적 근접성이 임베딩 근접성으로 이어짐
    3. 스케일 적응: 쿼드트리 레벨별 해상도 차이를 임베딩 노름으로 반영

인접 좌표 코사인 유사도 ≈ 0.52 의 유도 (v0.5.1 문서화)
----------------------------------------------------
좌표 (qx, qy) 에서 (qx+1, qy) 로 이동할 때 임베딩 유사도가 약 0.52 가
되는 이유는 `_spatial_encoding` 의 사인파 인코딩 구조에서 나온다.

셀 중심 정규화 좌표:
    (nx, ny)   = ((qx + 0.5) / 2^L, (qy + 0.5) / 2^L)
    (nx', ny') = ((qx + 1.5) / 2^L, (qy + 0.5) / 2^L)
    Δnx = 1 / 2^L,  Δny = 0

인코딩 벡터(비정규화):
    v[2i]   = sin(nx · 2π · f_i)
    v[2i+1] = cos(ny · 2π · f_i)         where f_i = 2^(6i / (d/2))

차이는 x 성분(짝수 인덱스)에만 나타나며:
    v'[2i]   = sin((nx + Δnx) · 2π · f_i)
    v'[2i+1] = v[2i+1]                    (y 성분 동일)

내적:
    <v, v'> = Σ_i [ sin(a_i)·sin(a_i + 2π·f_i·Δnx)
                  + cos(b_i)·cos(b_i) ]                      # y 성분 항등
            = Σ_i [ sin(a_i)·sin(a_i + θ_i) + cos²(b_i) ]
            where a_i = nx·2π·f_i,  θ_i = 2π·f_i·Δnx,  b_i = ny·2π·f_i

위상 Δnx · 2π·f_i 가 작은 저주파 성분에서는 sin·sin 이 cos(θ_i)/2 에 가깝고,
고주파 성분(f_i 최대 ~64)에서는 θ_i 가 크게 벌어져 평균 0 으로 수렴.

d=128 기준 수치 계산:
    L=6, 짧은 이동(Δnx=1/64) → <v,v'>/||v||||v'|| ≈ 0.52

이는 test_embedding.py 에서 관측된 0.5217 과 일치한다. 자세한 유도는
`tests/test_embedding.py` 의 "HMN 인접 좌표 평균 유사도" 블록 참조.

비고
----
값 0.52 는 (a) 임베딩 차원 d, (b) ARCS 리프 레벨, (c) 주파수 스케줄에
의해 결정되는 설계 산물이며 목표값이 아니다. 즉 구현이 변경되면
수치도 달라진다. 중요한 것은 "random init(≈0) 보다 유의하게 높고
인접도에 따라 부드럽게 감소한다" 는 귀납 편향의 존재이다.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class HMNInitializer:
    """
    GPL 임베딩을 위한 HMN 초기화기.

    사용법:
        initializer = HMNInitializer(max_coord_level=6, d_model=256)
        initializer.initialize(embedding_weight, vocab)
    """

    def __init__(self, max_coord_level: int = 6, d_model: int = 256,
                 coord_scale: float = 1.0, noise_std: float = 0.02):
        """
        Args:
            max_coord_level: 최대 ARCS 좌표 레벨
            d_model: 임베딩 차원
            coord_scale: 좌표 임베딩 초기화 스케일
            noise_std: 초기화 노이즈 표준편차
        """
        self.max_level = max_coord_level
        self.d_model = d_model
        self.coord_scale = coord_scale
        self.noise_std = noise_std

    @torch.no_grad()
    def initialize(self, weight: torch.Tensor,
                    vocab: 'GPLVocabulary') -> None:
        """
        임베딩 가중치를 HMN 방식으로 초기화.

        Args:
            weight: nn.Embedding의 weight 텐서 (vocab_size, d_model)
            vocab: GPLVocabulary 인스턴스
        """
        # 기본 Xavier 초기화
        nn.init.xavier_uniform_(weight)

        # 1. 특수 토큰: PAD = 0벡터
        weight[0].zero_()  # PAD

        # 2. 명령어 토큰: 의미 그룹 클러스터링
        self._init_command_tokens(weight)

        # 3. 복합 도형 토큰: 도형 유형별 클러스터링
        self._init_composite_tokens(weight)

        # 4. 공간 관계 토큰: 관계 유형별 클러스터링
        self._init_spatial_tokens(weight)

        # 5. 연속성 토큰: 순서 보존 (DISC < G0 < G1 < G2)
        self._init_continuity_tokens(weight)

        # 6. 곡률 토큰: 순서 보존 (직선 → 급곡선)
        self._init_curvature_tokens(weight)

        # 7. 좌표 토큰: 쿼드트리 위치 기반 — 핵심 HMN 초기화
        self._init_coord_tokens(weight, vocab)

    def _init_command_tokens(self, weight: torch.Tensor) -> None:
        """명령어 토큰을 의미 그룹으로 초기화."""
        d = self.d_model

        # 기본 방향 벡터 (정규화)
        base = torch.randn(d) * 0.1

        # 이동 명령: MOVE(10)
        weight[10] = base + self._semantic_vec(d, 0)

        # 직선 계열: LINE(11), HLINE(12), VLINE(13) — 유사 벡터
        line_center = self._semantic_vec(d, 1)
        weight[11] = base + line_center
        weight[12] = base + line_center + torch.randn(d) * self.noise_std
        weight[13] = base + line_center + torch.randn(d) * self.noise_std

        # 곡선 계열: CUBIC(14), QUADRATIC(15) — 유사 벡터
        curve_center = self._semantic_vec(d, 2)
        weight[14] = base + curve_center
        weight[15] = base + curve_center + torch.randn(d) * self.noise_std

        # 호/닫기: ARC(16), CLOSE(17)
        weight[16] = base + self._semantic_vec(d, 3)
        weight[17] = base + self._semantic_vec(d, 4)

    def _init_composite_tokens(self, weight: torch.Tensor) -> None:
        """복합 도형 토큰을 도형 유사성으로 초기화."""
        d = self.d_model
        shape_base = self._semantic_vec(d, 5)

        # CIRCLE(20), ELLIPSE(21) — 유사 (곡면 도형)
        weight[20] = shape_base + torch.randn(d) * self.noise_std
        weight[21] = shape_base + torch.randn(d) * self.noise_std

        # RECT(22), ROUND_RECT(23) — 유사 (각진 도형)
        rect_center = self._semantic_vec(d, 6)
        weight[22] = rect_center + torch.randn(d) * self.noise_std
        weight[23] = rect_center + torch.randn(d) * self.noise_std

    def _init_spatial_tokens(self, weight: torch.Tensor) -> None:
        """공간 관계 토큰을 관계 유형별로 초기화."""
        d = self.d_model

        # 정렬 계열: ALIGN_CENTER_H(60), ALIGN_CENTER_V(61)
        align_base = self._semantic_vec(d, 7)
        weight[60] = align_base + torch.randn(d) * self.noise_std
        weight[61] = align_base + torch.randn(d) * self.noise_std

        # 대칭 계열: SYM_REFLECT_X(62), SYM_REFLECT_Y(63)
        sym_base = self._semantic_vec(d, 8)
        weight[62] = sym_base + torch.randn(d) * self.noise_std
        weight[63] = sym_base + torch.randn(d) * self.noise_std

        # 간격/크기/반복 계열: 64-70
        dist_base = self._semantic_vec(d, 9)
        for tid in range(64, 71):
            if tid < weight.size(0):
                weight[tid] = dist_base + torch.randn(d) * self.noise_std

    def _init_continuity_tokens(self, weight: torch.Tensor) -> None:
        """연속성 토큰을 순서 보존으로 초기화: DISC(30) < G0(31) < G1(32) < G2(33)."""
        d = self.d_model
        cont_base = self._semantic_vec(d, 10)

        for i, tid in enumerate([30, 31, 32, 33]):
            # 순서 보존: 레벨이 높을수록 벡터가 한 방향으로 이동
            scale = i / 3.0  # 0.0, 0.33, 0.67, 1.0
            direction = self._semantic_vec(d, 11)
            weight[tid] = cont_base + direction * scale * 0.5

    def _init_curvature_tokens(self, weight: torch.Tensor) -> None:
        """곡률 토큰을 순서 보존으로 초기화: κ0(직선) → κ15(급곡선)."""
        d = self.d_model
        curv_base = self._semantic_vec(d, 12)
        curv_direction = self._semantic_vec(d, 13)

        for i in range(16):
            tid = 40 + i
            scale = i / 15.0  # 0.0 ~ 1.0
            weight[tid] = curv_base + curv_direction * scale * 0.5

    def _init_coord_tokens(self, weight: torch.Tensor,
                            vocab: 'GPLVocabulary') -> None:
        """
        좌표 토큰을 쿼드트리 위치 기반으로 초기화 — HMN의 핵심.

        원리:
            각 좌표 토큰 (level, qx, qy)에 대해:
            1. 정규화 위치 (nx, ny) = (qx/2^level, qy/2^level) ∈ [0,1)²
            2. 다중 스케일 사인파 인코딩으로 위치 벡터 생성
            3. 레벨별 스케일 조절: 높은 레벨(세밀) → 작은 노름

        결과:
            - 공간적으로 가까운 좌표 → 코사인 유사도 높음
            - 같은 레벨의 좌표들은 비슷한 노름
            - 학습 전에도 기하학적 귀납 편향이 존재
        """
        d = self.d_model
        half_d = d // 2

        for level in range(self.max_level + 1):
            grid_size = 2 ** level
            # 레벨별 스케일: 세밀한 레벨은 더 작은 초기 노름
            level_scale = self.coord_scale * (0.8 ** level)

            for qy in range(grid_size):
                for qx in range(grid_size):
                    tid = vocab.coord_to_id(level, qx, qy)
                    if tid is None or tid >= weight.size(0):
                        continue

                    # 정규화 좌표
                    nx = (qx + 0.5) / grid_size  # 셀 중심
                    ny = (qy + 0.5) / grid_size

                    # 다중 스케일 사인파 위치 인코딩
                    pos_enc = self._spatial_encoding(nx, ny, d)

                    # 레벨 인코딩 (레벨 정보도 벡터에 포함)
                    level_enc = torch.zeros(d)
                    level_idx = min(level, d - 1)
                    level_enc[level_idx] = 1.0

                    # 최종 초기화: 위치 인코딩 * 레벨 스케일 + 레벨 마커 + 노이즈
                    weight[tid] = (
                        pos_enc * level_scale +
                        level_enc * 0.1 +
                        torch.randn(d) * self.noise_std * 0.5
                    )

    def _spatial_encoding(self, nx: float, ny: float, d: int) -> torch.Tensor:
        """
        2D 공간 위치를 사인파 인코딩으로 벡터화.

        (nx, ny) ∈ [0,1]² → d차원 벡터

        사인/코사인 주파수가 다중 스케일로 공간 위치를 인코딩:
            enc[2i]   = sin(nx * 2π * freq_i)
            enc[2i+1] = cos(ny * 2π * freq_i)
        """
        enc = torch.zeros(d)
        half_d = d // 2

        for i in range(half_d):
            freq = 2.0 ** (i * 6.0 / half_d)  # 주파수: 1 ~ 64

            enc[2 * i] = math.sin(nx * 2 * math.pi * freq)
            enc[2 * i + 1] = math.cos(ny * 2 * math.pi * freq)

        # L2 정규화
        norm = enc.norm()
        if norm > 0:
            enc = enc / norm

        return enc

    def _semantic_vec(self, d: int, seed: int) -> torch.Tensor:
        """재현 가능한 시맨틱 방향 벡터 생성."""
        gen = torch.Generator()
        gen.manual_seed(42 + seed * 7)
        vec = torch.randn(d, generator=gen)
        return vec / vec.norm() * 0.3
