"""
합성 SVG 데이터셋 생성기
========================
프로그래밍으로 기본 도형/패턴 SVG를 생성하고,
GPL 토큰 시퀀스로 변환하여 PyTorch Dataset 제공.

생성 패턴:
    1. 단일 도형: 원, 사각형, 타원, 직선, 곡선
    2. 복합 경로: 연결된 선분/곡선
    3. 다중 요소: 여러 도형 조합
    4. 기하학 패턴: 정렬, 대칭, 등간격 배치

외부 데이터 불필요 — 모든 학습 데이터를 코드로 생성.
"""

import math
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ..tokenizer.vocabulary import (
    GPLVocabulary, SpecialToken, CommandToken, CompositeToken,
    ContinuityToken, CURVATURE_TOKEN_BASE, COORD_TOKEN_BASE,
)
from ..tokenizer.arcs import ARCS


@dataclass
class SyntheticSample:
    """합성 SVG 샘플."""
    svg_desc: str           # 사람이 읽을 수 있는 설명
    token_ids: List[int]    # GPL 토큰 ID 시퀀스
    category: str           # 카테고리 (circle, rect, path, multi, ...)


class SyntheticSVGGenerator:
    """
    기본 도형과 패턴을 GPL 토큰 시퀀스로 직접 생성.

    SVG 파싱 없이, 토큰 레벨에서 직접 시퀀스를 구성.
    → 파서/분석기 의존 없이 순수 토큰 생성.
    """

    def __init__(self, vocab: GPLVocabulary, arcs: ARCS,
                 max_coord_level: int = 6, seed: int = 42):
        self.vocab = vocab
        self.arcs = arcs
        self.max_level = max_coord_level
        self.rng = random.Random(seed)

    def _rand_coord_id(self, level: int = None) -> Optional[int]:
        """랜덤 좌표 토큰 ID 생성."""
        if level is None:
            level = self.rng.randint(2, min(5, self.max_level))
        grid = 2 ** level
        qx = self.rng.randint(0, grid - 1)
        qy = self.rng.randint(0, grid - 1)
        return self.vocab.coord_to_id(level, qx, qy)

    def _coord_id_at(self, x: float, y: float, level: int = 4) -> Tuple[int, int]:
        """정규화 좌표 (0~1) → 좌표 토큰 ID 쌍."""
        grid = 2 ** level
        qx = max(0, min(grid - 1, int(x * grid)))
        qy = max(0, min(grid - 1, int(y * grid)))
        idx = self.vocab.coord_to_id(level, qx, qy)
        idy = self.vocab.coord_to_id(level, qy, qy)  # y좌표용
        return idx, idy

    def _continuity(self, level: int = 0) -> int:
        """연속성 토큰."""
        return [30, 31, 32, 33][min(level, 3)]

    def _curvature(self, kappa: int = 0) -> int:
        """곡률 토큰."""
        return CURVATURE_TOKEN_BASE + max(0, min(15, kappa))

    # ─── 단일 도형 생성 ───

    def gen_circle(self) -> SyntheticSample:
        """원: [BOS][CIRCLE][cx][cy][r][EOS]"""
        level = self.rng.randint(3, 5)
        grid = 2 ** level
        cx = self.rng.randint(1, grid - 2)
        cy = self.rng.randint(1, grid - 2)
        r = self.rng.randint(0, min(cx, cy, grid - 1 - cx, grid - 1 - cy))
        r = max(1, min(r, grid // 4))

        tokens = [
            SpecialToken.BOS,
            CompositeToken.CIRCLE,
            self.vocab.coord_to_id(level, cx, cy),
            self.vocab.coord_to_id(level, r, 0),  # 반지름은 qx에 인코딩
            SpecialToken.EOS,
        ]
        tokens = [t for t in tokens if t is not None]
        return SyntheticSample("circle", tokens, "circle")

    def gen_rect(self) -> SyntheticSample:
        """사각형: [BOS][RECT][x][y][w][h][EOS]"""
        level = self.rng.randint(3, 5)
        grid = 2 ** level
        x = self.rng.randint(0, grid // 2)
        y = self.rng.randint(0, grid // 2)
        w = self.rng.randint(1, grid // 2)
        h = self.rng.randint(1, grid // 2)

        tokens = [
            SpecialToken.BOS,
            CompositeToken.RECT,
            self.vocab.coord_to_id(level, x, y),
            self.vocab.coord_to_id(level, w, h),
            SpecialToken.EOS,
        ]
        tokens = [t for t in tokens if t is not None]
        return SyntheticSample("rect", tokens, "rect")

    def gen_ellipse(self) -> SyntheticSample:
        """타원: [BOS][ELLIPSE][cx][cy][rx][ry][EOS]"""
        level = self.rng.randint(3, 5)
        grid = 2 ** level
        cx = self.rng.randint(2, grid - 3)
        cy = self.rng.randint(2, grid - 3)
        rx = self.rng.randint(1, grid // 4)
        ry = self.rng.randint(1, grid // 4)

        tokens = [
            SpecialToken.BOS,
            CompositeToken.ELLIPSE,
            self.vocab.coord_to_id(level, cx, cy),
            self.vocab.coord_to_id(level, rx, ry),
            SpecialToken.EOS,
        ]
        tokens = [t for t in tokens if t is not None]
        return SyntheticSample("ellipse", tokens, "ellipse")

    # ─── 경로 기반 도형 ───

    def gen_line_path(self) -> SyntheticSample:
        """직선 경로: [BOS][MOVE][p0][LINE][p1][G0|G1][κ0]...[CLOSE/EOS]"""
        level = self.rng.randint(3, 5)
        grid = 2 ** level
        n_segments = self.rng.randint(2, 5)

        tokens = [SpecialToken.BOS]

        # 시작점
        sx = self.rng.randint(0, grid - 1)
        sy = self.rng.randint(0, grid - 1)
        tokens.append(CommandToken.MOVE)
        tokens.append(self.vocab.coord_to_id(level, sx, sy))

        for i in range(n_segments):
            ex = self.rng.randint(0, grid - 1)
            ey = self.rng.randint(0, grid - 1)
            tokens.append(CommandToken.LINE)
            tokens.append(self.vocab.coord_to_id(level, ex, ey))
            # 연속성 + 곡률
            cont = self.rng.choice([30, 31])  # DISC or G0
            tokens.append(cont)
            tokens.append(CURVATURE_TOKEN_BASE)  # κ0 (직선)

        close = self.rng.random() < 0.4
        if close:
            tokens.append(CommandToken.CLOSE)
        tokens.append(SpecialToken.EOS)

        tokens = [t for t in tokens if t is not None]
        return SyntheticSample(f"line_path({n_segments})", tokens, "path")

    def gen_curve_path(self) -> SyntheticSample:
        """곡선 경로: [BOS][MOVE][p0][CUBIC][cp1][cp2][end][G1/G2][κ]...[EOS]"""
        level = self.rng.randint(3, 5)
        grid = 2 ** level
        n_curves = self.rng.randint(1, 3)

        tokens = [SpecialToken.BOS]

        sx = self.rng.randint(0, grid - 1)
        sy = self.rng.randint(0, grid - 1)
        tokens.append(CommandToken.MOVE)
        tokens.append(self.vocab.coord_to_id(level, sx, sy))

        for i in range(n_curves):
            tokens.append(CommandToken.CUBIC)
            # 3개 좌표: cp1, cp2, end
            for _ in range(3):
                px = self.rng.randint(0, grid - 1)
                py = self.rng.randint(0, grid - 1)
                tokens.append(self.vocab.coord_to_id(level, px, py))
            # 연속성 + 곡률
            cont = self.rng.choice([31, 32, 33])  # G0/G1/G2
            kappa = self.rng.randint(1, 12)
            tokens.append(cont)
            tokens.append(CURVATURE_TOKEN_BASE + kappa)

        tokens.append(SpecialToken.EOS)
        tokens = [t for t in tokens if t is not None]
        return SyntheticSample(f"curve_path({n_curves})", tokens, "path")

    def gen_mixed_path(self) -> SyntheticSample:
        """혼합 경로: 직선 + 곡선 혼합."""
        level = self.rng.randint(3, 5)
        grid = 2 ** level
        n_seg = self.rng.randint(3, 6)

        tokens = [SpecialToken.BOS]

        sx = self.rng.randint(0, grid - 1)
        sy = self.rng.randint(0, grid - 1)
        tokens.append(CommandToken.MOVE)
        tokens.append(self.vocab.coord_to_id(level, sx, sy))

        for i in range(n_seg):
            if self.rng.random() < 0.5:
                # 직선
                tokens.append(CommandToken.LINE)
                ex = self.rng.randint(0, grid - 1)
                ey = self.rng.randint(0, grid - 1)
                tokens.append(self.vocab.coord_to_id(level, ex, ey))
                tokens.append(self.rng.choice([30, 31]))
                tokens.append(CURVATURE_TOKEN_BASE)
            else:
                # 곡선
                tokens.append(CommandToken.CUBIC)
                for _ in range(3):
                    px = self.rng.randint(0, grid - 1)
                    py = self.rng.randint(0, grid - 1)
                    tokens.append(self.vocab.coord_to_id(level, px, py))
                tokens.append(self.rng.choice([31, 32, 33]))
                tokens.append(CURVATURE_TOKEN_BASE + self.rng.randint(1, 10))

        if self.rng.random() < 0.3:
            tokens.append(CommandToken.CLOSE)
        tokens.append(SpecialToken.EOS)

        tokens = [t for t in tokens if t is not None]
        return SyntheticSample(f"mixed_path({n_seg})", tokens, "path")

    # ─── 다중 요소 ───

    def gen_multi_shapes(self) -> SyntheticSample:
        """다중 도형: [BOS][shape1][SEP][shape2][SEP]...[EOS]"""
        n_shapes = self.rng.randint(2, 4)
        generators = [self.gen_circle, self.gen_rect, self.gen_ellipse]

        tokens = [SpecialToken.BOS]

        for i in range(n_shapes):
            gen = self.rng.choice(generators)
            sample = gen()
            # BOS/EOS 제거하고 내부 토큰만 추가
            inner = [t for t in sample.token_ids
                     if t != SpecialToken.BOS and t != SpecialToken.EOS]
            tokens.extend(inner)
            if i < n_shapes - 1:
                tokens.append(SpecialToken.SEP)

        tokens.append(SpecialToken.EOS)
        return SyntheticSample(f"multi({n_shapes})", tokens, "multi")

    # ─── 배치 생성 ───

    def generate_batch(self, n: int) -> List[SyntheticSample]:
        """n개의 랜덤 합성 샘플 생성."""
        generators = [
            (self.gen_circle, 0.15),
            (self.gen_rect, 0.15),
            (self.gen_ellipse, 0.10),
            (self.gen_line_path, 0.20),
            (self.gen_curve_path, 0.15),
            (self.gen_mixed_path, 0.15),
            (self.gen_multi_shapes, 0.10),
        ]
        gens, weights = zip(*generators)

        samples = []
        for _ in range(n):
            gen = self.rng.choices(gens, weights=weights, k=1)[0]
            samples.append(gen())
        return samples


class SyntheticSVGDataset(Dataset):
    """
    합성 SVG GPL 토큰 시퀀스 데이터셋.

    사용법:
        vocab = GPLVocabulary(max_coord_level=6)
        arcs = ARCS(max_level=6)
        dataset = SyntheticSVGDataset(vocab, arcs, n_samples=10000)
        loader = DataLoader(dataset, batch_size=32, collate_fn=SVGCollator(vocab))
    """

    def __init__(self, vocab: GPLVocabulary, arcs: ARCS,
                 n_samples: int = 10000, max_seq_len: int = 128,
                 seed: int = 42):
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        gen = SyntheticSVGGenerator(vocab, arcs, seed=seed)
        self.samples = gen.generate_batch(n_samples)

        # 최대 길이 초과 필터링
        self.samples = [s for s in self.samples
                        if len(s.token_ids) <= max_seq_len]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        token_ids = torch.tensor(sample.token_ids, dtype=torch.long)
        return {
            "token_ids": token_ids,
            "category": sample.category,
        }


class SVGCollator:
    """
    가변 길이 시퀀스를 배치로 패딩.

    입력: List[Dict] from SyntheticSVGDataset
    출력: Dict[str, Tensor] — input_ids, target_ids, attention_mask
    """

    def __init__(self, vocab: GPLVocabulary, max_seq_len: int = 128):
        self.pad_id = SpecialToken.PAD
        self.max_seq_len = max_seq_len

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        token_lists = [item["token_ids"] for item in batch]
        max_len = min(max(len(t) for t in token_lists), self.max_seq_len)

        input_ids = []
        target_ids = []
        attention_mask = []

        for tokens in token_lists:
            seq_len = min(len(tokens), max_len)
            # input = tokens[:-1], target = tokens[1:] (next-token prediction)
            inp = tokens[:seq_len - 1]
            tgt = tokens[1:seq_len]

            # 패딩
            pad_len = max_len - 1 - len(inp)
            mask = torch.ones(len(inp), dtype=torch.long)

            if pad_len > 0:
                inp = torch.cat([inp, torch.full((pad_len,), self.pad_id)])
                tgt = torch.cat([tgt, torch.full((pad_len,), self.pad_id)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])

            input_ids.append(inp)
            target_ids.append(tgt)
            attention_mask.append(mask)

        return {
            "input_ids": torch.stack(input_ids),
            "target_ids": torch.stack(target_ids),
            "attention_mask": torch.stack(attention_mask),
        }
