"""
GPL 어휘 관리기
===============
GPL 토큰의 어휘(vocabulary)를 정의하고 관리.

토큰 구조: [Type][Coords][DiffAttr]
    - Type: 명령어 유형 토큰 (MOVE, LINE, CUBIC, QUAD, ARC, CLOSE + 특수)
    - Coords: ARCS 양자화 좌표 토큰
    - DiffAttr: 연속성 레벨 + 양자화 곡률 클래스

비판 문서 반영 (섹션 2.3):
    "미분 기하학적 불변량 토큰화 — 곡률, 접선 벡터, 법선 벡터 등의
     불변량을 계산하고, 정규화하여 이산적인 토큰으로 변환"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum
import json


class SpecialToken(IntEnum):
    """특수 토큰."""
    PAD = 0
    BOS = 1      # Beginning of SVG
    EOS = 2      # End of SVG
    SEP = 3      # 요소(path) 경계 구분
    UNK = 4


class CommandToken(IntEnum):
    """명령어 유형 토큰."""
    MOVE = 10
    LINE = 11
    HLINE = 12
    VLINE = 13
    CUBIC = 14
    QUADRATIC = 15
    ARC = 16
    CLOSE = 17


class ContinuityToken(IntEnum):
    """연속성 레벨 토큰 (DiffAttr 일부)."""
    DISC = 30    # 불연속
    G0 = 31
    G1 = 32
    G2 = 33


# 곡률 클래스 토큰: 40 ~ 55 (16개 bin)
CURVATURE_TOKEN_BASE = 40
N_CURVATURE_BINS = 16

# 좌표 토큰 시작 ID
COORD_TOKEN_BASE = 100


@dataclass
class GPLToken:
    """단일 GPL 토큰."""
    token_id: int
    token_type: str      # "special", "command", "coord", "continuity", "curvature"
    value: any = None    # 토큰의 원본 값 (디버깅/역변환용)

    def __repr__(self):
        return f"T({self.token_type}:{self.token_id}|{self.value})"


class GPLVocabulary:
    """
    GPL 토큰 어휘 관리.

    토큰 ID 레이아웃:
        0-9:     특수 토큰 (PAD, BOS, EOS, SEP, UNK)
        10-19:   명령어 토큰 (MOVE, LINE, CUBIC, ...)
        30-33:   연속성 토큰 (DISC, G0, G1, G2)
        40-55:   곡률 클래스 토큰 (16 bins)
        100+:    좌표 토큰 (ARCS 양자화 좌표)

    사용법:
        vocab = GPLVocabulary(max_coord_level=6)
        token_id = vocab.coord_to_id(level=4, qx=10, qy=5)
        level, qx, qy = vocab.id_to_coord(token_id)
    """

    def __init__(self, max_coord_level: int = 6):
        self.max_coord_level = max_coord_level
        self._build_coord_table()

    def _build_coord_table(self):
        """좌표 토큰 ID 테이블 구축."""
        self._coord_to_id_map: Dict[Tuple[int, int, int], int] = {}
        self._id_to_coord_map: Dict[int, Tuple[int, int, int]] = {}

        current_id = COORD_TOKEN_BASE
        for level in range(self.max_coord_level + 1):
            grid_size = 2 ** level
            for qy in range(grid_size):
                for qx in range(grid_size):
                    key = (level, qx, qy)
                    self._coord_to_id_map[key] = current_id
                    self._id_to_coord_map[current_id] = key
                    current_id += 1

        self._total_coord_tokens = current_id - COORD_TOKEN_BASE
        self._vocab_size = current_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def total_coord_tokens(self) -> int:
        return self._total_coord_tokens

    # --- 토큰 생성 ---

    def special_token(self, st: SpecialToken) -> GPLToken:
        return GPLToken(token_id=int(st), token_type="special", value=st.name)

    def command_token(self, ct: CommandToken) -> GPLToken:
        return GPLToken(token_id=int(ct), token_type="command", value=ct.name)

    def continuity_token(self, cl: ContinuityToken) -> GPLToken:
        return GPLToken(token_id=int(cl), token_type="continuity", value=cl.name)

    def curvature_token(self, bin_idx: int) -> GPLToken:
        """곡률 클래스 토큰 생성."""
        bin_idx = max(0, min(bin_idx, N_CURVATURE_BINS - 1))
        tid = CURVATURE_TOKEN_BASE + bin_idx
        return GPLToken(token_id=tid, token_type="curvature", value=bin_idx)

    def coord_token(self, level: int, qx: int, qy: int) -> GPLToken:
        """ARCS 좌표 토큰 생성."""
        key = (level, qx, qy)
        tid = self._coord_to_id_map.get(key)
        if tid is None:
            return GPLToken(token_id=int(SpecialToken.UNK),
                            token_type="special", value="UNK_COORD")
        return GPLToken(token_id=tid, token_type="coord",
                        value=f"q{level}:{qx},{qy}")

    # --- 토큰 ID 디코딩 ---

    def decode_token_id(self, token_id: int) -> dict:
        """토큰 ID를 해석."""
        if token_id < 10:
            return {"type": "special", "value": SpecialToken(token_id).name}
        elif 10 <= token_id < 20:
            return {"type": "command", "value": CommandToken(token_id).name}
        elif 30 <= token_id < 34:
            return {"type": "continuity", "value": ContinuityToken(token_id).name}
        elif CURVATURE_TOKEN_BASE <= token_id < CURVATURE_TOKEN_BASE + N_CURVATURE_BINS:
            return {"type": "curvature", "bin": token_id - CURVATURE_TOKEN_BASE}
        elif token_id in self._id_to_coord_map:
            level, qx, qy = self._id_to_coord_map[token_id]
            return {"type": "coord", "level": level, "qx": qx, "qy": qy}
        else:
            return {"type": "unknown", "id": token_id}

    def id_to_coord(self, token_id: int) -> Optional[Tuple[int, int, int]]:
        """좌표 토큰 ID → (level, qx, qy)."""
        return self._id_to_coord_map.get(token_id)

    def coord_to_id(self, level: int, qx: int, qy: int) -> Optional[int]:
        """(level, qx, qy) → 좌표 토큰 ID."""
        return self._coord_to_id_map.get((level, qx, qy))

    # --- 통계 ---

    def summary(self) -> str:
        lines = [
            f"GPL Vocabulary Summary:",
            f"  Total vocab size: {self.vocab_size}",
            f"  Special tokens: 5 (PAD, BOS, EOS, SEP, UNK)",
            f"  Command tokens: 8 (MOVE~CLOSE)",
            f"  Continuity tokens: 4 (DISC, G0, G1, G2)",
            f"  Curvature bins: {N_CURVATURE_BINS}",
            f"  Coordinate tokens: {self.total_coord_tokens}",
            f"  Max coord level: {self.max_coord_level}",
            f"  Max coord grid: {2**self.max_coord_level}×{2**self.max_coord_level}",
        ]
        return "\n".join(lines)
