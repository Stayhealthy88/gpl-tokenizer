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


class CompositeToken(IntEnum):
    """Level 2 복합 도형 토큰."""
    CIRCLE = 20       # 원: [CIRCLE][cx,cy][r]
    ELLIPSE = 21      # 타원: [ELLIPSE][cx,cy][rx,ry]
    RECT = 22         # 사각형: [RECT][x,y][w,h]
    ROUND_RECT = 23   # 둥근 사각형: [ROUND_RECT][x,y][w,h][rx,ry]


class SpatialToken(IntEnum):
    """Level 3 공간 관계 토큰."""
    # 정렬 (Alignment)
    ALIGN_CENTER_H = 60    # 수평 중심 정렬: [ALIGN_H][axis_y_coord]
    ALIGN_CENTER_V = 61    # 수직 중심 정렬: [ALIGN_V][axis_x_coord]

    # 대칭 (Symmetry)
    SYM_REFLECT_X = 62     # 세로축 반사 대칭: [SYM_X][axis_x_coord]
    SYM_REFLECT_Y = 63     # 가로축 반사 대칭: [SYM_Y][axis_y_coord]

    # 등간격 (Distribution)
    EQUAL_SPACE_H = 64     # 수평 등간격: [EQUAL_SPACE_H][spacing_coord]
    EQUAL_SPACE_V = 65     # 수직 등간격: [EQUAL_SPACE_V][spacing_coord]

    # 크기 (Proportion)
    EQUAL_SIZE = 66        # 동일 크기 마커

    # 반복 카운트 (Repetition)
    REPEAT_2 = 67          # 2회 추가 반복 (총 3개)
    REPEAT_3 = 68          # 3회 추가 반복 (총 4개)
    REPEAT_4 = 69          # 4회 추가 반복 (총 5개)
    REPEAT_N = 70          # N회 반복: [REPEAT_N][count_as_coord]


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

    토큰 ID 레이아웃 (v0.5.1 명시):
        0-4     : 특수 토큰 (PAD, BOS, EOS, SEP, UNK)
        5-9     : [예약] 추가 특수 토큰 — MASK, CLS 등
        10-17   : 명령어 토큰 (MOVE~CLOSE, 8개)
        18-19   : [예약] 명령어 확장 (예: SMOOTH_CUBIC)
        20-23   : Level 2 복합 도형 (CIRCLE, ELLIPSE, RECT, ROUND_RECT)
        24-29   : [예약] 복합 도형 확장 (POLYGON, POLYLINE, STAR 등)
        30-33   : 연속성 토큰 (DISC, G0, G1, G2)
        34-39   : [예약] 연속성 확장 (G3 고차 도함수 등)
        40-55   : 곡률 클래스 토큰 (16 bins, κ0 ~ κ15)
        56-59   : [예약] 곡률 비닝 확장
        60-70   : Level 3 공간 관계 (ALIGN, SYM, EQUAL, REPEAT)
        71-99   : [예약] 공간 관계 확장 (GRID, RADIAL_SYM, FIBONACCI 등)
        100+    : ARCS 좌표 토큰 (쿼드트리 레벨 0~max_coord_level)

    예약 영역(reserved) 은 미래 확장을 위해 decode_token_id 에서 'unknown'
    으로 반환되며, 기존 학습 체크포인트를 깨지 않고 새 토큰을 안전하게
    추가할 수 있는 여유 공간이다. 새 토큰을 추가할 때는 이 표와 아래
    IntEnum 블록들, decode_token_id 의 분기, HMNInitializer 의 해당
    `_init_*_tokens` 메서드, 그리고 v0.1 문서 3곳(README.md, README.ko.md,
    RESEARCH_SUMMARY.md) 의 어휘 표를 함께 갱신할 것.

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

    def composite_token(self, ct: 'CompositeToken') -> GPLToken:
        """Level 2 복합 도형 토큰 생성."""
        return GPLToken(token_id=int(ct), token_type="composite", value=ct.name)

    def spatial_token(self, st: 'SpatialToken') -> GPLToken:
        """Level 3 공간 관계 토큰 생성."""
        return GPLToken(token_id=int(st), token_type="spatial", value=st.name)

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
        elif 10 <= token_id < 18:
            return {"type": "command", "value": CommandToken(token_id).name}
        elif 20 <= token_id < 24:
            return {"type": "composite", "value": CompositeToken(token_id).name}
        elif 60 <= token_id < 80:
            return {"type": "spatial", "value": SpatialToken(token_id).name}
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
