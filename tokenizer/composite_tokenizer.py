"""
Level 2 복합 토크나이저
=======================
Shape Detector가 인식한 기하 프리미티브를 단일 복합 토큰으로 압축.

v0.2 핵심:
    Level 1: circle = M + 4C + Z = [BOS][MOVE][coord][CUBIC][3coords][κ]×4[CLOSE][EOS] ≈ 28 tokens
    Level 2: circle = [BOS][CIRCLE][center_coord][radius_coord][EOS] = 5 tokens → 5.6× 압축

동작 방식:
    1. Level 1 토큰화 수행
    2. ShapeDetector로 원본 commands에서 도형 패턴 인식
    3. 인식된 도형 영역의 토큰을 복합 토큰으로 대체
    4. 비인식 영역은 Level 1 토큰 유지
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..parser.path_parser import PathCommand, CommandType
from ..analyzer.shape_detector import ShapeDetector, DetectedShape, ShapeType
from ..analyzer.curvature import CurvatureAnalyzer
from ..analyzer.continuity import ContinuityAnalyzer
from .arcs import ARCS
from .vocabulary import (
    GPLVocabulary, GPLToken, CompositeToken, SpecialToken, CommandToken
)
from .primitive_tokenizer import PrimitiveTokenizer, TokenizationResult


@dataclass
class Level2Result(TokenizationResult):
    """Level 2 토큰화 결과 (Level 1 확장)."""
    detected_shapes: List[DetectedShape] = None
    level1_n_tokens: int = 0         # Level 1 토큰 수 (비교용)
    compression_vs_level1: float = 0  # Level 1 대비 압축률


class CompositeTokenizer:
    """
    GPL Level 2 복합 토크나이저.

    Level 1 위에서 기하 프리미티브 패턴을 인식하고 압축.

    사용법:
        tokenizer = CompositeTokenizer(canvas_size=300, max_coord_level=6)
        result = tokenizer.tokenize(commands)
        print(f"Level 1: {result.level1_n_tokens} → Level 2: {result.n_tokens}")
    """

    def __init__(self, canvas_size: float = 300.0, max_coord_level: int = 6,
                 circle_tolerance: float = 5.0, rect_tolerance: float = 3.0):
        self.canvas_size = canvas_size
        self.vocab = GPLVocabulary(max_coord_level=max_coord_level)
        self.arcs = ARCS(canvas_size=canvas_size, max_level=max_coord_level)
        self.shape_detector = ShapeDetector(
            circle_tolerance=circle_tolerance,
            rect_tolerance=rect_tolerance
        )
        # Level 1 토크나이저 (비교용 + fallback)
        self.level1 = PrimitiveTokenizer(
            canvas_size=canvas_size,
            max_coord_level=max_coord_level
        )

    def tokenize(self, commands: List[PathCommand],
                 original_text: str = "") -> Level2Result:
        """
        PathCommand 시퀀스를 Level 2 GPL 토큰 시퀀스로 변환.

        1단계: Level 1 토큰화 (비교 기준)
        2단계: ShapeDetector로 도형 패턴 인식
        3단계: 인식된 영역 → 복합 토큰, 나머지 → Level 1 토큰
        """
        if not commands:
            return Level2Result(
                tokens=[], token_ids=[], n_commands=0, n_tokens=0,
                compression_ratio=0.0, detected_shapes=[], level1_n_tokens=0
            )

        # 1. Level 1 기준치
        level1_result = self.level1.tokenize(commands, original_text)
        # ARCS 동기화
        self.arcs = self.level1.arcs

        # 2. 도형 패턴 인식
        detected = self.shape_detector.detect(commands)

        # 3. 도형이 없으면 Level 1 결과 반환
        if not detected:
            return Level2Result(
                tokens=level1_result.tokens,
                token_ids=level1_result.token_ids,
                n_commands=level1_result.n_commands,
                n_tokens=level1_result.n_tokens,
                compression_ratio=level1_result.compression_ratio,
                metadata=level1_result.metadata,
                detected_shapes=[],
                level1_n_tokens=level1_result.n_tokens,
                compression_vs_level1=1.0
            )

        # 4. 인식된 도형 인덱스 세트
        shape_cmd_ranges = set()
        for shape in detected:
            for i in range(shape.start_index, shape.end_index + 1):
                shape_cmd_ranges.add(i)

        # 5. Level 2 토큰 시퀀스 구축
        tokens = [self.vocab.special_token(SpecialToken.BOS)]

        i = 0
        shape_idx = 0
        while i < len(commands):
            # 현재 인덱스가 도형 시작인지 확인
            current_shape = None
            for s in detected:
                if s.start_index == i:
                    current_shape = s
                    break

            if current_shape:
                # 복합 토큰으로 인코딩
                shape_tokens = self._encode_shape(current_shape)
                tokens.extend(shape_tokens)
                i = current_shape.end_index + 1
            else:
                # Level 1 토큰 사용 (개별 명령어)
                cmd = commands[i]
                cmd_tokens = self._tokenize_command_level1(i, cmd, commands)
                tokens.extend(cmd_tokens)
                i += 1

        tokens.append(self.vocab.special_token(SpecialToken.EOS))

        token_ids = [t.token_id for t in tokens]
        orig_len = len(original_text) if original_text else len(commands) * 20

        return Level2Result(
            tokens=tokens,
            token_ids=token_ids,
            n_commands=len(commands),
            n_tokens=len(token_ids),
            compression_ratio=len(token_ids) / max(orig_len, 1),
            metadata={
                "n_shapes_detected": len(detected),
                "shape_types": [s.shape_type.value for s in detected],
                "vocab_size": self.vocab.vocab_size,
            },
            detected_shapes=detected,
            level1_n_tokens=level1_result.n_tokens,
            compression_vs_level1=len(token_ids) / max(level1_result.n_tokens, 1)
        )

    def _encode_shape(self, shape: DetectedShape) -> List[GPLToken]:
        """인식된 도형을 복합 토큰 시퀀스로 인코딩."""
        tokens = []
        p = shape.params

        if shape.shape_type == ShapeType.CIRCLE:
            # [CIRCLE] [center_coord] [radius_as_coord]
            tokens.append(self.vocab.composite_token(CompositeToken.CIRCLE))
            tokens.append(self._q(p["cx"], p["cy"]))
            # 반지름을 좌표로 인코딩 (r, r)
            tokens.append(self._q(p["r"], p["r"]))

        elif shape.shape_type == ShapeType.ELLIPSE:
            # [ELLIPSE] [center_coord] [rx_ry_coord]
            tokens.append(self.vocab.composite_token(CompositeToken.ELLIPSE))
            tokens.append(self._q(p["cx"], p["cy"]))
            tokens.append(self._q(p["rx"], p["ry"]))

        elif shape.shape_type == ShapeType.RECT:
            # [RECT] [origin_coord] [size_coord]
            tokens.append(self.vocab.composite_token(CompositeToken.RECT))
            tokens.append(self._q(p["x"], p["y"]))
            tokens.append(self._q(p["width"], p["height"]))

        elif shape.shape_type == ShapeType.ROUND_RECT:
            # [ROUND_RECT] [origin_coord] [size_coord] [radius_coord]
            tokens.append(self.vocab.composite_token(CompositeToken.ROUND_RECT))
            tokens.append(self._q(p["x"], p["y"]))
            tokens.append(self._q(p["width"], p["height"]))
            tokens.append(self._q(p["rx"], p["ry"]))

        return tokens

    def _q(self, x: float, y: float) -> GPLToken:
        """좌표를 ARCS 양자화 후 토큰으로 변환."""
        qc = self.arcs.quantize(x, y)
        return self.vocab.coord_token(qc.level, qc.qx, qc.qy)

    def _tokenize_command_level1(self, index: int, cmd: PathCommand,
                                  all_cmds: List[PathCommand]) -> List[GPLToken]:
        """Level 1 방식으로 단일 명령어 토큰화 (fallback)."""
        from .primitive_tokenizer import _CMD_TYPE_TO_TOKEN
        tokens = []

        cmd_token_type = _CMD_TYPE_TO_TOKEN.get(cmd.command_type)
        if cmd_token_type is None:
            return tokens
        tokens.append(self.vocab.command_token(cmd_token_type))

        # 좌표 인코딩
        if cmd.command_type == CommandType.CLOSE:
            return tokens
        if cmd.abs_params is None:
            return tokens

        ap = cmd.abs_params
        if cmd.command_type == CommandType.MOVE:
            tokens.append(self._q(ap[0], ap[1]))
        elif cmd.command_type == CommandType.LINE:
            tokens.append(self._q(ap[0], ap[1]))
        elif cmd.command_type in (CommandType.HLINE, CommandType.VLINE):
            tokens.append(self._q(ap[0], ap[1]))
        elif cmd.command_type == CommandType.CUBIC:
            tokens.append(self._q(ap[0], ap[1]))
            tokens.append(self._q(ap[2], ap[3]))
            tokens.append(self._q(ap[4], ap[5]))
        elif cmd.command_type == CommandType.QUADRATIC:
            tokens.append(self._q(ap[0], ap[1]))
            tokens.append(self._q(ap[2], ap[3]))
        elif cmd.command_type == CommandType.ARC:
            tokens.append(self._q(ap[0], ap[1]))
            tokens.append(self._q(ap[5], ap[6]))

        return tokens
