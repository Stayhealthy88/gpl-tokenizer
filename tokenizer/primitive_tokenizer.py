"""
Level 1 프리미티브 토크나이저
=============================
SVG PathCommand 시퀀스를 GPL 토큰 시퀀스로 변환.

각 렌더링 가능 세그먼트(LINE, CUBIC, QUAD, ARC)를 하나의 GPL 프리미티브 토큰으로 패킹:
    [CommandToken] [CoordTokens...] [ContinuityToken] [CurvatureToken]

이것이 HiVG의 "세그먼트 토큰"에 대응하되,
미분 기하학적 속성(연속성, 곡률)을 명시적으로 부착하는 것이 GPL의 차별점.
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from ..parser.path_parser import PathCommand, CommandType
from ..analyzer.curvature import CurvatureAnalyzer, CurvatureInfo
from ..analyzer.continuity import ContinuityAnalyzer, ContinuityInfo, ContinuityLevel
from .arcs import ARCS, QuantizedCoord
from .vocabulary import (
    GPLVocabulary, GPLToken, CommandToken, ContinuityToken,
    SpecialToken, CURVATURE_TOKEN_BASE
)


# CommandType → CommandToken 매핑
_CMD_TYPE_TO_TOKEN = {
    CommandType.MOVE: CommandToken.MOVE,
    CommandType.LINE: CommandToken.LINE,
    CommandType.HLINE: CommandToken.HLINE,
    CommandType.VLINE: CommandToken.VLINE,
    CommandType.CUBIC: CommandToken.CUBIC,
    CommandType.QUADRATIC: CommandToken.QUADRATIC,
    CommandType.ARC: CommandToken.ARC,
    CommandType.CLOSE: CommandToken.CLOSE,
}

# ContinuityLevel → ContinuityToken 매핑
_CONT_LEVEL_TO_TOKEN = {
    ContinuityLevel.DISCONTINUOUS: ContinuityToken.DISC,
    ContinuityLevel.G0: ContinuityToken.G0,
    ContinuityLevel.G1: ContinuityToken.G1,
    ContinuityLevel.G2: ContinuityToken.G2,
}


@dataclass
class TokenizationResult:
    """토큰화 결과."""
    tokens: List[GPLToken]            # GPL 토큰 시퀀스
    token_ids: List[int]              # 토큰 ID 시퀀스 (LLM 입력용)
    n_commands: int                   # 원본 명령어 수
    n_tokens: int                     # 생성된 토큰 수
    compression_ratio: float          # 원본 문자 수 대비 토큰 수 비율
    metadata: Dict = None             # 추가 메타데이터


class PrimitiveTokenizer:
    """
    GPL Level 1 프리미티브 토크나이저.

    파이프라인:
        PathCommands → CurvatureAnalyzer → ContinuityAnalyzer
        → ARCS Quantization → Token Sequence

    사용법:
        tokenizer = PrimitiveTokenizer(canvas_size=300, max_coord_level=6)
        result = tokenizer.tokenize(commands)
        print(result.token_ids)
    """

    def __init__(self, canvas_size: float = 300.0, max_coord_level: int = 6,
                 use_adaptive_arcs: bool = True):
        """
        Args:
            canvas_size: SVG 캔버스 크기
            max_coord_level: 최대 ARCS 레벨
            use_adaptive_arcs: 적응적 ARCS 사용 여부 (False면 균일 양자화)
        """
        self.canvas_size = canvas_size
        self.vocab = GPLVocabulary(max_coord_level=max_coord_level)
        self.arcs = ARCS(canvas_size=canvas_size, max_level=max_coord_level)
        self.curv_analyzer = CurvatureAnalyzer()
        self.cont_analyzer = ContinuityAnalyzer()
        self.use_adaptive_arcs = use_adaptive_arcs

    def tokenize(self, commands: List[PathCommand],
                 original_text: str = "") -> TokenizationResult:
        """
        PathCommand 시퀀스를 GPL 토큰 시퀀스로 변환.

        Args:
            commands: resolve_to_absolute() 완료된 PathCommand 리스트
            original_text: 원본 SVG 텍스트 (압축률 계산용)
        Returns:
            TokenizationResult
        """
        if not commands:
            return TokenizationResult(
                tokens=[], token_ids=[], n_commands=0,
                n_tokens=0, compression_ratio=0.0
            )

        # 1. 곡률 분석
        curv_infos = self.curv_analyzer.analyze(commands)
        curv_map = {ci.command_index: ci for ci in curv_infos}

        # 2. 적응적 ARCS 구축 (곡률 기반)
        if self.use_adaptive_arcs and curv_infos:
            self._build_adaptive_arcs(commands, curv_infos)

        # 3. 연속성 분석
        cont_infos = self.cont_analyzer.analyze(commands, curv_infos)
        # 연결점 정보를 "다음 세그먼트 인덱스"로 매핑
        cont_map = {ci.segment_b_index: ci for ci in cont_infos}

        # 4. 토큰 시퀀스 생성
        tokens = [self.vocab.special_token(SpecialToken.BOS)]

        for i, cmd in enumerate(commands):
            cmd_tokens = self._tokenize_command(i, cmd, curv_map, cont_map)
            tokens.extend(cmd_tokens)

        tokens.append(self.vocab.special_token(SpecialToken.EOS))

        # 결과 조립
        token_ids = [t.token_id for t in tokens]
        orig_len = len(original_text) if original_text else len(commands) * 20  # 추정
        compression = len(token_ids) / max(orig_len, 1)

        return TokenizationResult(
            tokens=tokens,
            token_ids=token_ids,
            n_commands=len(commands),
            n_tokens=len(token_ids),
            compression_ratio=compression,
            metadata={
                "n_curvature_infos": len(curv_infos),
                "n_continuity_infos": len(cont_infos),
                "arcs_leaves": self.arcs.total_leaf_count(),
                "vocab_size": self.vocab.vocab_size,
            }
        )

    def _tokenize_command(self, index: int, cmd: PathCommand,
                          curv_map: Dict[int, CurvatureInfo],
                          cont_map: Dict[int, ContinuityInfo]) -> List[GPLToken]:
        """단일 PathCommand를 GPL 토큰 서브시퀀스로 변환."""
        tokens = []

        # 1. 명령어 타입 토큰
        cmd_token_type = _CMD_TYPE_TO_TOKEN.get(cmd.command_type)
        if cmd_token_type is None:
            return tokens
        tokens.append(self.vocab.command_token(cmd_token_type))

        # 2. 좌표 토큰들 (ARCS 양자화)
        coord_tokens = self._encode_coordinates(cmd)
        tokens.extend(coord_tokens)

        # 3. 미분 기하학적 속성 토큰 (DiffAttr)
        curv_info = curv_map.get(index)
        cont_info = cont_map.get(index)

        # 연속성 토큰
        if cont_info is not None:
            cl_token = _CONT_LEVEL_TO_TOKEN.get(cont_info.level, ContinuityToken.G0)
            tokens.append(self.vocab.continuity_token(cl_token))

        # 곡률 클래스 토큰
        if curv_info is not None and not curv_info.is_straight:
            curv_bin = curv_info.quantized_curvature_class(n_bins=16)
            tokens.append(self.vocab.curvature_token(curv_bin))

        return tokens

    def _encode_coordinates(self, cmd: PathCommand) -> List[GPLToken]:
        """명령어의 좌표를 ARCS 토큰으로 인코딩."""
        tokens = []

        if cmd.command_type == CommandType.CLOSE:
            return tokens  # Z는 좌표 없음

        if cmd.abs_params is None:
            return tokens

        ap = cmd.abs_params

        if cmd.command_type == CommandType.MOVE:
            # M: 끝점 좌표 1개
            tokens.append(self._quantize_to_token(ap[0], ap[1]))

        elif cmd.command_type == CommandType.LINE:
            # L: 끝점 좌표 1개
            tokens.append(self._quantize_to_token(ap[0], ap[1]))

        elif cmd.command_type in (CommandType.HLINE, CommandType.VLINE):
            # H/V: 끝점 좌표 1개
            tokens.append(self._quantize_to_token(ap[0], ap[1]))

        elif cmd.command_type == CommandType.CUBIC:
            # C: 제어점1 + 제어점2 + 끝점 = 3개 좌표
            tokens.append(self._quantize_to_token(ap[0], ap[1]))  # ctrl1
            tokens.append(self._quantize_to_token(ap[2], ap[3]))  # ctrl2
            tokens.append(self._quantize_to_token(ap[4], ap[5]))  # end

        elif cmd.command_type == CommandType.QUADRATIC:
            # Q: 제어점 + 끝점 = 2개 좌표
            tokens.append(self._quantize_to_token(ap[0], ap[1]))  # ctrl
            tokens.append(self._quantize_to_token(ap[2], ap[3]))  # end

        elif cmd.command_type == CommandType.ARC:
            # A: rx, ry, rotation, flags, 끝점
            # rx, ry는 별도 양자화 (반지름 토큰으로 확장 가능, 현재는 좌표로 처리)
            tokens.append(self._quantize_to_token(ap[0], ap[1]))  # rx, ry as coord
            tokens.append(self._quantize_to_token(ap[5], ap[6]))  # end point

        return tokens

    def _quantize_to_token(self, x: float, y: float) -> GPLToken:
        """좌표를 ARCS로 양자화한 후 토큰으로 변환."""
        qc = self.arcs.quantize(x, y)
        return self.vocab.coord_token(qc.level, qc.qx, qc.qy)

    def _build_adaptive_arcs(self, commands: List[PathCommand],
                             curv_infos: List[CurvatureInfo]):
        """곡률 정보 기반 적응적 ARCS 구축."""
        segment_data = []
        for ci in curv_infos:
            cmd = commands[ci.command_index]
            bbox = None
            if cmd.start_point and cmd.end_point:
                x1 = min(cmd.start_point[0], cmd.end_point[0])
                y1 = min(cmd.start_point[1], cmd.end_point[1])
                x2 = max(cmd.start_point[0], cmd.end_point[0])
                y2 = max(cmd.start_point[1], cmd.end_point[1])
                bbox = (x1, y1, x2, y2)

            segment_data.append({
                "bbox": bbox,
                "max_curvature": ci.max_abs_curvature,
                "arc_length": ci.arc_length,
            })

        if segment_data:
            self.arcs.build_from_curvatures(segment_data)
