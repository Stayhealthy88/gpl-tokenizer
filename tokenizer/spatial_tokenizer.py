"""
Level 3 공간 관계 토크나이저
============================
다중 요소의 공간 관계를 토큰으로 인코딩하여 추가 압축.

v0.3 핵심:
    Level 2 (5 circles): [BOS][CIRCLE][p][r][SEP]×4[CIRCLE][p][r][EOS] = 21 tokens
    Level 3 (5 circles): [BOS][CIRCLE][p][r][EQUAL_SIZE][ALIGN_H][y][EQUAL_SPACE_H][dx][REPEAT_4][EOS]
                        = 11 tokens → 1.9× vs L2

압축 전략:
    1. 선형 반복 (Linear Repeat): ALIGN + EQUAL_SPACE + EQUAL_SIZE → 앵커 + 관계 토큰 + REPEAT
    2. 축 대칭 (Symmetry): SYM_REFLECT → 한쪽만 인코딩 + SYM 토큰
    3. 그 외: Level 2 토큰 유지 (fallback)
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

from ..parser.path_parser import PathCommand, CommandType
from ..analyzer.spatial_analyzer import (
    SpatialAnalyzer, ElementInfo, SpatialRelation, RelationType
)
from ..analyzer.shape_detector import ShapeDetector, DetectedShape, ShapeType
from .composite_tokenizer import CompositeTokenizer, Level2Result
from .primitive_tokenizer import TokenizationResult
from .arcs import ARCS
from .vocabulary import (
    GPLVocabulary, GPLToken, SpatialToken, SpecialToken, CompositeToken
)


@dataclass
class Level3Result(TokenizationResult):
    """Level 3 토큰화 결과."""
    spatial_relations: List[SpatialRelation] = None
    element_infos: List[ElementInfo] = None
    level2_n_tokens: int = 0             # Level 2 총 토큰 수 (비교용)
    compression_vs_level2: float = 0.0   # Level 2 대비 압축률
    n_elements: int = 0                  # 입력 요소 수


class SpatialTokenizer:
    """
    GPL Level 3 공간 관계 토크나이저.

    다중 요소를 Level 2로 개별 토큰화한 후,
    요소 간 공간 관계를 탐지하여 압축 토큰으로 대체.

    사용법:
        tokenizer = SpatialTokenizer(canvas_size=300)
        result = tokenizer.tokenize_multi(element_commands_list)
        print(f"L2: {result.level2_n_tokens} → L3: {result.n_tokens}")
    """

    def __init__(self, canvas_size: float = 300.0, max_coord_level: int = 6,
                 align_tolerance: float = 3.0, spacing_tolerance: float = 3.0,
                 size_tolerance: float = 5.0, sym_tolerance: float = 5.0):
        self.canvas_size = canvas_size
        self.vocab = GPLVocabulary(max_coord_level=max_coord_level)
        self.arcs = ARCS(canvas_size=canvas_size, max_level=max_coord_level)
        self.level2 = CompositeTokenizer(
            canvas_size=canvas_size, max_coord_level=max_coord_level
        )
        self.spatial_analyzer = SpatialAnalyzer(
            align_tolerance=align_tolerance,
            size_tolerance=size_tolerance,
            spacing_tolerance=spacing_tolerance,
            sym_tolerance=sym_tolerance,
        )

    def tokenize_multi(self, elements_commands: List[List[PathCommand]],
                        original_text: str = "") -> Level3Result:
        """
        다중 요소를 공간 관계 포함하여 토큰화.

        Args:
            elements_commands: 각 요소의 PathCommand 리스트의 리스트
            original_text: 원본 SVG 텍스트 (압축률 계산용)
        Returns:
            Level3Result
        """
        if not elements_commands:
            return Level3Result(
                tokens=[], token_ids=[], n_commands=0, n_tokens=0,
                compression_ratio=0.0, spatial_relations=[], element_infos=[],
                level2_n_tokens=0, n_elements=0
            )

        # 1. 각 요소 Level 2 토큰화
        l2_results: List[Level2Result] = []
        for cmds in elements_commands:
            l2 = self.level2.tokenize(cmds)
            l2_results.append(l2)

        # ARCS 동기화
        self.arcs = self.level2.arcs

        # 2. 요소 정보 추출
        elem_infos = []
        for i, (cmds, l2) in enumerate(zip(elements_commands, l2_results)):
            info = self._extract_element_info(i, cmds, l2)
            elem_infos.append(info)

        # 3. Level 2 기준치 (모든 요소 연결)
        l2_total = self._count_l2_tokens(l2_results)

        # 4. 공간 관계 분석
        relations = self.spatial_analyzer.analyze(elem_infos)

        # 5. 관계 없으면 Level 2 결과 연결 반환
        if not relations:
            return self._build_l2_fallback(l2_results, elem_infos, l2_total,
                                            original_text)

        # 6. Level 3 압축 인코딩
        return self._encode_with_relations(
            l2_results, elem_infos, relations, l2_total, original_text
        )

    # ===================== 요소 정보 추출 =====================

    def _extract_element_info(self, index: int, commands: List[PathCommand],
                               l2_result: Level2Result) -> ElementInfo:
        """PathCommand 시퀀스에서 ElementInfo를 추출."""
        # Level 2에서 도형이 인식된 경우 그 정보 사용
        if l2_result.detected_shapes:
            shape = l2_result.detected_shapes[0]
            return self._shape_to_element_info(index, shape)

        # 인식되지 않은 경우 endpoint 기반 바운딩 박스 계산
        pts = []
        for cmd in commands:
            if cmd.end_point:
                pts.append(cmd.end_point)
            if cmd.start_point:
                pts.append(cmd.start_point)

        if not pts:
            return ElementInfo(
                index=index, element_type="path",
                center=(0, 0), bbox=(0, 0, 0, 0),
                width=0, height=0
            )

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        return ElementInfo(
            index=index, element_type="path",
            center=(cx, cy), bbox=(x_min, y_min, x_max, y_max),
            width=x_max - x_min, height=y_max - y_min
        )

    def _shape_to_element_info(self, index: int,
                                shape: DetectedShape) -> ElementInfo:
        """DetectedShape를 ElementInfo로 변환."""
        p = shape.params
        st = shape.shape_type

        if st == ShapeType.CIRCLE:
            cx, cy, r = p["cx"], p["cy"], p["r"]
            return ElementInfo(
                index=index, element_type="circle",
                center=(cx, cy),
                bbox=(cx - r, cy - r, cx + r, cy + r),
                width=2 * r, height=2 * r, params=p
            )
        elif st == ShapeType.ELLIPSE:
            cx, cy, rx, ry = p["cx"], p["cy"], p["rx"], p["ry"]
            return ElementInfo(
                index=index, element_type="ellipse",
                center=(cx, cy),
                bbox=(cx - rx, cy - ry, cx + rx, cy + ry),
                width=2 * rx, height=2 * ry, params=p
            )
        elif st in (ShapeType.RECT, ShapeType.ROUND_RECT):
            x, y = p["x"], p["y"]
            w, h = p["width"], p["height"]
            return ElementInfo(
                index=index,
                element_type="round_rect" if st == ShapeType.ROUND_RECT else "rect",
                center=(x + w / 2, y + h / 2),
                bbox=(x, y, x + w, y + h),
                width=w, height=h, params=p
            )

        # fallback
        return ElementInfo(
            index=index, element_type="path",
            center=(0, 0), bbox=(0, 0, 0, 0),
            width=0, height=0
        )

    # ===================== Level 2 기준치 =====================

    def _count_l2_tokens(self, l2_results: List[Level2Result]) -> int:
        """Level 2 결과들의 총 토큰 수 (multi-element 기준)."""
        # BOS + [elem1 inner] + SEP + [elem2 inner] + ... + EOS
        total = 2  # BOS + EOS
        for i, l2 in enumerate(l2_results):
            # inner = tokens without BOS/EOS
            inner_count = sum(1 for t in l2.tokens
                              if t.token_id not in (int(SpecialToken.BOS),
                                                     int(SpecialToken.EOS)))
            total += inner_count
            if i < len(l2_results) - 1:
                total += 1  # SEP
        return total

    def _build_l2_fallback(self, l2_results, elem_infos, l2_total,
                            original_text) -> Level3Result:
        """관계 없이 Level 2 결과를 단순 연결."""
        tokens = [self.vocab.special_token(SpecialToken.BOS)]
        for i, l2 in enumerate(l2_results):
            inner = [t for t in l2.tokens
                     if t.token_id not in (int(SpecialToken.BOS),
                                            int(SpecialToken.EOS))]
            tokens.extend(inner)
            if i < len(l2_results) - 1:
                tokens.append(self.vocab.special_token(SpecialToken.SEP))
        tokens.append(self.vocab.special_token(SpecialToken.EOS))

        token_ids = [t.token_id for t in tokens]
        n_cmds = sum(l2.n_commands for l2 in l2_results)
        orig_len = len(original_text) if original_text else n_cmds * 20

        return Level3Result(
            tokens=tokens, token_ids=token_ids,
            n_commands=n_cmds, n_tokens=len(token_ids),
            compression_ratio=len(token_ids) / max(orig_len, 1),
            spatial_relations=[], element_infos=elem_infos,
            level2_n_tokens=l2_total,
            compression_vs_level2=1.0,
            n_elements=len(l2_results)
        )

    # ===================== Level 3 인코딩 =====================

    def _encode_with_relations(self, l2_results, elem_infos, relations,
                                l2_total, original_text) -> Level3Result:
        """공간 관계를 활용한 압축 인코딩."""
        tokens = [self.vocab.special_token(SpecialToken.BOS)]
        encoded: Set[int] = set()

        # 우선순위: 선형 반복 > 대칭 > 나머지 (Level 2 fallback)

        # 1. 선형 반복 그룹 처리
        repeat_groups = self._find_repeat_groups(relations, elem_infos)
        for rg in repeat_groups:
            if encoded:
                tokens.append(self.vocab.special_token(SpecialToken.SEP))
            group_tokens = self._encode_repeat_group(rg, l2_results, elem_infos)
            tokens.extend(group_tokens)
            for idx in rg["indices"]:
                encoded.add(idx)

        # 2. 대칭 그룹 처리
        sym_relations = [r for r in relations
                         if r.relation_type in (RelationType.SYM_REFLECT_X,
                                                 RelationType.SYM_REFLECT_Y)]
        for rel in sym_relations:
            unencoded = [i for i in rel.element_indices if i not in encoded]
            if len(unencoded) >= 2:
                if encoded:
                    tokens.append(self.vocab.special_token(SpecialToken.SEP))
                sym_tokens = self._encode_symmetry(rel, l2_results,
                                                     elem_infos, encoded)
                tokens.extend(sym_tokens)
                for idx in rel.element_indices:
                    encoded.add(idx)

        # 3. 나머지 요소 Level 2 fallback
        for i, l2 in enumerate(l2_results):
            if i not in encoded:
                if encoded:
                    tokens.append(self.vocab.special_token(SpecialToken.SEP))
                inner = [t for t in l2.tokens
                         if t.token_id not in (int(SpecialToken.BOS),
                                                int(SpecialToken.EOS))]
                tokens.extend(inner)
                encoded.add(i)

        tokens.append(self.vocab.special_token(SpecialToken.EOS))

        token_ids = [t.token_id for t in tokens]
        n_cmds = sum(l2.n_commands for l2 in l2_results)
        orig_len = len(original_text) if original_text else n_cmds * 20

        return Level3Result(
            tokens=tokens, token_ids=token_ids,
            n_commands=n_cmds, n_tokens=len(token_ids),
            compression_ratio=len(token_ids) / max(orig_len, 1),
            metadata={
                "n_relations": len(relations),
                "relation_types": [r.relation_type.value for r in relations],
                "vocab_size": self.vocab.vocab_size,
            },
            spatial_relations=relations,
            element_infos=elem_infos,
            level2_n_tokens=l2_total,
            compression_vs_level2=len(token_ids) / max(l2_total, 1),
            n_elements=len(l2_results)
        )

    # ===================== 선형 반복 그룹 =====================

    def _find_repeat_groups(self, relations: List[SpatialRelation],
                             elem_infos: List[ElementInfo]) -> List[dict]:
        """ALIGN + EQUAL_SPACE + EQUAL_SIZE를 모두 만족하는 반복 그룹 탐색."""
        groups = []

        # EQUAL_SPACE가 있는 관계에서 시작 (3+ 요소 등간격)
        for rel in relations:
            if rel.relation_type not in (RelationType.EQUAL_SPACE_H,
                                          RelationType.EQUAL_SPACE_V):
                continue

            indices = rel.element_indices

            # EQUAL_SIZE도 만족하는지 확인
            has_equal_size = any(
                r.relation_type == RelationType.EQUAL_SIZE and
                set(indices).issubset(set(r.element_indices))
                for r in relations
            )

            # ALIGN도 만족하는지 확인
            if rel.relation_type == RelationType.EQUAL_SPACE_H:
                align_type = RelationType.ALIGN_CENTER_H
            else:
                align_type = RelationType.ALIGN_CENTER_V

            has_align = any(
                r.relation_type == align_type and
                set(indices).issubset(set(r.element_indices))
                for r in relations
            )

            if has_equal_size and has_align:
                # 정렬 축 값 찾기
                align_rel = next(
                    r for r in relations
                    if r.relation_type == align_type and
                    set(indices).issubset(set(r.element_indices))
                )
                groups.append({
                    "indices": indices,
                    "spacing": rel.spacing,
                    "direction": "H" if rel.relation_type == RelationType.EQUAL_SPACE_H else "V",
                    "axis_value": align_rel.axis_value,
                })

        return groups

    def _encode_repeat_group(self, group: dict, l2_results: List[Level2Result],
                              elem_infos: List[ElementInfo]) -> List[GPLToken]:
        """선형 반복 그룹을 Level 3 토큰으로 인코딩."""
        tokens = []
        indices = group["indices"]
        anchor_idx = indices[0]

        # 1. 앵커 요소 (Level 2 토큰, BOS/EOS 제외)
        l2 = l2_results[anchor_idx]
        inner = [t for t in l2.tokens
                 if t.token_id not in (int(SpecialToken.BOS),
                                        int(SpecialToken.EOS))]
        tokens.extend(inner)

        # 2. EQUAL_SIZE 마커
        tokens.append(self.vocab.spatial_token(SpatialToken.EQUAL_SIZE))

        # 3. ALIGN 토큰 + 축 좌표
        if group["direction"] == "H":
            tokens.append(self.vocab.spatial_token(SpatialToken.ALIGN_CENTER_H))
        else:
            tokens.append(self.vocab.spatial_token(SpatialToken.ALIGN_CENTER_V))
        tokens.append(self._q(group["axis_value"], group["axis_value"]))

        # 4. EQUAL_SPACE 토큰 + 간격 좌표
        if group["direction"] == "H":
            tokens.append(self.vocab.spatial_token(SpatialToken.EQUAL_SPACE_H))
        else:
            tokens.append(self.vocab.spatial_token(SpatialToken.EQUAL_SPACE_V))
        tokens.append(self._q(group["spacing"], group["spacing"]))

        # 5. REPEAT 카운트 (추가 복사 수 = 총 요소 수 - 1)
        repeat_count = len(indices) - 1
        if repeat_count == 2:
            tokens.append(self.vocab.spatial_token(SpatialToken.REPEAT_2))
        elif repeat_count == 3:
            tokens.append(self.vocab.spatial_token(SpatialToken.REPEAT_3))
        elif repeat_count == 4:
            tokens.append(self.vocab.spatial_token(SpatialToken.REPEAT_4))
        else:
            tokens.append(self.vocab.spatial_token(SpatialToken.REPEAT_N))
            tokens.append(self._q(float(repeat_count), float(repeat_count)))

        return tokens

    # ===================== 대칭 인코딩 =====================

    def _encode_symmetry(self, rel: SpatialRelation,
                          l2_results: List[Level2Result],
                          elem_infos: List[ElementInfo],
                          already_encoded: Set[int]) -> List[GPLToken]:
        """대칭 그룹을 Level 3 토큰으로 인코딩."""
        tokens = []
        indices = [i for i in rel.element_indices if i not in already_encoded]

        if not indices:
            return tokens

        # 축 기준으로 한쪽 요소만 인코딩
        axis_val = rel.axis_value
        is_x_sym = rel.relation_type == RelationType.SYM_REFLECT_X

        # 요소를 축 기준으로 분류
        left_or_top = []
        on_axis = []

        for idx in indices:
            elem = elem_infos[idx]
            coord = elem.center[0] if is_x_sym else elem.center[1]
            if abs(coord - axis_val) < 5.0:
                on_axis.append(idx)
            elif coord < axis_val:
                left_or_top.append(idx)

        # 축 위 요소 먼저 인코딩
        for idx in on_axis:
            l2 = l2_results[idx]
            inner = [t for t in l2.tokens
                     if t.token_id not in (int(SpecialToken.BOS),
                                            int(SpecialToken.EOS))]
            if tokens:
                tokens.append(self.vocab.special_token(SpecialToken.SEP))
            tokens.extend(inner)

        # 한쪽 요소만 인코딩
        for idx in left_or_top:
            l2 = l2_results[idx]
            inner = [t for t in l2.tokens
                     if t.token_id not in (int(SpecialToken.BOS),
                                            int(SpecialToken.EOS))]
            if tokens:
                tokens.append(self.vocab.special_token(SpecialToken.SEP))
            tokens.extend(inner)

        # 대칭 토큰 + 축 좌표
        if is_x_sym:
            tokens.append(self.vocab.spatial_token(SpatialToken.SYM_REFLECT_X))
        else:
            tokens.append(self.vocab.spatial_token(SpatialToken.SYM_REFLECT_Y))
        tokens.append(self._q(axis_val, axis_val))

        return tokens

    # ===================== 유틸리티 =====================

    def _q(self, x: float, y: float) -> GPLToken:
        """좌표를 ARCS 양자화 후 토큰으로 변환."""
        qc = self.arcs.quantize(x, y)
        return self.vocab.coord_token(qc.level, qc.qx, qc.qy)
