"""
Spatial Relation Analyzer — Level 3 공간 관계 분석
=================================================
다중 기하 요소 간 공간 관계(정렬, 대칭, 비례)를 탐지.

v0.3 핵심 모듈:
    ALIGN: 요소들이 수평/수직 중심축을 공유
    SYM: 요소들이 축 기준 반사 대칭
    EQUAL_SPACE: 요소들이 등간격 배치
    EQUAL_SIZE: 요소들이 동일 크기

탐지 알고리즘:
    1. 요소별 중심점·바운딩 박스 추출
    2. 값 기준 그룹화 (tolerance 이내 동일 그룹)
    3. 그룹 내 등간격·대칭 패턴 검증
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from enum import Enum
import math


class RelationType(Enum):
    """공간 관계 유형."""
    ALIGN_CENTER_H = "ALIGN_CENTER_H"   # 수평 중심 정렬 (같은 cy)
    ALIGN_CENTER_V = "ALIGN_CENTER_V"   # 수직 중심 정렬 (같은 cx)
    SYM_REFLECT_X = "SYM_REFLECT_X"     # 세로축 반사 대칭 (좌우 대칭)
    SYM_REFLECT_Y = "SYM_REFLECT_Y"     # 가로축 반사 대칭 (상하 대칭)
    EQUAL_SPACE_H = "EQUAL_SPACE_H"     # 수평 등간격
    EQUAL_SPACE_V = "EQUAL_SPACE_V"     # 수직 등간격
    EQUAL_SIZE = "EQUAL_SIZE"           # 동일 크기


@dataclass
class ElementInfo:
    """기하 요소 정보 — 공간 관계 분석의 기본 단위."""
    index: int                                      # 요소 인덱스
    element_type: str                               # "circle", "rect", "ellipse", "path"
    center: Tuple[float, float]                     # 중심점 (cx, cy)
    bbox: Tuple[float, float, float, float]         # (x_min, y_min, x_max, y_max)
    width: float                                    # 바운딩 박스 너비
    height: float                                   # 바운딩 박스 높이
    params: dict = field(default_factory=dict)       # 원본 도형 파라미터


@dataclass
class SpatialRelation:
    """탐지된 공간 관계."""
    relation_type: RelationType
    element_indices: List[int]                      # 관련 요소 인덱스 (정렬됨)
    axis_value: Optional[float] = None              # 정렬/대칭 축 좌표
    spacing: Optional[float] = None                 # 등간격 값
    confidence: float = 1.0


class SpatialAnalyzer:
    """
    다중 요소 간 공간 관계를 분석.

    사용법:
        analyzer = SpatialAnalyzer()
        elements = [ElementInfo(...), ...]
        relations = analyzer.analyze(elements)
        for rel in relations:
            print(f"{rel.relation_type}: indices={rel.element_indices}")
    """

    def __init__(self,
                 align_tolerance: float = 3.0,
                 size_tolerance: float = 5.0,
                 spacing_tolerance: float = 3.0,
                 sym_tolerance: float = 5.0):
        """
        Args:
            align_tolerance: 정렬 판정 오차 (px)
            size_tolerance: 크기 동일 판정 오차 (px)
            spacing_tolerance: 등간격 판정 오차 (px)
            sym_tolerance: 대칭 매칭 오차 (px)
        """
        self.align_tol = align_tolerance
        self.size_tol = size_tolerance
        self.spacing_tol = spacing_tolerance
        self.sym_tol = sym_tolerance

    def analyze(self, elements: List[ElementInfo]) -> List[SpatialRelation]:
        """
        전체 요소 목록에서 공간 관계를 탐지.

        Args:
            elements: ElementInfo 리스트 (최소 2개)
        Returns:
            SpatialRelation 리스트 (중복 없이)
        """
        if len(elements) < 2:
            return []

        relations = []

        # 1. 정렬 탐지
        relations.extend(self._detect_alignment(elements))

        # 2. 등간격 탐지 (정렬 그룹 내에서)
        relations.extend(self._detect_equal_spacing(elements))

        # 3. 대칭 탐지
        relations.extend(self._detect_symmetry(elements))

        # 4. 동일 크기 탐지
        relations.extend(self._detect_equal_size(elements))

        return relations

    # ===================== 정렬 탐지 =====================

    def _detect_alignment(self, elements: List[ElementInfo]) -> List[SpatialRelation]:
        """중심 정렬 그룹을 탐지."""
        relations = []

        # 수평 정렬 (같은 cy)
        h_groups = self._group_by_value(
            elements, key=lambda e: e.center[1], tolerance=self.align_tol
        )
        for group_val, indices in h_groups:
            if len(indices) >= 2:
                relations.append(SpatialRelation(
                    relation_type=RelationType.ALIGN_CENTER_H,
                    element_indices=sorted(indices),
                    axis_value=group_val,
                ))

        # 수직 정렬 (같은 cx)
        v_groups = self._group_by_value(
            elements, key=lambda e: e.center[0], tolerance=self.align_tol
        )
        for group_val, indices in v_groups:
            if len(indices) >= 2:
                relations.append(SpatialRelation(
                    relation_type=RelationType.ALIGN_CENTER_V,
                    element_indices=sorted(indices),
                    axis_value=group_val,
                ))

        return relations

    # ===================== 등간격 탐지 =====================

    def _detect_equal_spacing(self, elements: List[ElementInfo]) -> List[SpatialRelation]:
        """정렬된 그룹 내에서 등간격 배치를 탐지."""
        relations = []

        # 수평 정렬 그룹에서 수평 등간격 확인
        h_groups = self._group_by_value(
            elements, key=lambda e: e.center[1], tolerance=self.align_tol
        )
        for _, indices in h_groups:
            if len(indices) >= 3:
                group = [elements[i] for i in indices]
                sorted_group = sorted(group, key=lambda e: e.center[0])
                sorted_indices = [e.index for e in sorted_group]

                spacings = [
                    sorted_group[i + 1].center[0] - sorted_group[i].center[0]
                    for i in range(len(sorted_group) - 1)
                ]
                if self._all_similar(spacings, self.spacing_tol):
                    avg_spacing = sum(spacings) / len(spacings)
                    relations.append(SpatialRelation(
                        relation_type=RelationType.EQUAL_SPACE_H,
                        element_indices=sorted_indices,
                        spacing=avg_spacing,
                    ))

        # 수직 정렬 그룹에서 수직 등간격 확인
        v_groups = self._group_by_value(
            elements, key=lambda e: e.center[0], tolerance=self.align_tol
        )
        for _, indices in v_groups:
            if len(indices) >= 3:
                group = [elements[i] for i in indices]
                sorted_group = sorted(group, key=lambda e: e.center[1])
                sorted_indices = [e.index for e in sorted_group]

                spacings = [
                    sorted_group[i + 1].center[1] - sorted_group[i].center[1]
                    for i in range(len(sorted_group) - 1)
                ]
                if self._all_similar(spacings, self.spacing_tol):
                    avg_spacing = sum(spacings) / len(spacings)
                    relations.append(SpatialRelation(
                        relation_type=RelationType.EQUAL_SPACE_V,
                        element_indices=sorted_indices,
                        spacing=avg_spacing,
                    ))

        return relations

    # ===================== 대칭 탐지 =====================

    def _detect_symmetry(self, elements: List[ElementInfo]) -> List[SpatialRelation]:
        """축 반사 대칭을 탐지."""
        relations = []

        if len(elements) < 2:
            return relations

        # 전체 레이아웃의 중심축 계산
        all_cx = [e.center[0] for e in elements]
        all_cy = [e.center[1] for e in elements]
        mid_x = (min(all_cx) + max(all_cx)) / 2
        mid_y = (min(all_cy) + max(all_cy)) / 2

        # X축(세로축) 대칭: x = mid_x 기준 좌우 매칭
        x_sym = self._check_axis_symmetry(elements, axis='x', axis_value=mid_x)
        if x_sym:
            relations.append(SpatialRelation(
                relation_type=RelationType.SYM_REFLECT_X,
                element_indices=sorted(x_sym),
                axis_value=mid_x,
            ))

        # Y축(가로축) 대칭: y = mid_y 기준 상하 매칭
        y_sym = self._check_axis_symmetry(elements, axis='y', axis_value=mid_y)
        if y_sym:
            relations.append(SpatialRelation(
                relation_type=RelationType.SYM_REFLECT_Y,
                element_indices=sorted(y_sym),
                axis_value=mid_y,
            ))

        return relations

    def _check_axis_symmetry(self, elements: List[ElementInfo],
                              axis: str, axis_value: float) -> Optional[set]:
        """주어진 축에 대해 전체 대칭이 성립하는지 확인."""
        matched_pairs = self._find_symmetric_pairs(elements, axis, axis_value)
        all_indices = set()

        for a, b in matched_pairs:
            all_indices.add(a)
            all_indices.add(b)

        # 축 위 요소도 포함
        for e in elements:
            coord = e.center[0] if axis == 'x' else e.center[1]
            if abs(coord - axis_value) < self.sym_tol:
                all_indices.add(e.index)

        # 전체 요소가 대칭에 참여해야 함
        if len(all_indices) >= len(elements):
            return all_indices
        return None

    def _find_symmetric_pairs(self, elements: List[ElementInfo],
                               axis: str, axis_value: float) -> List[Tuple[int, int]]:
        """대칭축 기준으로 요소 쌍을 매칭."""
        pairs = []
        used = set()

        for i, ei in enumerate(elements):
            if i in used:
                continue

            # 축 위의 요소는 스킵 (자체 대칭)
            coord = ei.center[0] if axis == 'x' else ei.center[1]
            if abs(coord - axis_value) < self.sym_tol:
                used.add(i)
                continue

            # 대칭 위치 계산
            if axis == 'x':
                mirror = (2 * axis_value - ei.center[0], ei.center[1])
            else:
                mirror = (ei.center[0], 2 * axis_value - ei.center[1])

            # 매칭 요소 탐색
            best_match = None
            best_dist = float('inf')

            for j, ej in enumerate(elements):
                if j in used or j == i:
                    continue
                d = math.dist(mirror, ej.center)
                if d < self.sym_tol and d < best_dist:
                    # 크기도 유사해야 함
                    if (abs(ei.width - ej.width) < self.size_tol and
                            abs(ei.height - ej.height) < self.size_tol):
                        best_match = j
                        best_dist = d

            if best_match is not None:
                pairs.append((i, best_match))
                used.add(i)
                used.add(best_match)

        return pairs

    # ===================== 동일 크기 탐지 =====================

    def _detect_equal_size(self, elements: List[ElementInfo]) -> List[SpatialRelation]:
        """동일 크기 요소 그룹을 탐지."""
        relations = []
        used = set()

        for i, ei in enumerate(elements):
            if i in used:
                continue
            group = [i]
            for j, ej in enumerate(elements):
                if j <= i or j in used:
                    continue
                if (abs(ei.width - ej.width) < self.size_tol and
                        abs(ei.height - ej.height) < self.size_tol):
                    group.append(j)

            if len(group) >= 2:
                for idx in group:
                    used.add(idx)
                relations.append(SpatialRelation(
                    relation_type=RelationType.EQUAL_SIZE,
                    element_indices=sorted(group),
                ))

        return relations

    # ===================== 유틸리티 =====================

    def _group_by_value(self, elements: List[ElementInfo],
                        key: Callable, tolerance: float
                        ) -> List[Tuple[float, List[int]]]:
        """값 기준 그룹화 (tolerance 이내 같은 그룹)."""
        if not elements:
            return []

        indexed_vals = [(key(e), e.index) for e in elements]
        indexed_vals.sort(key=lambda x: x[0])

        groups = []
        current_group = [indexed_vals[0]]

        for i in range(1, len(indexed_vals)):
            if indexed_vals[i][0] - current_group[0][0] <= tolerance:
                current_group.append(indexed_vals[i])
            else:
                if len(current_group) >= 2:
                    avg_val = sum(v for v, _ in current_group) / len(current_group)
                    indices = [idx for _, idx in current_group]
                    groups.append((avg_val, indices))
                current_group = [indexed_vals[i]]

        if len(current_group) >= 2:
            avg_val = sum(v for v, _ in current_group) / len(current_group)
            indices = [idx for _, idx in current_group]
            groups.append((avg_val, indices))

        return groups

    def _all_similar(self, values: List[float], tolerance: float) -> bool:
        """모든 값이 tolerance 이내로 유사한지 확인."""
        if not values:
            return False
        avg = sum(values) / len(values)
        return all(abs(v - avg) <= tolerance for v in values)
