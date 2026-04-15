"""
ARCS — Adaptive Resolution Coordinate System
==============================================
쿼드트리 기반 적응적 좌표 양자화 시스템.

핵심 아이디어:
    시각적 복잡도가 높은 영역(높은 곡률, 경로 교차)에서는 높은 해상도로,
    단순한 영역에서는 낮은 해상도로 좌표를 양자화.

비판 문서 반영 (섹션 1.2):
    "적응적 해상도 양자화와 상대 좌표 인코딩을 결합한
     하이브리드 좌표 인코딩이 구체적으로 어떤 알고리즘으로 구현될 것인지"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class QuadNode:
    """쿼드트리 노드."""
    x: float          # 영역 좌상단 x
    y: float          # 영역 좌상단 y
    size: float        # 영역 크기 (정사각형 가정)
    level: int         # 트리 깊이 (0 = 루트)
    complexity: float = 0.0  # 시각적 복잡도
    children: Optional[List["QuadNode"]] = None  # 4 자식 (NW, NE, SW, SE)

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.size / 2, self.y + self.size / 2)

    @property
    def resolution(self) -> int:
        """이 노드의 해상도 (2^level)."""
        return 2 ** self.level


@dataclass
class QuantizedCoord:
    """ARCS로 양자화된 좌표."""
    level: int         # 쿼드트리 레벨 (해상도 지표)
    qx: int            # 양자화된 x (0 ~ 2^level - 1)
    qy: int            # 양자화된 y (0 ~ 2^level - 1)
    original_x: float  # 원본 x
    original_y: float  # 원본 y

    @property
    def token_str(self) -> str:
        """GPL 토큰 표현."""
        return f"q{self.level}:{self.qx},{self.qy}"

    def to_id(self, max_level: int = 8) -> int:
        """
        고유 토큰 ID 생성.
        ID 공간: level별로 분리하여 충돌 방지.
        """
        level_offset = sum(4 ** l for l in range(self.level))
        return level_offset + self.qy * (2 ** self.level) + self.qx


class ARCS:
    """
    Adaptive Resolution Coordinate System.

    사용법:
        arcs = ARCS(canvas_size=300, max_level=6)
        arcs.build_from_complexity(complexity_map)
        qcoord = arcs.quantize(127.4, 89.7)
        x, y = arcs.dequantize(qcoord)
    """

    def __init__(self, canvas_size: float = 300.0, max_level: int = 6,
                 min_level: int = 2, split_threshold: float = 0.5,
                 weights: Tuple[float, float, float] = (1.0, 0.5, 0.3)):
        """
        Args:
            canvas_size: SVG 캔버스 크기 (정사각형 가정)
            max_level: 최대 쿼드트리 깊이 (해상도 = 2^max_level)
            min_level: 최소 쿼드트리 깊이
            split_threshold: 분할 임계값
            weights: (w_curvature, w_intersection, w_style_var) 복잡도 가중치
        """
        self.canvas_size = canvas_size
        self.max_level = max_level
        self.min_level = min_level
        self.split_threshold = split_threshold
        self.w_curv, self.w_inter, self.w_style = weights
        self.root: Optional[QuadNode] = None

        # 기본 균일 트리 (폴백)
        self._build_uniform()

    def _build_uniform(self):
        """균일 쿼드트리 (기본 폴백)."""
        self.root = self._build_node(0.0, 0.0, self.canvas_size, 0, self.min_level)

    def _build_node(self, x: float, y: float, size: float,
                    level: int, target_level: int) -> QuadNode:
        node = QuadNode(x=x, y=y, size=size, level=level)
        if level < target_level:
            half = size / 2
            node.children = [
                self._build_node(x, y, half, level + 1, target_level),           # NW
                self._build_node(x + half, y, half, level + 1, target_level),     # NE
                self._build_node(x, y + half, half, level + 1, target_level),     # SW
                self._build_node(x + half, y + half, half, level + 1, target_level),  # SE
            ]
        return node

    def build_from_curvatures(self, segment_data: List[Dict]):
        """
        세그먼트 곡률 데이터로부터 적응적 쿼드트리 구축.

        Args:
            segment_data: 각 원소는
                {"bbox": (x1,y1,x2,y2), "max_curvature": float, "arc_length": float}
        """
        # 복잡도 맵 생성: 그리드 기반으로 누적
        grid_res = 2 ** self.max_level
        complexity_grid = np.zeros((grid_res, grid_res))

        for seg in segment_data:
            bbox = seg.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            kappa = seg.get("max_curvature", 0.0)

            # 복잡도 = 곡률 × 호 길이
            c = kappa * seg.get("arc_length", 1.0) * self.w_curv

            # 그리드 셀에 복잡도 누적
            gx1 = max(0, int(x1 / self.canvas_size * grid_res))
            gy1 = max(0, int(y1 / self.canvas_size * grid_res))
            gx2 = min(grid_res - 1, int(x2 / self.canvas_size * grid_res))
            gy2 = min(grid_res - 1, int(y2 / self.canvas_size * grid_res))
            if gx1 <= gx2 and gy1 <= gy2:
                complexity_grid[gy1:gy2 + 1, gx1:gx2 + 1] += c

        # 적응적 쿼드트리 구축
        self.root = self._adaptive_build(
            0.0, 0.0, self.canvas_size, 0, complexity_grid
        )

    def _adaptive_build(self, x: float, y: float, size: float,
                        level: int, complexity_grid: np.ndarray) -> QuadNode:
        """복잡도 기반 적응적 분할."""
        node = QuadNode(x=x, y=y, size=size, level=level)

        # 이 노드에 해당하는 그리드 영역의 복잡도 합산
        grid_res = complexity_grid.shape[0]
        gx1 = max(0, int(x / self.canvas_size * grid_res))
        gy1 = max(0, int(y / self.canvas_size * grid_res))
        gx2 = min(grid_res - 1, int((x + size) / self.canvas_size * grid_res))
        gy2 = min(grid_res - 1, int((y + size) / self.canvas_size * grid_res))

        if gx1 <= gx2 and gy1 <= gy2:
            node.complexity = float(np.sum(complexity_grid[gy1:gy2 + 1, gx1:gx2 + 1]))
        else:
            node.complexity = 0.0

        # 분할 조건: 최소 레벨 미만이면 무조건 분할, 최대 레벨이면 정지
        should_split = (
            (level < self.min_level) or
            (level < self.max_level and node.complexity > self.split_threshold)
        )

        if should_split:
            half = size / 2
            node.children = [
                self._adaptive_build(x, y, half, level + 1, complexity_grid),
                self._adaptive_build(x + half, y, half, level + 1, complexity_grid),
                self._adaptive_build(x, y + half, half, level + 1, complexity_grid),
                self._adaptive_build(x + half, y + half, half, level + 1, complexity_grid),
            ]

        return node

    def quantize(self, x: float, y: float) -> QuantizedCoord:
        """
        연속 좌표를 ARCS로 양자화.

        Args:
            x, y: 원본 좌표
        Returns:
            QuantizedCoord — 적응적 레벨에서 양자화된 좌표
        """
        # 좌표를 [0, canvas_size] 범위로 클램프
        x = max(0.0, min(x, self.canvas_size - 1e-6))
        y = max(0.0, min(y, self.canvas_size - 1e-6))

        # 쿼드트리 탐색하여 리프 노드 찾기
        leaf = self._find_leaf(self.root, x, y)

        # 리프 노드의 레벨에서 양자화
        level = leaf.level
        cell_size = self.canvas_size / (2 ** level)
        qx = int(x / cell_size)
        qy = int(y / cell_size)

        # 범위 클램프
        max_val = (2 ** level) - 1
        qx = min(qx, max_val)
        qy = min(qy, max_val)

        return QuantizedCoord(
            level=level, qx=qx, qy=qy,
            original_x=x, original_y=y
        )

    def dequantize(self, qcoord: QuantizedCoord) -> Tuple[float, float]:
        """
        양자화된 좌표를 연속 좌표로 복원.
        셀 중심으로 복원하여 양자화 오차 최소화.
        """
        cell_size = self.canvas_size / (2 ** qcoord.level)
        x = (qcoord.qx + 0.5) * cell_size
        y = (qcoord.qy + 0.5) * cell_size
        return (x, y)

    def _find_leaf(self, node: QuadNode, x: float, y: float) -> QuadNode:
        """좌표 (x,y)가 속하는 리프 노드 탐색."""
        if node.is_leaf:
            return node
        half = node.size / 2
        # 사분면 결정
        if x < node.x + half:
            if y < node.y + half:
                return self._find_leaf(node.children[0], x, y)  # NW
            else:
                return self._find_leaf(node.children[2], x, y)  # SW
        else:
            if y < node.y + half:
                return self._find_leaf(node.children[1], x, y)  # NE
            else:
                return self._find_leaf(node.children[3], x, y)  # SE

    def quantization_error(self, x: float, y: float) -> float:
        """양자화 왕복 오차 (유클리드 거리)."""
        qc = self.quantize(x, y)
        rx, ry = self.dequantize(qc)
        return np.sqrt((x - rx) ** 2 + (y - ry) ** 2)

    def total_leaf_count(self) -> int:
        """리프 노드 총 수 (=좌표 어휘 크기의 지표)."""
        return self._count_leaves(self.root)

    def _count_leaves(self, node: QuadNode) -> int:
        if node.is_leaf:
            return 1
        return sum(self._count_leaves(c) for c in node.children)

    def level_distribution(self) -> Dict[int, int]:
        """레벨별 리프 노드 수."""
        dist = {}
        self._collect_level_dist(self.root, dist)
        return dist

    def _collect_level_dist(self, node: QuadNode, dist: Dict[int, int]):
        if node.is_leaf:
            dist[node.level] = dist.get(node.level, 0) + 1
        else:
            for c in node.children:
                self._collect_level_dist(c, dist)
