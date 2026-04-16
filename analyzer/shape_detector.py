"""
Shape Detector — Level 2 기하 프리미티브 인식
=============================================
PathCommand 시퀀스에서 원, 타원, 사각형 등 고수준 기하 도형 패턴을 인식.

v0.2 핵심 모듈:
    4 cubic beziers → CIRCLE (단일 토큰)
    4 lines + close → RECT (단일 토큰)
    4 cubics (비균일) → ELLIPSE (단일 토큰)
    4 lines + 4 arcs + close → ROUND_RECT (단일 토큰)

인식 알고리즘:
    1. 명령어 시퀀스 패턴 매칭 (M + 4C + Z → 원 후보)
    2. 기하학적 검증 (중심점, 반지름 일관성, 대칭성)
    3. 허용 오차(tolerance) 기반 매칭
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import math

from ..parser.path_parser import PathCommand, CommandType


class ShapeType(Enum):
    """인식 가능한 도형 유형."""
    CIRCLE = "CIRCLE"
    ELLIPSE = "ELLIPSE"
    RECT = "RECT"
    ROUND_RECT = "ROUND_RECT"


@dataclass
class DetectedShape:
    """인식된 도형 정보."""
    shape_type: ShapeType
    start_index: int       # commands 내 시작 인덱스 (MOVE 포함)
    end_index: int         # commands 내 끝 인덱스 (CLOSE 포함)
    n_commands: int        # 대체되는 명령어 수
    params: dict           # 도형별 파라미터

    # CIRCLE: {"cx", "cy", "r"}
    # ELLIPSE: {"cx", "cy", "rx", "ry"}
    # RECT: {"x", "y", "width", "height"}
    # ROUND_RECT: {"x", "y", "width", "height", "rx", "ry"}

    confidence: float = 1.0  # 인식 신뢰도 (0-1)


class ShapeDetector:
    """
    PathCommand 시퀀스에서 기하 프리미티브를 인식.

    사용법:
        detector = ShapeDetector()
        shapes = detector.detect(commands)
        for shape in shapes:
            print(f"{shape.shape_type}: {shape.params}")
    """

    def __init__(self,
                 circle_tolerance: float = 5.0,
                 rect_tolerance: float = 3.0,
                 angle_tolerance: float = 0.15):
        """
        Args:
            circle_tolerance: 원 인식 반지름 오차 허용 (px)
            rect_tolerance: 사각형 인식 변 길이 오차 허용 (px)
            angle_tolerance: 직각 판정 각도 오차 허용 (rad, ~8.6°)
        """
        self.circle_tol = circle_tolerance
        self.rect_tol = rect_tolerance
        self.angle_tol = angle_tolerance

    def detect(self, commands: List[PathCommand]) -> List[DetectedShape]:
        """
        전체 명령어 시퀀스에서 도형 패턴을 탐지.

        Returns:
            DetectedShape 리스트 (겹치지 않는 도형만 반환)
        """
        if len(commands) < 3:
            return []

        shapes = []
        used = set()  # 이미 도형으로 인식된 인덱스

        # 패스 단위로 분리 (MOVE로 시작, CLOSE로 끝나는 서브패스)
        subpaths = self._split_subpaths(commands)

        for sp_start, sp_end in subpaths:
            if sp_start in used:
                continue

            sp_cmds = commands[sp_start:sp_end + 1]

            # 원/타원 검출: M + 4C + Z
            shape = self._detect_circle_or_ellipse(sp_cmds, sp_start)
            if shape:
                shapes.append(shape)
                for i in range(shape.start_index, shape.end_index + 1):
                    used.add(i)
                continue

            # 사각형 검출: M + 3-4L + Z
            shape = self._detect_rect(sp_cmds, sp_start)
            if shape:
                shapes.append(shape)
                for i in range(shape.start_index, shape.end_index + 1):
                    used.add(i)
                continue

            # 둥근 사각형 검출: M + (L+A)*4 + Z 또는 유사 패턴
            shape = self._detect_round_rect(sp_cmds, sp_start)
            if shape:
                shapes.append(shape)
                for i in range(shape.start_index, shape.end_index + 1):
                    used.add(i)
                continue

        return shapes

    # ===================== 서브패스 분리 =====================

    def _split_subpaths(self, commands: List[PathCommand]) -> List[Tuple[int, int]]:
        """MOVE~CLOSE 단위 서브패스로 분리."""
        subpaths = []
        current_start = None

        for i, cmd in enumerate(commands):
            if cmd.command_type == CommandType.MOVE:
                current_start = i
            elif cmd.command_type == CommandType.CLOSE and current_start is not None:
                subpaths.append((current_start, i))
                current_start = None

        return subpaths

    # ===================== 원/타원 검출 =====================

    def _detect_circle_or_ellipse(self, cmds: List[PathCommand],
                                   offset: int) -> Optional[DetectedShape]:
        """
        M + 4C + Z 패턴에서 원 또는 타원을 검출.

        원 검출 원리:
            SVG에서 원은 4개의 cubic bezier로 근사됨 (kappa = 0.5522847498).
            4개 세그먼트의 끝점이 원의 동/남/서/북 점에 위치하고,
            중심점까지의 거리가 일정하면 원으로 판정.

        타원 검출:
            중심점까지의 x축/y축 거리가 각각 일정하면 타원으로 판정.
        """
        # 패턴 검사: M + 4C + Z = 6개 명령어
        if len(cmds) != 6:
            return None
        if cmds[0].command_type != CommandType.MOVE:
            return None
        if cmds[-1].command_type != CommandType.CLOSE:
            return None
        for i in range(1, 5):
            if cmds[i].command_type != CommandType.CUBIC:
                return None

        # 4개 cubic의 끝점 추출
        endpoints = []
        start_pt = cmds[0].end_point
        if start_pt is None:
            return None
        endpoints.append(start_pt)

        for i in range(1, 5):
            ep = cmds[i].end_point
            if ep is None:
                return None
            endpoints.append(ep)

        # 끝점 5개: [시작점, C1끝, C2끝, C3끝, C4끝]
        # C4의 끝점은 시작점과 일치해야 함 (CLOSE)
        p0, p1, p2, p3, p4 = endpoints

        # 폐합 검사
        close_dist = math.dist(p0, p4)
        if close_dist > self.circle_tol * 2:
            return None

        # 중심점 계산 (대향 끝점의 중점)
        # p0-p2가 대향, p1-p3이 대향
        cx1 = (p0[0] + p2[0]) / 2
        cy1 = (p0[1] + p2[1]) / 2
        cx2 = (p1[0] + p3[0]) / 2
        cy2 = (p1[1] + p3[1]) / 2

        # 두 중심 추정치가 일치하는지 확인
        center_dist = math.dist((cx1, cy1), (cx2, cy2))
        if center_dist > self.circle_tol:
            return None

        cx = (cx1 + cx2) / 2
        cy = (cy1 + cy2) / 2

        # 각 끝점에서 중심까지의 거리
        dists = [math.dist((cx, cy), p) for p in [p0, p1, p2, p3]]

        # x축/y축 반지름 분리 (타원 검출)
        rx_candidates = [abs(p[0] - cx) for p in [p0, p2]]
        ry_candidates = [abs(p[1] - cy) for p in [p1, p3]]

        # p0, p2는 x축 방향 (동/서), p1, p3은 y축 방향 (남/북)
        # 또는 반대일 수 있으므로 둘 다 확인
        rx1 = max(abs(p0[0] - cx), abs(p2[0] - cx))
        ry1 = max(abs(p1[1] - cy), abs(p3[1] - cy))
        rx2 = max(abs(p0[1] - cy), abs(p2[1] - cy))
        ry2 = max(abs(p1[0] - cx), abs(p3[0] - cx))

        # Case 1: p0/p2 = x축, p1/p3 = y축
        if rx2 < self.circle_tol and ry2 < self.circle_tol:
            rx, ry = rx1, ry1
        # Case 2: p0/p2 = y축, p1/p3 = x축
        elif rx1 < self.circle_tol and ry1 < self.circle_tol:
            rx, ry = ry2, rx2
        else:
            # 대향점 체크: 모든 점에서 중심까지 거리로 판단
            avg_r = sum(dists) / 4
            r_spread = max(dists) - min(dists)
            if r_spread < self.circle_tol:
                rx = ry = avg_r
            else:
                return None

        # 반지름 유효성 (너무 작으면 제외)
        if rx < 1.0 or ry < 1.0:
            return None

        # 원 vs 타원 판정
        r_diff = abs(rx - ry)
        confidence = max(0.5, 1.0 - center_dist / self.circle_tol)

        if r_diff < self.circle_tol:
            # 원
            r = (rx + ry) / 2
            return DetectedShape(
                shape_type=ShapeType.CIRCLE,
                start_index=offset,
                end_index=offset + 5,
                n_commands=6,
                params={"cx": cx, "cy": cy, "r": r},
                confidence=confidence
            )
        else:
            # 타원
            return DetectedShape(
                shape_type=ShapeType.ELLIPSE,
                start_index=offset,
                end_index=offset + 5,
                n_commands=6,
                params={"cx": cx, "cy": cy, "rx": rx, "ry": ry},
                confidence=confidence
            )

    # ===================== 사각형 검출 =====================

    def _detect_rect(self, cmds: List[PathCommand],
                     offset: int) -> Optional[DetectedShape]:
        """
        M + 3~4L + Z 패턴에서 사각형을 검출.

        직사각형 검출 원리:
            - 4개 꼭짓점이 직교 배치 (인접 변이 수직)
            - 대향 변 길이 일치
        """
        # M + 3L + Z (4개 점, 마지막 CLOSE가 자동 연결) 또는 M + 4L + Z
        n = len(cmds)
        if n not in (5, 6):
            return None

        if cmds[0].command_type != CommandType.MOVE:
            return None
        if cmds[-1].command_type != CommandType.CLOSE:
            return None

        # 중간 명령어가 모두 LINE인지 확인 (HLINE/VLINE 포함)
        line_types = {CommandType.LINE, CommandType.HLINE, CommandType.VLINE}
        for i in range(1, n - 1):
            if cmds[i].command_type not in line_types:
                return None

        # 꼭짓점 추출
        vertices = []
        start = cmds[0].end_point
        if start is None:
            return None
        vertices.append(start)

        for i in range(1, n - 1):
            ep = cmds[i].end_point
            if ep is None:
                return None
            vertices.append(ep)

        # 3L인 경우 4번째 꼭짓점 → 시작점 (CLOSE가 연결)
        # 4L인 경우 마지막 L의 끝점이 시작점 근처여야 함
        if len(vertices) == 4:
            close_dist = math.dist(vertices[3], vertices[0])
            if close_dist > self.rect_tol:
                pass  # 4개 꼭짓점이 정확히 닫히지 않아도 사각형일 수 있음
        elif len(vertices) == 3:
            # CLOSE가 자동 연결하므로 4번째 = 시작점
            vertices.append(vertices[0])

        if len(vertices) < 4:
            return None

        # 직각 검증: 인접 변이 수직인지
        v = vertices[:4]
        edges = []
        for i in range(4):
            j = (i + 1) % 4
            dx = v[j][0] - v[i][0]
            dy = v[j][1] - v[i][1]
            edges.append((dx, dy))

        # 각 인접 변의 내적 검사 (직각이면 0)
        for i in range(4):
            j = (i + 1) % 4
            dot = edges[i][0] * edges[j][0] + edges[i][1] * edges[j][1]
            len_a = math.sqrt(edges[i][0]**2 + edges[i][1]**2)
            len_b = math.sqrt(edges[j][0]**2 + edges[j][1]**2)
            if len_a < 0.1 or len_b < 0.1:
                return None
            cos_angle = dot / (len_a * len_b)
            if abs(cos_angle) > self.angle_tol:
                return None

        # 대향 변 길이 검사
        len_edges = [math.sqrt(e[0]**2 + e[1]**2) for e in edges]
        if abs(len_edges[0] - len_edges[2]) > self.rect_tol:
            return None
        if abs(len_edges[1] - len_edges[3]) > self.rect_tol:
            return None

        # 바운딩 박스 계산
        xs = [p[0] for p in v]
        ys = [p[1] for p in v]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width = x_max - x_min
        height = y_max - y_min

        if width < 1.0 or height < 1.0:
            return None

        return DetectedShape(
            shape_type=ShapeType.RECT,
            start_index=offset,
            end_index=offset + n - 1,
            n_commands=n,
            params={"x": x_min, "y": y_min, "width": width, "height": height},
            confidence=1.0
        )

    # ===================== 둥근 사각형 검출 =====================

    def _detect_round_rect(self, cmds: List[PathCommand],
                            offset: int) -> Optional[DetectedShape]:
        """
        M + (L + C/A)*4 + Z 패턴에서 둥근 사각형을 검출.

        일반적인 SVG rounded rect 패턴:
            M (x+rx) y → H (x+w-rx) → A rx ry ... → V (y+h-ry) → A ... → H (x+rx) → A ... → V (y+ry) → A ... → Z
        """
        n = len(cmds)
        if n < 8:  # 최소 M + 4L + 4C/A + Z
            return None

        if cmds[0].command_type != CommandType.MOVE:
            return None
        if cmds[-1].command_type != CommandType.CLOSE:
            return None

        # 중간 명령어에 ARC 또는 CUBIC이 정확히 4개 있는지 확인
        curve_types = {CommandType.ARC, CommandType.CUBIC}
        line_types = {CommandType.LINE, CommandType.HLINE, CommandType.VLINE}
        curve_count = sum(1 for c in cmds[1:-1] if c.command_type in curve_types)
        line_count = sum(1 for c in cmds[1:-1] if c.command_type in line_types)

        if curve_count != 4 or line_count < 3:
            return None

        # 모든 끝점 추출
        pts = []
        for cmd in cmds[:-1]:  # CLOSE 제외
            if cmd.end_point:
                pts.append(cmd.end_point)

        if len(pts) < 8:
            return None

        # 바운딩 박스
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width = x_max - x_min
        height = y_max - y_min

        if width < 2.0 or height < 2.0:
            return None

        # ARC/CUBIC 세그먼트의 시작/끝점에서 코너 반지름 추정
        curve_cmds = [c for c in cmds[1:-1] if c.command_type in curve_types]
        rx_estimates = []
        ry_estimates = []

        for cc in curve_cmds:
            if cc.start_point and cc.end_point:
                dx = abs(cc.end_point[0] - cc.start_point[0])
                dy = abs(cc.end_point[1] - cc.start_point[1])
                rx_estimates.append(dx)
                ry_estimates.append(dy)

        if not rx_estimates:
            return None

        rx = sum(rx_estimates) / len(rx_estimates)
        ry = sum(ry_estimates) / len(ry_estimates)

        # 코너 반지름이 너무 크거나 작으면 제외
        if rx < 0.5 or ry < 0.5:
            return None
        if rx > width / 2 + self.rect_tol or ry > height / 2 + self.rect_tol:
            return None

        return DetectedShape(
            shape_type=ShapeType.ROUND_RECT,
            start_index=offset,
            end_index=offset + n - 1,
            n_commands=n,
            params={
                "x": x_min, "y": y_min,
                "width": width, "height": height,
                "rx": rx, "ry": ry
            },
            confidence=0.85
        )
