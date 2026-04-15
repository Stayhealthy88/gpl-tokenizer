"""
곡률 분석기
==========
SVG path 명령어 시퀀스에서 각 세그먼트의 곡률 프로파일, 최대 곡률,
접선 방향 등 미분 기하학적 속성을 계산.

비판 문서 반영 (섹션 2.3):
    "미분 기하학적 불변량을 토큰화" — 곡률(κ), 접선 각도(θ),
    호 길이(s) 등을 각 프리미티브에 대해 계산하여 토큰 메타데이터로 부착.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..parser.path_parser import PathCommand, CommandType
from ..utils.math_utils import BezierMath


@dataclass
class CurvatureInfo:
    """하나의 path 세그먼트에 대한 곡률 분석 결과."""
    command_index: int            # 원본 PathCommand 리스트 내 인덱스
    command_type: CommandType
    max_abs_curvature: float      # |κ|의 최대값
    mean_abs_curvature: float     # |κ|의 평균
    curvature_at_start: float     # t=0에서의 곡률
    curvature_at_end: float       # t=1에서의 곡률
    tangent_angle_start: float    # t=0에서의 접선 각도 (rad)
    tangent_angle_end: float      # t=1에서의 접선 각도 (rad)
    arc_length: float             # 호 길이
    is_straight: bool             # 직선 여부 (곡률 ≈ 0)

    @property
    def curvature_complexity(self) -> float:
        """
        곡률 복잡도 — ARCS 쿼드트리 분할의 입력.
        높을수록 더 높은 좌표 해상도가 필요함을 의미.
        """
        return self.max_abs_curvature * self.arc_length

    def quantized_curvature_class(self, n_bins: int = 16) -> int:
        """곡률을 이산 클래스로 양자화 (토큰 DiffAttr용)."""
        # log 스케일로 양자화: 0(직선) ~ n_bins-1(매우 큰 곡률)
        if self.max_abs_curvature < 1e-6:
            return 0
        log_k = np.log1p(self.max_abs_curvature * 100)  # 정규화
        bin_idx = min(int(log_k / 5.0 * n_bins), n_bins - 1)
        return max(0, bin_idx)


class CurvatureAnalyzer:
    """
    PathCommand 시퀀스의 곡률 분석.

    사용법:
        analyzer = CurvatureAnalyzer()
        infos = analyzer.analyze(commands)
    """

    def __init__(self, n_samples: int = 30):
        """
        Args:
            n_samples: 곡률 샘플링 포인트 수 (높을수록 정밀, 느림).
        """
        self.n_samples = n_samples

    def analyze(self, commands: List[PathCommand]) -> List[CurvatureInfo]:
        """
        PathCommand 리스트의 각 세그먼트에 대한 곡률 정보 계산.

        Args:
            commands: resolve_to_absolute() 완료된 PathCommand 리스트
        Returns:
            CurvatureInfo 리스트 (MOVE, CLOSE 제외)
        """
        results = []
        for i, cmd in enumerate(commands):
            info = self._analyze_command(i, cmd)
            if info is not None:
                results.append(info)
        return results

    def _analyze_command(self, index: int, cmd: PathCommand) -> Optional[CurvatureInfo]:
        """개별 PathCommand의 곡률 분석."""
        if cmd.abs_params is None or cmd.start_point is None:
            return None

        if cmd.command_type == CommandType.CUBIC:
            return self._analyze_cubic(index, cmd)
        elif cmd.command_type == CommandType.QUADRATIC:
            return self._analyze_quadratic(index, cmd)
        elif cmd.command_type == CommandType.LINE:
            return self._analyze_line(index, cmd)
        elif cmd.command_type in (CommandType.HLINE, CommandType.VLINE):
            return self._analyze_line(index, cmd)
        elif cmd.command_type == CommandType.ARC:
            return self._analyze_arc(index, cmd)
        else:
            return None  # MOVE, CLOSE

    def _analyze_cubic(self, index: int, cmd: PathCommand) -> CurvatureInfo:
        """큐빅 베지에의 곡률 분석."""
        p0 = np.array(cmd.start_point)
        p1 = np.array(cmd.abs_params[0:2])
        p2 = np.array(cmd.abs_params[2:4])
        p3 = np.array(cmd.abs_params[4:6])

        ts = np.linspace(0, 1, self.n_samples)
        curvatures = np.array([
            BezierMath.curvature_cubic_at(p0, p1, p2, p3, t) for t in ts
        ])
        abs_curvatures = np.abs(curvatures)

        arc_len = BezierMath.arc_length_cubic(p0, p1, p2, p3, self.n_samples * 2)
        theta_start = BezierMath.tangent_angle_cubic(p0, p1, p2, p3, 0.0)
        theta_end = BezierMath.tangent_angle_cubic(p0, p1, p2, p3, 1.0)

        return CurvatureInfo(
            command_index=index,
            command_type=CommandType.CUBIC,
            max_abs_curvature=float(np.max(abs_curvatures)),
            mean_abs_curvature=float(np.mean(abs_curvatures)),
            curvature_at_start=float(curvatures[0]),
            curvature_at_end=float(curvatures[-1]),
            tangent_angle_start=theta_start,
            tangent_angle_end=theta_end,
            arc_length=arc_len,
            is_straight=(float(np.max(abs_curvatures)) < 1e-4)
        )

    def _analyze_quadratic(self, index: int, cmd: PathCommand) -> CurvatureInfo:
        """쿼드라틱 베지에의 곡률 분석."""
        p0 = np.array(cmd.start_point)
        p1 = np.array(cmd.abs_params[0:2])
        p2 = np.array(cmd.abs_params[2:4])

        ts = np.linspace(0, 1, self.n_samples)
        curvatures = np.array([
            BezierMath.curvature_quadratic_at(p0, p1, p2, t) for t in ts
        ])
        abs_curvatures = np.abs(curvatures)

        # 호 길이: 쿼드라틱을 큐빅으로 승격하여 계산
        # Q(P0,P1,P2) → C(P0, P0+2/3*(P1-P0), P2+2/3*(P1-P2), P2)
        cp1 = p0 + (2.0 / 3.0) * (p1 - p0)
        cp2 = p2 + (2.0 / 3.0) * (p1 - p2)
        arc_len = BezierMath.arc_length_cubic(p0, cp1, cp2, p2, self.n_samples * 2)

        d1_start = BezierMath.deriv1_quadratic(p0, p1, p2, 0.0)
        d1_end = BezierMath.deriv1_quadratic(p0, p1, p2, 1.0)
        theta_start = float(np.arctan2(d1_start[1], d1_start[0]))
        theta_end = float(np.arctan2(d1_end[1], d1_end[0]))

        return CurvatureInfo(
            command_index=index,
            command_type=CommandType.QUADRATIC,
            max_abs_curvature=float(np.max(abs_curvatures)),
            mean_abs_curvature=float(np.mean(abs_curvatures)),
            curvature_at_start=float(curvatures[0]),
            curvature_at_end=float(curvatures[-1]),
            tangent_angle_start=theta_start,
            tangent_angle_end=theta_end,
            arc_length=arc_len,
            is_straight=False  # 쿼드라틱은 일반적으로 곡선
        )

    def _analyze_line(self, index: int, cmd: PathCommand) -> CurvatureInfo:
        """직선(L, H, V)의 곡률 분석 — 곡률 0."""
        p0 = np.array(cmd.start_point)
        p1 = np.array(cmd.end_point) if cmd.end_point else np.array(cmd.abs_params[:2])
        length = float(np.linalg.norm(p1 - p0))

        diff = p1 - p0
        if np.linalg.norm(diff) < 1e-12:
            theta = 0.0
        else:
            theta = float(np.arctan2(diff[1], diff[0]))

        return CurvatureInfo(
            command_index=index,
            command_type=cmd.command_type,
            max_abs_curvature=0.0,
            mean_abs_curvature=0.0,
            curvature_at_start=0.0,
            curvature_at_end=0.0,
            tangent_angle_start=theta,
            tangent_angle_end=theta,
            arc_length=length,
            is_straight=True
        )

    def _analyze_arc(self, index: int, cmd: PathCommand) -> CurvatureInfo:
        """원호(A)의 곡률 분석 — 1/r로 근사."""
        ap = cmd.abs_params
        rx, ry = ap[0], ap[1]

        # 타원호의 곡률은 위치에 따라 변하지만, 근사로 평균 반지름의 역수 사용
        if rx < 1e-6 or ry < 1e-6:
            kappa = 0.0
        else:
            avg_r = (rx + ry) / 2.0
            kappa = 1.0 / avg_r

        p0 = np.array(cmd.start_point)
        p1 = np.array(cmd.end_point) if cmd.end_point else np.array(ap[5:7])
        chord_len = float(np.linalg.norm(p1 - p0))

        diff = p1 - p0
        theta = float(np.arctan2(diff[1], diff[0])) if np.linalg.norm(diff) > 1e-12 else 0.0

        return CurvatureInfo(
            command_index=index,
            command_type=CommandType.ARC,
            max_abs_curvature=kappa,
            mean_abs_curvature=kappa,
            curvature_at_start=kappa,
            curvature_at_end=kappa,
            tangent_angle_start=theta,
            tangent_angle_end=theta,
            arc_length=chord_len * 1.2,  # 호 길이 ≈ 현 길이 × 근사 보정
            is_straight=False
        )
