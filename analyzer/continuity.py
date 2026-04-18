"""
연속성 분석기
=============
인접한 path 세그먼트 간의 G0/G1/G2 연속성을 판정.
GPL 토큰의 DiffAttr 필드에 연속성 타입을 부착하는 데 사용.

비판 문서 반영 (섹션 1.1):
    "G1/G2 연속성 같은 미분 기하학적 속성이 토큰화 과정에서
     어떻게 보존되거나 인코딩될 수 있는지에 대한 수학적 모델링"
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import IntEnum

from ..parser.path_parser import PathCommand, CommandType
from ..utils.constants import GeometricConstants, DEFAULT_CONSTANTS
from .curvature import CurvatureInfo


class ContinuityLevel(IntEnum):
    """기하학적 연속성 수준."""
    DISCONTINUOUS = -1  # 불연속 (끝점 불일치)
    G0 = 0              # 위치 연속 (끝점 일치)
    G1 = 1              # 접선 방향 연속
    G2 = 2              # 곡률 연속


@dataclass
class ContinuityInfo:
    """인접 세그먼트 간 연속성 분석 결과."""
    segment_a_index: int         # 이전 세그먼트 인덱스
    segment_b_index: int         # 다음 세그먼트 인덱스
    level: ContinuityLevel       # 연속성 수준
    position_error: float        # 끝점-시작점 거리
    angle_error: float           # 접선 각도 차이 (rad)
    curvature_error: float       # 곡률 차이
    junction_point: Tuple[float, float]  # 연결점 좌표


class ContinuityAnalyzer:
    """
    Path 세그먼트 시퀀스의 연속성 분석.

    사용법:
        c_analyzer = ContinuityAnalyzer()
        joints = c_analyzer.analyze(commands, curvature_infos)
    """

    def __init__(self,
                 g0_threshold: Optional[float] = None,
                 g1_angle_threshold: Optional[float] = None,
                 g2_curvature_threshold: Optional[float] = None,
                 constants: Optional[GeometricConstants] = None):
        """
        Args:
            g0_threshold: G0 판정 거리 임계값 (미지정 시 constants 값 사용)
            g1_angle_threshold: G1 판정 접선 각도 차이 임계값 (라디안)
            g2_curvature_threshold: G2 판정 곡률 차이 임계값
            constants: GeometricConstants 인스턴스. 미지정 시 DEFAULT_CONSTANTS 사용.

        Note:
            명시적 인자가 constants 보다 우선한다. 이는 기존 API 호환성을 유지한다.
        """
        c = constants if constants is not None else DEFAULT_CONSTANTS
        self.constants = c
        self.g0_threshold = g0_threshold if g0_threshold is not None else c.g0_distance_threshold
        self.g1_angle_threshold = (
            g1_angle_threshold if g1_angle_threshold is not None else c.g1_angle_threshold
        )
        self.g2_curvature_threshold = (
            g2_curvature_threshold if g2_curvature_threshold is not None else c.g2_curvature_threshold
        )

    def analyze(self, commands: List[PathCommand],
                curvature_infos: List[CurvatureInfo]) -> List[ContinuityInfo]:
        """
        인접한 렌더링 가능 세그먼트 쌍에 대해 연속성을 분석.

        Args:
            commands: resolve_to_absolute() 완료된 PathCommand 리스트
            curvature_infos: CurvatureAnalyzer에서 얻은 곡률 정보
        Returns:
            ContinuityInfo 리스트 (각 연결점에 대해)
        """
        # 곡률 정보를 command_index로 매핑
        curv_map = {ci.command_index: ci for ci in curvature_infos}

        # 렌더링 가능 세그먼트만 추출 (MOVE, CLOSE 제외)
        renderable = [(i, cmd) for i, cmd in enumerate(commands)
                      if cmd.command_type not in (CommandType.MOVE, CommandType.CLOSE)]

        results = []
        for k in range(len(renderable) - 1):
            idx_a, cmd_a = renderable[k]
            idx_b, cmd_b = renderable[k + 1]

            # MOVE가 사이에 있으면 서브패스 경계 → 분석 스킵
            if any(commands[j].command_type == CommandType.MOVE
                   for j in range(idx_a + 1, idx_b)):
                continue

            ci = self._analyze_junction(
                idx_a, cmd_a, curv_map.get(idx_a),
                idx_b, cmd_b, curv_map.get(idx_b)
            )
            results.append(ci)

        return results

    def _analyze_junction(self, idx_a: int, cmd_a: PathCommand,
                          curv_a: CurvatureInfo,
                          idx_b: int, cmd_b: PathCommand,
                          curv_b: CurvatureInfo) -> ContinuityInfo:
        """단일 연결점의 연속성 판정."""

        # --- G0: 끝점-시작점 일치 ---
        end_a = np.array(cmd_a.end_point) if cmd_a.end_point else np.zeros(2)
        start_b = np.array(cmd_b.start_point) if cmd_b.start_point else np.zeros(2)
        pos_err = float(np.linalg.norm(end_a - start_b))

        junction = tuple(((end_a + start_b) / 2).tolist())

        if pos_err > self.g0_threshold:
            return ContinuityInfo(
                segment_a_index=idx_a, segment_b_index=idx_b,
                level=ContinuityLevel.DISCONTINUOUS,
                position_error=pos_err, angle_error=float('inf'),
                curvature_error=float('inf'), junction_point=junction
            )

        # --- G1: 접선 방향 일치 ---
        if curv_a is not None and curv_b is not None:
            theta_a_end = curv_a.tangent_angle_end
            theta_b_start = curv_b.tangent_angle_start
            # 각도 차이 (0~π 범위로 정규화)
            angle_diff = abs(theta_a_end - theta_b_start)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            angle_err = angle_diff
        else:
            angle_err = float('inf')

        if angle_err > self.g1_angle_threshold:
            return ContinuityInfo(
                segment_a_index=idx_a, segment_b_index=idx_b,
                level=ContinuityLevel.G0,
                position_error=pos_err, angle_error=angle_err,
                curvature_error=float('inf'), junction_point=junction
            )

        # --- G2: 곡률 일치 ---
        if curv_a is not None and curv_b is not None:
            kappa_a_end = curv_a.curvature_at_end
            kappa_b_start = curv_b.curvature_at_start
            curv_err = abs(kappa_a_end - kappa_b_start)
        else:
            curv_err = float('inf')

        # 상대 모드: |Δκ| / max(|κ_a|, |κ_b|, ε) ≤ tol
        # 절대 모드: |Δκ| ≤ g2_curvature_threshold (기본)
        if self.constants.use_relative_g2 and curv_a is not None and curv_b is not None:
            denom = max(abs(curv_a.curvature_at_end),
                        abs(curv_b.curvature_at_start),
                        self.constants.g2_epsilon)
            g2_fail = (curv_err / denom) > self.constants.g2_relative_tolerance
        else:
            g2_fail = curv_err > self.g2_curvature_threshold

        if g2_fail:
            return ContinuityInfo(
                segment_a_index=idx_a, segment_b_index=idx_b,
                level=ContinuityLevel.G1,
                position_error=pos_err, angle_error=angle_err,
                curvature_error=curv_err, junction_point=junction
            )

        return ContinuityInfo(
            segment_a_index=idx_a, segment_b_index=idx_b,
            level=ContinuityLevel.G2,
            position_error=pos_err, angle_error=angle_err,
            curvature_error=curv_err, junction_point=junction
        )
