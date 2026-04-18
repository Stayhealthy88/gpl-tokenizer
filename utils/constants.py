"""
기하 임계값 상수 정의
=====================
GPL 프로젝트 전반의 수치 임계값(numerical thresholds)을 한 곳에서 관리.

비판 문서 반영 (섹션 1.4):
    "임계값의 일관성과 상대 오차 기준 도입은 기하 판정의 재현성을 보장한다."

v0.5 분석 보고서 개선 #2:
    이전 버전에서는 continuity.py 와 math_utils.py 에 동일한 성격의 임계값이
    서로 다른 값으로 하드코딩되어 있었다. 이 모듈은 그 값을 단일 출처로 통합한다.

사용법:
    from gpl_tokenizer.utils.constants import GeometricConstants, DEFAULT_CONSTANTS

    # 기본값 사용
    analyzer = ContinuityAnalyzer()   # DEFAULT_CONSTANTS 이 자동 적용

    # 커스텀 값
    strict = GeometricConstants(g1_angle_threshold=0.05)
    analyzer = ContinuityAnalyzer(constants=strict)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeometricConstants:
    """
    GPL 기하 판정에 사용되는 임계값 묶음.

    모든 값은 캔버스 기준 절대값(거리는 픽셀, 각도는 라디안, 곡률은 1/픽셀)이며
    canvas_size ≈ 300 을 기준으로 조정되어 있다. 상대 모드를 사용하면 스케일이
    다른 캔버스에서도 일관된 판정이 가능하다.

    Attributes:
        g0_distance_threshold:
            G0 연속성(끝점-시작점 일치) 판정 거리 임계값 (픽셀).
            기본 0.5 는 300 캔버스에서 약 1/600 상대 오차.

        g1_angle_threshold:
            G1 연속성(접선 방향 일치) 판정 각도 임계값 (라디안).
            기본 0.1 rad ≈ 5.73°. 실무에서 사람 눈으로 꺾임이 보이기 직전 수준.

        g2_curvature_threshold:
            G2 연속성(곡률 일치) 판정 곡률 차이 임계값 (1/픽셀).
            기본 0.05 는 반지름 20px 원 수준의 곡률 격차.

        use_relative_g2:
            True 로 설정하면 |κ_a - κ_b| / max(|κ_a|, |κ_b|, ε) 로 상대 판정.
            곡률이 매우 작은 구간(κ≈0.001)에서 과민 판정을 방지한다.

        g2_relative_tolerance:
            상대 모드에서 허용하는 최대 상대 곡률 오차 (기본 0.2 = 20%).

        g2_epsilon:
            상대 모드에서 0으로 나눔을 방지하는 최소 곡률 (기본 1e-6).

        degenerate_speed_cubed:
            속도^3 가 이 값 이하면 곡률을 0 으로 반환 (퇴화점 처리).

        tangent_zero_norm:
            접선 벡터 norm 이 이 값 이하면 퇴화 접선으로 처리.
    """
    # G0/G1/G2 — 절대 임계값
    g0_distance_threshold: float = 0.5          # 픽셀
    g1_angle_threshold: float = 0.1             # 라디안 (~5.73°)
    g2_curvature_threshold: float = 0.05        # 1/픽셀

    # G2 — 상대 모드 옵션
    use_relative_g2: bool = False
    g2_relative_tolerance: float = 0.2          # 20 %
    g2_epsilon: float = 1e-6

    # 수치 안정성 기준
    degenerate_speed_cubed: float = 1e-12
    tangent_zero_norm: float = 1e-12


# 프로젝트 전역 기본값 — 기존 v0.5 동작과 하위 호환
DEFAULT_CONSTANTS = GeometricConstants()
