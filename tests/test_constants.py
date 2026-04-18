"""
GeometricConstants 통합 테스트
==============================
v0.5.1 개선 #1: continuity.py 와 math_utils.py 임계값 단일 출처 통합.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpl_tokenizer.utils.constants import GeometricConstants, DEFAULT_CONSTANTS
from gpl_tokenizer.utils.math_utils import BezierMath
from gpl_tokenizer.analyzer.continuity import ContinuityAnalyzer
from gpl_tokenizer.analyzer.curvature import CurvatureInfo


passed = failed = 0


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


print("=" * 60)
print("v0.5.1 Constants Integration Tests")
print("=" * 60)

# ------------------------------------------------------------
# 1. 기본값 일관성
# ------------------------------------------------------------
print("\n[1] 기본값 일관성")

ana = ContinuityAnalyzer()
check(ana.g0_threshold == DEFAULT_CONSTANTS.g0_distance_threshold,
      f"ContinuityAnalyzer.g0 == DEFAULT.g0 ({ana.g0_threshold})")
check(ana.g1_angle_threshold == DEFAULT_CONSTANTS.g1_angle_threshold,
      f"ContinuityAnalyzer.g1 == DEFAULT.g1 ({ana.g1_angle_threshold})")
check(ana.g2_curvature_threshold == DEFAULT_CONSTANTS.g2_curvature_threshold,
      f"ContinuityAnalyzer.g2 == DEFAULT.g2 ({ana.g2_curvature_threshold})")

# BezierMath 헬퍼도 같은 기본값을 사용해야 함
e_tangent = np.array([1.0, 0.0])
s_tangent = np.array([np.cos(0.08), np.sin(0.08)])  # ~4.58° 차이
# ContinuityAnalyzer 기본(0.1 rad) 에서 G1 로 판정되므로 math_utils 도 True 여야 함
check(BezierMath.check_g1_continuity(e_tangent, s_tangent) == True,
      "math_utils.check_g1 기본 임계값이 ContinuityAnalyzer 와 일치")

# 반대로 0.15 rad 차이는 두 쪽 모두 False 여야 함
s_tangent_big = np.array([np.cos(0.15), np.sin(0.15)])
check(BezierMath.check_g1_continuity(e_tangent, s_tangent_big) == False,
      "math_utils.check_g1 은 0.15 rad 을 G1 아님으로 판정")

# G2 동일
check(BezierMath.check_g2_continuity(0.01, 0.03) == True,
      "math_utils.check_g2 기본값(0.05) 에서 Δκ=0.02 는 G2")
check(BezierMath.check_g2_continuity(0.01, 0.07) == False,
      "math_utils.check_g2 기본값(0.05) 에서 Δκ=0.06 는 G2 아님")


# ------------------------------------------------------------
# 2. 명시적 임계값 우선권
# ------------------------------------------------------------
print("\n[2] 명시적 인자 우선권")

strict = ContinuityAnalyzer(g1_angle_threshold=0.01)
check(strict.g1_angle_threshold == 0.01,
      "명시 인자가 constants 보다 우선")

# 커스텀 constants
custom = GeometricConstants(g1_angle_threshold=0.5)
loose = ContinuityAnalyzer(constants=custom)
check(loose.g1_angle_threshold == 0.5,
      "커스텀 constants 의 값이 적용")


# ------------------------------------------------------------
# 3. 상대 G2 모드
# ------------------------------------------------------------
print("\n[3] 상대 G2 모드")

# 절대 모드: |0.001 - 0.06| = 0.059 > 0.05 → G2 실패
abs_mode = GeometricConstants(use_relative_g2=False)
rel_mode = GeometricConstants(use_relative_g2=True, g2_relative_tolerance=0.2)

# 매우 작은 곡률(κ≈0.001)에서 큰 상대 차이는 상대 모드에서 G2 실패
# 절대 차 0.02 (|0.001-0.021|) — 절대 기준(0.05)은 통과, 상대 기준(|Δκ|/max(|κ|)=0.02/0.021≈0.95>0.2) 은 실패
from gpl_tokenizer.parser.path_parser import PathCommand, CommandType

curv_a = CurvatureInfo(
    command_index=0, command_type=CommandType.CUBIC,
    curvature_at_start=0.001, curvature_at_end=0.001,
    max_abs_curvature=0.001, mean_abs_curvature=0.001,
    tangent_angle_start=0.0, tangent_angle_end=0.0,
    arc_length=10.0, is_straight=False,
)
curv_b = CurvatureInfo(
    command_index=1, command_type=CommandType.CUBIC,
    curvature_at_start=0.021, curvature_at_end=0.021,
    max_abs_curvature=0.021, mean_abs_curvature=0.021,
    tangent_angle_start=0.0, tangent_angle_end=0.0,
    arc_length=10.0, is_straight=False,
)

cmd_a = PathCommand(
    command_type=CommandType.CUBIC,
    is_relative=False,
    params=[],
    start_point=(0.0, 0.0),
    end_point=(10.0, 10.0),
)
cmd_b = PathCommand(
    command_type=CommandType.CUBIC,
    is_relative=False,
    params=[],
    start_point=(10.0, 10.0),   # G0 성립
    end_point=(20.0, 20.0),
)

ana_abs = ContinuityAnalyzer(constants=abs_mode)
result_abs = ana_abs._analyze_junction(0, cmd_a, curv_a, 1, cmd_b, curv_b)

ana_rel = ContinuityAnalyzer(constants=rel_mode)
result_rel = ana_rel._analyze_junction(0, cmd_a, curv_a, 1, cmd_b, curv_b)

# 절대 모드에서는 G2 성립 (|Δκ|=0.02 < 0.05)
from gpl_tokenizer.analyzer.continuity import ContinuityLevel
check(result_abs.level == ContinuityLevel.G2,
      f"절대 모드: Δκ=0.02 < 0.05 → G2 (실제: {result_abs.level.name})")
# 상대 모드에서는 G1 으로 강등 (상대 차이 95% > 20%)
check(result_rel.level == ContinuityLevel.G1,
      f"상대 모드: 상대 차이 95% > 20% → G1 로 강등 (실제: {result_rel.level.name})")


# ------------------------------------------------------------
# 4. frozen dataclass — 실수 방지
# ------------------------------------------------------------
print("\n[4] GeometricConstants 는 불변")

c = GeometricConstants()
try:
    c.g0_distance_threshold = 5.0  # type: ignore
    mutated = True
except Exception:
    mutated = False

check(mutated == False, "frozen=True — 필드 수정 불가")


# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
total = passed + failed
print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed}")

if failed == 0:
    print(f"\n  * ALL {total} TESTS PASSED *")
else:
    print(f"\n  x {failed} TESTS FAILED")

sys.exit(0 if failed == 0 else 1)
