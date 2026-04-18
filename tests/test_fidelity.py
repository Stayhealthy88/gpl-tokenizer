"""
왕복 충실도(Round-trip Fidelity) 정량 측정 테스트
================================================
v0.5.1 개선 #3: 기존 roundtrip 테스트는 위상(topology)만 검증하였으나,
이 테스트는 양자화로 인한 절대 좌표 오차를 픽셀 단위로 측정한다.

이론적 상한:
    ARCS 리프 레벨 L 에서 셀 크기 = canvas_size / 2^L
    최대 유클리드 오차 = (cell_size / 2) · sqrt(2)   (셀 모서리)

v0.5 기본값 (canvas=300, max_level=6):
    최대 오차 = (300/64)/2 · sqrt(2) ≈ 3.31 px
"""

import os
import sys
import math
import random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpl_tokenizer.tokenizer.arcs import ARCS, QuantizedCoord
from gpl_tokenizer.tokenizer.vocabulary import GPLVocabulary
from gpl_tokenizer.tokenizer.detokenizer import Detokenizer


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
print("v0.5.1 Round-trip Fidelity Tests")
print("=" * 60)

# ------------------------------------------------------------
# 1. 이론적 오차 상한
# ------------------------------------------------------------
print("\n[1] 이론적 오차 상한")

arcs = ARCS(canvas_size=300.0, max_level=6)
# max_level=6 → cell = 300/64 = 4.6875 → max L2 = 2.34375 · sqrt(2) ≈ 3.314
bound6 = arcs.theoretical_max_error(6)
expected = (300.0 / 64) / 2 * math.sqrt(2)
check(abs(bound6 - expected) < 1e-9,
      f"level=6 에서 이론 상한 = {bound6:.4f} (예상 {expected:.4f})")

bound3 = arcs.theoretical_max_error(3)
expected3 = (300.0 / 8) / 2 * math.sqrt(2)
check(abs(bound3 - expected3) < 1e-9,
      f"level=3 에서 이론 상한 = {bound3:.4f}")

# 레벨이 얕을수록 상한이 커야 함
check(bound3 > bound6, "상한은 얕은 레벨에서 더 큼(coarser cells)")


# ------------------------------------------------------------
# 2. 경험적 오차 ≤ 이론 상한
# ------------------------------------------------------------
print("\n[2] 경험적 오차 ≤ 이론 상한 (랜덤 1000점)")

rng = random.Random(42)
points = [(rng.uniform(0, 299.99), rng.uniform(0, 299.99)) for _ in range(1000)]

metrics = arcs.roundtrip_fidelity(points)
check(metrics["within_bound"] == True,
      f"최대 경험 오차 {metrics['max_error']:.4f} "
      f"≤ 이론 상한 {metrics['theoretical_bound']:.4f}")
check(metrics["n_points"] == 1000, "측정 점 수 일치")
check(metrics["mean_error"] < metrics["max_error"],
      f"평균({metrics['mean_error']:.4f}) < 최대({metrics['max_error']:.4f})")


# ------------------------------------------------------------
# 3. Detokenizer 측 측정 API
# ------------------------------------------------------------
print("\n[3] Detokenizer.extract_coordinates & measure_fidelity")

# adaptive tree 빌드 없이 min_level=2 만 사용하는 기본 상태
vocab = GPLVocabulary(max_coord_level=6)
detok = Detokenizer(vocab, arcs)

orig = [(50.0, 50.0), (150.5, 80.2), (200.1, 200.9), (280.0, 10.0)]
token_ids = []
for x, y in orig:
    qc = arcs.quantize(x, y)
    tid = vocab.coord_to_id(qc.level, qc.qx, qc.qy)
    token_ids.append(tid)

recovered = detok.extract_coordinates(token_ids)
check(len(recovered) == 4, "4개 좌표 추출")

fidelity = detok.measure_fidelity(orig, token_ids)
check(fidelity["n_points"] == 4 and fidelity["n_recovered"] == 4,
      "n_points == n_recovered == 4")
check(fidelity["within_bound"] == True,
      f"Detokenizer 측정 max_error={fidelity['max_error']:.4f} 이 상한 내 "
      f"(bound={fidelity['theoretical_bound']:.4f}, "
      f"coarsest_level={fidelity['coarsest_level']})")
# min_level=2 기본 — cell=75, max_err = 75/2*sqrt(2) ≈ 53
check(fidelity["coarsest_level"] == 2,
      f"adaptive tree 미빌드 상태에서 coarsest=min_level=2 "
      f"(실제 {fidelity['coarsest_level']})")
check(fidelity["max_error"] < 55.0,
      f"max_error={fidelity['max_error']:.4f} 는 coarse(level=2) 상한 내")


# ------------------------------------------------------------
# 3b. adaptive tree 빌드 후 더 정밀한 리프 사용
# ------------------------------------------------------------
print("\n[3b] adaptive tree 빌드 시 max_error 감소")

arcs_adaptive = ARCS(canvas_size=300.0, max_level=6, min_level=2,
                     split_threshold=0.01)  # 매우 낮은 threshold → 항상 분할
# 캔버스 전역에 높은 곡률을 가진 세그먼트를 넣어 전 영역이 max_level까지 분할되게 함
segments = [{
    "bbox": (0.0, 0.0, 300.0, 300.0),
    "max_curvature": 100.0,
    "arc_length": 500.0,
}]
arcs_adaptive.build_from_curvatures(segments)

detok_ad = Detokenizer(vocab, arcs_adaptive)
token_ids_ad = []
for x, y in orig:
    qc = arcs_adaptive.quantize(x, y)
    tid = vocab.coord_to_id(qc.level, qc.qx, qc.qy)
    token_ids_ad.append(tid)
fid_ad = detok_ad.measure_fidelity(orig, token_ids_ad)

check(fid_ad["coarsest_level"] >= 4,
      f"adaptive build 후 coarsest_level={fid_ad['coarsest_level']} ≥ 4 "
      f"(더 정밀한 리프 사용)")
check(fid_ad["max_error"] < fidelity["max_error"],
      f"adaptive build 후 max_error {fid_ad['max_error']:.4f} "
      f"< 기본 {fidelity['max_error']:.4f}")


# ------------------------------------------------------------
# 4. 좌표 토큰 외에는 무시
# ------------------------------------------------------------
print("\n[4] 비좌표 토큰은 추출에서 제외")

# [BOS, MOVE, coord, coord, LINE, coord, coord, EOS] — coord 4개만 추출되어야
from gpl_tokenizer.tokenizer.vocabulary import SpecialToken, CommandToken
tok_mixed = [
    int(SpecialToken.BOS),
    int(CommandToken.MOVE),
    token_ids[0], token_ids[1],
    int(CommandToken.LINE),
    token_ids[2], token_ids[3],
    int(SpecialToken.EOS),
]
rec = detok.extract_coordinates(tok_mixed)
check(len(rec) == 4, f"mixed 시퀀스에서 좌표 4개만 추출(실제 {len(rec)})")


# ------------------------------------------------------------
# 5. 빈 입력 / 엣지 케이스
# ------------------------------------------------------------
print("\n[5] 엣지 케이스")

empty = arcs.roundtrip_fidelity([])
check(empty["n_points"] == 0 and empty["max_error"] == 0.0,
      "빈 좌표 리스트 처리")

empty_fid = detok.measure_fidelity([], [])
check(empty_fid["within_bound"] == True and empty_fid["max_error"] == 0.0,
      "빈 Detokenizer 입력 처리")

# 경계 좌표 — canvas_size 바로 아래
corner = arcs.roundtrip_fidelity([(299.999, 299.999), (0.0, 0.0)])
check(corner["within_bound"] == True,
      f"캔버스 모서리 좌표 왕복 — max_err={corner['max_error']:.4f}")


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
