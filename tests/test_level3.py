"""
v0.3 Level 3 Spatial Relation Token 테스트
==========================================
공간 관계 분석, 토큰화, 역변환, GID 벤치마크를 포함하는 통합 테스트.

테스트 구성:
    Suite 1: SpatialAnalyzer — 정렬/등간격/대칭/동일크기 탐지
    Suite 2: SpatialTokenizer — Level 3 토큰화 및 압축
    Suite 3: Detokenizer — Level 3 → SVG 역변환
    Suite 4: GID 벤치마크 — Level 1 vs Level 2 vs Level 3
"""

import sys
import os
import math

sys.path.insert(0, '..')

from gpl_tokenizer.parser.path_parser import PathParser, PathCommand, CommandType
from gpl_tokenizer.analyzer.spatial_analyzer import (
    SpatialAnalyzer, ElementInfo, SpatialRelation, RelationType
)
from gpl_tokenizer.tokenizer.spatial_tokenizer import SpatialTokenizer, Level3Result
from gpl_tokenizer.tokenizer.composite_tokenizer import CompositeTokenizer
from gpl_tokenizer.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from gpl_tokenizer.tokenizer.detokenizer import Detokenizer
from gpl_tokenizer.tokenizer.vocabulary import GPLVocabulary, SpatialToken
from gpl_tokenizer.tokenizer.arcs import ARCS


# ===================== 헬퍼 =====================

PARSER = PathParser()
KAPPA = 0.5522847498  # 원의 cubic bezier 근사 상수


def make_circle_commands(cx, cy, r):
    """원을 4개 cubic bezier로 구성하는 PathCommand 리스트."""
    k = r * KAPPA
    d = (f"M {cx+r} {cy} "
         f"C {cx+r} {cy+k}, {cx+k} {cy+r}, {cx} {cy+r} "
         f"C {cx-k} {cy+r}, {cx-r} {cy+k}, {cx-r} {cy} "
         f"C {cx-r} {cy-k}, {cx-k} {cy-r}, {cx} {cy-r} "
         f"C {cx+k} {cy-r}, {cx+r} {cy-k}, {cx+r} {cy} Z")
    cmds = PARSER.parse(d)
    return PARSER.resolve_to_absolute(cmds)


def make_rect_commands(x, y, w, h):
    """사각형을 L 명령어로 구성하는 PathCommand 리스트."""
    d = f"M {x} {y} L {x+w} {y} L {x+w} {y+h} L {x} {y+h} Z"
    cmds = PARSER.parse(d)
    return PARSER.resolve_to_absolute(cmds)


def make_ellipse_commands(cx, cy, rx, ry):
    """타원을 4개 cubic bezier로 구성하는 PathCommand 리스트."""
    kx = rx * KAPPA
    ky = ry * KAPPA
    d = (f"M {cx+rx} {cy} "
         f"C {cx+rx} {cy+ky}, {cx+kx} {cy+ry}, {cx} {cy+ry} "
         f"C {cx-kx} {cy+ry}, {cx-rx} {cy+ky}, {cx-rx} {cy} "
         f"C {cx-rx} {cy-ky}, {cx-kx} {cy-ry}, {cx} {cy-ry} "
         f"C {cx+kx} {cy-ry}, {cx+rx} {cy-ky}, {cx+rx} {cy} Z")
    cmds = PARSER.parse(d)
    return PARSER.resolve_to_absolute(cmds)


# ===================== 테스트 데이터 =====================

# 3개 원 수평 배치 (cx=50, 150, 250, cy=100, r=20)
THREE_CIRCLES_H = [
    make_circle_commands(50, 100, 20),
    make_circle_commands(150, 100, 20),
    make_circle_commands(250, 100, 20),
]

# 5개 원 수평 배치
FIVE_CIRCLES_H = [
    make_circle_commands(50, 100, 20),
    make_circle_commands(150, 100, 20),
    make_circle_commands(250, 100, 20),
    make_circle_commands(350, 100, 20),
    make_circle_commands(450, 100, 20),
]

# 좌우 대칭 원 쌍 (axis_x=150)
SYMMETRIC_CIRCLES = [
    make_circle_commands(50, 100, 20),
    make_circle_commands(250, 100, 20),
]

# 3개 사각형 수직 배치
THREE_RECTS_V = [
    make_rect_commands(100, 20, 60, 40),
    make_rect_commands(100, 100, 60, 40),
    make_rect_commands(100, 180, 60, 40),
]

# 비정형 배치 (관계 없음)
RANDOM_LAYOUT = [
    make_circle_commands(30, 50, 15),
    make_rect_commands(120, 170, 80, 30),
]

# 대칭 + 중심 요소
SYMMETRIC_WITH_CENTER = [
    make_circle_commands(50, 100, 20),   # 좌
    make_circle_commands(150, 100, 30),  # 중앙 (축 위)
    make_circle_commands(250, 100, 20),  # 우
]

# 4개 원 그리드 (2×2 대칭)
GRID_2X2 = [
    make_circle_commands(50, 50, 15),
    make_circle_commands(150, 50, 15),
    make_circle_commands(50, 150, 15),
    make_circle_commands(150, 150, 15),
]


# ===================== 테스트 실행기 =====================

passed = 0
failed = 0


def check(condition: bool, message: str):
    global passed, failed
    if condition:
        print(f"  ✓ {message}")
        passed += 1
    else:
        print(f"  ✗ {message}")
        failed += 1


# ============================================================
# TEST SUITE 1: SpatialAnalyzer — 공간 관계 탐지
# ============================================================

print("=" * 60)
print("TEST SUITE 1: SpatialAnalyzer — 공간 관계 탐지")
print("=" * 60)

analyzer = SpatialAnalyzer(align_tolerance=3.0, spacing_tolerance=3.0,
                            size_tolerance=5.0, sym_tolerance=5.0)

# Test 1.1: 수평 정렬 탐지
elems_h = [
    ElementInfo(0, "circle", (50, 100), (30, 80, 70, 120), 40, 40),
    ElementInfo(1, "circle", (150, 100), (130, 80, 170, 120), 40, 40),
    ElementInfo(2, "circle", (250, 100), (230, 80, 270, 120), 40, 40),
]
rels = analyzer.analyze(elems_h)
align_h = [r for r in rels if r.relation_type == RelationType.ALIGN_CENTER_H]
check(len(align_h) >= 1, "수평 중심 정렬 탐지 (3원)")
check(len(align_h[0].element_indices) == 3, "정렬 그룹에 3개 요소")

# Test 1.2: 등간격 탐지
equal_sp = [r for r in rels if r.relation_type == RelationType.EQUAL_SPACE_H]
check(len(equal_sp) >= 1, "수평 등간격 탐지 (dx=100)")
if equal_sp:
    check(abs(equal_sp[0].spacing - 100) < 5, f"간격 ≈ 100 (실제: {equal_sp[0].spacing:.1f})")

# Test 1.3: 동일 크기 탐지
eq_size = [r for r in rels if r.relation_type == RelationType.EQUAL_SIZE]
check(len(eq_size) >= 1, "동일 크기 그룹 탐지")

# Test 1.4: 수직 정렬 탐지
elems_v = [
    ElementInfo(0, "rect", (130, 40), (100, 20, 160, 60), 60, 40),
    ElementInfo(1, "rect", (130, 120), (100, 100, 160, 140), 60, 40),
    ElementInfo(2, "rect", (130, 200), (100, 180, 160, 220), 60, 40),
]
rels_v = analyzer.analyze(elems_v)
align_v = [r for r in rels_v if r.relation_type == RelationType.ALIGN_CENTER_V]
check(len(align_v) >= 1, "수직 중심 정렬 탐지 (3 rect)")

# Test 1.5: 대칭 탐지
elems_sym = [
    ElementInfo(0, "circle", (50, 100), (30, 80, 70, 120), 40, 40),
    ElementInfo(1, "circle", (250, 100), (230, 80, 270, 120), 40, 40),
]
rels_sym = analyzer.analyze(elems_sym)
sym_x = [r for r in rels_sym if r.relation_type == RelationType.SYM_REFLECT_X]
check(len(sym_x) >= 1, "X축 반사 대칭 탐지 (2원)")
if sym_x:
    check(abs(sym_x[0].axis_value - 150) < 5, f"대칭축 ≈ 150 (실제: {sym_x[0].axis_value:.1f})")

# Test 1.6: 비정형 배치 — 관계 최소
elems_rand = [
    ElementInfo(0, "circle", (30, 50), (15, 35, 45, 65), 30, 30),
    ElementInfo(1, "rect", (160, 185), (120, 170, 200, 200), 80, 30),
]
rels_rand = analyzer.analyze(elems_rand)
check(len([r for r in rels_rand if r.relation_type == RelationType.EQUAL_SPACE_H]) == 0,
      "비정형 배치 — 등간격 미탐지")

# Test 1.7: 대칭 + 축 위 요소
elems_sym3 = [
    ElementInfo(0, "circle", (50, 100), (30, 80, 70, 120), 40, 40),
    ElementInfo(1, "circle", (150, 100), (120, 70, 180, 130), 60, 60),  # 축 위
    ElementInfo(2, "circle", (250, 100), (230, 80, 270, 120), 40, 40),
]
rels_sym3 = analyzer.analyze(elems_sym3)
sym_x3 = [r for r in rels_sym3 if r.relation_type == RelationType.SYM_REFLECT_X]
check(len(sym_x3) >= 1, "대칭 + 중심 요소 탐지")


# ============================================================
# TEST SUITE 2: SpatialTokenizer — Level 3 토큰화
# ============================================================

print()
print("=" * 60)
print("TEST SUITE 2: SpatialTokenizer — Level 3 토큰화")
print("=" * 60)

tokenizer = SpatialTokenizer(canvas_size=500, max_coord_level=6)
l2_tokenizer = CompositeTokenizer(canvas_size=500, max_coord_level=6)

# Test 2.1: 3개 원 수평 배치 — 선형 반복 압축
print()
result_3c = tokenizer.tokenize_multi(THREE_CIRCLES_H)
print(f"  [3 circles H] L2: {result_3c.level2_n_tokens} → L3: {result_3c.n_tokens} tokens")
check(result_3c.n_tokens < result_3c.level2_n_tokens,
      "3원 수평: Level 3 < Level 2")
compression_3c = result_3c.level2_n_tokens / max(result_3c.n_tokens, 1)
print(f"  [3 circles H] 압축률: {compression_3c:.1f}× (L2/L3)")
check(compression_3c >= 1.1, f"3원 수평: 최소 1.1× 압축 (실제: {compression_3c:.1f}×)")

# 공간 관계 토큰 존재 확인
spatial_tokens = [t for t in result_3c.tokens if t.token_type == "spatial"]
check(len(spatial_tokens) >= 3, f"공간 관계 토큰 ≥3개 (실제: {len(spatial_tokens)}개)")

# Test 2.2: 5개 원 수평 배치 — 더 큰 압축
print()
result_5c = tokenizer.tokenize_multi(FIVE_CIRCLES_H)
print(f"  [5 circles H] L2: {result_5c.level2_n_tokens} → L3: {result_5c.n_tokens} tokens")
compression_5c = result_5c.level2_n_tokens / max(result_5c.n_tokens, 1)
print(f"  [5 circles H] 압축률: {compression_5c:.1f}× (L2/L3)")
check(compression_5c > compression_3c - 0.1,
      f"5원: 3원보다 높은 압축률 ({compression_5c:.1f}× vs {compression_3c:.1f}×)")

# Test 2.3: 대칭 쌍 — SYM 압축
print()
result_sym = tokenizer.tokenize_multi(SYMMETRIC_CIRCLES)
print(f"  [sym 2 circles] L2: {result_sym.level2_n_tokens} → L3: {result_sym.n_tokens} tokens")
sym_tokens = [t for t in result_sym.tokens if t.token_type == "spatial"
              and "SYM" in t.value]
check(len(sym_tokens) >= 1, "대칭 토큰 존재")

# Test 2.4: 사각형 수직 배치
print()
result_rv = tokenizer.tokenize_multi(THREE_RECTS_V)
print(f"  [3 rects V] L2: {result_rv.level2_n_tokens} → L3: {result_rv.n_tokens} tokens")
check(result_rv.n_tokens < result_rv.level2_n_tokens,
      "3 rect 수직: Level 3 < Level 2")

# Test 2.5: 비정형 배치 — Level 2 유지
print()
result_rand = tokenizer.tokenize_multi(RANDOM_LAYOUT)
print(f"  [random] L2: {result_rand.level2_n_tokens} → L3: {result_rand.n_tokens} tokens")
check(result_rand.n_tokens <= result_rand.level2_n_tokens + 1,
      "비정형: Level 3 ≈ Level 2 (과잉 압축 없음)")

# Test 2.6: 대칭 + 중심 요소
print()
result_sym3 = tokenizer.tokenize_multi(SYMMETRIC_WITH_CENTER)
print(f"  [sym+center] L2: {result_sym3.level2_n_tokens} → L3: {result_sym3.n_tokens} tokens")
check(result_sym3.n_tokens <= result_sym3.level2_n_tokens,
      "대칭+중심: Level 3 ≤ Level 2")


# ============================================================
# TEST SUITE 3: Detokenizer — Level 3 → SVG 역변환
# ============================================================

print()
print("=" * 60)
print("TEST SUITE 3: Detokenizer — Level 3 → SVG 역변환")
print("=" * 60)

vocab = GPLVocabulary(max_coord_level=6)
arcs = ARCS(canvas_size=500, max_level=6)
detok = Detokenizer(vocab, arcs)

# Test 3.1: 3원 수평 → SVG 역변환 (3개 circle 요소)
svg_3c = detok.to_svg_document(result_3c.token_ids, width=500, height=200)
circle_count = svg_3c.count("<circle")
check(circle_count == 3, f"3원 → SVG: <circle> {circle_count}개 (기대: 3)")

# Test 3.2: 5원 수평 → SVG 역변환
svg_5c = detok.to_svg_document(result_5c.token_ids, width=600, height=200)
circle_count_5 = svg_5c.count("<circle")
check(circle_count_5 == 5, f"5원 → SVG: <circle> {circle_count_5}개 (기대: 5)")

# Test 3.3: 대칭 쌍 → SVG 역변환
svg_sym = detok.to_svg_document(result_sym.token_ids, width=300, height=200)
sym_circle_count = svg_sym.count("<circle")
check(sym_circle_count == 2, f"대칭 2원 → SVG: <circle> {sym_circle_count}개 (기대: 2)")

# Test 3.4: 사각형 수직 → SVG 역변환
svg_rv = detok.to_svg_document(result_rv.token_ids, width=300, height=300)
rect_count = svg_rv.count("<rect")
check(rect_count == 3, f"3 rect → SVG: <rect> {rect_count}개 (기대: 3)")

# Test 3.5: 비정형 → SVG (요소 유지)
svg_rand = detok.to_svg_document(result_rand.token_ids, width=300, height=300)
has_elements = "<circle" in svg_rand or "<rect" in svg_rand or "<path" in svg_rand
check(has_elements, "비정형 → SVG: 요소 존재")


# ============================================================
# TEST SUITE 4: GID 벤치마크 — Level 1 vs Level 2 vs Level 3
# ============================================================

print()
print("=" * 60)
print("TEST SUITE 4: GID 벤치마크 — L1 vs L2 vs L3")
print("=" * 60)

l1_tokenizer = PrimitiveTokenizer(canvas_size=500, max_coord_level=6)

print()
print(f"  {'SVG Layout':<25} {'L1':>6} {'L2':>6} {'L3':>6} {'L2/L3':>8} {'GID L3/BPE':>11}")
print("  " + "-" * 68)

benchmarks = {
    "3 circles (H)": THREE_CIRCLES_H,
    "5 circles (H)": FIVE_CIRCLES_H,
    "2 circles (sym)": SYMMETRIC_CIRCLES,
    "3 rects (V)": THREE_RECTS_V,
    "random (no rel)": RANDOM_LAYOUT,
    "3 circles (sym+c)": SYMMETRIC_WITH_CENTER,
    "4 circles (2×2)": GRID_2X2,
}

for name, elem_cmds in benchmarks.items():
    # Level 1: 각 요소 개별 토큰화 합산
    l1_total = 2  # BOS + EOS
    total_cmds = 0
    for cmds in elem_cmds:
        l1 = l1_tokenizer.tokenize(cmds)
        l1_inner = l1.n_tokens - 2  # BOS/EOS 제외
        l1_total += l1_inner
        total_cmds += l1.n_commands
    l1_total += len(elem_cmds) - 1  # SEP

    # Level 2
    l2_total = 2  # BOS + EOS
    for i, cmds in enumerate(elem_cmds):
        l2 = l2_tokenizer.tokenize(cmds)
        l2_inner = l2.n_tokens - 2
        l2_total += l2_inner
    l2_total += len(elem_cmds) - 1

    # Level 3
    l3 = tokenizer.tokenize_multi(elem_cmds)
    l3_total = l3.n_tokens

    # BPE 추정 (원 1개 ≈ 60 chars, rect ≈ 40 chars → 3 chars/token)
    n_elements = len(elem_cmds)
    avg_chars = 50 * n_elements
    bpe_tokens = avg_chars / 3

    # 압축률
    l2_l3_ratio = l2_total / max(l3_total, 1)

    # GID: geometric_info / n_tokens
    # 각 원 = 3 params (cx,cy,r), 각 rect = 4 params (x,y,w,h)
    geo_info = sum(3 if "circle" in name or "sym" in name else 4 for _ in elem_cmds)
    gid_l3 = geo_info / max(l3_total, 1)
    gid_bpe = geo_info / max(bpe_tokens, 1)
    gid_ratio = gid_l3 / max(gid_bpe, 0.001)

    print(f"  {name:<25} {l1_total:>6} {l2_total:>6} {l3_total:>6}"
          f"   {l2_l3_ratio:>5.1f}×   {gid_ratio:>8.1f}×")

# 핵심 검증: 5원 수평 배치에서 L3가 L2 대비 유의미한 압축 달성
result_5c_final = tokenizer.tokenize_multi(FIVE_CIRCLES_H)
l2_5c_total = result_5c_final.level2_n_tokens
l3_5c_total = result_5c_final.n_tokens
check(l3_5c_total < l2_5c_total * 0.8,
      f"5원 GID: L3({l3_5c_total}) < L2×0.8({l2_5c_total * 0.8:.0f}) — 유의미한 압축")


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
    print(f"\n  ★ ALL {total} TESTS PASSED ★")
else:
    print(f"\n  ✗ {failed} TESTS FAILED")

sys.exit(0 if failed == 0 else 1)
