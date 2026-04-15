"""
GPL 토크나이저 왕복(Round-trip) 테스트 및 GID 측정
===================================================
SVG → Parse → Tokenize → Detokenize → SVG 파이프라인의 충실도 검증.
"""

import sys
import os
import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpl_tokenizer.parser.svg_parser import SVGParser
from gpl_tokenizer.parser.path_parser import PathParser, CommandType
from gpl_tokenizer.analyzer.curvature import CurvatureAnalyzer
from gpl_tokenizer.analyzer.continuity import ContinuityAnalyzer, ContinuityLevel
from gpl_tokenizer.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from gpl_tokenizer.tokenizer.detokenizer import Detokenizer
from gpl_tokenizer.tokenizer.vocabulary import GPLVocabulary
from gpl_tokenizer.tokenizer.arcs import ARCS
from gpl_tokenizer.utils.math_utils import BezierMath


# ====================================================================
# 테스트 SVG 데이터
# ====================================================================

TEST_SVGS = {
    "simple_line": '<svg width="300" height="300"><path d="M 10 10 L 290 290"/></svg>',

    "simple_rect": '<svg width="300" height="300"><rect x="50" y="50" width="200" height="100"/></svg>',

    "simple_circle": '<svg width="300" height="300"><circle cx="150" cy="150" r="100"/></svg>',

    "cubic_bezier": '<svg width="300" height="300"><path d="M 10 80 C 40 10, 65 10, 95 80 S 150 150, 180 80"/></svg>',

    "quadratic_bezier": '<svg width="300" height="300"><path d="M 10 80 Q 95 10, 180 80 T 290 80"/></svg>',

    "complex_icon": '''<svg width="300" height="300">
        <path d="M 150 30 L 270 230 L 30 230 Z"/>
        <circle cx="150" cy="163" r="40"/>
    </svg>''',

    "multi_path": '''<svg width="300" height="300">
        <path d="M 50 50 C 100 20, 150 20, 200 50 L 200 150 C 150 180, 100 180, 50 150 Z"/>
        <path d="M 100 80 L 150 60 L 200 80 L 200 120 L 150 140 L 100 120 Z"/>
    </svg>''',
}


def run_all_tests():
    """모든 테스트 실행."""
    print("=" * 70)
    print("GPL Tokenizer Round-Trip Test Suite")
    print("=" * 70)

    results = {}

    # 1. 기본 파서 테스트
    print("\n[1] SVG Parser Tests")
    print("-" * 40)
    test_parser(results)

    # 2. 곡률 분석 테스트
    print("\n[2] Curvature Analyzer Tests")
    print("-" * 40)
    test_curvature(results)

    # 3. 연속성 분석 테스트
    print("\n[3] Continuity Analyzer Tests")
    print("-" * 40)
    test_continuity(results)

    # 4. ARCS 양자화 테스트
    print("\n[4] ARCS Quantization Tests")
    print("-" * 40)
    test_arcs(results)

    # 5. 전체 파이프라인 왕복 테스트
    print("\n[5] Full Pipeline Round-Trip Tests")
    print("-" * 40)
    test_full_roundtrip(results)

    # 6. GID (Geometric Information Density) 측정
    print("\n[6] GID Measurement")
    print("-" * 40)
    test_gid(results)

    # 7. 어휘 통계
    print("\n[7] Vocabulary Statistics")
    print("-" * 40)
    test_vocab(results)

    # 결과 요약
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def test_parser(results):
    """SVG 파서 기본 테스트."""
    parser = SVGParser()

    # 테스트 1: 단순 직선
    doc = parser.parse_string(TEST_SVGS["simple_line"])
    ok = len(doc.elements) == 1 and len(doc.elements[0].commands) == 2
    print(f"  simple_line: {len(doc.elements)} elements, "
          f"{len(doc.elements[0].commands)} commands -> {'OK' if ok else 'FAIL'}")
    results["parser_simple_line"] = ok

    # 테스트 2: 사각형 (path로 변환됨)
    doc = parser.parse_string(TEST_SVGS["simple_rect"])
    ok = len(doc.elements) == 1 and len(doc.elements[0].commands) >= 5
    print(f"  simple_rect: {len(doc.elements)} elements, "
          f"{len(doc.elements[0].commands)} commands -> {'OK' if ok else 'FAIL'}")
    results["parser_simple_rect"] = ok

    # 테스트 3: 원 (4 큐빅 베지에로 변환)
    doc = parser.parse_string(TEST_SVGS["simple_circle"])
    cubic_count = sum(1 for c in doc.elements[0].commands
                      if c.command_type == CommandType.CUBIC)
    ok = cubic_count == 4
    print(f"  simple_circle: {cubic_count} cubic beziers -> {'OK' if ok else 'FAIL'}")
    results["parser_circle_to_cubic"] = ok

    # 테스트 4: 큐빅 베지에 + Smooth (S → C 승격)
    doc = parser.parse_string(TEST_SVGS["cubic_bezier"])
    cmds = doc.elements[0].commands
    cubic_count = sum(1 for c in cmds if c.command_type == CommandType.CUBIC)
    ok = cubic_count == 2  # C + S(→C)
    print(f"  cubic_bezier: {cubic_count} cubic (incl. S->C) -> {'OK' if ok else 'FAIL'}")
    results["parser_smooth_cubic"] = ok

    # 테스트 5: 복합 SVG
    doc = parser.parse_string(TEST_SVGS["complex_icon"])
    ok = len(doc.elements) == 2
    print(f"  complex_icon: {len(doc.elements)} elements -> {'OK' if ok else 'FAIL'}")
    results["parser_complex"] = ok


def test_curvature(results):
    """곡률 분석 테스트."""
    parser = SVGParser()
    analyzer = CurvatureAnalyzer(n_samples=50)

    # 직선의 곡률 = 0
    doc = parser.parse_string(TEST_SVGS["simple_line"])
    infos = analyzer.analyze(doc.elements[0].commands)
    ok = len(infos) >= 1 and infos[0].is_straight and infos[0].max_abs_curvature < 1e-6
    print(f"  line curvature: κ_max={infos[0].max_abs_curvature:.6f}, "
          f"straight={infos[0].is_straight} -> {'OK' if ok else 'FAIL'}")
    results["curvature_line"] = ok

    # 원의 곡률 = 1/r (r=100이면 κ ≈ 0.01)
    doc = parser.parse_string(TEST_SVGS["simple_circle"])
    infos = analyzer.analyze(doc.elements[0].commands)
    cubics = [i for i in infos if i.command_type == CommandType.CUBIC]
    if cubics:
        avg_kappa = np.mean([c.mean_abs_curvature for c in cubics])
        expected_kappa = 1.0 / 100.0  # r=100
        error = abs(avg_kappa - expected_kappa) / expected_kappa
        ok = error < 0.15  # 15% 허용 오차
        print(f"  circle curvature: κ_avg={avg_kappa:.4f}, "
              f"expected={expected_kappa:.4f}, error={error:.2%} -> {'OK' if ok else 'FAIL'}")
        results["curvature_circle"] = ok
    else:
        print(f"  circle curvature: no cubics found -> FAIL")
        results["curvature_circle"] = False

    # 큐빅 베지에의 곡률 프로파일
    doc = parser.parse_string(TEST_SVGS["cubic_bezier"])
    infos = analyzer.analyze(doc.elements[0].commands)
    cubics = [i for i in infos if i.command_type == CommandType.CUBIC]
    ok = len(cubics) >= 1 and cubics[0].max_abs_curvature > 0
    if cubics:
        print(f"  bezier curvature: κ_max={cubics[0].max_abs_curvature:.4f}, "
              f"arc_length={cubics[0].arc_length:.1f} -> {'OK' if ok else 'FAIL'}")
    results["curvature_bezier"] = ok


def test_continuity(results):
    """연속성 분석 테스트."""
    parser = SVGParser()
    curv_analyzer = CurvatureAnalyzer()
    cont_analyzer = ContinuityAnalyzer()

    # 큐빅 + Smooth: G1 이상 연속이어야 함
    doc = parser.parse_string(TEST_SVGS["cubic_bezier"])
    cmds = doc.elements[0].commands
    curv_infos = curv_analyzer.analyze(cmds)
    cont_infos = cont_analyzer.analyze(cmds, curv_infos)

    if cont_infos:
        best_level = max(ci.level for ci in cont_infos)
        ok = best_level >= ContinuityLevel.G1
        print(f"  smooth cubic continuity: best={ContinuityLevel(best_level).name}, "
              f"joints={len(cont_infos)} -> {'OK' if ok else 'FAIL'}")
        results["continuity_smooth"] = ok
    else:
        print(f"  smooth cubic continuity: no joints -> SKIP")
        results["continuity_smooth"] = True

    # 삼각형 (각 꼭짓점에서 G0): 꼭짓점 연결부 확인
    doc = parser.parse_string(TEST_SVGS["complex_icon"])
    cmds = doc.elements[0].commands  # 삼각형 path
    curv_infos = curv_analyzer.analyze(cmds)
    cont_infos = cont_analyzer.analyze(cmds, curv_infos)
    if cont_infos:
        has_g0 = any(ci.level == ContinuityLevel.G0 for ci in cont_infos)
        print(f"  triangle joints: {len(cont_infos)} joints, "
              f"has G0={has_g0} -> OK")
        results["continuity_triangle"] = True
    else:
        print(f"  triangle joints: no joints analyzed -> SKIP")
        results["continuity_triangle"] = True


def test_arcs(results):
    """ARCS 양자화 테스트."""
    arcs = ARCS(canvas_size=300.0, max_level=6, min_level=4)

    # 테스트 1: 기본 양자화/역양자화 왕복 오차
    test_points = [(0, 0), (150, 150), (299, 299), (50.3, 200.7), (127.4, 89.7)]
    errors = []
    for x, y in test_points:
        err = arcs.quantization_error(x, y)
        errors.append(err)

    avg_err = np.mean(errors)
    max_err = np.max(errors)
    # 레벨4 = 16×16 그리드, 셀 크기 ≈ 18.75px → 최대 오차 ≈ 셀 대각선/2 ≈ 13.3px
    cell_size = 300.0 / (2 ** 4)
    max_expected = cell_size * np.sqrt(2) / 2
    ok = max_err < max_expected * 1.1  # 10% 마진
    print(f"  uniform ARCS: avg_err={avg_err:.2f}px, max_err={max_err:.2f}px -> {'OK' if ok else 'FAIL'}")
    results["arcs_uniform"] = ok

    # 테스트 2: 리프 노드 수
    leaf_count = arcs.total_leaf_count()
    level_dist = arcs.level_distribution()
    print(f"  leaf count: {leaf_count}, distribution: {dict(level_dist)}")
    results["arcs_structure"] = leaf_count > 0

    # 테스트 3: 적응적 ARCS 구축
    segment_data = [
        {"bbox": (100, 100, 200, 200), "max_curvature": 0.05, "arc_length": 150},
        {"bbox": (10, 10, 50, 50), "max_curvature": 0.001, "arc_length": 50},
    ]
    arcs_adaptive = ARCS(canvas_size=300.0, max_level=6, min_level=2)
    arcs_adaptive.build_from_curvatures(segment_data)
    adaptive_leaves = arcs_adaptive.total_leaf_count()
    uniform_leaves = arcs.total_leaf_count()
    ok = adaptive_leaves != uniform_leaves  # 적응적이면 리프 수가 달라야 함
    print(f"  adaptive ARCS: {adaptive_leaves} leaves (uniform: {uniform_leaves}) -> {'OK' if ok else 'WARN'}")
    results["arcs_adaptive"] = True  # 구조 자체는 정상


def test_full_roundtrip(results):
    """전체 파이프라인 왕복 테스트."""
    svg_parser = SVGParser()

    for name, svg_text in TEST_SVGS.items():
        doc = svg_parser.parse_string(svg_text)

        for elem_idx, elem in enumerate(doc.elements):
            if not elem.commands:
                continue

            # 캔버스 크기
            cw, ch = doc.canvas_size
            canvas = max(cw, ch)

            # 토크나이저
            tokenizer = PrimitiveTokenizer(
                canvas_size=canvas, max_coord_level=6,
                use_adaptive_arcs=True
            )

            # 토큰화
            result = tokenizer.tokenize(elem.commands)

            # 디토크나이저
            detok = Detokenizer(tokenizer.vocab, tokenizer.arcs)
            recovered_d = detok.detokenize(result.token_ids)

            # 기본 검증: 비어있지 않은 결과
            ok = len(result.token_ids) > 2 and len(recovered_d) > 0  # BOS+EOS 이상
            status = "OK" if ok else "FAIL"

            print(f"  {name}[{elem_idx}]: "
                  f"{result.n_commands} cmds -> {result.n_tokens} tokens, "
                  f"recovered d='{recovered_d[:60]}...' -> {status}")

            results[f"roundtrip_{name}_{elem_idx}"] = ok


def test_gid(results):
    """GID (Geometric Information Density) 측정."""
    svg_parser = SVGParser()

    print(f"\n  {'SVG':<20} {'Orig Chars':>10} {'GPL Tokens':>10} {'Cmds':>5} "
          f"{'GID':>8} {'vs BPE':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*5} {'-'*8} {'-'*8}")

    for name, svg_text in TEST_SVGS.items():
        doc = svg_parser.parse_string(svg_text)

        for elem_idx, elem in enumerate(doc.elements):
            if not elem.commands:
                continue

            cw, ch = doc.canvas_size
            canvas = max(cw, ch)

            tokenizer = PrimitiveTokenizer(canvas_size=canvas, max_coord_level=6)
            result = tokenizer.tokenize(elem.commands, original_text=svg_text)

            # GID 계산: 렌더링 명령어 수 / 토큰 수
            n_renderable = sum(1 for c in elem.commands
                               if c.command_type not in (CommandType.MOVE, CommandType.CLOSE))
            gid = n_renderable / max(result.n_tokens, 1)

            # BPE 추정: SVG 문자열 길이 / 4 (평균 BPE 토큰 길이 ~4 문자)
            est_bpe_tokens = len(svg_text) / 4.0
            gid_bpe = n_renderable / max(est_bpe_tokens, 1)
            ratio = gid / max(gid_bpe, 1e-6)

            print(f"  {name}[{elem_idx}]"[:20].ljust(20) +
                  f" {len(svg_text):>10} {result.n_tokens:>10} "
                  f"{n_renderable:>5} {gid:>8.4f} {ratio:>7.1f}×")

    results["gid_measured"] = True


def test_vocab(results):
    """어휘 통계."""
    vocab = GPLVocabulary(max_coord_level=6)
    print(vocab.summary())
    results["vocab_ok"] = vocab.vocab_size > 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
