"""
Level 2 복합 토크나이저 테스트
==============================
v0.2: 원/사각형 → 단일 토큰 압축 검증

테스트 항목:
    1. ShapeDetector — 원, 타원, 사각형, 둥근 사각형 인식
    2. CompositeTokenizer — Level 2 토큰화 + 압축률
    3. Detokenizer — Level 2 토큰 → SVG 역변환
    4. GID 벤치마크 — Level 1 vs Level 2 비교
    5. Round-trip — 원본 → Level 2 토큰 → SVG 재구성
"""

import sys
import math
sys.path.insert(0, '..')

from gpl_tokenizer.parser import SVGParser
from gpl_tokenizer.analyzer.shape_detector import ShapeDetector, ShapeType
from gpl_tokenizer.tokenizer import PrimitiveTokenizer, CompositeTokenizer, Detokenizer


# ===================== 테스트 SVG 데이터 =====================

TEST_SVGS = {
    "circle_100": '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">'
                  '<circle cx="150" cy="150" r="100" fill="none" stroke="black"/></svg>',

    "circle_50": '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">'
                 '<circle cx="100" cy="100" r="50" fill="none" stroke="black"/></svg>',

    "ellipse": '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">'
               '<ellipse cx="150" cy="150" rx="120" ry="80" fill="none" stroke="black"/></svg>',

    "rect_simple": '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">'
                   '<rect x="50" y="50" width="200" height="150" fill="none" stroke="black"/></svg>',

    "rect_square": '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">'
                   '<rect x="75" y="75" width="150" height="150" fill="none" stroke="black"/></svg>',

    "rect_rounded": '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">'
                    '<rect x="50" y="50" width="200" height="150" rx="20" ry="20" fill="none" stroke="black"/></svg>',

    "triangle": '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">'
                '<path d="M 150 30 L 270 250 L 30 250 Z" fill="none" stroke="black"/></svg>',

    "mixed_shapes": '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">'
                    '<circle cx="75" cy="75" r="50" fill="none" stroke="black"/>'
                    '<rect x="150" y="50" width="100" height="100" fill="none" stroke="black"/></svg>',
}


def run_tests():
    """전체 테스트 실행."""
    parser = SVGParser()
    results = {"passed": 0, "failed": 0, "total": 0}

    def check(name, condition, msg=""):
        results["total"] += 1
        if condition:
            results["passed"] += 1
            print(f"  ✓ {name}")
        else:
            results["failed"] += 1
            print(f"  ✗ {name} — {msg}")

    # ============================================================
    print("\n" + "="*60)
    print("TEST SUITE 1: ShapeDetector — 도형 인식")
    print("="*60)
    # ============================================================

    detector = ShapeDetector()

    # Circle 인식
    doc = parser.parse_string(TEST_SVGS["circle_100"])
    shapes = detector.detect(doc.elements[0].commands)
    check("circle_100: 원 1개 인식", len(shapes) == 1,
          f"expected 1, got {len(shapes)}")
    if shapes:
        check("circle_100: ShapeType=CIRCLE",
              shapes[0].shape_type == ShapeType.CIRCLE,
              f"got {shapes[0].shape_type}")
        check("circle_100: 중심 (150,150) 근처",
              abs(shapes[0].params["cx"] - 150) < 10 and
              abs(shapes[0].params["cy"] - 150) < 10,
              f"got cx={shapes[0].params['cx']:.1f}, cy={shapes[0].params['cy']:.1f}")
        check("circle_100: 반지름 100 근처",
              abs(shapes[0].params["r"] - 100) < 10,
              f"got r={shapes[0].params['r']:.1f}")

    # Ellipse 인식
    doc = parser.parse_string(TEST_SVGS["ellipse"])
    shapes = detector.detect(doc.elements[0].commands)
    check("ellipse: 타원 1개 인식", len(shapes) == 1,
          f"expected 1, got {len(shapes)}")
    if shapes:
        check("ellipse: ShapeType=ELLIPSE",
              shapes[0].shape_type == ShapeType.ELLIPSE,
              f"got {shapes[0].shape_type}")

    # Rect 인식
    doc = parser.parse_string(TEST_SVGS["rect_simple"])
    shapes = detector.detect(doc.elements[0].commands)
    check("rect_simple: 사각형 1개 인식", len(shapes) == 1,
          f"expected 1, got {len(shapes)}")
    if shapes:
        check("rect_simple: ShapeType=RECT",
              shapes[0].shape_type == ShapeType.RECT,
              f"got {shapes[0].shape_type}")
        check("rect_simple: width=200 근처",
              abs(shapes[0].params["width"] - 200) < 10,
              f"got width={shapes[0].params['width']:.1f}")

    # Triangle — 도형 미인식 (삼각형은 RECT/CIRCLE이 아님)
    doc = parser.parse_string(TEST_SVGS["triangle"])
    shapes = detector.detect(doc.elements[0].commands)
    check("triangle: 기하 프리미티브 미인식 (삼각형은 Level 1 유지)",
          len(shapes) == 0, f"got {len(shapes)} shapes")

    # ============================================================
    print("\n" + "="*60)
    print("TEST SUITE 2: CompositeTokenizer — Level 2 토큰화")
    print("="*60)
    # ============================================================

    l1 = PrimitiveTokenizer(canvas_size=300, max_coord_level=6)
    l2 = CompositeTokenizer(canvas_size=300, max_coord_level=6)

    # Circle: Level 1 vs Level 2 토큰 수 비교
    doc = parser.parse_string(TEST_SVGS["circle_100"])
    l1_result = l1.tokenize(doc.elements[0].commands)
    l2_result = l2.tokenize(doc.elements[0].commands)

    print(f"\n  [circle] Level 1: {l1_result.n_tokens} tokens → Level 2: {l2_result.n_tokens} tokens")
    check("circle: Level 2 < Level 1",
          l2_result.n_tokens < l1_result.n_tokens,
          f"L1={l1_result.n_tokens}, L2={l2_result.n_tokens}")
    check("circle: 도형 1개 인식",
          len(l2_result.detected_shapes) == 1,
          f"got {len(l2_result.detected_shapes)}")

    compression = l1_result.n_tokens / max(l2_result.n_tokens, 1)
    print(f"  [circle] 압축률: {compression:.1f}× (Level 1 / Level 2)")
    check("circle: 최소 3× 압축",
          compression >= 3.0,
          f"got {compression:.1f}×")

    # Rect: Level 1 vs Level 2
    doc = parser.parse_string(TEST_SVGS["rect_simple"])
    l1_result = l1.tokenize(doc.elements[0].commands)
    l2_result = l2.tokenize(doc.elements[0].commands)

    print(f"\n  [rect] Level 1: {l1_result.n_tokens} tokens → Level 2: {l2_result.n_tokens} tokens")
    check("rect: Level 2 < Level 1",
          l2_result.n_tokens < l1_result.n_tokens,
          f"L1={l1_result.n_tokens}, L2={l2_result.n_tokens}")
    compression = l1_result.n_tokens / max(l2_result.n_tokens, 1)
    print(f"  [rect] 압축률: {compression:.1f}× (Level 1 / Level 2)")
    check("rect: 최소 2× 압축",
          compression >= 2.0,
          f"got {compression:.1f}×")

    # Ellipse: Level 2
    doc = parser.parse_string(TEST_SVGS["ellipse"])
    l1_result = l1.tokenize(doc.elements[0].commands)
    l2_result = l2.tokenize(doc.elements[0].commands)

    print(f"\n  [ellipse] Level 1: {l1_result.n_tokens} tokens → Level 2: {l2_result.n_tokens} tokens")
    check("ellipse: Level 2 < Level 1",
          l2_result.n_tokens < l1_result.n_tokens,
          f"L1={l1_result.n_tokens}, L2={l2_result.n_tokens}")

    # Triangle: Level 2 = Level 1 (미인식)
    doc = parser.parse_string(TEST_SVGS["triangle"])
    l1_result = l1.tokenize(doc.elements[0].commands)
    l2_result = l2.tokenize(doc.elements[0].commands)

    print(f"\n  [triangle] Level 1: {l1_result.n_tokens} tokens → Level 2: {l2_result.n_tokens} tokens")
    check("triangle: Level 2 ≈ Level 1 (삼각형 미압축)",
          abs(l2_result.n_tokens - l1_result.n_tokens) <= 2,
          f"L1={l1_result.n_tokens}, L2={l2_result.n_tokens}")

    # ============================================================
    print("\n" + "="*60)
    print("TEST SUITE 3: Detokenizer — Level 2 → SVG 역변환")
    print("="*60)
    # ============================================================

    # Circle round-trip
    doc = parser.parse_string(TEST_SVGS["circle_100"])
    l2_result = l2.tokenize(doc.elements[0].commands)
    detok = Detokenizer(l2.vocab, l2.arcs)
    svg_doc = detok.to_svg_document(l2_result.token_ids, width=300, height=300)

    check("circle L2 → SVG: <circle> 요소 포함",
          "<circle" in svg_doc,
          f"output: {svg_doc[:100]}...")
    check("circle L2 → SVG: cx 속성 존재",
          'cx="' in svg_doc,
          f"no cx attribute")

    # Rect round-trip
    doc = parser.parse_string(TEST_SVGS["rect_simple"])
    l2_result = l2.tokenize(doc.elements[0].commands)
    detok = Detokenizer(l2.vocab, l2.arcs)
    svg_doc = detok.to_svg_document(l2_result.token_ids, width=300, height=300)

    check("rect L2 → SVG: <rect> 요소 포함",
          "<rect" in svg_doc,
          f"output: {svg_doc[:100]}...")
    check("rect L2 → SVG: width 속성 존재",
          'width="' in svg_doc,
          f"no width attribute")

    # Ellipse round-trip
    doc = parser.parse_string(TEST_SVGS["ellipse"])
    l2_result = l2.tokenize(doc.elements[0].commands)
    detok = Detokenizer(l2.vocab, l2.arcs)
    svg_doc = detok.to_svg_document(l2_result.token_ids, width=300, height=300)

    check("ellipse L2 → SVG: <ellipse> 요소 포함",
          "<ellipse" in svg_doc,
          f"output: {svg_doc[:100]}...")

    # ============================================================
    print("\n" + "="*60)
    print("TEST SUITE 4: GID 벤치마크 — Level 1 vs Level 2")
    print("="*60)
    # ============================================================

    print(f"\n  {'SVG Type':<20} {'L1 Tokens':>10} {'L2 Tokens':>10} {'Compression':>12} {'GID L2/BPE':>12}")
    print(f"  {'-'*64}")

    benchmark_results = {}
    for name, svg_str in TEST_SVGS.items():
        if name == "mixed_shapes":
            continue  # multi-element은 별도 처리

        doc = parser.parse_string(svg_str)
        if not doc.elements:
            continue

        l1 = PrimitiveTokenizer(canvas_size=300, max_coord_level=6)
        l2 = CompositeTokenizer(canvas_size=300, max_coord_level=6)

        l1_r = l1.tokenize(doc.elements[0].commands)
        l2_r = l2.tokenize(doc.elements[0].commands)

        # 예상 BPE 토큰 수 (문자열 길이 / 4 근사)
        svg_text = svg_str
        bpe_est = max(len(svg_text) // 4, l1_r.n_tokens)

        compress = l1_r.n_tokens / max(l2_r.n_tokens, 1)
        gid_vs_bpe = bpe_est / max(l2_r.n_tokens, 1)

        benchmark_results[name] = {
            "l1": l1_r.n_tokens, "l2": l2_r.n_tokens,
            "compress": compress, "gid": gid_vs_bpe,
            "shapes": len(l2_r.detected_shapes)
        }

        shapes_str = f" ({l2_r.detected_shapes[0].shape_type.value})" if l2_r.detected_shapes else ""
        print(f"  {name:<20} {l1_r.n_tokens:>10} {l2_r.n_tokens:>10} {compress:>10.1f}× {gid_vs_bpe:>10.1f}×{shapes_str}")

    # Circle GID 개선 확인 (v0.1에서 0.6×이었던 것이 개선)
    if "circle_100" in benchmark_results:
        br = benchmark_results["circle_100"]
        check("circle GID vs BPE > 1.0× (v0.1에서 0.6×이었음)",
              br["gid"] > 1.0,
              f"got {br['gid']:.1f}×")

    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    # ============================================================

    print(f"\n  Total: {results['total']} | Passed: {results['passed']} | Failed: {results['failed']}")
    if results["failed"] == 0:
        print(f"\n  ★ ALL {results['total']} TESTS PASSED ★")
    else:
        print(f"\n  ⚠ {results['failed']} TESTS FAILED")

    return results


if __name__ == "__main__":
    run_tests()
