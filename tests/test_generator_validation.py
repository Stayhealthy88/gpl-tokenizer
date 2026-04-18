"""
Generator._validate_svg 강화 테스트
==================================
v0.5.1 개선 #3: 과거에는 "M 로 시작 + 숫자 포함" 만 확인하던 검증을
PathParser 기반의 실제 구조 검증으로 교체.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpl_tokenizer.tokenizer.vocabulary import GPLVocabulary
from gpl_tokenizer.tokenizer.arcs import ARCS
from gpl_tokenizer.training.gpl_transformer import GPLTransformer, GPLTransformerConfig
from gpl_tokenizer.training.generator import GPLGenerator


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
print("v0.5.1 Generator Validation Tests")
print("=" * 60)

# 테스트용 소형 generator (실제 생성은 하지 않고 _validate_svg 만 테스트)
vocab = GPLVocabulary(max_coord_level=4)
arcs = ARCS(canvas_size=256.0, max_level=4)
config = GPLTransformerConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64)
model = GPLTransformer(vocab, config)
gen = GPLGenerator(model, vocab, arcs)


# ------------------------------------------------------------
# 1. 유효한 SVG path
# ------------------------------------------------------------
print("\n[1] 유효한 SVG path")

valid_cases = [
    "M 10 10 L 20 20",
    "M 0 0 C 10 0 20 10 30 30",
    "M 5 5 Q 10 5 15 10",
    "M 0 0 L 10 10 L 10 20 Z",
    "m 0 0 l 5 5 l 5 -5",  # 상대 좌표
]

for case in valid_cases:
    check(gen._validate_svg(case) == True,
          f"유효: '{case[:40]}...'")


# ------------------------------------------------------------
# 2. 무효한 SVG path — 이전 검증은 통과했으나 실제로는 불완전
# ------------------------------------------------------------
print("\n[2] 이전 약한 검증을 통과하던 무효 케이스")

# "M 10 10" — 숫자도 있고 M으로 시작하지만 렌더링 가능 명령이 없음
check(gen._validate_svg("M 10 10") == False,
      "M 만 있고 LINE/CURVE 등 렌더 명령 없음")

# "M 0 0 0 0 0" — 숫자 나열만 있는 비정상 케이스
# 실제로는 파서가 암묵적 LINE 으로 해석할 수 있음 — 해석 결과에 따라 True 일 수 있음
# 그래서 이 케이스는 파서 동작에 의존

# "MX" — 알파벳만
check(gen._validate_svg("MX") == False,
      "알파벳만으로 구성된 무효 경로")

# 빈 문자열
check(gen._validate_svg("") == False, "빈 문자열")
check(gen._validate_svg("   ") == False, "공백만")

# M 으로 시작하지 않음
check(gen._validate_svg("L 10 10 L 20 20") == False,
      "M 없이 시작하는 경로")

# 잘못된 문자
check(gen._validate_svg("@@##") == False, "비정상 문자")

# None-like
check(gen._validate_svg("M") == False, "M 하나만")


# ------------------------------------------------------------
# 3. 예외 처리 — 파싱 실패 시 False
# ------------------------------------------------------------
print("\n[3] 파싱 실패 시 안전하게 False 반환")

# 비정상적으로 긴 입력도 예외 없이 False 반환
very_long = "M " + " ".join(["X"] * 1000)
try:
    result = gen._validate_svg(very_long)
    check(result == False, "비정상 입력도 예외 없이 False")
except Exception as e:
    check(False, f"예외 발생: {type(e).__name__}")


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
