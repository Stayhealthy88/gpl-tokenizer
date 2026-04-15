"""
GPL (Geometric Primitive Language) Tokenizer
=============================================
SVG의 기하학적 구조를 보존하는 토큰화 시스템.

Architecture:
    SVG Text → SVGParser → PathCommands → GeometricAnalyzer → AnnotatedCommands
    → PrimitiveTokenizer → GPL Tokens → Detokenizer → SVG Text

Modules:
    parser/      : SVG 파싱 및 path 명령어 분해
    analyzer/    : 베지에 곡률, G1/G2 연속성, 공간 관계 분석
    tokenizer/   : Level 1 프리미티브 토큰화, ARCS, 어휘 관리
    utils/       : 수학 유틸리티
"""

__version__ = "0.1.0"
