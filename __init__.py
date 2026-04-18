"""
GPL (Geometric Primitive Language) Tokenizer
=============================================
SVG의 기하학적 구조를 보존하는 토큰화 시스템.

Architecture:
    SVG Text → SVGParser → PathCommands → GeometricAnalyzer → AnnotatedCommands
    → PrimitiveTokenizer → GPL Tokens → Detokenizer → SVG Text
    → GPLEmbedding → GPLTransformer → Generated GPL Tokens → SVG

Modules:
    parser/      : SVG 파싱 및 path 명령어 분해
    analyzer/    : 베지에 곡률, G1/G2 연속성, 공간 관계 분석
    tokenizer/   : Level 1-3 토큰화, ARCS, 어휘 관리
    embedding/   : GPLEmbedding + HMN 초기화 (v0.4)
    training/    : 학습 파이프라인 — 데이터셋, Transformer, 생성, 평가 (v0.5)
    utils/       : 수학 유틸리티
"""

__version__ = "0.5.0"
