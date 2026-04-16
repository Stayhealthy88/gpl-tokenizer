# GPL Tokenizer — Geometric Primitive Language for SVG

<p align="center">
  <strong>SVG의 기하학적 구조를 보존하는 차세대 토큰화 시스템</strong>
</p>

<p align="center">
  <a href="./README.md">English</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

## Problem

LLM이 SVG를 생성할 때 **좌표 환각(coordinate hallucination)** 이 발생하는 근본 원인은 BPE 토큰화가 2D 기하학적 정보를 파괴하기 때문입니다.

<p align="center">
  <img src="assets/bpe_vs_gpl.svg" alt="BPE vs GPL Tokenization" width="750"/>
</p>

```
SVG:    C 40 10, 65 10, 95 80
BPE:    ["C", " 4", "0", " 1", "0", ",", " 6", "5", ...]  → 15+ tokens (기하학적 의미 파괴)
GPL:    [CUBIC] [q4:5,1] [q4:8,1] [q4:12,10] [G1] [κ3]    → 6 tokens (기하학적 구조 보존)
```

## Features

- **SVG 1.1 Full Parser** — path, circle, rect, ellipse, line, polygon, polyline 지원
- **Geometric Analyzer** — 베지에 곡률, G0/G1/G2 연속성, 호 길이, 접선 각도 분석
- **ARCS (Adaptive Resolution Coordinate System)** — 쿼드트리 기반 적응적 좌표 양자화
- **Level 1 Primitive Tokenizer** — [Command][Coords][Continuity][Curvature] 복합 토큰 구조
- **Detokenizer** — GPL 토큰 → 유효 SVG 역변환
- **Round-trip verified** — 24/24 테스트 통과

## Round-trip Demo

GPL 토큰화 후 역변환한 결과 — 원본과 복원 SVG의 시각적 비교:

<p align="center">
  <img src="assets/roundtrip_demo.svg" alt="Round-trip Tokenization Demo" width="750"/>
</p>

## Quick Start

```python
from gpl_tokenizer.parser import SVGParser
from gpl_tokenizer.tokenizer import PrimitiveTokenizer, Detokenizer

# 1. SVG 파싱
parser = SVGParser()
doc = parser.parse_string('<svg width="300" height="300"><path d="M 10 80 C 40 10, 65 10, 95 80"/></svg>')

# 2. GPL 토큰화
tokenizer = PrimitiveTokenizer(canvas_size=300, max_coord_level=6)
result = tokenizer.tokenize(doc.elements[0].commands)

print(f"Commands: {result.n_commands}")   # 2
print(f"Tokens: {result.n_tokens}")       # ~8
print(f"Token IDs: {result.token_ids}")

# 3. 역변환 (Detokenize)
detok = Detokenizer(tokenizer.vocab, tokenizer.arcs)
svg_d = detok.detokenize(result.token_ids)
print(f"Recovered: {svg_d}")

# 4. 완전한 SVG 문서로 복원
svg_doc = detok.to_svg_document(result.token_ids, width=300, height=300)
```

## Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="GPL Tokenizer Architecture" width="850"/>
</p>

```
SVG Text
  → SVGParser (XML → structured elements)
  → PathParser (d attribute → PathCommands)
  → CurvatureAnalyzer (κ, θ, arc length per segment)
  → ContinuityAnalyzer (G0/G1/G2 at junctions)
  → ARCS (adaptive quadtree coordinate quantization)
  → PrimitiveTokenizer (commands + coords + diff attrs → token IDs)
  → [LLM Training / Inference]
  → Detokenizer (token IDs → valid SVG)
```

### Token Structure

각 렌더링 가능 세그먼트는 다음 구조의 토큰 서브시퀀스로 변환됩니다:

| Field | Description | Example |
|:---|:---|:---|
| `CommandToken` | 명령어 유형 | `CUBIC`, `LINE`, `ARC` |
| `CoordTokens` | ARCS 양자화 좌표 | `q4:5,1`, `q4:8,1`, `q4:12,10` |
| `ContinuityToken` | 이전 세그먼트와의 연속성 | `G0`, `G1`, `G2` |
| `CurvatureToken` | 양자화된 곡률 클래스 | `κ0`(직선) ~ `κ15`(급곡선) |

### Vocabulary Layout

| Range | Type | Count |
|:---|:---|:---|
| 0-4 | Special (PAD, BOS, EOS, SEP, UNK) | 5 |
| 10-17 | Commands (MOVE ~ CLOSE) | 8 |
| 30-33 | Continuity (DISC, G0, G1, G2) | 4 |
| 40-55 | Curvature bins | 16 |
| 100+ | ARCS coordinates | 5,461 |
| **Total** | | **5,561** |

## Benchmarks

GID (Geometric Information Density) — BPE 대비:

<p align="center">
  <img src="assets/gid_benchmark.svg" alt="GID Benchmark vs BPE" width="650"/>
</p>

| SVG Type | GPL Tokens | GID vs BPE |
|:---|:---|:---|
| Simple line | 6 | **2.7×** |
| Rectangle | 13 | **1.6×** |
| Cubic Bezier | 15 | **1.6×** |
| Triangle icon | 10 | **3.4×** |
| Multi-path | 19 | **2.8×** |

## Real-world SVG Examples

실제 SVG 애니메이션 패턴(anime.js 참고)에 GPL 토큰화를 적용한 예시:

### Shape Morphing

<p align="center">
  <img src="assets/example_morphing.svg" alt="SVG Shape Morphing — GPL Token Sequence" width="750"/>
</p>

### Line Drawing (Stroke Animation)

<p align="center">
  <img src="assets/example_line_drawing.svg" alt="SVG Line Drawing — Stroke Animation Stages" width="750"/>
</p>

### Motion Path (Trajectory Encoding)

<p align="center">
  <img src="assets/example_motion_path.svg" alt="SVG Motion Path — Geometric Trajectory Encoding" width="750"/>
</p>

## Project Structure

```
gpl_tokenizer/
├── parser/
│   ├── svg_parser.py          # SVG XML → structured elements
│   └── path_parser.py         # path d → PathCommands
├── analyzer/
│   ├── curvature.py           # Bézier curvature analysis
│   └── continuity.py          # G0/G1/G2 continuity detection
├── tokenizer/
│   ├── arcs.py                # Adaptive Resolution Coordinate System
│   ├── vocabulary.py          # Token vocabulary management
│   ├── primitive_tokenizer.py # Level 1 tokenization
│   └── detokenizer.py         # GPL → SVG reconstruction
├── utils/
│   └── math_utils.py          # Bézier math utilities
└── tests/
    └── test_roundtrip.py      # Round-trip & GID tests (24/24 pass)
```

## Roadmap

- [x] **v0.1** — Core tokenizer (parser, analyzer, ARCS, tokenizer, detokenizer)
- [x] **v0.2** — Level 2 segment merging (circle/rect → single token)
- [ ] **v0.3** — Spatial relation tokens (ALIGN, SYM, PROP)
- [ ] **v0.4** — PyTorch embedding layer with HMN initialization
- [ ] **v0.5** — LLM fine-tuning pipeline (Qwen2.5-1.5B)
- [ ] **v1.0** — API server + Figma plugin

## Key References

1. **HiVG** (Xing et al., 2026) — Hierarchical SVG Tokenization
2. **StrokeNUWA** (Tang et al., 2024) — Stroke tokenization for VG synthesis
3. **LLM4SVG** (Xing et al., 2025) — Empowering LLMs for SVG
4. **VectorGym** (Rodriguez et al., 2026) — SVG multitask benchmark

## License

This project is proprietary. All rights reserved.

---

*Built with the goal of enabling LLMs to truly understand 2D geometry.*
