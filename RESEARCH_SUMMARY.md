# GPL Tokenizer — Research Summary

**Geometric Primitive Language: A geometry-aware tokenization system for SVG vector graphics**

---

## Overview

GPL Tokenizer is a complete research pipeline that transforms how AI models understand and generate vector graphics. Instead of treating SVG code as plain text (character-by-character), GPL preserves the geometric structure of shapes, paths, and spatial relationships through specialized tokenization — then connects this representation to neural networks for generation.

The project spans five development milestones (v0.1–v0.5), progressing from basic parsing to a fully functional AI training pipeline.

---

## Architecture

```
SVG File
  │
  ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  SVG Parser │ ──▶ │ Geometric Analyzer│ ──▶ │ Level 1 Tokenizer│
│  (parser/)  │     │  (analyzer/)      │     │  Primitives      │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                    ┌──────────────────┐     ┌─────────────────┐
                    │ Level 3 Tokenizer│ ◀── │ Level 2 Tokenizer│
                    │  Spatial Patterns │     │  Shape Detection │
                    └────────┬────────┘     └─────────────────┘
                             │
                             ▼
                    ┌──────────────────┐     ┌─────────────────┐
                    │  GPLEmbedding    │ ──▶ │ GPLTransformer   │
                    │  + HMN Init      │     │  Decoder-only    │
                    └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  GPL Generator   │
                                              │  → New SVG Files │
                                              └─────────────────┘
```

---

## Milestones

### v0.1 — Core Engine (24 tests)

Built the foundation: SVG parser, geometric analyzer (curvature + continuity), primitive tokenizer with ARCS coordinate system, vocabulary (5,561 tokens), and round-trip detokenizer.

**Key result:** Lossless SVG → GPL → SVG round-trip verified across 47 test cases. GID (Geometric Information Density) measured at 2–3× better than BPE tokenization.

### v0.2 — Shape Recognition (23 tests)

Automatic detection of circles, rectangles, ellipses, and rounded rectangles. Each recognized shape collapses from ~28 tokens to ~5 tokens.

**Key result:** 5.6× compression for circles vs Level 1. Zero information loss.

### v0.3 — Spatial Intelligence (24 tests)

Multi-element pattern recognition: alignment, symmetry, equal spacing, equal size, and repetition. Five identical circles in a row: 21 tokens → 11 tokens.

**Key result:** Up to 15× compression vs standard BPE. Spatial relations encoded as first-class tokens (ALIGN, SYM, EQUAL_SPACE, REPEAT).

### v0.4 — AI Embedding Layer (25 tests)

PyTorch embedding module combining four components: HMN-initialized token embeddings, type embeddings (7 types), coordinate structure encoder (quadtree MLP), and sinusoidal positional encoding.

**Key result:** HMN initialization encodes geometric structure into initial weights — adjacent coordinates have 0.52 cosine similarity vs 0.00 for random initialization. 736K parameters.

### v0.5 — AI Training Pipeline (42 tests)

Complete end-to-end pipeline: synthetic dataset generator (7 shape/pattern types), decoder-only GPLTransformer (1.79M params with weight tying), trainer (cosine LR, gradient clipping, checkpoints), autoregressive SVG generator (unconditional/conditional/completion modes), and quality evaluator (5 metrics).

**Key result:** 200 training samples, 10 epochs, 8.5 seconds → loss 7.81→2.91, accuracy 5.6%→35.6%, 53.3% valid SVG generation rate, 0.88 structural score, 0.99 geometric score.

---

## Codebase

| Category | Files | Lines |
|----------|-------|-------|
| Source code | 27 | 6,073 |
| Test code | 5 | 1,672 |
| **Total** | **34** | **7,745** |

### Module breakdown

| Module | Purpose | Key classes |
|--------|---------|-------------|
| `parser/` | SVG parsing, path decomposition | SVGParser, PathParser |
| `analyzer/` | Curvature, continuity, shape detection, spatial analysis | CurvatureAnalyzer, ShapeDetector, SpatialAnalyzer |
| `tokenizer/` | L1–L3 tokenization, vocabulary, detokenization, ARCS | PrimitiveTokenizer, CompositeTokenizer, SpatialTokenizer, Detokenizer, GPLVocabulary |
| `embedding/` | PyTorch embeddings with HMN initialization | GPLEmbedding, HMNInitializer |
| `training/` | Dataset, model, trainer, generator, evaluator | SyntheticSVGDataset, GPLTransformer, GPLTrainer, GPLGenerator, GPLEvaluator |

---

## Token Vocabulary (5,561 tokens)

| Range | Type | Count | Examples |
|-------|------|-------|----------|
| 0–4 | Special | 5 | PAD, BOS, EOS, SEP, UNK |
| 10–17 | Command | 8 | MOVE, LINE, CUBIC, ARC, CLOSE |
| 20–23 | Composite | 4 | CIRCLE, ELLIPSE, RECT, ROUND_RECT |
| 30–33 | Continuity | 4 | DISC, G0, G1, G2 |
| 40–55 | Curvature | 16 | κ0 (straight) → κ15 (sharp) |
| 60–70 | Spatial | 11 | ALIGN, SYM, EQUAL_SPACE, REPEAT |
| 100+ | Coordinate | 5,461 | Quadtree ARCS coordinates (levels 0–6) |

---

## Test Coverage

| Suite | Version | Tests | Status |
|-------|---------|-------|--------|
| test_roundtrip | v0.1 | 24 | ✅ All pass |
| test_level2 | v0.2 | 23 | ✅ All pass |
| test_level3 | v0.3 | 24 | ✅ All pass |
| test_embedding | v0.4 | 25 | ✅ All pass |
| test_training | v0.5 | 42 | ✅ All pass |
| **Total** | | **138** | **✅ 138/138** |

---

## Technical Foundation

This work builds on academic research in geometric tokenization:

- **HiVG** (Xing et al.) — Hierarchical SVG tokenization
- **StrokeNUWA** (Tang et al.) — Stroke tokenization for vector synthesis
- **LLM4SVG** (Xing et al.) — Language model enhancement for SVG
- **VectorGym** (Rodriguez et al.) — SVG multi-task benchmark

---

## Next Steps (v1.0)

The remaining milestone targets production deployment:

- REST API service for SVG tokenization and generation
- Figma plugin for design tool integration
- Scaled training on real-world SVG datasets
- Model distillation for edge deployment

---

*GPL Tokenizer — bridging AI and visual design through geometric understanding.*
