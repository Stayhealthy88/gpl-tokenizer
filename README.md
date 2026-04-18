# GPL Tokenizer

**Teaching AI to truly understand shapes and drawings.**

<p align="center">
  <a href="./README.ko.md">한국어</a>
</p>

---

## The Problem

Today's AI models (like ChatGPT and Claude) are remarkably good at understanding and generating text. But when it comes to vector graphics — the crisp, scalable images used in every app, website, and design tool — they consistently fail. Circles come out wobbly. Rectangles don't close. Coordinates are just... wrong.

Why? Because current AI systems read graphics code the same way they read English — one text fragment at a time. A coordinate like `150.5` gets chopped into meaningless pieces like `"1"`, `"50"`, `"."`, `"5"`. The AI never sees that these fragments represent a *point in space*. It's like trying to navigate a city by reading a map one letter at a time.

<p align="center">
  <img src="assets/hero_concept.svg" alt="The Problem and Our Solution" width="800"/>
</p>

## Our Solution

GPL Tokenizer is a **geometry-aware translation layer** that sits between vector graphics and AI models. Instead of letting AI read raw code character by character, we first translate graphics into a language designed for geometric understanding.

A circle isn't 28 text fragments anymore — it's a single "circle" token with a center point and radius. A row of five identical buttons isn't 167 text fragments — it's one button definition plus "repeat 4 times, evenly spaced." The AI sees shapes, positions, and spatial relationships — not scrambled digits.

<p align="center">
  <img src="assets/how_it_works.svg" alt="How GPL Tokenizer Works" width="800"/>
</p>

## Key Results

We've built three levels of compression, each adding a new layer of geometric intelligence:

**Level 1 — Basic Geometry.** Each drawing command (lines, curves, arcs) becomes a structured token that preserves its mathematical properties: position, curvature, and smoothness. This alone is 2-3x more efficient than standard text tokenization.

**Level 2 — Shape Recognition.** The system automatically detects common shapes — circles, rectangles, ellipses — and compresses them into single tokens. A circle that required 28 tokens at Level 1 becomes just 5 tokens. That's a **5.6x compression** with zero information loss.

**Level 3 — Spatial Intelligence.** When multiple elements share a pattern (aligned in a row, evenly spaced, symmetric), the system captures these relationships. Five identical circles in a row? Instead of describing each one separately (21 tokens), it says "one circle, repeat 4 times with this spacing" (11 tokens). **Up to 15x fewer tokens** compared to standard AI.

<p align="center">
  <img src="assets/compression_results.svg" alt="Efficiency Gains" width="800"/>
</p>

The fewer tokens AI needs to process, the faster it runs, the less it costs, and the more accurately it draws. This isn't just an optimization — it's a fundamental shift in how AI understands visual content.

## How It's Built

The tokenizer pipeline has four main stages:

**Parsing** reads any SVG file and breaks it into structured geometric commands — understanding the difference between straight lines, curves, and arcs.

**Analysis** examines each piece: How curved is this segment? Does it connect smoothly to the next one? Is this actually a circle drawn as four curves? Are these shapes aligned or symmetric?

**Tokenization** converts everything into compact, meaningful tokens at three levels — from individual commands (L1) to recognized shapes (L2) to spatial patterns (L3).

**Reconstruction** reverses the process perfectly: tokens become valid SVG graphics again. This round-trip fidelity is verified by 47 automated tests.

## What's Next

<p align="center">
  <img src="assets/roadmap_visual.svg" alt="Development Roadmap" width="800"/>
</p>

- [x] **v0.1** — Core engine: parser, geometric analyzer, tokenizer, reconstruction
- [x] **v0.2** — Shape recognition: circles, rectangles compressed to single tokens
- [x] **v0.3** — Spatial intelligence: alignment, symmetry, spacing patterns
- [x] **v0.4** — AI embedding layer: connect tokens to neural networks (PyTorch)
- [ ] **v0.5** — AI training pipeline: fine-tune language models for SVG generation
- [ ] **v1.0** — Product: API service + Figma design tool plugin

## Why This Matters

Vector graphics are everywhere — app icons, logos, UI components, illustrations, data visualizations, maps. The global design tools market is valued at $13B+ and growing. Yet AI still can't reliably create or edit vector content.

GPL Tokenizer solves the foundational bottleneck: giving AI models a native understanding of 2D geometry. This unlocks capabilities like AI-powered design generation, automated icon creation, intelligent SVG editing, and design-to-code workflows that actually produce correct output.

## Technical Foundation

Built on peer-reviewed research in geometric tokenization:

- **HiVG** (Xing et al.) — Hierarchical SVG tokenization
- **StrokeNUWA** (Tang et al.) — Stroke-level tokenization for vector synthesis
- **LLM4SVG** (Xing et al.) — Empowering language models for SVG
- **VectorGym** (Rodriguez et al.) — SVG multitask benchmarking

## License

This project is proprietary. All rights reserved.

---

*Building the bridge between AI and visual design.*
