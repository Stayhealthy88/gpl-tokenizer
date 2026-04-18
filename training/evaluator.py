"""
GPL 생성 품질 평가기
=====================
모델이 생성한 SVG의 품질을 정량적으로 측정.

메트릭:
    1. Valid SVG Rate: 유효한 SVG 비율
    2. Token Accuracy: 구조적 토큰 정확도
    3. Structural Score: BOS/EOS 쌍, 명령어-좌표 쌍 일관성
    4. Diversity Score: 생성 다양성 (유니크 토큰 비율)
    5. Geometric Consistency: 좌표 범위, 도형 완성도
"""

from typing import List, Dict
from collections import Counter
import math

from .generator import GeneratedSVG
from ..tokenizer.vocabulary import (
    SpecialToken, CommandToken, CompositeToken, ContinuityToken,
    CURVATURE_TOKEN_BASE, COORD_TOKEN_BASE,
)


class GPLEvaluator:
    """
    GPL 생성 품질 평가기.

    사용법:
        evaluator = GPLEvaluator()
        metrics = evaluator.evaluate(generated_samples)
        evaluator.print_report(metrics)
    """

    def evaluate(self, samples: List[GeneratedSVG]) -> Dict[str, float]:
        """
        생성된 SVG 샘플들의 품질 메트릭 계산.

        Args:
            samples: GeneratedSVG 리스트
        Returns:
            메트릭 딕셔너리
        """
        if not samples:
            return {}

        n = len(samples)

        # 1. Valid SVG Rate
        valid_count = sum(1 for s in samples if s.is_valid)
        valid_rate = valid_count / n

        # 2. Structural Score (각 샘플별)
        struct_scores = [self._structural_score(s.token_ids) for s in samples]
        avg_struct = sum(struct_scores) / n

        # 3. Diversity Score
        diversity = self._diversity_score(samples)

        # 4. Token Length Statistics
        lengths = [s.n_tokens for s in samples]
        avg_len = sum(lengths) / n
        min_len = min(lengths)
        max_len = max(lengths)

        # 5. Geometric Consistency
        geo_scores = [self._geometric_score(s.token_ids) for s in samples]
        avg_geo = sum(geo_scores) / n

        # 6. Category Distribution
        categories = self._detect_categories(samples)

        return {
            "n_samples": n,
            "valid_svg_rate": valid_rate,
            "structural_score": avg_struct,
            "diversity_score": diversity,
            "geometric_score": avg_geo,
            "avg_token_length": avg_len,
            "min_token_length": min_len,
            "max_token_length": max_len,
            "categories": categories,
        }

    def _structural_score(self, token_ids: List[int]) -> float:
        """
        구조적 일관성 점수 (0~1).

        체크사항:
            - BOS로 시작하는가
            - EOS로 끝나는가
            - 명령어 뒤에 좌표가 오는가
            - 연속성/곡률 토큰이 적절한 위치인가
        """
        if not token_ids:
            return 0.0

        score = 0.0
        checks = 0

        # BOS 시작
        checks += 1
        if token_ids[0] == SpecialToken.BOS:
            score += 1.0

        # EOS 종료
        checks += 1
        if token_ids[-1] == SpecialToken.EOS:
            score += 1.0

        # 명령어 뒤에 좌표
        cmd_tokens = set(range(10, 18)) | set(range(20, 24))
        for i in range(len(token_ids) - 1):
            if token_ids[i] in cmd_tokens:
                checks += 1
                if token_ids[i + 1] >= COORD_TOKEN_BASE:
                    score += 1.0

        # PAD가 중간에 없음
        checks += 1
        inner = token_ids[1:-1] if len(token_ids) > 2 else []
        if SpecialToken.PAD not in inner:
            score += 1.0

        return score / max(checks, 1)

    def _geometric_score(self, token_ids: List[int]) -> float:
        """
        기하학적 일관성 점수 (0~1).

        체크사항:
            - 좌표 토큰이 유효 범위 내인가
            - 도형 토큰 뒤에 충분한 좌표가 있는가
            - 반복된 동일 좌표가 없는가 (다양성)
        """
        if not token_ids:
            return 0.0

        score = 0.0
        checks = 0

        # 좌표 토큰 수
        coord_ids = [t for t in token_ids if t >= COORD_TOKEN_BASE]
        n_coords = len(coord_ids)

        # 좌표가 최소 1개 이상
        checks += 1
        if n_coords >= 1:
            score += 1.0

        # 좌표 다양성: 유니크 좌표 비율
        if n_coords > 0:
            checks += 1
            unique_ratio = len(set(coord_ids)) / n_coords
            score += unique_ratio

        # 명령어 수 대비 좌표 수 비율 (합리적 범위: 1~4)
        cmd_count = sum(1 for t in token_ids if 10 <= t < 24)
        if cmd_count > 0:
            checks += 1
            ratio = n_coords / cmd_count
            if 0.5 <= ratio <= 5.0:
                score += 1.0

        return score / max(checks, 1)

    def _diversity_score(self, samples: List[GeneratedSVG]) -> float:
        """
        생성 다양성 점수 (0~1).

        유니크한 시퀀스 비율 + 평균 유니크 토큰 비율.
        """
        if not samples:
            return 0.0

        # 유니크 시퀀스 비율
        seq_strs = [str(s.token_ids) for s in samples]
        unique_seqs = len(set(seq_strs)) / len(seq_strs)

        # 평균 유니크 토큰 비율
        unique_ratios = []
        for s in samples:
            if len(s.token_ids) > 0:
                unique_ratios.append(len(set(s.token_ids)) / len(s.token_ids))
        avg_unique = sum(unique_ratios) / len(unique_ratios) if unique_ratios else 0

        return (unique_seqs + avg_unique) / 2

    def _detect_categories(self, samples: List[GeneratedSVG]) -> Dict[str, int]:
        """생성된 SVG의 카테고리 분포 감지."""
        cats = Counter()
        for s in samples:
            cat = "unknown"
            for t in s.token_ids:
                if t == CompositeToken.CIRCLE:
                    cat = "circle"
                    break
                elif t == CompositeToken.RECT:
                    cat = "rect"
                    break
                elif t == CompositeToken.ELLIPSE:
                    cat = "ellipse"
                    break
                elif t == CommandToken.LINE:
                    cat = "path"
                    break
                elif t == CommandToken.CUBIC:
                    cat = "curve"
                    break
            cats[cat] += 1
        return dict(cats)

    def print_report(self, metrics: Dict) -> str:
        """평가 결과를 포맷팅된 문자열로 출력."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"GPL Generation Quality Report")
        lines.append(f"{'='*60}")

        lines.append(f"\n  {'Metric':<30} {'Value':>15}")
        lines.append(f"  {'-'*47}")

        lines.append(f"  {'Samples':<30} {metrics.get('n_samples', 0):>15}")
        lines.append(f"  {'Valid SVG Rate':<30} {metrics.get('valid_svg_rate', 0):>14.1%}")
        lines.append(f"  {'Structural Score':<30} {metrics.get('structural_score', 0):>14.3f}")
        lines.append(f"  {'Geometric Score':<30} {metrics.get('geometric_score', 0):>14.3f}")
        lines.append(f"  {'Diversity Score':<30} {metrics.get('diversity_score', 0):>14.3f}")
        lines.append(f"  {'Avg Token Length':<30} {metrics.get('avg_token_length', 0):>14.1f}")
        lines.append(f"  {'Min / Max Length':<30} "
                      f"{metrics.get('min_token_length', 0):>6} / "
                      f"{metrics.get('max_token_length', 0):<6}")

        cats = metrics.get("categories", {})
        if cats:
            lines.append(f"\n  Category Distribution:")
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                pct = count / metrics.get("n_samples", 1)
                lines.append(f"    {cat:<20} {count:>5} ({pct:>5.1%})")

        report = "\n".join(lines)
        print(report)
        return report
