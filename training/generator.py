"""
GPL SVG 생성기
===============
학습된 GPLTransformer로 새로운 SVG를 생성.

파이프라인:
    [BOS] prompt → GPLTransformer (autoregressive) → GPL 토큰 시퀀스
    → Detokenizer → SVG path d → SVG 파일

생성 모드:
    1. 무조건 생성: [BOS]부터 자유 생성
    2. 조건부 생성: [BOS][CIRCLE] 등 도형 유형 지정
    3. 완성: 부분 시퀀스 → 나머지 생성
"""

from typing import List, Optional, Dict
from dataclasses import dataclass

import torch

from .gpl_transformer import GPLTransformer
from ..tokenizer.vocabulary import (
    GPLVocabulary, SpecialToken, CommandToken, CompositeToken,
)
from ..tokenizer.detokenizer import Detokenizer
from ..tokenizer.arcs import ARCS


@dataclass
class GeneratedSVG:
    """생성된 SVG 결과."""
    token_ids: List[int]     # GPL 토큰 시퀀스
    svg_path: str            # SVG path d 문자열
    svg_full: str            # 완전한 SVG 파일 문자열
    is_valid: bool           # 유효한 SVG 여부
    n_tokens: int            # 토큰 수


class GPLGenerator:
    """
    GPL 모델 기반 SVG 생성기.

    사용법:
        gen = GPLGenerator(model, vocab, arcs)
        results = gen.generate_batch(n=10)
        gen.save_svg(results[0], "output.svg")
    """

    def __init__(self,
                 model: GPLTransformer,
                 vocab: GPLVocabulary,
                 arcs: ARCS,
                 detokenizer: Optional[Detokenizer] = None):
        self.model = model
        self.vocab = vocab
        self.arcs = arcs
        self.detokenizer = detokenizer or Detokenizer(vocab, arcs)

    def generate_unconditional(self,
                                max_len: int = 64,
                                temperature: float = 0.8,
                                top_k: int = 50,
                                top_p: float = 0.9,
                                ) -> GeneratedSVG:
        """
        무조건 생성: [BOS]부터 자유 생성.

        Returns:
            GeneratedSVG 결과
        """
        prompt = torch.tensor([[SpecialToken.BOS]], dtype=torch.long)
        return self._generate_from_prompt(prompt, max_len, temperature, top_k, top_p)

    def generate_shape(self,
                       shape_type: str = "circle",
                       max_len: int = 32,
                       temperature: float = 0.7,
                       top_k: int = 30,
                       top_p: float = 0.9,
                       ) -> GeneratedSVG:
        """
        도형 유형 지정 생성.

        Args:
            shape_type: "circle", "rect", "ellipse", "line", "curve"
        """
        type_map = {
            "circle": CompositeToken.CIRCLE,
            "rect": CompositeToken.RECT,
            "ellipse": CompositeToken.ELLIPSE,
            "line": CommandToken.MOVE,
            "curve": CommandToken.MOVE,
        }
        token = type_map.get(shape_type, CompositeToken.CIRCLE)
        prompt = torch.tensor([[SpecialToken.BOS, int(token)]], dtype=torch.long)
        return self._generate_from_prompt(prompt, max_len, temperature, top_k, top_p)

    def generate_completion(self,
                            partial_ids: List[int],
                            max_len: int = 64,
                            temperature: float = 0.8,
                            top_k: int = 50,
                            top_p: float = 0.9,
                            ) -> GeneratedSVG:
        """
        부분 시퀀스 완성.

        Args:
            partial_ids: 시작 토큰 시퀀스
        """
        prompt = torch.tensor([partial_ids], dtype=torch.long)
        return self._generate_from_prompt(prompt, max_len, temperature, top_k, top_p)

    def generate_batch(self,
                       n: int = 10,
                       mode: str = "unconditional",
                       **kwargs,
                       ) -> List[GeneratedSVG]:
        """
        N개 SVG 일괄 생성.

        Args:
            n: 생성 수
            mode: "unconditional" | "circle" | "rect" | "ellipse" | ...
        """
        results = []
        for _ in range(n):
            if mode == "unconditional":
                result = self.generate_unconditional(**kwargs)
            else:
                result = self.generate_shape(shape_type=mode, **kwargs)
            results.append(result)
        return results

    def _generate_from_prompt(self,
                               prompt: torch.Tensor,
                               max_len: int,
                               temperature: float,
                               top_k: int,
                               top_p: float,
                               ) -> GeneratedSVG:
        """프롬프트에서 토큰 생성 → SVG 변환."""
        # 토큰 생성
        generated = self.model.generate(
            prompt=prompt,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=SpecialToken.EOS,
        )

        token_ids = generated[0].tolist()

        # SVG 변환 시도
        svg_path = ""
        svg_full = ""
        is_valid = False

        try:
            svg_path = self.detokenizer.detokenize(token_ids)
            svg_full = self._wrap_svg(svg_path)
            is_valid = self._validate_svg(svg_path)
        except Exception:
            svg_full = self._wrap_svg("")  # 빈 SVG
            is_valid = False

        return GeneratedSVG(
            token_ids=token_ids,
            svg_path=svg_path,
            svg_full=svg_full,
            is_valid=is_valid,
            n_tokens=len(token_ids),
        )

    def _wrap_svg(self, path_d: str, width: int = 256, height: int = 256) -> str:
        """SVG path를 완전한 SVG 파일로 래핑."""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">\n'
            f'  <path d="{path_d}" fill="none" stroke="black" stroke-width="2"/>\n'
            f'</svg>'
        )

    def _validate_svg(self, svg_path: str) -> bool:
        """
        SVG path 유효성 검사.

        v0.5.1 개선 — 과거 버전은 "M 로 시작" + "숫자 포함" 만 확인했으나,
        이는 `M` 하나만으로도 True 를 반환하는 약한 검증이었다.
        이제 실제 PathParser 로 파싱해 적어도 1개의 렌더링 가능 명령이
        성공적으로 해석되는지 검증한다. 파싱 실패 또는 0 명령이면 False.
        """
        if not svg_path or len(svg_path.strip()) == 0:
            return False

        stripped = svg_path.strip()
        if not (stripped.startswith('M') or stripped.startswith('m')):
            return False

        # PathParser 로 실제 해석 시도
        try:
            from ..parser.path_parser import PathParser, CommandType
            parser = PathParser()
            commands = parser.parse(svg_path)
        except Exception:
            return False

        # 렌더링 가능 명령(MOVE/CLOSE 이외) 이 최소 1개 있어야 진짜 도형
        renderable = [
            c for c in commands
            if c.command_type not in (CommandType.MOVE, CommandType.CLOSE)
        ]
        return len(renderable) >= 1

    @staticmethod
    def save_svg(result: GeneratedSVG, filepath: str):
        """SVG 파일 저장."""
        with open(filepath, 'w') as f:
            f.write(result.svg_full)
