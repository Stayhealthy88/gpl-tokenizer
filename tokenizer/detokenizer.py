"""
GPL 디토크나이저
================
GPL 토큰 시퀀스를 유효한 SVG path 문자열로 역변환.

비판 문서 반영 (검증 관점):
    "디토크나이저의 역변환 충실도" — 왕복(round-trip) 테스트로 검증 가능하도록 설계.
"""

from typing import List, Tuple, Optional
from .vocabulary import (
    GPLVocabulary, GPLToken, SpecialToken, CommandToken, CompositeToken,
    ContinuityToken, CURVATURE_TOKEN_BASE, N_CURVATURE_BINS,
    COORD_TOKEN_BASE
)
from .arcs import ARCS


class Detokenizer:
    """
    GPL 토큰 시퀀스를 SVG path 'd' 문자열로 역변환.

    사용법:
        detok = Detokenizer(vocab, arcs)
        svg_d = detok.detokenize(token_ids)
        svg_full = detok.to_svg_element(token_ids)
    """

    def __init__(self, vocab: GPLVocabulary, arcs: ARCS):
        self.vocab = vocab
        self.arcs = arcs

    def detokenize(self, token_ids: List[int]) -> str:
        """
        토큰 ID 시퀀스를 SVG path 'd' 문자열로 역변환.

        Args:
            token_ids: GPL 토큰 ID 리스트
        Returns:
            SVG path d 속성 문자열
        """
        # 토큰 해석
        decoded = [self.vocab.decode_token_id(tid) for tid in token_ids]

        # 명령어 + 좌표 그룹으로 분리
        segments = self._group_into_segments(decoded)

        # 각 세그먼트를 SVG path 명령어 문자열로 변환
        parts = []
        for seg in segments:
            part = self._segment_to_path_str(seg)
            if part:
                parts.append(part)

        return " ".join(parts)

    def to_svg_element(self, token_ids: List[int],
                       fill: str = "none", stroke: str = "black",
                       stroke_width: float = 1.0) -> str:
        """토큰 시퀀스를 완전한 SVG <path> 요소로 변환."""
        d = self.detokenize(token_ids)
        return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{stroke_width}"/>')

    def to_svg_document(self, token_ids: List[int],
                        width: float = 300, height: float = 300) -> str:
        """토큰 시퀀스를 완전한 SVG 문서로 변환 (Level 2 복합 토큰 지원)."""
        elements = self._to_svg_elements(token_ids)
        inner = "\n  ".join(elements)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}">\n'
                f'  {inner}\n'
                f'</svg>')

    def _to_svg_elements(self, token_ids: List[int]) -> List[str]:
        """토큰 시퀀스를 SVG 요소 리스트로 변환 (path + 기본 도형)."""
        decoded = [self.vocab.decode_token_id(tid) for tid in token_ids]
        elements = []
        path_segments = []
        i = 0

        while i < len(decoded):
            d = decoded[i]

            if d["type"] == "special":
                i += 1
                continue

            if d["type"] == "composite":
                # 현재까지 쌓인 path segments가 있으면 먼저 출력
                if path_segments:
                    path_d = self._segments_to_path_d(path_segments)
                    if path_d:
                        elements.append(
                            f'<path d="{path_d}" fill="none" stroke="black" stroke-width="1"/>'
                        )
                    path_segments = []

                # 복합 토큰 처리
                elem, consumed = self._decode_composite(decoded, i)
                if elem:
                    elements.append(elem)
                i += consumed
                continue

            if d["type"] == "command":
                seg = {"command": d["value"], "coords": [], "continuity": None, "curvature_bin": None}
                i += 1
                while i < len(decoded):
                    dd = decoded[i]
                    if dd["type"] == "coord":
                        from .arcs import QuantizedCoord
                        qc = QuantizedCoord(level=dd["level"], qx=dd["qx"], qy=dd["qy"],
                                            original_x=0, original_y=0)
                        x, y = self.arcs.dequantize(qc)
                        seg["coords"].append((x, y))
                        i += 1
                    elif dd["type"] == "continuity":
                        seg["continuity"] = dd["value"]
                        i += 1
                    elif dd["type"] == "curvature":
                        seg["curvature_bin"] = dd["bin"]
                        i += 1
                    else:
                        break
                path_segments.append(seg)
                continue

            i += 1

        # 남은 path segments 출력
        if path_segments:
            path_d = self._segments_to_path_d(path_segments)
            if path_d:
                elements.append(
                    f'<path d="{path_d}" fill="none" stroke="black" stroke-width="1"/>'
                )

        return elements if elements else [self.to_svg_element(token_ids)]

    def _decode_composite(self, decoded: List[dict], start: int) -> Tuple[str, int]:
        """복합 토큰을 SVG 기본 도형 요소로 변환. (consumed 수 반환)"""
        d = decoded[start]
        shape = d["value"]

        def read_coord(idx):
            """다음 coord 토큰을 읽어 (x, y)로 변환."""
            if idx >= len(decoded) or decoded[idx]["type"] != "coord":
                return None, idx
            dd = decoded[idx]
            from .arcs import QuantizedCoord
            qc = QuantizedCoord(level=dd["level"], qx=dd["qx"], qy=dd["qy"],
                                original_x=0, original_y=0)
            x, y = self.arcs.dequantize(qc)
            return (x, y), idx + 1

        def fmt(v):
            return f"{v:.1f}".rstrip('0').rstrip('.')

        idx = start + 1

        if shape == "CIRCLE":
            center, idx = read_coord(idx)
            radius, idx = read_coord(idx)
            if center and radius:
                cx, cy = center
                r = (radius[0] + radius[1]) / 2  # (r, r)로 인코딩됨
                return (f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(r)}" '
                        f'fill="none" stroke="black" stroke-width="1"/>',
                        idx - start)

        elif shape == "ELLIPSE":
            center, idx = read_coord(idx)
            radii, idx = read_coord(idx)
            if center and radii:
                cx, cy = center
                rx, ry = radii
                return (f'<ellipse cx="{fmt(cx)}" cy="{fmt(cy)}" rx="{fmt(rx)}" ry="{fmt(ry)}" '
                        f'fill="none" stroke="black" stroke-width="1"/>',
                        idx - start)

        elif shape == "RECT":
            origin, idx = read_coord(idx)
            size, idx = read_coord(idx)
            if origin and size:
                x, y = origin
                w, h = size
                return (f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
                        f'fill="none" stroke="black" stroke-width="1"/>',
                        idx - start)

        elif shape == "ROUND_RECT":
            origin, idx = read_coord(idx)
            size, idx = read_coord(idx)
            radii, idx = read_coord(idx)
            if origin and size and radii:
                x, y = origin
                w, h = size
                rx, ry = radii
                return (f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
                        f'rx="{fmt(rx)}" ry="{fmt(ry)}" '
                        f'fill="none" stroke="black" stroke-width="1"/>',
                        idx - start)

        return None, 1

    def _segments_to_path_d(self, segments: List[dict]) -> str:
        """세그먼트 리스트를 path d 문자열로 변환."""
        parts = []
        for seg in segments:
            part = self._segment_to_path_str(seg)
            if part:
                parts.append(part)
        return " ".join(parts)

    def _group_into_segments(self, decoded: List[dict]) -> List[dict]:
        """디코딩된 토큰을 명령어 세그먼트로 그룹화."""
        segments = []
        current_segment = None

        for d in decoded:
            if d["type"] == "special":
                continue  # BOS, EOS, SEP, PAD 스킵

            if d["type"] == "command":
                if current_segment is not None:
                    segments.append(current_segment)
                current_segment = {
                    "command": d["value"],
                    "coords": [],
                    "continuity": None,
                    "curvature_bin": None,
                }

            elif d["type"] == "coord" and current_segment is not None:
                # ARCS 좌표를 연속 좌표로 역변환
                from .arcs import QuantizedCoord
                qc = QuantizedCoord(
                    level=d["level"], qx=d["qx"], qy=d["qy"],
                    original_x=0, original_y=0
                )
                x, y = self.arcs.dequantize(qc)
                current_segment["coords"].append((x, y))

            elif d["type"] == "continuity" and current_segment is not None:
                current_segment["continuity"] = d["value"]

            elif d["type"] == "curvature" and current_segment is not None:
                current_segment["curvature_bin"] = d["bin"]

        if current_segment is not None:
            segments.append(current_segment)

        return segments

    def _segment_to_path_str(self, seg: dict) -> str:
        """단일 세그먼트를 SVG path 명령어 문자열로 변환."""
        cmd = seg["command"]
        coords = seg["coords"]

        def fmt(x, y):
            """좌표 포맷팅 — 불필요한 소수점 제거."""
            fx = f"{x:.1f}".rstrip('0').rstrip('.')
            fy = f"{y:.1f}".rstrip('0').rstrip('.')
            return f"{fx} {fy}"

        if cmd == "MOVE":
            if len(coords) >= 1:
                return f"M {fmt(*coords[0])}"
        elif cmd == "LINE":
            if len(coords) >= 1:
                return f"L {fmt(*coords[0])}"
        elif cmd == "HLINE":
            if len(coords) >= 1:
                fx = f"{coords[0][0]:.1f}".rstrip('0').rstrip('.')
                return f"H {fx}"
        elif cmd == "VLINE":
            if len(coords) >= 1:
                fy = f"{coords[0][1]:.1f}".rstrip('0').rstrip('.')
                return f"V {fy}"
        elif cmd == "CUBIC":
            if len(coords) >= 3:
                return (f"C {fmt(*coords[0])}, "
                        f"{fmt(*coords[1])}, "
                        f"{fmt(*coords[2])}")
        elif cmd == "QUADRATIC":
            if len(coords) >= 2:
                return f"Q {fmt(*coords[0])}, {fmt(*coords[1])}"
        elif cmd == "ARC":
            if len(coords) >= 2:
                # coords[0] = (rx, ry), coords[1] = endpoint
                rx, ry = coords[0]
                ex, ey = coords[1]
                frx = f"{rx:.1f}".rstrip('0').rstrip('.')
                fry = f"{ry:.1f}".rstrip('0').rstrip('.')
                return f"A {frx} {fry} 0 0 1 {fmt(ex, ey)}"
        elif cmd == "CLOSE":
            return "Z"

        return ""
