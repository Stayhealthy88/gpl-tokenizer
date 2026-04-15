"""
SVG 문서 파서
=============
SVG XML 문서를 구조화된 요소 트리로 파싱.
<path>, <circle>, <rect>, <ellipse>, <line>, <polygon>, <polyline> 지원.
모든 기본 도형은 내부적으로 path 명령어로 정규화(normalize).
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from .path_parser import PathParser, PathCommand, CommandType


@dataclass
class SVGElement:
    """파싱된 SVG 요소."""
    tag: str                               # 원본 태그 (path, circle, rect, ...)
    element_id: Optional[str] = None       # id 속성
    commands: List[PathCommand] = field(default_factory=list)  # path 명령어 시퀀스
    attributes: Dict[str, str] = field(default_factory=dict)   # 스타일/속성
    group_id: Optional[str] = None         # 소속 그룹 id
    transform: Optional[str] = None        # transform 속성 (미처리, 추후 확장)

    @property
    def bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        """(min_x, min_y, max_x, max_y) 바운딩 박스."""
        points = []
        for cmd in self.commands:
            if cmd.end_point:
                points.append(cmd.end_point)
            if cmd.start_point:
                points.append(cmd.start_point)
        if not points:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class SVGDocument:
    """파싱된 SVG 문서."""
    width: float = 300.0
    height: float = 300.0
    viewbox: Optional[Tuple[float, float, float, float]] = None
    elements: List[SVGElement] = field(default_factory=list)

    @property
    def canvas_size(self) -> Tuple[float, float]:
        if self.viewbox:
            return (self.viewbox[2], self.viewbox[3])
        return (self.width, self.height)

    @property
    def total_commands(self) -> int:
        return sum(len(e.commands) for e in self.elements)


class SVGParser:
    """
    SVG XML 문서를 SVGDocument로 파싱.

    사용법:
        parser = SVGParser()
        doc = parser.parse_string('<svg><path d="M0 0 L10 10"/></svg>')
        doc = parser.parse_file("icon.svg")
    """

    SVG_NS = "http://www.w3.org/2000/svg"

    def __init__(self):
        self._path_parser = PathParser()

    def parse_string(self, svg_text: str) -> SVGDocument:
        """SVG XML 문자열을 파싱."""
        # namespace 제거 (간소화)
        svg_text = re.sub(r'\sxmlns="[^"]*"', '', svg_text, count=1)
        svg_text = re.sub(r'\sxmlns:xlink="[^"]*"', '', svg_text)

        root = ET.fromstring(svg_text)
        return self._parse_root(root)

    def parse_file(self, filepath: str) -> SVGDocument:
        """SVG 파일을 파싱."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return self.parse_string(f.read())

    def _parse_root(self, root: ET.Element) -> SVGDocument:
        doc = SVGDocument()

        # viewBox 파싱
        vb = root.get("viewBox")
        if vb:
            parts = re.split(r'[\s,]+', vb.strip())
            if len(parts) == 4:
                doc.viewbox = tuple(float(p) for p in parts)

        # width/height
        w = root.get("width")
        h = root.get("height")
        if w:
            doc.width = self._parse_length(w)
        if h:
            doc.height = self._parse_length(h)

        # 재귀적 요소 수집
        self._collect_elements(root, doc, group_id=None)
        return doc

    def _collect_elements(self, node: ET.Element, doc: SVGDocument,
                          group_id: Optional[str]):
        tag = self._strip_ns(node.tag)

        if tag == "g":
            gid = node.get("id", group_id)
            for child in node:
                self._collect_elements(child, doc, group_id=gid)
            return

        elem = None
        if tag == "path":
            elem = self._parse_path(node)
        elif tag == "circle":
            elem = self._parse_circle(node)
        elif tag == "rect":
            elem = self._parse_rect(node)
        elif tag == "ellipse":
            elem = self._parse_ellipse(node)
        elif tag == "line":
            elem = self._parse_line(node)
        elif tag == "polygon":
            elem = self._parse_polygon(node)
        elif tag == "polyline":
            elem = self._parse_polyline(node)

        if elem is not None:
            elem.group_id = group_id
            elem.element_id = node.get("id")
            elem.transform = node.get("transform")
            # 스타일 속성 수집
            for attr in ["fill", "stroke", "stroke-width", "opacity",
                         "fill-opacity", "stroke-opacity", "style", "class"]:
                val = node.get(attr)
                if val:
                    elem.attributes[attr] = val
            doc.elements.append(elem)

        # 자식 노드 재귀 처리
        for child in node:
            self._collect_elements(child, doc, group_id=group_id)

    def _parse_path(self, node: ET.Element) -> SVGElement:
        d = node.get("d", "")
        commands = self._path_parser.parse(d)
        commands = self._path_parser.resolve_to_absolute(commands)
        return SVGElement(tag="path", commands=commands)

    def _parse_circle(self, node: ET.Element) -> SVGElement:
        """<circle> → 4개의 큐빅 베지에로 근사."""
        cx = float(node.get("cx", 0))
        cy = float(node.get("cy", 0))
        r = float(node.get("r", 0))
        commands = self._circle_to_commands(cx, cy, r)
        return SVGElement(tag="circle", commands=commands,
                          attributes={"cx": str(cx), "cy": str(cy), "r": str(r)})

    def _parse_rect(self, node: ET.Element) -> SVGElement:
        """<rect> → M, L, L, L, Z 명령어."""
        x = float(node.get("x", 0))
        y = float(node.get("y", 0))
        w = float(node.get("width", 0))
        h = float(node.get("height", 0))
        rx = float(node.get("rx", 0))
        ry = float(node.get("ry", rx))

        if rx == 0 and ry == 0:
            commands = self._rect_to_commands(x, y, w, h)
        else:
            commands = self._rounded_rect_to_commands(x, y, w, h, rx, ry)
        return SVGElement(tag="rect", commands=commands)

    def _parse_ellipse(self, node: ET.Element) -> SVGElement:
        """<ellipse> → 4개의 큐빅 베지에로 근사."""
        cx = float(node.get("cx", 0))
        cy = float(node.get("cy", 0))
        rx = float(node.get("rx", 0))
        ry = float(node.get("ry", 0))
        commands = self._ellipse_to_commands(cx, cy, rx, ry)
        return SVGElement(tag="ellipse", commands=commands)

    def _parse_line(self, node: ET.Element) -> SVGElement:
        x1 = float(node.get("x1", 0))
        y1 = float(node.get("y1", 0))
        x2 = float(node.get("x2", 0))
        y2 = float(node.get("y2", 0))
        commands = [
            PathCommand(CommandType.MOVE, False, [x1, y1],
                        abs_params=[x1, y1], start_point=(x1, y1),
                        end_point=(x1, y1)),
            PathCommand(CommandType.LINE, False, [x2, y2],
                        abs_params=[x2, y2], start_point=(x1, y1),
                        end_point=(x2, y2)),
        ]
        return SVGElement(tag="line", commands=commands)

    def _parse_polygon(self, node: ET.Element) -> SVGElement:
        return self._parse_poly(node, close=True)

    def _parse_polyline(self, node: ET.Element) -> SVGElement:
        return self._parse_poly(node, close=False)

    def _parse_poly(self, node: ET.Element, close: bool) -> SVGElement:
        points_str = node.get("points", "")
        nums = [float(n) for n in re.findall(r'[+-]?[\d.]+', points_str)]
        if len(nums) < 4:
            return SVGElement(tag="polygon" if close else "polyline", commands=[])

        commands = []
        # 첫 점: M
        commands.append(PathCommand(
            CommandType.MOVE, False, [nums[0], nums[1]],
            abs_params=[nums[0], nums[1]],
            start_point=(nums[0], nums[1]),
            end_point=(nums[0], nums[1])
        ))
        # 나머지: L
        prev = (nums[0], nums[1])
        for i in range(2, len(nums) - 1, 2):
            x, y = nums[i], nums[i + 1]
            commands.append(PathCommand(
                CommandType.LINE, False, [x, y],
                abs_params=[x, y],
                start_point=prev,
                end_point=(x, y)
            ))
            prev = (x, y)

        if close:
            commands.append(PathCommand(
                CommandType.CLOSE, False, [],
                abs_params=[],
                start_point=prev,
                end_point=(nums[0], nums[1])
            ))

        return SVGElement(tag="polygon" if close else "polyline", commands=commands)

    # --- 기본 도형 → path 명령어 변환 유틸리티 ---

    @staticmethod
    def _rect_to_commands(x, y, w, h) -> List[PathCommand]:
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        cmds = [PathCommand(CommandType.MOVE, False, [pts[0][0], pts[0][1]],
                            abs_params=[pts[0][0], pts[0][1]],
                            start_point=pts[0], end_point=pts[0])]
        for i in range(1, 4):
            cmds.append(PathCommand(CommandType.LINE, False,
                                    [pts[i][0], pts[i][1]],
                                    abs_params=[pts[i][0], pts[i][1]],
                                    start_point=pts[i - 1], end_point=pts[i]))
        cmds.append(PathCommand(CommandType.CLOSE, False, [],
                                abs_params=[], start_point=pts[3], end_point=pts[0]))
        return cmds

    @staticmethod
    def _rounded_rect_to_commands(x, y, w, h, rx, ry) -> List[PathCommand]:
        """둥근 모서리 사각형 → path 명령어."""
        rx = min(rx, w / 2)
        ry = min(ry, h / 2)
        # 간소화: 아크 명령어로 모서리 표현
        cmds = []
        cmds.append(PathCommand(CommandType.MOVE, False,
                                [x + rx, y],
                                abs_params=[x + rx, y],
                                start_point=(x + rx, y), end_point=(x + rx, y)))
        # 상단 변
        cmds.append(PathCommand(CommandType.LINE, False,
                                [x + w - rx, y],
                                abs_params=[x + w - rx, y],
                                start_point=(x + rx, y), end_point=(x + w - rx, y)))
        # 우상 모서리 (아크)
        cmds.append(PathCommand(CommandType.ARC, False,
                                [rx, ry, 0, 0, 1, x + w, y + ry],
                                abs_params=[rx, ry, 0, 0, 1, x + w, y + ry],
                                start_point=(x + w - rx, y), end_point=(x + w, y + ry)))
        # 우측 변
        cmds.append(PathCommand(CommandType.LINE, False,
                                [x + w, y + h - ry],
                                abs_params=[x + w, y + h - ry],
                                start_point=(x + w, y + ry), end_point=(x + w, y + h - ry)))
        # 우하 모서리
        cmds.append(PathCommand(CommandType.ARC, False,
                                [rx, ry, 0, 0, 1, x + w - rx, y + h],
                                abs_params=[rx, ry, 0, 0, 1, x + w - rx, y + h],
                                start_point=(x + w, y + h - ry), end_point=(x + w - rx, y + h)))
        # 하단 변
        cmds.append(PathCommand(CommandType.LINE, False,
                                [x + rx, y + h],
                                abs_params=[x + rx, y + h],
                                start_point=(x + w - rx, y + h), end_point=(x + rx, y + h)))
        # 좌하 모서리
        cmds.append(PathCommand(CommandType.ARC, False,
                                [rx, ry, 0, 0, 1, x, y + h - ry],
                                abs_params=[rx, ry, 0, 0, 1, x, y + h - ry],
                                start_point=(x + rx, y + h), end_point=(x, y + h - ry)))
        # 좌측 변
        cmds.append(PathCommand(CommandType.LINE, False,
                                [x, y + ry],
                                abs_params=[x, y + ry],
                                start_point=(x, y + h - ry), end_point=(x, y + ry)))
        # 좌상 모서리
        cmds.append(PathCommand(CommandType.ARC, False,
                                [rx, ry, 0, 0, 1, x + rx, y],
                                abs_params=[rx, ry, 0, 0, 1, x + rx, y],
                                start_point=(x, y + ry), end_point=(x + rx, y)))
        cmds.append(PathCommand(CommandType.CLOSE, False, [],
                                abs_params=[], start_point=(x + rx, y), end_point=(x + rx, y)))
        return cmds

    @staticmethod
    def _circle_to_commands(cx, cy, r) -> List[PathCommand]:
        """원 → 4개의 큐빅 베지에 근사. Kappa = 4(√2-1)/3 ≈ 0.5522847."""
        k = 0.5522847498
        cmds = []
        cmds.append(PathCommand(
            CommandType.MOVE, False, [cx + r, cy],
            abs_params=[cx + r, cy],
            start_point=(cx + r, cy), end_point=(cx + r, cy)))
        # 상단 (시계 방향)
        segments = [
            ([cx + r, cy + r * k, cx + r * k, cy + r, cx, cy + r], (cx, cy + r)),
            ([cx - r * k, cy + r, cx - r, cy + r * k, cx - r, cy], (cx - r, cy)),
            ([cx - r, cy - r * k, cx - r * k, cy - r, cx, cy - r], (cx, cy - r)),
            ([cx + r * k, cy - r, cx + r, cy - r * k, cx + r, cy], (cx + r, cy)),
        ]
        prev = (cx + r, cy)
        for params, end in segments:
            cmds.append(PathCommand(
                CommandType.CUBIC, False, params,
                abs_params=params,
                start_point=prev, end_point=end))
            prev = end
        cmds.append(PathCommand(CommandType.CLOSE, False, [],
                                abs_params=[], start_point=prev, end_point=(cx + r, cy)))
        return cmds

    @staticmethod
    def _ellipse_to_commands(cx, cy, rx, ry) -> List[PathCommand]:
        """타원 → 4개의 큐빅 베지에 근사."""
        k = 0.5522847498
        cmds = []
        cmds.append(PathCommand(
            CommandType.MOVE, False, [cx + rx, cy],
            abs_params=[cx + rx, cy],
            start_point=(cx + rx, cy), end_point=(cx + rx, cy)))
        segments = [
            ([cx + rx, cy + ry * k, cx + rx * k, cy + ry, cx, cy + ry], (cx, cy + ry)),
            ([cx - rx * k, cy + ry, cx - rx, cy + ry * k, cx - rx, cy], (cx - rx, cy)),
            ([cx - rx, cy - ry * k, cx - rx * k, cy - ry, cx, cy - ry], (cx, cy - ry)),
            ([cx + rx * k, cy - ry, cx + rx, cy - ry * k, cx + rx, cy], (cx + rx, cy)),
        ]
        prev = (cx + rx, cy)
        for params, end in segments:
            cmds.append(PathCommand(
                CommandType.CUBIC, False, params,
                abs_params=params,
                start_point=prev, end_point=end))
            prev = end
        cmds.append(PathCommand(CommandType.CLOSE, False, [],
                                abs_params=[], start_point=prev, end_point=(cx + rx, cy)))
        return cmds

    @staticmethod
    def _parse_length(s: str) -> float:
        """CSS 길이 값을 float으로 파싱 (단위 무시)."""
        m = re.match(r'([+-]?[\d.]+)', s)
        return float(m.group(1)) if m else 0.0

    @staticmethod
    def _strip_ns(tag: str) -> str:
        """XML namespace 제거."""
        if "}" in tag:
            return tag.split("}")[-1]
        return tag
