"""
SVG Path 명령어 파서
====================
SVG <path d="..."> 속성의 'd' 문자열을 구조화된 PathCommand 리스트로 분해.

지원 명령어:
    M/m (moveto), L/l (lineto), H/h (horizontal), V/v (vertical),
    C/c (cubic bezier), S/s (smooth cubic), Q/q (quadratic bezier),
    T/t (smooth quadratic), A/a (arc), Z/z (closepath)

참고: SVG 1.1 명세 (Section 8.3) 준수.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class CommandType(Enum):
    """SVG path 명령어 유형."""
    MOVE = "M"
    LINE = "L"
    HLINE = "H"
    VLINE = "V"
    CUBIC = "C"
    SMOOTH_CUBIC = "S"
    QUADRATIC = "Q"
    SMOOTH_QUADRATIC = "T"
    ARC = "A"
    CLOSE = "Z"


# 각 명령어가 필요로 하는 좌표 매개변수 수
COMMAND_PARAM_COUNTS = {
    CommandType.MOVE: 2,
    CommandType.LINE: 2,
    CommandType.HLINE: 1,
    CommandType.VLINE: 1,
    CommandType.CUBIC: 6,
    CommandType.SMOOTH_CUBIC: 4,
    CommandType.QUADRATIC: 4,
    CommandType.SMOOTH_QUADRATIC: 2,
    CommandType.ARC: 7,
    CommandType.CLOSE: 0,
}


@dataclass
class PathCommand:
    """
    하나의 SVG path 명령어를 표현.

    Attributes:
        command_type: 명령어 유형 (M, L, C, ...)
        is_relative: 소문자 명령어 여부 (상대 좌표)
        params: 매개변수 리스트 (좌표값들)
        abs_params: 절대 좌표로 변환된 매개변수 (resolve 후 설정)
        start_point: 이 명령어의 시작점 (절대 좌표)
        end_point: 이 명령어의 끝점 (절대 좌표)
    """
    command_type: CommandType
    is_relative: bool
    params: List[float]
    abs_params: Optional[List[float]] = None
    start_point: Optional[Tuple[float, float]] = None
    end_point: Optional[Tuple[float, float]] = None

    def __repr__(self):
        cmd_char = self.command_type.value
        if self.is_relative:
            cmd_char = cmd_char.lower()
        params_str = ", ".join(f"{p:.2f}" for p in self.params[:6])
        if len(self.params) > 6:
            params_str += ", ..."
        return f"PathCmd({cmd_char} [{params_str}])"


class PathParser:
    """
    SVG path 'd' 속성 문자열을 PathCommand 리스트로 파싱.

    사용법:
        parser = PathParser()
        commands = parser.parse("M 10 80 C 40 10, 65 10, 95 80 Z")
        resolved = parser.resolve_to_absolute(commands)
    """

    # 명령어 문자를 CommandType으로 매핑
    _CMD_MAP = {
        'M': CommandType.MOVE, 'm': CommandType.MOVE,
        'L': CommandType.LINE, 'l': CommandType.LINE,
        'H': CommandType.HLINE, 'h': CommandType.HLINE,
        'V': CommandType.VLINE, 'v': CommandType.VLINE,
        'C': CommandType.CUBIC, 'c': CommandType.CUBIC,
        'S': CommandType.SMOOTH_CUBIC, 's': CommandType.SMOOTH_CUBIC,
        'Q': CommandType.QUADRATIC, 'q': CommandType.QUADRATIC,
        'T': CommandType.SMOOTH_QUADRATIC, 't': CommandType.SMOOTH_QUADRATIC,
        'A': CommandType.ARC, 'a': CommandType.ARC,
        'Z': CommandType.CLOSE, 'z': CommandType.CLOSE,
    }

    # 숫자 토큰화 정규식: 부호, 소수점, 지수 표기 지원
    _NUM_RE = re.compile(
        r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
    )

    # 명령어 문자 정규식
    _CMD_RE = re.compile(r'[MmLlHhVvCcSsQqTtAaZz]')

    def parse(self, d_string: str) -> List[PathCommand]:
        """
        SVG path 'd' 문자열을 PathCommand 리스트로 파싱.

        Args:
            d_string: SVG path의 d 속성 문자열
        Returns:
            PathCommand 리스트 (원본 좌표, 상대/절대 그대로)
        """
        if not d_string or not d_string.strip():
            return []

        commands = []
        # 명령어 문자 기준으로 분리
        tokens = self._tokenize(d_string)

        for cmd_char, params in tokens:
            cmd_type = self._CMD_MAP.get(cmd_char)
            if cmd_type is None:
                continue

            is_relative = cmd_char.islower()
            param_count = COMMAND_PARAM_COUNTS[cmd_type]

            if param_count == 0:
                # Z/z — 매개변수 없음
                commands.append(PathCommand(
                    command_type=cmd_type,
                    is_relative=is_relative,
                    params=[]
                ))
            else:
                # 반복 명령어 처리: 매개변수가 param_count의 배수일 때
                # 예: M 10 20 30 40 → M(10,20) L(30,40)
                i = 0
                first_in_group = True
                while i + param_count <= len(params):
                    chunk = params[i:i + param_count]

                    # M/m 뒤의 추가 좌표쌍은 암묵적 L/l
                    if cmd_type == CommandType.MOVE and not first_in_group:
                        actual_type = CommandType.LINE
                    else:
                        actual_type = cmd_type

                    commands.append(PathCommand(
                        command_type=actual_type,
                        is_relative=is_relative,
                        params=chunk
                    ))
                    i += param_count
                    first_in_group = False

        return commands

    def _tokenize(self, d_string: str) -> List[Tuple[str, List[float]]]:
        """d 문자열을 (명령어 문자, 매개변수 리스트) 쌍으로 분리."""
        result = []
        # 명령어 문자 위치 탐색
        cmd_positions = [(m.start(), m.group()) for m in self._CMD_RE.finditer(d_string)]

        for idx, (pos, cmd_char) in enumerate(cmd_positions):
            # 이 명령어와 다음 명령어 사이의 문자열에서 숫자 추출
            if idx + 1 < len(cmd_positions):
                segment = d_string[pos + 1:cmd_positions[idx + 1][0]]
            else:
                segment = d_string[pos + 1:]

            nums = [float(m.group()) for m in self._NUM_RE.finditer(segment)]
            result.append((cmd_char, nums))

        return result

    def resolve_to_absolute(self, commands: List[PathCommand]) -> List[PathCommand]:
        """
        모든 명령어를 절대 좌표로 변환하고, 각 명령어의 시작/끝점을 설정.

        Args:
            commands: parse()로 얻은 PathCommand 리스트
        Returns:
            절대 좌표가 설정된 PathCommand 리스트 (동일 객체 변이)
        """
        cx, cy = 0.0, 0.0  # 현재 점 (current point)
        sx, sy = 0.0, 0.0  # 서브패스 시작점 (subpath start)
        last_ctrl = None     # 마지막 제어점 (S/T 명령어용)

        for cmd in commands:
            cmd.start_point = (cx, cy)
            p = cmd.params

            if cmd.command_type == CommandType.MOVE:
                if cmd.is_relative:
                    ax, ay = cx + p[0], cy + p[1]
                else:
                    ax, ay = p[0], p[1]
                cmd.abs_params = [ax, ay]
                cmd.end_point = (ax, ay)
                cx, cy = ax, ay
                sx, sy = ax, ay
                last_ctrl = None

            elif cmd.command_type == CommandType.LINE:
                if cmd.is_relative:
                    ax, ay = cx + p[0], cy + p[1]
                else:
                    ax, ay = p[0], p[1]
                cmd.abs_params = [ax, ay]
                cmd.end_point = (ax, ay)
                cx, cy = ax, ay
                last_ctrl = None

            elif cmd.command_type == CommandType.HLINE:
                if cmd.is_relative:
                    ax = cx + p[0]
                else:
                    ax = p[0]
                cmd.abs_params = [ax, cy]
                cmd.end_point = (ax, cy)
                cx = ax
                last_ctrl = None

            elif cmd.command_type == CommandType.VLINE:
                if cmd.is_relative:
                    ay = cy + p[0]
                else:
                    ay = p[0]
                cmd.abs_params = [cx, ay]
                cmd.end_point = (cx, ay)
                cy = ay
                last_ctrl = None

            elif cmd.command_type == CommandType.CUBIC:
                if cmd.is_relative:
                    abs_p = [
                        cx + p[0], cy + p[1],  # x1, y1 (첫 번째 제어점)
                        cx + p[2], cy + p[3],  # x2, y2 (두 번째 제어점)
                        cx + p[4], cy + p[5],  # x, y (끝점)
                    ]
                else:
                    abs_p = list(p)
                cmd.abs_params = abs_p
                cmd.end_point = (abs_p[4], abs_p[5])
                last_ctrl = (abs_p[2], abs_p[3])  # 두 번째 제어점
                cx, cy = abs_p[4], abs_p[5]

            elif cmd.command_type == CommandType.SMOOTH_CUBIC:
                # S: 첫 번째 제어점은 이전 C/S의 두 번째 제어점의 반사
                if last_ctrl is not None:
                    x1 = 2 * cx - last_ctrl[0]
                    y1 = 2 * cy - last_ctrl[1]
                else:
                    x1, y1 = cx, cy

                if cmd.is_relative:
                    abs_p = [
                        x1, y1,
                        cx + p[0], cy + p[1],
                        cx + p[2], cy + p[3],
                    ]
                else:
                    abs_p = [x1, y1, p[0], p[1], p[2], p[3]]
                cmd.abs_params = abs_p
                cmd.end_point = (abs_p[4], abs_p[5])
                last_ctrl = (abs_p[2], abs_p[3])
                cx, cy = abs_p[4], abs_p[5]
                # S를 내부적으로 CUBIC으로 승격
                cmd.command_type = CommandType.CUBIC

            elif cmd.command_type == CommandType.QUADRATIC:
                if cmd.is_relative:
                    abs_p = [
                        cx + p[0], cy + p[1],  # 제어점
                        cx + p[2], cy + p[3],  # 끝점
                    ]
                else:
                    abs_p = list(p)
                cmd.abs_params = abs_p
                cmd.end_point = (abs_p[2], abs_p[3])
                last_ctrl = (abs_p[0], abs_p[1])
                cx, cy = abs_p[2], abs_p[3]

            elif cmd.command_type == CommandType.SMOOTH_QUADRATIC:
                if last_ctrl is not None:
                    qx = 2 * cx - last_ctrl[0]
                    qy = 2 * cy - last_ctrl[1]
                else:
                    qx, qy = cx, cy

                if cmd.is_relative:
                    abs_p = [qx, qy, cx + p[0], cy + p[1]]
                else:
                    abs_p = [qx, qy, p[0], p[1]]
                cmd.abs_params = abs_p
                cmd.end_point = (abs_p[2], abs_p[3])
                last_ctrl = (abs_p[0], abs_p[1])
                cx, cy = abs_p[2], abs_p[3]
                cmd.command_type = CommandType.QUADRATIC

            elif cmd.command_type == CommandType.ARC:
                if cmd.is_relative:
                    abs_p = [
                        p[0], p[1],        # rx, ry
                        p[2],              # x-rotation
                        p[3], p[4],        # large-arc-flag, sweep-flag
                        cx + p[5], cy + p[6],  # 끝점
                    ]
                else:
                    abs_p = list(p)
                cmd.abs_params = abs_p
                cmd.end_point = (abs_p[5], abs_p[6])
                cx, cy = abs_p[5], abs_p[6]
                last_ctrl = None

            elif cmd.command_type == CommandType.CLOSE:
                cmd.abs_params = []
                cmd.end_point = (sx, sy)
                cx, cy = sx, sy
                last_ctrl = None

        return commands
