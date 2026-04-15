"""
베지에 곡선 수학 유틸리티
========================
큐빅/쿼드라틱 베지에 곡선의 평가, 미분, 곡률 계산 등.

수학적 기반:
    큐빅 베지에: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    곡률: κ(t) = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class Point:
    """2D 점."""
    x: float
    y: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Point":
        return cls(x=float(arr[0]), y=float(arr[1]))

    def distance_to(self, other: "Point") -> float:
        return float(np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))

    def __repr__(self):
        return f"P({self.x:.2f}, {self.y:.2f})"


class BezierMath:
    """베지에 곡선 수학 연산 모음."""

    # --- 곡선 평가 ---

    @staticmethod
    def eval_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                   p3: np.ndarray, t: float) -> np.ndarray:
        """큐빅 베지에 곡선 B(t) 평가."""
        u = 1.0 - t
        return (u**3) * p0 + 3 * (u**2) * t * p1 + 3 * u * (t**2) * p2 + (t**3) * p3

    @staticmethod
    def eval_quadratic(p0: np.ndarray, p1: np.ndarray,
                       p2: np.ndarray, t: float) -> np.ndarray:
        """쿼드라틱 베지에 곡선 B(t) 평가."""
        u = 1.0 - t
        return (u**2) * p0 + 2 * u * t * p1 + (t**2) * p2

    # --- 1차 도함수 ---

    @staticmethod
    def deriv1_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                     p3: np.ndarray, t: float) -> np.ndarray:
        """큐빅 베지에 B'(t) — 1차 도함수 (접선 벡터)."""
        u = 1.0 - t
        return (3 * (u**2) * (p1 - p0) +
                6 * u * t * (p2 - p1) +
                3 * (t**2) * (p3 - p2))

    @staticmethod
    def deriv1_quadratic(p0: np.ndarray, p1: np.ndarray,
                         p2: np.ndarray, t: float) -> np.ndarray:
        """쿼드라틱 베지에 B'(t) — 1차 도함수."""
        u = 1.0 - t
        return 2 * u * (p1 - p0) + 2 * t * (p2 - p1)

    # --- 2차 도함수 ---

    @staticmethod
    def deriv2_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                     p3: np.ndarray, t: float) -> np.ndarray:
        """큐빅 베지에 B''(t) — 2차 도함수."""
        u = 1.0 - t
        return (6 * u * (p2 - 2 * p1 + p0) +
                6 * t * (p3 - 2 * p2 + p1))

    @staticmethod
    def deriv2_quadratic(p0: np.ndarray, p1: np.ndarray,
                         p2: np.ndarray, t: float) -> np.ndarray:
        """쿼드라틱 베지에 B''(t) — 2차 도함수 (상수)."""
        return 2 * (p2 - 2 * p1 + p0)

    # --- 곡률 (Curvature) ---

    @staticmethod
    def curvature_2d(d1: np.ndarray, d2: np.ndarray) -> float:
        """
        2D 곡률 계산.
        κ = (x'·y'' - y'·x'') / (x'² + y'²)^(3/2)

        Args:
            d1: 1차 도함수 벡터 [x', y']
            d2: 2차 도함수 벡터 [x'', y'']
        Returns:
            부호 있는 곡률 (signed curvature). 양수=반시계, 음수=시계 방향.
        """
        cross = d1[0] * d2[1] - d1[1] * d2[0]  # x'y'' - y'x''
        speed_sq = d1[0] ** 2 + d1[1] ** 2
        speed_cubed = speed_sq ** 1.5

        if speed_cubed < 1e-12:
            return 0.0  # 퇴화점 (degenerate point)
        return float(cross / speed_cubed)

    @classmethod
    def curvature_cubic_at(cls, p0: np.ndarray, p1: np.ndarray,
                           p2: np.ndarray, p3: np.ndarray, t: float) -> float:
        """큐빅 베지에 곡선의 파라미터 t에서의 곡률."""
        d1 = cls.deriv1_cubic(p0, p1, p2, p3, t)
        d2 = cls.deriv2_cubic(p0, p1, p2, p3, t)
        return cls.curvature_2d(d1, d2)

    @classmethod
    def curvature_quadratic_at(cls, p0: np.ndarray, p1: np.ndarray,
                               p2: np.ndarray, t: float) -> float:
        """쿼드라틱 베지에 곡선의 파라미터 t에서의 곡률."""
        d1 = cls.deriv1_quadratic(p0, p1, p2, t)
        d2 = cls.deriv2_quadratic(p0, p1, p2, t)
        return cls.curvature_2d(d1, d2)

    # --- 곡률 프로파일 ---

    @classmethod
    def curvature_profile_cubic(cls, p0: np.ndarray, p1: np.ndarray,
                                p2: np.ndarray, p3: np.ndarray,
                                n_samples: int = 20) -> np.ndarray:
        """
        큐빅 베지에 곡선의 곡률 프로파일 샘플링.
        Returns: shape (n_samples,) — 각 t에서의 곡률 값.
        """
        ts = np.linspace(0, 1, n_samples)
        return np.array([cls.curvature_cubic_at(p0, p1, p2, p3, t) for t in ts])

    @classmethod
    def max_abs_curvature_cubic(cls, p0: np.ndarray, p1: np.ndarray,
                                p2: np.ndarray, p3: np.ndarray,
                                n_samples: int = 50) -> float:
        """큐빅 베지에의 최대 절대 곡률."""
        profile = cls.curvature_profile_cubic(p0, p1, p2, p3, n_samples)
        return float(np.max(np.abs(profile)))

    # --- 접선 방향 ---

    @classmethod
    def tangent_angle_cubic(cls, p0: np.ndarray, p1: np.ndarray,
                            p2: np.ndarray, p3: np.ndarray, t: float) -> float:
        """큐빅 베지에 곡선의 파라미터 t에서의 접선 각도 (라디안)."""
        d1 = cls.deriv1_cubic(p0, p1, p2, p3, t)
        if np.linalg.norm(d1) < 1e-12:
            return 0.0
        return float(np.arctan2(d1[1], d1[0]))

    # --- 호 길이 근사 ---

    @classmethod
    def arc_length_cubic(cls, p0: np.ndarray, p1: np.ndarray,
                         p2: np.ndarray, p3: np.ndarray,
                         n_samples: int = 100) -> float:
        """큐빅 베지에 곡선의 호 길이 (수치 적분 근사)."""
        ts = np.linspace(0, 1, n_samples)
        points = np.array([cls.eval_cubic(p0, p1, p2, p3, t) for t in ts])
        diffs = np.diff(points, axis=0)
        lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        return float(np.sum(lengths))

    # --- 바운딩 박스 ---

    @classmethod
    def bounding_box_cubic(cls, p0: np.ndarray, p1: np.ndarray,
                           p2: np.ndarray, p3: np.ndarray,
                           n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        큐빅 베지에의 바운딩 박스.
        Returns: (min_point, max_point) — 각각 [x, y].
        """
        ts = np.linspace(0, 1, n_samples)
        points = np.array([cls.eval_cubic(p0, p1, p2, p3, t) for t in ts])
        return points.min(axis=0), points.max(axis=0)

    # --- G1/G2 연속성 판정 ---

    @classmethod
    def check_g1_continuity(cls, end_tangent: np.ndarray,
                            start_tangent: np.ndarray,
                            angle_threshold: float = 0.05) -> bool:
        """
        G1 연속성 검사: 두 접선 벡터가 같은 방향인지 판정.

        Args:
            end_tangent: 이전 세그먼트 끝점의 접선 벡터
            start_tangent: 다음 세그먼트 시작점의 접선 벡터
            angle_threshold: 허용 각도 오차 (라디안), 기본 ~2.87°
        Returns:
            G1 연속 여부.
        """
        norm1 = np.linalg.norm(end_tangent)
        norm2 = np.linalg.norm(start_tangent)
        if norm1 < 1e-12 or norm2 < 1e-12:
            return True  # 퇴화 케이스 — G1 성립으로 간주
        u1 = end_tangent / norm1
        u2 = start_tangent / norm2
        dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
        angle = np.arccos(dot)
        return angle < angle_threshold

    @classmethod
    def check_g2_continuity(cls, kappa_end: float, kappa_start: float,
                            threshold: float = 0.01) -> bool:
        """
        G2 연속성 검사: 연결점에서의 곡률 일치 여부.

        Args:
            kappa_end: 이전 세그먼트 끝점의 곡률
            kappa_start: 다음 세그먼트 시작점의 곡률
            threshold: 허용 곡률 차이
        Returns:
            G2 연속 여부.
        """
        return abs(kappa_end - kappa_start) < threshold
