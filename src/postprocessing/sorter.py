"""Reading-order sorting.

Re-orders detected elements into natural reading order (top-to-bottom,
left-to-right), with a heuristic for multi-column layouts: when the page splits
into columns, each column is read top-to-bottom before moving to the next.

Elements are sorted by their relative ``bbox`` ([x1, y1, x2, y2] in 0-1) and the
``reading_order`` field is reassigned from 1.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ReadingOrderSorter:
    """Sort layout elements into reading order and reassign ``reading_order``."""

    def sort(self, elements: list[dict]) -> list[dict]:
        """Return ``elements`` sorted into reading order with fresh indices.

        Args:
            elements: Element dicts each carrying a relative ``bbox``.

        Returns:
            A new sorted list; each element's ``reading_order`` is set from 1.
        """
        if not elements:
            return []

        n_columns = self._detect_columns(elements)
        if n_columns > 1:
            ordered = self._sort_multi_column(elements, n_columns)
        else:
            ordered = self._sort_single_column(elements)

        # Reassign reading_order on shallow copies to avoid mutating inputs.
        result: list[dict] = []
        for i, elem in enumerate(ordered, start=1):
            new_elem = dict(elem)
            new_elem["reading_order"] = i
            result.append(new_elem)
        return result

    # -- sorting strategies ----------------------------------------------- #

    def _sort_single_column(self, elements: list[dict]) -> list[dict]:
        """Group elements into rows (by vertical overlap), order rows top-down."""
        remaining = sorted(elements, key=self._get_center_y)
        rows: list[list[dict]] = []
        for elem in remaining:
            placed = False
            for row in rows:
                if self._is_same_row(elem, row[0]):
                    row.append(elem)
                    placed = True
                    break
            if not placed:
                rows.append([elem])
        # Order rows by their top edge, and within a row left-to-right.
        rows.sort(key=lambda r: min(self._get_top(e) for e in r))
        ordered: list[dict] = []
        for row in rows:
            row.sort(key=self._get_center_x)
            ordered.extend(row)
        return ordered

    def _sort_multi_column(
        self, elements: list[dict], n_columns: int
    ) -> list[dict]:
        """Assign elements to columns by center_x, read each column top-down."""
        # Build evenly-spaced column boundaries across [0, 1].
        buckets: list[list[dict]] = [[] for _ in range(n_columns)]
        for elem in elements:
            cx = self._get_center_x(elem)
            col = min(int(cx * n_columns), n_columns - 1)
            buckets[col].append(elem)

        ordered: list[dict] = []
        for col in buckets:
            ordered.extend(self._sort_single_column(col))
        return ordered

    # -- geometry helpers -------------------------------------------------- #

    def _get_center_y(self, element: dict) -> float:
        """Vertical center of the element's relative bbox."""
        _, y1, _, y2 = self._bbox(element)
        return (y1 + y2) / 2.0

    def _get_center_x(self, element: dict) -> float:
        """Horizontal center of the element's relative bbox."""
        x1, _, x2, _ = self._bbox(element)
        return (x1 + x2) / 2.0

    def _get_top(self, element: dict) -> float:
        """Top edge (y1) of the element's relative bbox."""
        return self._bbox(element)[1]

    def _is_same_row(
        self, elem_a: dict, elem_b: dict, threshold: float = 0.02
    ) -> bool:
        """Return True if two elements substantially overlap vertically.

        Treated as the same row when their vertical center distance is within
        ``threshold`` or their vertical spans overlap.
        """
        _, a_y1, _, a_y2 = self._bbox(elem_a)
        _, b_y1, _, b_y2 = self._bbox(elem_b)
        # Vertical overlap?
        overlap = min(a_y2, b_y2) - max(a_y1, b_y1)
        if overlap > 0:
            return True
        return abs(self._get_center_y(elem_a) - self._get_center_y(elem_b)) <= threshold

    def _detect_columns(self, elements: list[dict]) -> int:
        """Estimate the number of columns from the spread of center_x values.

        Heuristic: if enough elements cluster on the left and right halves with
        few straddling the middle, treat the page as two columns. Otherwise one.
        """
        if len(elements) < 4:
            return 1

        centers = [self._get_center_x(e) for e in elements]
        widths = [self._bbox(e)[2] - self._bbox(e)[0] for e in elements]
        # Full-width elements (e.g. titles, tables) argue against columns.
        wide = sum(1 for w in widths if w > 0.65)
        if wide > len(elements) * 0.3:
            return 1

        left = sum(1 for c in centers if c < 0.45)
        right = sum(1 for c in centers if c > 0.55)
        middle = sum(1 for c in centers if 0.45 <= c <= 0.55)
        if left >= 2 and right >= 2 and middle <= len(elements) * 0.25:
            return 2
        return 1

    @staticmethod
    def _bbox(element: dict) -> list[float]:
        """Return the relative bbox, defaulting to a full-page box if missing."""
        bbox = element.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return [float(v) for v in bbox]
        return [0.0, 0.0, 1.0, 1.0]


def sort_elements(elements: list[dict]) -> list[dict]:
    """Convenience function: sort ``elements`` into reading order."""
    return ReadingOrderSorter().sort(elements)
