#!/usr/bin/env python
# coding: utf-8

"""
Plot focused comparison figures for TNC and QHD-SB over plot_iteration = 0..230.

This version intentionally uses Pillow instead of Matplotlib because the
QHD_SB_OJ_GBY conda environment can crash inside Matplotlib's Windows drawing
stack while saving figures.

Inputs:
- output/3bus_tnc_qhdsb_230_start_metrics.csv
- output/3bus_tnc_qhdsb_230_objective_comparison.csv
- output/3bus_tnc_qhdsb_230_iteration_metrics.csv
- output/3bus_objective_comparison.csv

Outputs:
- output/3bus_tnc_qhdsb_230_start_metrics.png
- output/3bus_tnc_qhdsb_230_objective_selection_comparison.png
- output/3bus_tnc_qhdsb_230_iteration_metrics.png
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent
START_CSV = OUT_DIR / "3bus_tnc_qhdsb_230_start_metrics.csv"
OBJECTIVE_CSV = OUT_DIR / "3bus_tnc_qhdsb_230_objective_comparison.csv"
ITERATION_CSV = OUT_DIR / "3bus_tnc_qhdsb_230_iteration_metrics.csv"
REFERENCE_OBJECTIVE_CSV = OUT_DIR / "3bus_objective_comparison.csv"

START_FIG = OUT_DIR / "3bus_tnc_qhdsb_230_start_metrics.png"
OBJECTIVE_FIG = OUT_DIR / "3bus_tnc_qhdsb_230_objective_selection_comparison.png"
ITERATION_FIG = OUT_DIR / "3bus_tnc_qhdsb_230_iteration_metrics.png"

TNC_COLOR = (245, 133, 24)
QHDSB_COLOR = (76, 120, 168)
REFERENCE_COLOR = (100, 100, 100)
GRID_COLOR = (224, 224, 224)
AXIS_COLOR = (55, 55, 55)
TEXT_COLOR = (32, 32, 32)
CONNECTOR_COLOR = (170, 170, 170)
WHITE = (255, 255, 255)

FONT_DIR = Path("C:/Windows/Fonts")


@dataclass
class PlotArea:
    left: int
    top: int
    width: int
    height: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    log_y: bool = False

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def x(self, value: float) -> int:
        frac = (value - self.x_min) / (self.x_max - self.x_min)
        return int(round(self.left + frac * self.width))

    def y(self, value: float) -> int:
        if self.log_y:
            value = max(value, 1e-300)
            lo = math.log10(self.y_min)
            hi = math.log10(self.y_max)
            frac = (math.log10(value) - lo) / (hi - lo)
        else:
            frac = (value - self.y_min) / (self.y_max - self.y_min)
        return int(round(self.bottom - frac * self.height))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def reference_objective() -> float:
    for row in read_csv(REFERENCE_OBJECTIVE_CSV):
        if row.get("method") == "SQP (SLSQP)":
            return float(row["objective"])
    raise RuntimeError(f"SQP reference objective not found in {REFERENCE_OBJECTIVE_CSV}")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        FONT_DIR / ("arialbd.ttf" if bold else "arial.ttf"),
        FONT_DIR / ("calibrib.ttf" if bold else "calibri.ttf"),
        FONT_DIR / "segoeui.ttf",
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TICK = font(34)
FONT_LABEL = font(42)
FONT_LEGEND = font(38)
FONT_NOTE = font(31)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=6)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, fill=TEXT_COLOR) -> None:
    w, h = text_size(draw, text, fnt)
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=fnt, fill=fill)


def draw_right_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, fill=TEXT_COLOR) -> None:
    w, h = text_size(draw, text, fnt)
    draw.text((xy[0] - w, xy[1] - h / 2), text, font=fnt, fill=fill)


def draw_vertical_label(img: Image.Image, center: tuple[int, int], text: str) -> None:
    fnt = FONT_LABEL
    probe = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    probe_draw = ImageDraw.Draw(probe)
    w, h = text_size(probe_draw, text, fnt)
    tmp = Image.new("RGBA", (w + 70, h + 70), (255, 255, 255, 0))
    td = ImageDraw.Draw(tmp)
    td.multiline_text(
        ((tmp.width - w) / 2, (tmp.height - h) / 2),
        text,
        font=fnt,
        fill=TEXT_COLOR + (255,),
        align="center",
        spacing=8,
    )
    rotated = tmp.rotate(90, expand=True)
    img.alpha_composite(rotated, (int(center[0] - rotated.width / 2), int(center[1] - rotated.height / 2)))


def draw_axes(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    area: PlotArea,
    y_ticks: list[float],
    y_tick_labels: list[str],
    x_ticks: list[float],
    x_tick_labels: list[str],
    y_label: str,
    x_label: str | None = None,
) -> None:
    for tick, label in zip(y_ticks, y_tick_labels):
        y = area.y(tick)
        draw.line((area.left, y, area.right, y), fill=GRID_COLOR, width=2)
        draw.line((area.left - 8, y, area.left, y), fill=AXIS_COLOR, width=3)
        draw_right_text(draw, (area.left - 16, y), label, FONT_TICK)

    draw.line((area.left, area.top, area.left, area.bottom), fill=AXIS_COLOR, width=3)
    draw.line((area.left, area.bottom, area.right, area.bottom), fill=AXIS_COLOR, width=3)

    for tick, label in zip(x_ticks, x_tick_labels):
        x = area.x(tick)
        draw.line((x, area.bottom, x, area.bottom + 8), fill=AXIS_COLOR, width=3)
        draw.multiline_text(
            (x, area.bottom + 22),
            label,
            font=FONT_TICK,
            fill=TEXT_COLOR,
            anchor="ma",
            align="center",
            spacing=4,
        )

    draw_vertical_label(img, (area.left - 125, area.top + area.height // 2), y_label)
    if x_label:
        draw_centered_text(draw, (area.left + area.width // 2, area.bottom + 100), x_label, FONT_LABEL)


def draw_polyline(draw: ImageDraw.ImageDraw, area: PlotArea, xs: list[float], ys: list[float], color, width: int) -> None:
    points = [(area.x(x), area.y(y)) for x, y in zip(xs, ys) if math.isfinite(y)]
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_marker(draw: ImageDraw.ImageDraw, x: int, y: int, color, radius: int = 10) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=WHITE, width=3)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    items: list[tuple[str, tuple[int, int, int], str]],
    x: int,
    y: int,
    line_width: int = 6,
) -> None:
    row_h = 54
    max_w = max(text_size(draw, label, FONT_LEGEND)[0] for label, _color, _kind in items)
    box_w = 88 + max_w + 34
    box_h = row_h * len(items) + 22
    draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=8, fill=WHITE, outline=(210, 210, 210), width=2)
    for idx, (label, color, kind) in enumerate(items):
        yy = y + 26 + idx * row_h
        if kind == "point":
            draw_marker(draw, x + 34, yy, color, radius=12)
        elif kind == "dash":
            draw.line((x + 15, yy, x + 60, yy), fill=color, width=4)
            for sx in range(x + 15, x + 60, 18):
                draw.line((sx, yy, sx + 9, yy), fill=WHITE, width=5)
        else:
            draw.line((x + 15, yy, x + 60, yy), fill=color, width=line_width)
        draw.text((x + 80, yy - 23), label, font=FONT_LEGEND, fill=TEXT_COLOR)


def draw_reference_line(draw: ImageDraw.ImageDraw, area: PlotArea, y_value: float, dashed: bool = True) -> None:
    y = area.y(y_value)
    if dashed:
        x = area.left
        while x < area.right:
            draw.line((x, y, min(x + 18, area.right), y), fill=REFERENCE_COLOR, width=4)
            x += 32
    else:
        draw.line((area.left, y, area.right, y), fill=REFERENCE_COLOR, width=3)


def fmt_sci(value: float) -> str:
    return f"{value:.2e}"


def plot_start_metrics() -> None:
    rows = read_csv(START_CSV)
    specs = [
        (
            "objective",
            "Normalized\nmonetary units (NMU)",
            False,
            0.0,
            0.62,
            [0.0, 0.2, 0.4, 0.6],
            ["0.0", "0.2", "0.4", "0.6"],
        ),
        ("l2_h", "L2 equality residual", True, 8e-2, 1.2, [1.0, 1e-1], ["1e0", "1e-1"]),
        ("max_abs_h", "Max abs residual", True, 8e-2, 1.2, [1.0, 1e-1], ["1e0", "1e-1"]),
        ("load_supplied_pct", "Load supplied (%)", False, 0.0, 110.0, [0, 50, 100], ["0", "50", "100"]),
    ]
    img = Image.new("RGBA", (2800, 760), WHITE + (255,))
    draw = ImageDraw.Draw(img)
    panel_w = 520
    gap = 115
    top = 55
    height = 460
    left0 = 185
    labels = ["TNC", "QHD-SB"]
    colors = [TNC_COLOR, QHDSB_COLOR]

    for idx, (key, y_label, log_y, y_min, y_max, y_ticks, y_tick_labels) in enumerate(specs):
        area = PlotArea(left0 + idx * (panel_w + gap), top, panel_w, height, -0.4, 1.4, y_min, y_max, log_y)
        draw_axes(img, draw, area, y_ticks, y_tick_labels, [0, 1], labels, y_label)
        baseline = y_min if log_y else 0.0
        for xpos, row, color in zip([0, 1], rows, colors):
            value = as_float(row, key)
            x = area.x(xpos)
            draw.line((x, area.y(baseline), x, area.y(value)), fill=color, width=8)
            draw_marker(draw, x, area.y(value), color, radius=13)
            text = fmt_sci(value) if log_y else f"{value:.3f}" if key != "load_supplied_pct" else f"{value:.1f}"
            draw_centered_text(draw, (x, area.y(value) - 36), text, FONT_NOTE)

    img.convert("RGB").save(START_FIG)


def plot_objective_selection_comparison(reference_obj: float) -> None:
    rows = [
        row
        for row in read_csv(OBJECTIVE_CSV)
        if row["selection"]
        in {
            "first_l2_le_1e-3",
            "best_l2_residual",
            "final_plot_iteration_230",
            "min_objective_with_l2_le_1e-3",
        }
    ]
    x_labels = ["First L2<=1e-3", "Best L2", "Final @230", "Min obj\nwith L2<=1e-3"]
    img = Image.new("RGBA", (2400, 1500), WHITE + (255,))
    draw = ImageDraw.Draw(img)

    area_obj = PlotArea(230, 70, 1970, 540, -0.5, 3.5, 0.520, 0.550, False)
    area_res = PlotArea(230, 805, 1970, 430, -0.5, 3.5, 4.8e-4, 1.08e-3, True)
    x_ticks = list(range(4))

    draw_axes(
        img,
        draw,
        area_obj,
        [0.520, 0.525, 0.530, 0.535, 0.540, 0.545, 0.550],
        ["0.520", "0.525", "0.530", "0.535", "0.540", "0.545", "0.550"],
        x_ticks,
        ["", "", "", ""],
        "Normalized\nmonetary units (NMU)",
    )
    draw_reference_line(draw, area_obj, reference_obj)
    draw_legend(
        draw,
        [
            ("Truncated Newton", TNC_COLOR, "point"),
            ("QHD-SB coarse-only", QHDSB_COLOR, "point"),
            ("Optimum reference", REFERENCE_COLOR, "dash"),
        ],
        1580,
        95,
    )

    draw_axes(
        img,
        draw,
        area_res,
        [5e-4, 7e-4, 1e-3],
        ["5e-4", "7e-4", "1e-3"],
        x_ticks,
        x_labels,
        "L2 equality residual",
    )
    draw_reference_line(draw, area_res, 1e-3)

    for idx, row in enumerate(rows):
        tnc_obj = as_float(row, "tnc_objective")
        qhd_obj = as_float(row, "qhdsb_objective")
        tnc_l2 = as_float(row, "tnc_l2_h")
        qhd_l2 = as_float(row, "qhdsb_l2_h")
        x_tnc = idx - 0.18
        x_qhd = idx + 0.18
        draw.line((area_obj.x(idx), area_obj.y(min(tnc_obj, qhd_obj)), area_obj.x(idx), area_obj.y(max(tnc_obj, qhd_obj))), fill=CONNECTOR_COLOR, width=3)
        draw.line((area_res.x(idx), area_res.y(min(tnc_l2, qhd_l2)), area_res.x(idx), area_res.y(max(tnc_l2, qhd_l2))), fill=CONNECTOR_COLOR, width=3)
        draw_marker(draw, area_obj.x(x_tnc), area_obj.y(tnc_obj), TNC_COLOR, radius=13)
        draw_marker(draw, area_obj.x(x_qhd), area_obj.y(qhd_obj), QHDSB_COLOR, radius=13)
        draw_marker(draw, area_res.x(x_tnc), area_res.y(tnc_l2), TNC_COLOR, radius=13)
        draw_marker(draw, area_res.x(x_qhd), area_res.y(qhd_l2), QHDSB_COLOR, radius=13)

    img.convert("RGB").save(OBJECTIVE_FIG)


def plot_iteration_metrics(reference_obj: float) -> None:
    rows = read_csv(ITERATION_CSV)
    xs = [int(row["plot_iteration"]) for row in rows]
    tnc_l2 = [as_float(row, "tnc_l2_h") for row in rows]
    qhd_l2 = [as_float(row, "qhdsb_l2_h") for row in rows]
    tnc_max = [as_float(row, "tnc_max_abs_h") for row in rows]
    qhd_max = [as_float(row, "qhdsb_max_abs_h") for row in rows]
    tnc_obj = [as_float(row, "tnc_objective") for row in rows]
    qhd_obj = [as_float(row, "qhdsb_objective") for row in rows]

    img = Image.new("RGBA", (2400, 2050), WHITE + (255,))
    draw = ImageDraw.Draw(img)
    x_ticks = [0, 50, 100, 150, 200]
    x_tick_labels = ["0", "50", "100", "150", "200"]
    left = 245
    width = 1980
    height = 460
    gap = 145

    area_l2 = PlotArea(left, 70, width, height, 0, 230, 3e-4, 3.0, True)
    area_max = PlotArea(left, 70 + height + gap, width, height, 0, 230, 8e-5, 1.2, True)
    area_obj = PlotArea(left, 70 + 2 * (height + gap), width, height, 0, 230, -0.02, 0.60, False)

    draw_axes(
        img,
        draw,
        area_l2,
        [1.0, 1e-1, 1e-2, 1e-3],
        ["1e0", "1e-1", "1e-2", "1e-3"],
        x_ticks,
        ["", "", "", "", ""],
        "L2 equality residual",
    )
    draw_reference_line(draw, area_l2, 1e-3)
    draw_polyline(draw, area_l2, xs, tnc_l2, TNC_COLOR, 6)
    draw_polyline(draw, area_l2, xs, qhd_l2, QHDSB_COLOR, 6)
    draw_legend(
        draw,
        [
            ("Truncated Newton", TNC_COLOR, "line"),
            ("QHD-SB coarse-only", QHDSB_COLOR, "line"),
            ("1e-3 residual", REFERENCE_COLOR, "dash"),
        ],
        1570,
        95,
    )

    draw_axes(
        img,
        draw,
        area_max,
        [1.0, 1e-1, 1e-2, 1e-3, 1e-4],
        ["1e0", "1e-1", "1e-2", "1e-3", "1e-4"],
        x_ticks,
        ["", "", "", "", ""],
        "Max abs residual",
    )
    draw_polyline(draw, area_max, xs, tnc_max, TNC_COLOR, 6)
    draw_polyline(draw, area_max, xs, qhd_max, QHDSB_COLOR, 6)
    draw_legend(
        draw,
        [
            ("Truncated Newton", TNC_COLOR, "line"),
            ("QHD-SB coarse-only", QHDSB_COLOR, "line"),
        ],
        1570,
        70 + height + gap + 25,
    )

    draw_axes(
        img,
        draw,
        area_obj,
        [0.0, 0.2, 0.4, 0.6],
        ["0.0", "0.2", "0.4", "0.6"],
        x_ticks,
        x_tick_labels,
        "Normalized\nmonetary units (NMU)",
        x_label="Iteration",
    )
    draw_reference_line(draw, area_obj, reference_obj)
    draw_polyline(draw, area_obj, xs, tnc_obj, TNC_COLOR, 6)
    draw_polyline(draw, area_obj, xs, qhd_obj, QHDSB_COLOR, 6)
    draw_legend(
        draw,
        [
            ("Truncated Newton", TNC_COLOR, "line"),
            ("QHD-SB coarse-only", QHDSB_COLOR, "line"),
            ("Optimum reference", REFERENCE_COLOR, "dash"),
        ],
        1570,
        70 + 2 * (height + gap) + 250,
    )

    img.convert("RGB").save(ITERATION_FIG)


def main() -> None:
    ref_obj = reference_objective()
    plot_start_metrics()
    plot_objective_selection_comparison(ref_obj)
    plot_iteration_metrics(ref_obj)
    print(f"Wrote {START_FIG}")
    print(f"Wrote {OBJECTIVE_FIG}")
    print(f"Wrote {ITERATION_FIG}")


if __name__ == "__main__":
    main()
