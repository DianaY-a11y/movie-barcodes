"""
CLI entry point for movie-barcode.

Usage:
    moviebarcode VIDEO [OPTIONS]

Full pipeline:
  1. Probe video metadata
  2. (Optional) Detect letterbox black bars
  3. Detect shot boundaries
  4. Extract dominant color per shot (K-means or trimmed mean)
  5. Blend shots into time-equal stripes
  6. Render barcode image (+ optional palette)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

app = typer.Typer(
    name="moviebarcode",
    help="Generate a movie barcode from a video file.",
    add_completion=False,
)
console = Console(stderr=True)


@app.command()
def main(
    video: Path = typer.Argument(..., help="Input video file.", exists=True, dir_okay=False),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output image path (PNG/JPEG). Defaults to <video_stem>_barcode.png.",
    ),
    stripes: int = typer.Option(
        500,
        "--stripes", "-n",
        min=1,
        help="Number of vertical color stripes.",
    ),
    stripe_width: int = typer.Option(
        3,
        "--stripe-width", "-w",
        min=1,
        help="Pixel width of each stripe.",
    ),
    height: int = typer.Option(
        400,
        "--height",
        min=1,
        help="Pixel height of the output image.",
    ),
    smooth: int = typer.Option(
        0,
        "--smooth",
        min=0,
        help="Gaussian blur radius applied to the barcode (0 = sharp stripes).",
    ),
    mode: str = typer.Option(
        "natural",
        "--mode",
        help="Color extraction mode: 'natural' (K-means dominant), 'vivid' (most saturated cluster), or 'fast' (trimmed mean).",
    ),
    k: int = typer.Option(
        3,
        "--k",
        min=1,
        help="K-means clusters per shot (natural and vivid modes only).",
    ),
    sample_frames: int = typer.Option(
        5,
        "--sample-frames",
        min=1,
        help="Frames sampled per shot for color extraction.",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        min=1,
        help="Parallel worker processes for color extraction.",
    ),
    sensitivity: str = typer.Option(
        "medium",
        "--sensitivity",
        help="Shot detection sensitivity: 'low', 'medium', or 'high'.",
    ),
    no_letterbox: bool = typer.Option(
        False,
        "--no-letterbox",
        help="Skip letterbox (black bar) detection.",
    ),
    palette: Optional[Path] = typer.Option(
        None,
        "--palette",
        help="Also save a dominant-color palette image to this path.",
    ),
    palette_colors: int = typer.Option(
        16,
        "--palette-colors",
        min=1,
        help="Number of colors in the palette image.",
    ),
    start: float = typer.Option(
        0.0,
        "--start",
        min=0.0,
        help="Start time in seconds (skip opening credits).",
    ),
    end: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds (skip closing credits).",
    ),
) -> None:
    """Generate a movie barcode from VIDEO."""

    if mode not in ("natural", "fast", "vivid"):
        console.print(f"[red]Error:[/red] --mode must be 'natural', 'vivid', or 'fast', got '{mode}'.")
        raise typer.Exit(1)

    if sensitivity not in ("low", "medium", "high"):
        console.print(f"[red]Error:[/red] --sensitivity must be 'low', 'medium', or 'high'.")
        raise typer.Exit(1)

    # Default output path
    if output is None:
        output = video.with_name(video.stem + "_barcode.png")

    # ------------------------------------------------------------------ #
    # Step 1 — probe video metadata                                        #
    # ------------------------------------------------------------------ #
    from moviebarcode.utils.av_helpers import open_video_meta

    console.print(f"\n[bold cyan]movie-barcode[/bold cyan]  {video.name}")
    console.rule()

    try:
        meta = open_video_meta(video)
    except Exception as exc:
        console.print(f"[red]Cannot open video:[/red] {exc}")
        raise typer.Exit(1)

    duration = meta.duration
    t_start = max(0.0, start)
    t_end = min(duration, end) if end is not None else duration
    if t_end <= t_start:
        console.print("[red]Error:[/red] --end must be greater than --start.")
        raise typer.Exit(1)

    span = t_end - t_start
    stripe_duration = span / stripes

    console.print(
        f"  [dim]Duration:[/dim] {duration:.1f}s  "
        f"[dim]FPS:[/dim] {meta.fps:.2f}  "
        f"[dim]Size:[/dim] {meta.width}×{meta.height}  "
        f"[dim]Codec:[/dim] {meta.codec}"
    )
    console.print(
        f"  [dim]Range:[/dim] {t_start:.1f}s – {t_end:.1f}s  "
        f"[dim]Stripes:[/dim] {stripes}  "
        f"[dim]Mode:[/dim] {mode}  "
        f"[dim]Sensitivity:[/dim] {sensitivity}"
    )
    console.print()

    # ------------------------------------------------------------------ #
    # Step 2 — letterbox detection                                         #
    # ------------------------------------------------------------------ #
    crop: Optional[tuple[float, float]] = None

    if not no_letterbox:
        with _spinner("Detecting letterbox…"):
            from moviebarcode.pipeline.letterbox import detect_letterbox
            top_frac, bot_frac = detect_letterbox(video)

        if top_frac > 0.0 or bot_frac < 1.0:
            console.print(
                f"  [green]✓[/green] Letterbox detected: "
                f"top {top_frac*100:.1f}%  bottom {(1-bot_frac)*100:.1f}%"
            )
            crop = (top_frac, bot_frac)
        else:
            console.print("  [green]✓[/green] No letterbox detected.")
    else:
        console.print("  [dim]Letterbox detection skipped.[/dim]")

    # ------------------------------------------------------------------ #
    # Step 3 — shot detection                                              #
    # ------------------------------------------------------------------ #
    console.print()
    console.print("[bold]Detecting shots…[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analyzing frames…", total=100)

        def shot_progress(frac: float) -> None:
            progress.update(task, completed=int(frac * 100))

        from moviebarcode.pipeline.shot_detect import detect_shots
        shots = detect_shots(
            video,
            sensitivity=sensitivity,
            crop=crop,
            start=t_start,
            end=t_end,
            progress_callback=shot_progress,
        )

    console.print(f"  [green]✓[/green] {len(shots)} shot(s) found.")

    # ------------------------------------------------------------------ #
    # Step 4 — color extraction                                            #
    # ------------------------------------------------------------------ #
    console.print()
    console.print("[bold]Extracting colors…[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Processing shots…", total=len(shots))

        def color_progress(done: int, total: int) -> None:
            progress.update(task, completed=done)

        from moviebarcode.pipeline.color_extract import extract_shot_colors
        shots = extract_shot_colors(
            video,
            shots,
            mode=mode,
            crop=crop,
            k=k,
            sample_frames=sample_frames,
            workers=workers,
            progress_callback=color_progress,
        )

    colored = sum(1 for s in shots if s.color_rgb is not None)
    console.print(f"  [green]✓[/green] Colors extracted for {colored}/{len(shots)} shot(s).")

    # ------------------------------------------------------------------ #
    # Step 5 — stripe blending                                             #
    # ------------------------------------------------------------------ #
    with _spinner("Blending stripes…"):
        from moviebarcode.pipeline.blending import compute_stripes
        stripe_list = compute_stripes(
            shots,
            total_duration=duration,
            stripe_duration=stripe_duration,
            start=t_start,
            end=t_end,
        )

    console.print(f"  [green]✓[/green] {len(stripe_list)} stripes computed.")

    # ------------------------------------------------------------------ #
    # Step 6 — render & save                                               #
    # ------------------------------------------------------------------ #
    with _spinner("Rendering barcode…"):
        from moviebarcode.pipeline.render import render_barcode
        img = render_barcode(stripe_list, stripe_width=stripe_width, height=height, smooth=smooth)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == '.webp':
        img.save(output, lossless=True)
    else:
        img.save(output)
    console.print(f"\n  [bold green]Saved:[/bold green] {output}  ({img.width}×{img.height} px)")

    # Write sidecar JSON for per-stripe hover data
    sidecar = output.with_suffix('.json')
    sidecar.write_text(json.dumps([
        {"hex": "#{:02x}{:02x}{:02x}".format(*s.color_rgb.tolist()), "t": round(s.start, 2)}
        for s in stripe_list
    ]))
    console.print(f"  [dim]Colors:[/dim]        {sidecar.name}  ({len(stripe_list)} stripes)")

    if palette is not None:
        with _spinner("Rendering palette…"):
            from moviebarcode.pipeline.render import render_palette
            pal_img = render_palette(stripe_list, n_colors=palette_colors)
        palette.parent.mkdir(parents=True, exist_ok=True)
        if palette.suffix.lower() == '.webp':
            pal_img.save(palette, lossless=True)
        else:
            pal_img.save(palette)
        console.print(f"  [bold green]Palette:[/bold green]  {palette}")

    console.print()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

class _spinner:
    """Context manager that shows a Rich spinner for blocking work."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._progress: Progress | None = None
        self._task_id = None

    def __enter__(self) -> "_spinner":
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(self._label)
        return self

    def __exit__(self, *_) -> None:
        if self._progress is not None:
            self._progress.stop()
