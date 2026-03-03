# movie-barcode

Generate a movie barcode from any video file — a visualization of a film's color palette over time as vertical stripes.

## Installation

```sh
pip install -e .
```

## Usage

```sh
moviebarcode VIDEO [OPTIONS]
```

`VIDEO` is the path to your video file (MP4, MKV, AVI, etc.). It is required.

```sh
moviebarcode movie.mp4
moviebarcode /path/to/film.mkv -o barcode.png
```

## Options

| Option | Default | Description |
|---|---|---|
| `VIDEO` | *(required)* | Path to the input video file |
| `-o, --output PATH` | `<name>_barcode.png` | Output image path (PNG or JPEG) |
| `-n, --stripes INT` | `500` | Number of vertical color stripes |
| `-w, --stripe-width INT` | `3` | Pixel width of each stripe |
| `--height INT` | `400` | Pixel height of the output image |
| `--smooth INT` | `0` | Gaussian blur radius (0 = sharp stripes) |
| `--mode TEXT` | `natural` | Color extraction mode: `natural`, `vivid`, or `fast` (see below) |
| `--k INT` | `3` | K-means clusters per shot (`natural` and `vivid` modes only) |
| `--sample-frames INT` | `5` | Frames sampled per shot for color extraction |
| `--workers INT` | `1` | Parallel worker processes for color extraction |
| `--sensitivity TEXT` | `medium` | Shot detection sensitivity: `low`, `medium`, or `high` |
| `--no-letterbox` | off | Skip automatic letterbox (black bar) detection |
| `--palette PATH` | *(none)* | Also save a dominant-color palette image to this path |
| `--palette-colors INT` | `16` | Number of colors in the palette image |
| `--start FLOAT` | `0.0` | Start time in seconds (e.g. skip opening credits) |
| `--end FLOAT` | *(end of video)* | End time in seconds (e.g. skip closing credits) |
| `--help` | | Show help and exit |

## Examples

**Basic — generate a barcode with defaults:**
```sh
moviebarcode movie.mp4
# saves movie_barcode.png
```

**Custom output path and size:**
```sh
moviebarcode movie.mp4 -o output/barcode.png --stripes 800 --height 600
```

**Smooth gradient look:**
```sh
moviebarcode movie.mp4 --smooth 15
```

**Vivid mode — for dark films with saturated accent colors:**
```sh
moviebarcode movie.mp4 --mode vivid --k 5
```

**Fast mode for long videos:**
```sh
moviebarcode movie.mp4 --mode fast --workers 4
```

**Skip credits (start at 2 min, end at 1h 55 min):**
```sh
moviebarcode movie.mp4 --start 120 --end 6900
```

**Also generate a color palette:**
```sh
moviebarcode movie.mp4 --palette palette.png --palette-colors 12
```

**High sensitivity shot detection (catches soft transitions):**
```sh
moviebarcode movie.mp4 --sensitivity high
```

**All options combined:**
```sh
moviebarcode film.mkv \
  -o barcode.png \
  --stripes 1000 \
  --stripe-width 2 \
  --height 500 \
  --smooth 10 \
  --mode natural \
  --k 5 \
  --sample-frames 8 \
  --workers 4 \
  --sensitivity medium \
  --start 120 \
  --end 7080 \
  --palette palette.png \
  --palette-colors 16
```

## Color Extraction Modes

| Mode | Algorithm | Best for |
|---|---|---|
| `natural` | K-means in Lab space; returns the **largest cluster by pixel count** | Most films — accurately reflects the scene's overall color |
| `vivid` | K-means in Lab space; returns the **most chromatically salient cluster** (`chroma × weight^0.25`) | Dark films with saturated accent colors (e.g. a single red light in a dark scene) — the dominant black is ignored in favour of the most saturated colour present |
| `fast` | Trimmed mean in Lab space; no K-means | Long videos where speed matters; less precise |

`natural` and `vivid` both use K-means and accept `--k`. The difference is only in which cluster is selected: largest (`natural`) vs most saturated (`vivid`).

## How It Works

1. **Probe** — reads video metadata (duration, fps, resolution, codec)
2. **Letterbox detection** — finds and crops horizontal black bars automatically
3. **Shot detection** — identifies cut points using HSV histogram distances with adaptive thresholding
4. **Color extraction** — samples frames per shot and extracts dominant color per chosen mode
5. **Stripe blending** — blends shot colors into time-equal stripes, weighted by overlap duration
6. **Render** — draws vertical stripes and saves the image

## License

MIT
# movie-barcodes
