# mb-featuretrack

Annotate video features with descriptive, tracking overlays.

Point at a feature in a video, give it a label, and `mb-featuretrack` tracks that feature
for the rest of the shot — the circle follows it, the label stays put, and the connecting
line stretches between them. Export the result as a new video with the overlays burned in
and the original audio intact.

![example](assets/mb-featuretrack-example-01.gif)

## Features

- **Drag-to-annotate authoring** — mark a feature, drag to where the label should sit, type
  the text in-window. No config files to hand-write (though the output is plain JSON you
  *can* hand-edit).
- **Automatic tracking** — each annotation seeds an OpenCV CSRT tracker that follows its
  feature frame to frame. Position smoothing keeps the circle from jittering.
- **Fixed labels, moving features** — the label is anchored to an absolute position so it
  never wanders; only the circle tracks, and the connector rubber-bands between them.
- **Reveal animation** — annotations draw themselves in: circle → line → underline → text
  (typewriter), over a configurable number of frames.
- **Per-annotation styling** — color, text color, radius, font size, and which box edge
  the connector attaches to, all editable per feature.
- **Start/stop control** — each annotation appears at the frame you created it and can be
  told to stop drawing at any later frame.
- **Real fonts** — label text renders through a TTF (bundled: Varela Round) via
  `cv2.freetype`, not OpenCV's built-in vector font.
- **Video export** — burn the overlays into `<video>.annotated.mp4` at full resolution,
  with the source audio muxed back in.

## Requirements

- Python 3
- **OpenCV** with the `freetype` module (`cv2.freetype`) — used for TTF label text.
  If it's unavailable the tool still runs, falling back to OpenCV's built-in Hershey font.
- **ffmpeg** — only for `--mode render`, to encode the video and copy the source audio.
  Without it, export falls back to an OpenCV writer that produces **video with no audio**.

No build step and no dependency manifest — it's a single script.

## Usage

```bash
# edit mode (default): navigate, annotate, track
python3 mb-featuretrack.py my-video.mp4

# render mode: export my-video.annotated.mp4 with overlays + audio (headless, no window)
python3 mb-featuretrack.py my-video.mp4 --mode render
```

Annotations are saved to a sidecar JSON next to the video — `my-video.mp4` →
`my-video.annotations.json` (override with `--json`). It saves on quit (and on `s`), and
loads automatically on the next run.

### Two modes

| Mode | What it does |
|------|--------------|
| `edit` (default) | Full authoring UI: seek bar, status line, annotation ids, mouse + keyboard editing. |
| `render` | Headless export. Burns the saved overlays into `<video>.annotated.mp4` at full resolution with the original audio. No window, no editing, no chrome. |

## Controls

### Navigation

| Key | Action |
|-----|--------|
| `SPACE` / `p` | pause / resume |
| `d` / `→` | step forward one frame (pauses) |
| `a` / `←` | step back one frame (pauses) |
| drag **Frame** slider | seek anywhere (keeps playing if it was) |
| `q` / `ESC` | quit (auto-saves) |

### Creating an annotation

1. Pause on a frame where the feature is clearly visible.
2. Press **`f`** to arm feature placement (press `f` again to cancel). Until armed, clicks
   do nothing — this keeps stray clicks from creating features.
3. **Left-drag** from the feature to where you want the label.
4. **Type the label** in the window, then `Enter` to confirm (`Esc` cancels, `Backspace` edits).
5. Play forward — the tracker follows the feature and records its path.

### Editing an annotation

Each annotation shows its **id** next to its circle (edit mode only). Type the id, then
press a key:

| Key | Action |
|-----|--------|
| `e` | stop this track at the current frame (press `e` again at that same frame to clear) |
| `r` | remove the stop (draw to the end) |
| `t` | toggle the text label (off → circle only) |
| `l` | toggle the underline |
| `b` | toggle the rounded text box |
| `c` | cycle where the line connects: bottom → top → left → right |
| `Backspace` | clear the typed id |

Other keys: `u` undo the most recent annotation, `s` save now.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {edit,render}` | `edit` | Authoring UI, or headless export with audio. |
| `--json PATH` | `<video>.annotations.json` | Annotations file. |
| `--box N` | `48` | Size (px) of the tracker box seeded around the clicked feature. |
| `--radius N` | `12` | Circle radius in px (display only, independent of `--box`). |
| `--color R G B` | `255 255 255` | Default circle/line/box color for new annotations. |
| `--text-color R G B` | `100 161 157` | Default label text color (teal `0x64A19D`). |
| `--font-scale F` | `1.2` | Default text size for new annotations. |
| `--font PATH` | bundled Varela Round | TTF for label text. |
| `--line-side {bottom,top,left,right}` | `bottom` | Which box edge the connector attaches to for new annotations. |
| `--smooth N` | `2` | Steady the circle by averaging position over ±N frames (`0` = off). |
| `--reveal N` | `12` | Frames over which an annotation reveals (`0` = instant). |
| `--max-width N` | `1280` | Downscale the *display* only; all coordinates stay full-resolution. |

`--color`, `--text-color`, `--radius`, `--font-scale`, and `--line-side` only seed **new**
annotations — they never override values already saved in the JSON.

## The annotations file

Plain JSON, safe to hand-edit. Each annotation stores its seed, its display specs, and the
recorded per-frame track:

```jsonc
{
  "video": "my-video.mp4",
  "box": 48,
  "annotations": [
    {
      "id": 1,
      "text": "PSU (Seasonic Prime PX-1600)",
      "start_frame": 188,            // first frame it appears
      "end_frame": 864,              // stop drawing after this frame (null = never)
      "color": [255, 255, 255],      // RGB - circle, line, box
      "text_color": [100, 161, 157], // RGB - label text only
      "radius": 12,                  // circle radius (px)
      "font_scale": 1.2,             // text size
      "show_text": true,             // false -> circle only
      "show_underline": true,
      "show_box": true,              // rounded box behind the text
      "line_side": "top",            // bottom | top | left | right
      "text_anchor": [1516, 537],    // FIXED label position (frame coords)
      "seed_bbox": [1586, 394, 48, 48],
      "track": {                     // recorded tracker output, per frame
        "188": [1610.0, 418.0, 48.0, 48.0],
        "189": [1610.0, 418.0, 46.0, 46.0]
      }
    }
  ]
}
```

Because the track is recorded to disk, rendering is deterministic and fully seekable — no
tracker has to re-run at export time. Every display spec has a default, so older files
missing a field still load.

To restyle a feature, edit its values and re-run — for example set `"color": [255, 0, 0]`
for a red marker, or `"show_text": false` to leave just a tracked circle.

## Tips

- **Tracking drifts or is lost?** A lost tracker freezes at its last position and turns
  gray. Undo (`u`) and re-place the feature on a frame where it's crisp, or try a larger
  `--box` for low-contrast features.
- **Circle looks jittery?** Raise `--smooth`. It's applied at render time only, so the
  recorded track is untouched and you can retune freely.
- **Label collides with the subject?** Cycle the connector side with `c`, or edit
  `text_anchor` in the JSON.
- **Export has no audio?** ffmpeg wasn't found — install it and re-run `--mode render`.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
The bundled **Varela Round** font is under the SIL Open Font License; see
[fonts/OFL.txt](fonts/OFL.txt).
