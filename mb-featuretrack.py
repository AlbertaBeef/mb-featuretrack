#!/usr/bin/env python3
"""mb-featuretrack — navigate a video and annotate tracked features.

Two major modes (--mode):
  edit    (default) full navigator + annotation authoring + tracking, with the
          seek trackbar, status bar and annotation ids on screen.
  render  headless export: burns the feature-track overlays into a new video
          (with the source audio, via ffmpeg), named
          <video-without-ext>.annotated.mp4. No window, no editing.

Video navigator (built on the proven OpenCV navigation pattern) plus mouse-driven
feature annotation:

  * click on a feature   -> starts an annotation (auto-pauses); this is the circle
  * drag                 -> positions the text label, with a connecting line preview
  * release              -> type the label text in-window (Enter confirm / Esc cancel)
  * subsequent frames    -> a per-annotation tracker follows the feature; the circle
                            moves with it and the connecting line stretches, while the
                            text label stays fixed at where it was placed

Annotations are saved to a sidecar `<video-without-ext>.annotations.json` (seed + recorded
per-frame track) so playback is deterministic and fully seekable.

Navigation controls:
  SPACE / p        toggle pause / resume
  d / RIGHT arrow  step forward one frame (pauses)
  a / LEFT arrow   step back one frame (pauses)
  drag "Frame"     seek anywhere (keeps playing if it was)
  q / ESC          quit (auto-saves)

Annotation controls:
  f                arm feature placement (press again to cancel); the NEXT left-drag
                   creates an annotation. Clicks do nothing until armed, so stray
                   clicks can't create features.
  left-drag        (once armed) feature -> text position, then type the label
  digits + e       stop a track: type an annotation's id (shown by each circle),
                   then 'e' to stop it at the current frame (press 'e' again at
                   that same frame to clear); Backspace clears the typed id
  digits + r       remove a track's stop (type its id, then 'r')
  digits + t       toggle a track's text label on/off (off -> circle only)
  digits + l       toggle a track's underline on/off
  digits + b       toggle a track's rounded text box on/off
  digits + c       cycle where the line connects (box bottom/top/left/right)
  u                undo (remove the most recent annotation)
  s                save annotations to JSON now
  Enter/Esc/Bksp   confirm / cancel / edit while typing a label

Each annotation stores its own display specs in the JSON, editable per feature:
color ([R,G,B], default white; circle/line/box), text_color ([R,G,B], default
teal 0x64A19D), radius (px), font_scale, show_text / show_underline / show_box
(bools). New annotations take their defaults from --color / --text-color /
--radius / --font-scale.

On appearance each annotation reveals in stages (circle -> line -> underline ->
text) over --reveal frames. Annotation ids are shown next to each circle in edit
mode only (never in render mode).
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

WINDOW = "mb-featuretrack"

MODE_NAV = "nav"
MODE_TEXT = "text"

# Connector attaches to this box edge; label sits on the opposite side of the feature.
LINE_SIDES = ("bottom", "top", "left", "right")

COLOR_LOST = (150, 150, 150)  # gray (BGR) — tracking lost (edit mode status cue)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Per-annotation spec defaults (stored in JSON, editable per feature).
DEFAULT_COLOR = [255, 255, 255]  # RGB, white — circle, line, box
DEFAULT_TEXT_COLOR = [0x64, 0xA1, 0x9D]  # RGB teal (0x64A19D) — label text
DEFAULT_FONT_SCALE = 1.2         # font size multiplier (per annotation)
BASE_FONT_PX = 32                # pixel height at font_scale=1.0 for a TTF (cv2.freetype)
# Reference glyphs for fixed font metrics: caps + ascenders (top) and descenders
# (bottom). Measured once per size so box padding / baseline don't shift with the
# actual glyphs in a label (a 'p'/'q'/'(' must not move the text). No brackets here
# on purpose — parens overshoot the cap/descender lines and just hang into the margin.
METRIC_REF = "AbdfhklMgjpqy"
DEFAULT_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fonts", "VarelaRound-Regular.ttf")


def _bgr(rgb):
    """Convert a JSON [R, G, B] color to an OpenCV (B, G, R) tuple."""
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

# Reveal-animation phase boundaries, as fractions of the reveal progress (0..1):
# circle is drawn immediately, then the line grows, then the underline, then text.
R_LINE = 0.15    # line starts growing after this
R_UNDER = 0.60   # underline starts growing after this
R_TEXT = 0.75    # text starts typing in after this


def _seg(p, a, b):
    """Fraction (0..1) of progress `p` through the sub-interval [a, b]."""
    if p <= a:
        return 0.0
    if p >= b:
        return 1.0
    return (p - a) / (b - a)


def make_tracker():
    """Create the best available OpenCV tracker (CSRT > KCF > MIL)."""
    for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        ctor = getattr(cv2, name, None)
        if ctor is not None:
            return ctor()
    raise RuntimeError("No OpenCV tracker available (need CSRT, KCF or MIL).")


def position_at(ann, f):
    """Position [cx, cy, w, h] of an annotation at frame f, or None if not yet visible.

    Uses the exact recorded position when present, otherwise holds the last
    recorded position at or before f. Returns None when f precedes the annotation.
    """
    track = ann["track"]
    key = str(f)
    if key in track:
        return track[key]
    prev = [k for k in (int(t) for t in track) if k <= f]
    if not prev:
        return None
    return track[str(max(prev))]


class App:
    def __init__(self, video_path, json_path, box, radius, max_width, render, smooth,
                 reveal, color, font_scale, font, text_color, line_side):
        self.video_path = video_path
        self.json_path = json_path
        self.box = box                  # tracker seed box size (feature detection)
        self.radius = max(1, radius)    # default circle radius for new annotations
        self.default_color = list(color)  # default RGB color (circle/line/box)
        self.default_text_color = list(text_color)  # default RGB text color
        self.default_line_side = line_side  # default connect side for new annotations
        self.font_scale = font_scale    # default font scale for new annotations
        self._load_font(font)           # sets self.ft (cv2.freetype) or None (Hershey fallback)
        self.render = render            # render mode: overlays only, no chrome, no editing
        self.show_chrome = not render   # status bar + seek trackbar shown only when editing
        self.smooth = smooth       # half-window (frames) for display smoothing; 0 = off
        self.reveal = reveal       # frames over which an annotation reveals; 0 = instant

        self.source = cv2.VideoCapture(video_path)
        if not self.source.isOpened():
            raise SystemExit(f"Error: could not open video: {video_path}")

        self.fps = self.source.get(cv2.CAP_PROP_FPS)
        if self.fps <= 1.0:
            self.fps = 30.0
        self.total_frames = int(self.source.get(cv2.CAP_PROP_FRAME_COUNT))
        self.is_seekable = self.total_frames > 0
        width = self.source.get(cv2.CAP_PROP_FRAME_WIDTH) or max_width
        self.display_scale = min(1.0, max_width / width) if width else 1.0

        self.frame = None          # raw BGR frame currently displayed
        self.frame_idx = -1        # 0-based index of the displayed frame
        self.paused = False
        self.mode = MODE_NAV
        self.pending = None        # {"feature": (x,y), "text": (x,y)} while creating
        self.placing = False       # left button held, dragging text position
        self.armed = False         # 'f' arms the next drag to create an annotation
        self.text_buffer = ""
        self.sel_buffer = ""       # digits typed to select an annotation by id
        self.seek_target = None    # pending frame to seek to (trackbar-independent)
        self.annotations = []
        self.dirty = False

        self.load()

    # ---- persistence -------------------------------------------------------

    def load(self):
        if not os.path.exists(self.json_path):
            return
        try:
            with open(self.json_path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not read {self.json_path}: {exc}", file=sys.stderr)
            return
        for a in data.get("annotations", []):
            if "text_anchor" not in a and "text_offset" in a:   # migrate to a fixed anchor
                a["text_anchor"] = list(self.text_anchor(a))
                a.pop("text_offset", None)
            a.setdefault("_tracker", None)
            a.setdefault("_tracker_frame", None)
            a.setdefault("_lost", False)
            self.annotations.append(a)
        print(f"Loaded {len(self.annotations)} annotation(s) from {self.json_path}")

    def save(self):
        out = {
            "video": os.path.basename(self.video_path),
            "box": self.box,
            "annotations": [
                {k: v for k, v in a.items() if not k.startswith("_")}
                for a in self.annotations
            ],
        }
        with open(self.json_path, "w") as fh:
            json.dump(out, fh, indent=2)
        self.dirty = False
        print(f"Saved {len(self.annotations)} annotation(s) to {self.json_path}")

    # ---- mouse -------------------------------------------------------------

    def on_mouse(self, event, x, y, flags, param):
        if self.render or self.mode != MODE_NAV or self.frame is None:
            return
        # Map display-space coords back to full-resolution frame coords.
        fx = int(round(x / self.display_scale))
        fy = int(round(y / self.display_scale))
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.armed:                     # press 'f' first — stray clicks do nothing
                return
            self.paused = True                     # freeze the feature while placing
            self.pending = {"feature": (fx, fy), "text": (fx, fy)}
            self.placing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.placing:
            self.pending["text"] = (fx, fy)
        elif event == cv2.EVENT_LBUTTONUP and self.placing:
            self.pending["text"] = (fx, fy)
            self.placing = False
            self.mode = MODE_TEXT                  # release -> query text from user
            self.text_buffer = ""

    def select_digit(self, d):
        """Add a digit to the id selection, keeping multi-digit ids typeable without
        letting the buffer run away: the digit is appended only while the result is
        still a prefix of some real id, otherwise it starts a fresh selection. So with
        ids 1-5, '1' then '5' selects 5 (not 15); with an id 12, '1' then '2' selects 12.
        """
        cand = self.sel_buffer + d
        self.sel_buffer = cand if any(str(a["id"]).startswith(cand)
                                      for a in self.annotations) else d

    def selected_ann(self):
        """The annotation whose id matches the current selection buffer, or None."""
        if not self.sel_buffer.isdigit():
            return None
        sid = int(self.sel_buffer)
        return next((a for a in self.annotations if a["id"] == sid), None)

    def set_end_selected(self):
        """Set (or toggle off, at the same frame) the stop frame of the selected
        annotation at the current frame. Select first by typing its id number."""
        if not self.sel_buffer:
            print("Type an annotation id first, then 'e' to stop it here.")
            return
        ann = self.selected_ann()
        if ann is None:
            print(f"No annotation #{self.sel_buffer}.")
        elif ann.get("end_frame") == self.frame_idx:
            ann["end_frame"] = None
            print(f"Cleared stop for #{ann['id']} ({ann['text']!r})")
            self.dirty = True
        else:
            ann["end_frame"] = self.frame_idx
            print(f"#{ann['id']} ({ann['text']!r}) stops at frame {self.frame_idx}")
            self.dirty = True
        self.sel_buffer = ""

    def clear_end_selected(self):
        """Remove the stop frame of the selected annotation (drawn to the end)."""
        ann = self.selected_ann()
        if ann is None:
            print(f"No annotation #{self.sel_buffer}." if self.sel_buffer
                  else "Type an annotation id first, then 'r' to clear its stop.")
        else:
            ann["end_frame"] = None
            print(f"Cleared stop for #{ann['id']} ({ann['text']!r})")
            self.dirty = True
        self.sel_buffer = ""

    def finalize_annotation(self):
        text = self.text_buffer.strip()
        if not text or self.pending is None:
            self.cancel_pending()
            return
        fx, fy = self.pending["feature"]
        tx, ty = self.pending["text"]
        b = self.box
        ann = {
            "id": max((a["id"] for a in self.annotations), default=0) + 1,
            "text": text,
            "start_frame": self.frame_idx,
            "end_frame": None,        # frame after which the track stops drawing
            "color": list(self.default_color),  # [R, G, B]
            "radius": self.radius,              # circle radius (px)
            "font_scale": self.font_scale,      # text size (cv2 font scale)
            "show_text": True,                  # draw the label; off -> circle only
            "show_underline": True,             # draw the underline under the text
            "show_box": True,                   # rounded box (in `color`) behind the text
            "line_side": self.default_line_side,  # "bottom" (label above) | "top" (label below)
            "text_color": list(self.default_text_color),  # [R, G, B] (text only)
            "text_anchor": [tx, ty],            # FIXED label position (frame coords)
            "seed_bbox": [fx - b // 2, fy - b // 2, b, b],
            "track": {str(self.frame_idx): [float(fx), float(fy), float(b), float(b)]},
            "_tracker": make_tracker(),
            "_tracker_frame": self.frame_idx,
            "_lost": False,
        }
        ann["_tracker"].init(self.frame, (fx - b // 2, fy - b // 2, b, b))
        self.annotations.append(ann)
        self.dirty = True
        self.cancel_pending()

    def cancel_pending(self):
        self.pending = None
        self.placing = False
        self.mode = MODE_NAV
        self.text_buffer = ""
        self.armed = False          # one annotation per 'f' — re-arm for the next

    # ---- tracking ----------------------------------------------------------

    def advance_trackers(self, f):
        """Extend trackers by one frame when f is exactly the next frame."""
        if self.render or self.frame is None:
            return
        for ann in self.annotations:
            tr = ann["_tracker"]
            tf = ann["_tracker_frame"]
            if tr is None and not ann["_lost"]:
                # Lazily re-acquire a lost-free annotation at its high-water mark,
                # so a loaded annotation can be extended by playing past its end.
                hw = max(int(k) for k in ann["track"])
                if f == hw:
                    cx, cy, w, h = ann["track"][str(hw)]
                    bbox = (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
                    tr = make_tracker()
                    tr.init(self.frame, bbox)
                    ann["_tracker"], ann["_tracker_frame"], tf = tr, hw, hw
            if tr is not None and tf is not None and f == tf + 1:
                ok, bbox = tr.update(self.frame)
                if ok:
                    x, y, w, h = bbox
                    ann["track"][str(f)] = [x + w / 2, y + h / 2, float(w), float(h)]
                    ann["_tracker_frame"] = f
                    self.dirty = True
                else:
                    ann["_tracker"] = None            # lost — stop, hold last position
                    ann["_lost"] = True

    # ---- rendering ---------------------------------------------------------

    def _load_font(self, font_path):
        """Load a TTF via cv2.freetype for label text; fall back to a Hershey font."""
        self.ft = None
        self.font_path = font_path
        if not font_path:
            return
        if not hasattr(cv2, "freetype"):
            print("Warning: cv2.freetype not available; using built-in font.", file=sys.stderr)
        elif not os.path.exists(font_path):
            print(f"Warning: font not found: {font_path}; using built-in font.", file=sys.stderr)
        else:
            try:
                ft = cv2.freetype.createFreeType2()
                ft.loadFontData(font_path, 0)
                self.ft = ft
            except Exception as exc:
                print(f"Warning: could not load font {font_path}: {exc}; using built-in font.",
                      file=sys.stderr)

    @staticmethod
    def _font_px(font_scale):
        return max(8, int(round(font_scale * BASE_FONT_PX)))

    def text_size(self, text, font_scale):
        """(width, height) of `text` at `font_scale`, matching the active renderer."""
        if self.ft is not None:
            (w, h), _ = self.ft.getTextSize(text, self._font_px(font_scale), -1)
            return w, h
        (w, h), _ = cv2.getTextSize(text, FONT, font_scale, 2)
        return w, h

    def _put_text(self, img, text, org, color, font_scale):
        """Draw `text` with its baseline at `org` (bottom-left), TTF or Hershey."""
        if self.ft is not None:
            self.ft.putText(img, text, org, self._font_px(font_scale), color, -1,
                            cv2.LINE_AA, True)
        else:
            cv2.putText(img, text, org, FONT, font_scale, color, 2, cv2.LINE_AA)

    def _text_extent(self, text, font_scale):
        """(ascent, descent) in px: how far the glyphs of `text` actually reach above /
        below the baseline. Measured by rendering once (cached), so the box can pad
        symmetrically instead of relying on font-metric estimates."""
        ph = self._font_px(font_scale)
        cache = getattr(self, "_extent_cache", None)
        if cache is None:
            cache = self._extent_cache = {}
        key = (text, ph, self.ft is not None)
        if key in cache:
            return cache[key]
        tw, _ = self.text_size(text, font_scale)
        w, h, base = max(2, tw + 4), max(2, ph * 3), ph * 2
        buf = np.zeros((h, w, 3), np.uint8)
        self._put_text(buf, text, (2, base), (255, 255, 255), font_scale)
        rows = np.where(buf.any(axis=2).any(axis=1))[0]
        ext = ((base - int(rows.min()), int(rows.max()) - base) if len(rows)
               else (int(ph * 0.72), int(ph * 0.28)))
        cache[key] = ext
        return ext

    def _font_metrics(self, font_scale):
        """(cap, desc): fixed cap/ascender height and descender depth for the font at
        this size (from METRIC_REF, not the label's own glyphs), so box padding and the
        text baseline stay put regardless of which characters a label contains."""
        return self._text_extent(METRIC_REF, font_scale)

    @staticmethod
    def _rounded_rect(img, x1, y1, x2, y2, r, color):
        """Filled rounded rectangle (OpenCV has no primitive): center rects + AA corners."""
        r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
        if r == 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            return
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cxr, cyr in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cxr, cyr), r, color, -1, cv2.LINE_AA)

    def draw_label(self, img, text, anchor, color, underline_frac=1.0, nchars=None,
                   font_scale=DEFAULT_FONT_SCALE, underline=True, box=True, text_color=None,
                   line_side="bottom"):
        """The label: an optional rounded box (in `color`) behind the text, an optional
        underline, and the text (in `text_color`, defaulting to `color`).

        `anchor` is where the connector line attaches; `line_side` picks which box edge
        it attaches to, and thus where the label sits relative to the feature:
        "bottom" -> label ABOVE, "top" -> label BELOW, "left" -> label to the RIGHT,
        "right" -> label to the LEFT. `underline_frac`/`nchars` drive the reveal.
        Text renders in the loaded TTF (cv2.freetype) or a Hershey fallback.
        """
        ax, ay = anchor
        ph = self._font_px(font_scale)
        tw, th = self.text_size(text, font_scale)   # full-text width for centering
        half = tw // 2
        text_col = text_color if text_color is not None else color
        # Fixed metrics (not this label's glyphs): stable baseline + equal borders; the
        # visual block is [cap-line .. baseline], equal margin all around, descenders
        # hang into the bottom margin (never move the text).
        cap, desc = self._font_metrics(font_scale)
        pad_x, pad_y = max(4, int(ph * 0.37)), max(3, int(ph * 0.18))
        margin = desc + pad_y
        box_h = cap + 2 * margin
        box_w = 2 * half + 2 * pad_x

        # Place the box so the anchor is centered on the chosen edge.
        if line_side == "top":
            box_left, box_top = ax - box_w // 2, ay
        elif line_side == "left":
            box_left, box_top = ax, ay - box_h // 2
        elif line_side == "right":
            box_left, box_top = ax - box_w, ay - box_h // 2
        else:  # bottom
            box_left, box_top = ax - box_w // 2, ay - box_h
        box_right, box_bottom = box_left + box_w, box_top + box_h
        cx = box_left + box_w // 2                    # text center x

        if box:
            self._rounded_rect(img, box_left, box_top, box_right, box_bottom,
                               max(4, int(ph * 0.35)), color)
        baseline = box_bottom - margin
        underline_y = box_bottom if box else baseline + max(3, int(ph * 0.16))

        if underline and underline_frac > 0:
            h = max(1, int(half * underline_frac))
            cv2.line(img, (cx - h, underline_y), (cx + h, underline_y), color, 2, cv2.LINE_AA)

        shown = text if nchars is None else text[:nchars]
        if shown:
            self._put_text(img, shown, (cx - half, baseline), text_col, font_scale)

    def display_position(self, ann, f):
        """Smoothed [cx, cy, w, h] for display: mean over recorded frames in a
        +/- self.smooth window around f. Non-destructive (the stored track keeps
        the raw tracker output) and seek-safe (looks up neighbors, no filter state).
        """
        base = position_at(ann, f)
        if base is None or self.smooth <= 0:
            return base
        acc = [[], [], [], []]
        for g in range(f - self.smooth, f + self.smooth + 1):
            p = ann["track"].get(str(g))
            if p:
                for i in range(4):
                    acc[i].append(p[i])
        if not acc[0]:
            return base
        return [sum(c) / len(c) for c in acc]

    def reveal_progress(self, ann, f):
        """0..1 reveal progress since the annotation's start_frame (1 = fully drawn)."""
        if self.reveal <= 0:
            return 1.0
        d = f - ann["start_frame"]
        if d < 0:
            return 0.0
        return min(1.0, d / self.reveal)

    @staticmethod
    def text_anchor(ann):
        """Fixed label position in frame coords. New annotations store `text_anchor`
        directly; legacy files with a feature-relative `text_offset` are converted
        against the seed feature position so the label is still anchored, not moving."""
        if "text_anchor" in ann:
            return int(ann["text_anchor"][0]), int(ann["text_anchor"][1])
        x, y, w, h = ann["seed_bbox"]
        dx, dy = ann.get("text_offset", (0, 0))
        return int(x + w / 2 + dx), int(y + h / 2 + dy)

    def draw_annotation(self, img, ann, f):
        pos = self.display_position(ann, f)
        if pos is None:
            return
        cx, cy = int(round(pos[0])), int(round(pos[1]))
        tx, ty = self.text_anchor(ann)            # FIXED — does not move with the feature
        end = ann.get("end_frame")
        stopped = end is not None and f > end     # past its stop frame -> hide the overlay
        r = int(ann.get("radius", self.radius))
        color = COLOR_LOST if ann["_lost"] else _bgr(ann.get("color", DEFAULT_COLOR))
        show_text = ann.get("show_text", True)

        if not stopped:
            if show_text:
                p = self.reveal_progress(ann, f)
                line_f = _seg(p, R_LINE, R_UNDER)     # line grows first
                und_f = _seg(p, R_UNDER, R_TEXT)      # then the underline
                txt_f = _seg(p, R_TEXT, 1.0)          # finally the text types in
                self.draw_connector(img, cx, cy, r, tx, ty, color, frac=line_f)
                if und_f > 0:
                    nchars = None if txt_f >= 1.0 else int(math.ceil(len(ann["text"]) * txt_f))
                    self.draw_label(img, ann["text"], (tx, ty), color, underline_frac=und_f,
                                    nchars=nchars, font_scale=ann.get("font_scale", DEFAULT_FONT_SCALE),
                                    underline=ann.get("show_underline", True),
                                    box=ann.get("show_box", True),
                                    text_color=_bgr(ann.get("text_color", DEFAULT_TEXT_COLOR)),
                                    line_side=ann.get("line_side", "bottom"))
            else:
                self.draw_connector(img, cx, cy, r, tx, ty, color, frac=0.0)  # circle only

        # Authoring aid (edit mode only): show the id for keyboard selection, and a
        # ring on the selected one. Stopped annotations still show a dim tag so they
        # can be re-selected to move or clear their stop.
        if not self.render:
            self.draw_id_tag(img, cx, cy, ann, stopped, r)

    def draw_id_tag(self, img, cx, cy, ann, stopped, r):
        col = COLOR_LOST if stopped else _bgr(ann.get("color", DEFAULT_COLOR))
        if self.sel_buffer.isdigit() and int(self.sel_buffer) == ann["id"]:
            cv2.circle(img, (cx, cy), r + 3, col, 1, cv2.LINE_AA)
        tag = str(ann["id"])
        if ann.get("end_frame") is not None:
            tag += f":stop{ann['end_frame']}"
        if not ann.get("show_text", True):
            tag += " (text off)"
        elif not ann.get("show_underline", True):
            tag += " (ul off)"
        cv2.putText(img, tag, (cx + r + 4, cy + 5), FONT, 0.5, col, 1, cv2.LINE_AA)

    def draw_connector(self, img, cx, cy, r, tx, ty, color, frac=1.0):
        """Circle at (cx,cy); line from the circle edge toward (tx,ty), drawn to
        `frac` of the way there (1.0 = all the way to the underline midpoint)."""
        cv2.circle(img, (cx, cy), r, color, 2, cv2.LINE_AA)
        if frac <= 0:
            return
        vx, vy = tx - cx, ty - cy
        dist = math.hypot(vx, vy) or 1.0
        ex, ey = cx + vx / dist * r, cy + vy / dist * r
        gx, gy = ex + (tx - ex) * frac, ey + (ty - ey) * frac
        cv2.line(img, (int(ex), int(ey)), (int(gx), int(gy)), color, 2, cv2.LINE_AA)

    def draw_pending(self, img):
        if self.pending is None:
            return
        fx, fy = self.pending["feature"]
        tx, ty = self.pending["text"]
        col = _bgr(self.default_color)
        self.draw_connector(img, fx, fy, self.radius, tx, ty, col)
        if self.mode == MODE_TEXT:
            self.draw_label(img, self.text_buffer + "_", (tx, ty), col,
                            font_scale=self.font_scale,
                            text_color=_bgr(self.default_text_color),
                            line_side=self.default_line_side)

    def draw_status(self, img):
        parts = []
        if self.is_seekable:
            parts.append(f"Frame {self.frame_idx}/{self.total_frames - 1}")
        if self.paused:
            parts.append("PAUSED")
        parts.append(f"ann:{len(self.annotations)}")
        if self.armed and self.mode == MODE_NAV:
            parts.append("PLACE FEATURE: drag (f=cancel)")
        if not self.render and self.sel_buffer:
            parts.append(f"SEL #{self.sel_buffer} (e/r=stop  t/l/b  c=side)")
        if self.mode == MODE_TEXT:
            parts.append(f"TEXT: {self.text_buffer}_")
        text = "   ".join(parts)
        cv2.putText(img, text, (10, 25), FONT, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, 25), FONT, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    def compose_frame(self):
        """Full-resolution frame with the overlays drawn on it (no display scaling)."""
        display = self.frame.copy()
        for ann in self.annotations:
            self.draw_annotation(display, ann, self.frame_idx)
        if self.show_chrome:                 # render mode: overlays only, no chrome
            self.draw_pending(display)
            self.draw_status(display)
        return display

    def show_frame(self):
        display = self.compose_frame()
        if self.display_scale != 1.0:
            display = cv2.resize(display, None, fx=self.display_scale,
                                 fy=self.display_scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW, display)

    # ---- main loop ---------------------------------------------------------

    # ---- render / export ---------------------------------------------------

    def export(self):
        """Render mode: burn overlays into a new video (with the source audio),
        named <video-without-ext>.annotated.mp4. Runs headless (no window)."""
        out_path = os.path.splitext(self.video_path)[0] + ".annotated.mp4"
        self._last_pct = -1
        if not self.annotations:
            print("Warning: no annotations loaded — exporting with no overlays.", file=sys.stderr)
        self.source.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first = self.source.read()
        if not ret:
            print(f"Error: could not read {self.video_path}", file=sys.stderr)
            return 1
        h, w = first.shape[:2]
        n = self.total_frames

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            code = self._export_ffmpeg(ffmpeg, out_path, first, w, h, n)
        else:
            print("ffmpeg not found — writing video WITHOUT audio "
                  "(install ffmpeg to mux the original audio).", file=sys.stderr)
            code = self._export_opencv(out_path, first, w, h, n)
        self.source.release()
        return code

    def _composed_frames(self, first):
        """Yield (idx, composed full-res frame) for the whole video from frame 0."""
        self.frame, self.frame_idx = first, 0
        yield 0, self.compose_frame()
        idx = 1
        while True:
            ret, frame = self.source.read()
            if not ret:
                return
            self.frame, self.frame_idx = frame, idx
            yield idx, self.compose_frame()
            idx += 1

    def _progress(self, idx, n):
        if n:
            pct = 100 * (idx + 1) // n
            if pct != self._last_pct:                 # only reprint when the percent changes
                self._last_pct = pct
                print(f"\rExporting {idx + 1}/{n} ({pct}%)", end="", flush=True)
        elif (idx + 1) % 30 == 0:
            print(f"\rExporting {idx + 1}", end="", flush=True)

    def _export_ffmpeg(self, ffmpeg, out_path, first, w, h, n):
        # Pipe raw BGR frames to ffmpeg; it encodes H.264 and copies the source audio.
        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{self.fps}", "-i", "-",          # 0: piped video
               "-i", self.video_path,                    # 1: source (for audio)
               "-map", "0:v:0", "-map", "1:a:0?",        # audio optional (?)
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
               "-c:a", "aac", "-b:a", "192k", "-shortest", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for idx, composed in self._composed_frames(first):
                proc.stdin.write(composed.tobytes())
                self._progress(idx, n)
        except BrokenPipeError:
            pass
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            proc.wait()
        print()
        if proc.returncode != 0:
            print(f"Error: ffmpeg failed (exit {proc.returncode}).", file=sys.stderr)
            return 1
        print(f"Wrote {out_path} (with audio)")
        return 0

    def _export_opencv(self, out_path, first, w, h, n):
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not writer.isOpened():
            print(f"Error: could not open VideoWriter for {out_path}", file=sys.stderr)
            return 1
        for idx, composed in self._composed_frames(first):
            writer.write(composed)
            self._progress(idx, n)
        writer.release()
        print()
        print(f"Wrote {out_path} (no audio)")
        return 0

    def run(self):
        if self.render:
            return self.export()
        cv2.namedWindow(WINDOW)
        if self.show_chrome and self.is_seekable:
            cv2.createTrackbar("Frame", WINDOW, 0, self.total_frames - 1, lambda v: None)
        if not self.render:
            cv2.setMouseCallback(WINDOW, self.on_mouse)

        while True:
            need_read = False
            # Only navigate in NAV mode; stay on the current frame while typing.
            if self.mode == MODE_NAV:
                if self.show_chrome and self.is_seekable:  # Rule 2: read trackbar after waitKey
                    tb = cv2.getTrackbarPos("Frame", WINDOW)
                    if tb != self.frame_idx:
                        self.seek_target = tb
                if self.seek_target is not None:           # keyboard/trackbar seek
                    self.source.set(cv2.CAP_PROP_POS_FRAMES, self.seek_target)
                    self.seek_target = None
                    need_read = True
                if not need_read and not self.paused:      # Rule 3: if, not elif
                    need_read = True

            if need_read:
                ret, frame = self.source.read()
                if not ret:
                    break
                self.frame = frame
                # Rule 1: never use POS_FRAMES directly for UI state.
                self.frame_idx = int(self.source.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if self.show_chrome and self.is_seekable:
                    cv2.setTrackbarPos("Frame", WINDOW, self.frame_idx)
                self.advance_trackers(self.frame_idx)

            if self.frame is None:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key in (32, ord('p')):
                    self.paused = not self.paused
                continue

            self.show_frame()

            # Rule 5: responsive when paused/typing, frame-rate-paced when playing.
            wait_ms = 30 if (self.paused or self.mode == MODE_TEXT) else max(1, int(1000 / self.fps))
            key = cv2.waitKey(wait_ms) & 0xFF

            if self.mode == MODE_TEXT:
                if not self.handle_text_key(key):
                    break
            else:
                if not self.handle_nav_key(key):
                    break

        if not self.render and self.dirty:
            self.save()
        self.source.release()
        cv2.destroyAllWindows()
        return 0

    def handle_text_key(self, key):
        if key == 13:                       # Enter — confirm
            self.finalize_annotation()
        elif key == 27:                     # Esc — cancel this annotation
            self.cancel_pending()
        elif key in (8, 127):               # Backspace / Del
            self.text_buffer = self.text_buffer[:-1]
        elif 32 <= key <= 126:              # printable ASCII
            self.text_buffer += chr(key)
        return True

    def handle_nav_key(self, key):
        if key in (ord('q'), 27):
            return False
        elif key in (32, ord('p')):                 # SPACE / p
            self.paused = not self.paused
        elif key in (ord('d'), 83) and self.is_seekable:   # d / RIGHT
            self.paused = True
            if self.frame_idx < self.total_frames - 1:
                self.seek_target = self.frame_idx + 1
        elif key in (ord('a'), 81) and self.is_seekable:   # a / LEFT
            self.paused = True
            if self.frame_idx > 0:
                self.seek_target = self.frame_idx - 1
        elif key == ord('u') and not self.render:     # undo last annotation
            if self.annotations:
                removed = self.annotations.pop()
                self.dirty = True
                print(f"Removed annotation #{removed['id']} ({removed['text']!r})")
        elif key == ord('s') and not self.render:     # save now
            self.save()
        elif 48 <= key <= 57 and not self.render:     # digit -> select an annotation by id
            self.select_digit(chr(key))
        elif key in (8, 127) and not self.render:     # backspace -> clear id selection
            self.sel_buffer = ""
        elif key == ord('e') and not self.render:     # stop selected track at current frame
            self.set_end_selected()
        elif key == ord('r') and not self.render:     # remove selected track's stop
            self.clear_end_selected()
        elif key == ord('t') and not self.render:     # toggle selected annotation's text
            self.toggle_spec_selected("show_text", "text", "'t'")
        elif key == ord('l') and not self.render:     # toggle selected annotation's underline
            self.toggle_spec_selected("show_underline", "underline", "'l'")
        elif key == ord('b') and not self.render:     # toggle selected annotation's box
            self.toggle_spec_selected("show_box", "box", "'b'")
        elif key == ord('c') and not self.render:     # flip selected annotation's connect side
            self.flip_line_side_selected()
        elif key == ord('f') and not self.render:     # arm the next drag to place a feature
            self.armed = not self.armed
            print("Feature placement ARMED — drag from the feature to the label position."
                  if self.armed else "Feature placement cancelled.")
        return True

    def flip_line_side_selected(self):
        """Cycle the selected annotation's connector side: bottom -> top -> left -> right."""
        ann = self.selected_ann()
        if ann is None:
            print(f"No annotation #{self.sel_buffer}." if self.sel_buffer
                  else "Type an annotation id first, then 'c' to change its line side.")
        else:
            cur = ann.get("line_side", "bottom")
            i = LINE_SIDES.index(cur) if cur in LINE_SIDES else 0
            ann["line_side"] = LINE_SIDES[(i + 1) % len(LINE_SIDES)]
            print(f"#{ann['id']} ({ann['text']!r}) line connects at the {ann['line_side']}")
            self.dirty = True
        self.sel_buffer = ""

    def toggle_spec_selected(self, key, label, keyhint):
        """Toggle a boolean display spec (show_text / show_underline) of the selected
        annotation. `label`/`keyhint` are only used for console feedback."""
        ann = self.selected_ann()
        if ann is None:
            print(f"No annotation #{self.sel_buffer}." if self.sel_buffer
                  else f"Type an annotation id first, then {keyhint} to toggle its {label}.")
        else:
            ann[key] = not ann.get(key, True)
            print(f"#{ann['id']} ({ann['text']!r}) {label} {'on' if ann[key] else 'off'}")
            self.dirty = True
        self.sel_buffer = ""


def main():
    parser = argparse.ArgumentParser(
        description="Navigate a video and annotate tracked features.")
    parser.add_argument("video", help="path to the video file")
    parser.add_argument("--json",
                        help="annotations file (default: the video path with its extension "
                             "replaced by .annotations.json)")
    parser.add_argument("--box", type=int, default=48,
                        help="size (px) of the tracker box seeded around a click (default 48)")
    parser.add_argument("--radius", type=int, default=12,
                        help="fixed circle radius in px (default 12; display only, independent of --box)")
    parser.add_argument("--max-width", type=int, default=1280,
                        help="downscale display so width <= this (default 1280); coords stay full-res")
    parser.add_argument("--smooth", type=int, default=2,
                        help="steady the circle: average position over +/-N frames (default 2, 0=off)")
    parser.add_argument("--reveal", type=int, default=12,
                        help="frames over which an annotation reveals (circle->line->underline->text); 0=instant")
    parser.add_argument("--mode", choices=("edit", "render"), default="edit",
                        help="edit: annotate + track (default); render: export "
                             "<video>.annotated.mp4 with overlays + audio (headless)")
    parser.add_argument("--color", type=int, nargs=3, metavar=("R", "G", "B"),
                        default=DEFAULT_COLOR,
                        help="default circle/line/box color for new annotations (default 255 255 255)")
    parser.add_argument("--text-color", type=int, nargs=3, metavar=("R", "G", "B"),
                        default=DEFAULT_TEXT_COLOR,
                        help="default text color for new annotations (default 100 161 157 = teal 0x64A19D)")
    parser.add_argument("--font-scale", type=float, default=DEFAULT_FONT_SCALE,
                        help=f"default text size for new annotations (default {DEFAULT_FONT_SCALE})")
    parser.add_argument("--font", default=DEFAULT_FONT,
                        help="TTF for label text via cv2.freetype (default: bundled Varela Round; "
                             "falls back to a built-in Hershey font if missing/unloadable)")
    parser.add_argument("--line-side", choices=LINE_SIDES, default="bottom",
                        help="which box edge the connector attaches to for new annotations: "
                             "bottom=label above (default), top=below, left=right of, right=left of")
    args = parser.parse_args()

    json_path = args.json or (os.path.splitext(args.video)[0] + ".annotations.json")
    app = App(args.video, json_path, args.box, args.radius, args.max_width,
              args.mode == "render", args.smooth, args.reveal, args.color, args.font_scale,
              args.font, args.text_color, args.line_side)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
