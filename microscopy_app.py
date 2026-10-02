import dash
from dash import dcc, html, Input, Output, State, callback_context
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import requests
import tifffile
import hashlib
import traceback
import time
import os
from io import BytesIO
import base64
import json

app = dash.Dash(__name__, title="Microscopy Grid Aligner")

# ── Settings JSON schema version ────────────────────────────────────────
# 1 (implicit — the key is absent from the JSON entirely): crop sliders were purely cosmetic
#   and never restricted anything (pre-"Section Crop"). Files saved by that older version of
#   this app have no `schema_version` key at all.
# 2: "Section Crop" is functional — crop_top/bottom/left/right are real bounds that restrict
#   the grid/wells/exports.
# 3: the grid was aligned against an uploaded TIF (viewing one channel at a time) instead of the
#   merge JPG, because the JPG export isn't always pixel-perfect against the tile-stitched TIF.
#   These files record an explicit `source` field ("tif" or "jpg") alongside
#   `schema_version: 3`. Every save from this version of the app writes `schema_version: 3` and
#   `source`. Files with schema_version < 3 (or no key at all) always imply `source == "jpg"`,
#   since TIF-source alignment didn't exist yet when they were written.
#
# `CROP_SCHEMA_VERSION` (kept separate from `SETTINGS_SCHEMA_VERSION`, which just tracks the
# CURRENT format this app writes) is the fixed threshold `load_settings` uses to stay backwards
# compatible: legacy files (schema_version < CROP_SCHEMA_VERSION) have their stored crop values
# ignored on load (reset to 0), so old alignments open exactly as they looked when they were
# saved, instead of suddenly being clipped by cosmetic leftovers. Bumping
# `SETTINGS_SCHEMA_VERSION` for the new `source` field must NOT change that threshold.
CROP_SCHEMA_VERSION = 2
SETTINGS_SCHEMA_VERSION = 3

# ── Load sample image ──────────────────────────────────────────────────
try:
    url = "https://raw.githubusercontent.com/scikit-image/scikit-image/main/skimage/data/immunohistochemistry.png"
    response = requests.get(url, timeout=5)
    original_image = ImageOps.exif_transpose(Image.open(BytesIO(response.content))).convert('RGB')
except Exception:
    img_array = np.zeros((800, 800, 3), dtype=np.uint8)
    for i in range(0, 800, 80):
        img_array[i:i+40, :] = [50, 0, 100]
        img_array[:, i:i+40] = [0, 100, 50]
    original_image = Image.fromarray(img_array)

# ── Rotation cache ─────────────────────────────────────────────────────
_rotation_cache = {}
_uploaded_image = None

# ── TIF upload state ───────────────────────────────────────────────────
# When the active upload is a TIF, `_uploaded_image` is (re)built from a SINGLE selected
# channel (as a grayscale 'L' PIL image) of `_uploaded_tif_array`, so every other callback in
# this app -- which all just read the `_uploaded_image` global -- keeps working unchanged.
# `_uploaded_tif_array` (Y, X, C) holds the FULL set of channels so switching the "Channel"
# control doesn't require re-uploading.
_uploaded_tif_array = None
_tif_decode_cache = {}
# Cache for TIFs loaded from a local path (keyed by path + mtime + downsample factor), bounded
# to the most recent one. This exists so `update_image` and `reset_vmin_vmax_for_channel` --
# two INDEPENDENT callbacks both triggered by the same `btn-load-tif-path.n_clicks` click, with
# no Output/Input relationship between them -- each resolve the SAME array on their own rather
# than one reading a global the other may not have written yet (Dash does not guarantee which
# of two independently-triggered callbacks runs first). Whichever callback runs first does the
# actual disk read and populates this cache; the other just hits it.
_tif_path_cache = {}
# Rotated copy of `_uploaded_tif_array` (ALL channels), cached by rotation angle -- separate
# from `_rotation_cache` (which caches a single-channel PIL preview) because switching the
# "Channel" selector or dragging the vmin/vmax sliders (both far more frequent than changing
# rotation) would otherwise force a full re-rotation of the whole multi-channel array on every
# such change. Only the most recent rotation is kept (bounded memory, one upload at a time).
_tif_rotated_cache = {}


def _load_tif_array(file_obj):
    """Load a (possibly multi-channel, possibly OME-) TIFF into a (Y, X, C) array, collapsing
    any extra axes (Z, T, S, ...) by taking their first index -- mirrors scoper.py's
    `TifImage._to_yxc` so a grid aligned here translates directly for that package.
    """
    with tifffile.TiffFile(file_obj) as tif:
        series = tif.series[0]
        array = series.asarray()
        axes = series.axes.upper()

    extra_axes = [a for a in axes if a not in "YXC"]
    for ax in extra_axes:
        idx = axes.index(ax)
        array = np.take(array, 0, axis=idx)
        axes = axes[:idx] + axes[idx + 1:]

    y_idx, x_idx = axes.index("Y"), axes.index("X")
    if "C" in axes:
        c_idx = axes.index("C")
        array = np.moveaxis(array, (y_idx, x_idx, c_idx), (0, 1, 2))
    else:
        array = np.moveaxis(array, (y_idx, x_idx), (0, 1))
        array = array[..., np.newaxis]
    return array


def _decode_tif_from_b64(content_string):
    """Decode+parse a TIF upload's base64 payload, cached by content hash so switching the
    "Channel" selector (which re-triggers `update_image`) doesn't re-parse the whole TIF.
    Only the single most recent upload is kept (bounded memory, matches `_uploaded_tif_array`
    tracking one upload at a time).
    """
    key = hashlib.md5(content_string.encode('ascii')).hexdigest()
    if key not in _tif_decode_cache:
        print(f"[update_image] decoding TIF upload: {len(content_string) / 1e6:.1f} MB of "
              "base64 text received from the browser", flush=True)
        t0 = time.time()
        decoded = base64.b64decode(content_string)
        print(f"[update_image] base64-decoded to {len(decoded) / 1e6:.1f} MB of raw bytes "
              f"({time.time() - t0:.2f}s)", flush=True)
        t0 = time.time()
        array = _load_tif_array(BytesIO(decoded))
        print(f"[update_image] tifffile parsed array shape={array.shape} dtype={array.dtype} "
              f"({time.time() - t0:.2f}s)", flush=True)
        _tif_decode_cache.clear()
        _tif_decode_cache[key] = array
    return _tif_decode_cache[key]


def _load_tif_path_cached(path, step):
    """Load (and downsample) a TIF from a local path, cached by (path, mtime, step) so repeat
    calls for the same file+downsample (e.g. from `update_image` and
    `reset_vmin_vmax_for_channel` reacting to the same click) only hit disk once.
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    key = (path, mtime, step)
    if key not in _tif_path_cache:
        array = _load_tif_array(path)
        if step > 1:
            array = array[::step, ::step, :]
        _tif_path_cache.clear()
        _tif_path_cache[key] = array
    return _tif_path_cache[key]


def _normalize_to_uint8(plane, low_pct=1.0, high_pct=99.5):
    """Percentile-stretch a single channel plane (any dtype) to a viewable uint8 grayscale
    image. Microscopy TIFs are commonly 16-bit with a handful of hot outlier pixels, so a
    plain min/max stretch tends to wash out everything else -- percentile clipping keeps the
    preview usable for alignment purposes. Used as a fallback before the vmin/vmax slider has
    a real value (e.g. right after a TIF first loads); see `_normalize_to_uint8_range` for the
    user-controlled version used everywhere else.
    """
    plane = plane.astype(np.float32)
    lo, hi = np.percentile(plane, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros(plane.shape, dtype=np.uint8)
    stretched = np.clip((plane - lo) / (hi - lo), 0, 1) * 255
    return stretched.astype(np.uint8)


def _normalize_to_uint8_range(plane, vmin, vmax):
    """Linearly stretch a single channel plane to uint8 using an EXPLICIT (vmin, vmax) range
    -- i.e. whatever the user has set via the vmin/vmax slider -- instead of automatic
    percentile clipping. Values at/below vmin go to 0, at/above vmax go to 255.
    """
    plane = plane.astype(np.float32)
    if vmax <= vmin:
        return np.zeros(plane.shape, dtype=np.uint8)
    stretched = np.clip((plane - vmin) / (vmax - vmin), 0, 1) * 255
    return stretched.astype(np.uint8)


def _rotate_multichannel(array, angle):
    """Rotate each channel plane of a (Y, X, C) array independently via PIL -- same
    `rotate(-angle, expand=True, resample=BILINEAR)` call (note the sign) used by
    `_get_rotated_data` for single-channel/RGB previews, so both paths agree on rotation
    direction.
    """
    if (angle or 0) % 360 == 0:
        return array
    dtype = array.dtype
    channels = []
    for i in range(array.shape[2]):
        plane = Image.fromarray(array[:, :, i].astype(np.float32), mode='F')
        rotated = plane.rotate(-angle, expand=True, resample=Image.BILINEAR)
        channels.append(np.array(rotated))
    stacked = np.stack(channels, axis=-1)
    if np.issubdtype(dtype, np.integer):
        stacked = np.round(stacked)
    return stacked.astype(dtype)


def _get_rotated_tif_array(rotation):
    """Rotated (all channels) version of `_uploaded_tif_array`, cached by rotation angle so
    channel switches / vmin-vmax slider drags don't re-pay for a full-array rotation.
    """
    key = round((rotation or 0) % 360, 6)
    if key not in _tif_rotated_cache:
        _tif_rotated_cache.clear()
        _tif_rotated_cache[key] = _rotate_multichannel(_uploaded_tif_array, rotation or 0)
    return _tif_rotated_cache[key]


def _pil_to_b64(img, fmt='JPEG'):
    buf = BytesIO()
    if fmt == 'JPEG':
        img.save(buf, format='JPEG', quality=85)
        mime = 'image/jpeg'
    else:
        img.save(buf, format='PNG')
        mime = 'image/png'
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:{mime};base64,{b64}"


def _get_rotated_data(img, rotation):
    cache_key = (id(img), rotation)
    if cache_key not in _rotation_cache:
        if rotation == 0:
            rotated = img
        else:
            rotated = img.rotate(-rotation, expand=True, resample=Image.BILINEAR)
        # Create a lightweight preview for the browser UI to avoid WebGL lag
        preview = rotated.copy()
        preview.thumbnail((2000, 2000), Image.BILINEAR)

        _rotation_cache[cache_key] = {
            'b64': _pil_to_b64(preview),
            'w': rotated.size[0],
            'h': rotated.size[1],
            'pw': preview.size[0],
            'ph': preview.size[1],
            'img': rotated
        }
        if len(_rotation_cache) > 3:
            oldest = next(iter(_rotation_cache))
            del _rotation_cache[oldest]
    return _rotation_cache[cache_key]


def _get_rotated_pil(img, rotation):
    """Return the cached PIL image to avoid re-rotation."""
    return _get_rotated_data(img, rotation)['img'].copy()


# ── Shared styles ──────────────────────────────────────────────────────
_btn_style = {
    'backgroundColor': '#333', 'color': '#ddd', 'border': '1px solid #555',
    'borderRadius': '4px', 'padding': '8px 12px', 'cursor': 'pointer',
    'fontFamily': 'sans-serif', 'fontSize': '0.85em', 'width': '100%',
    'marginBottom': '6px'
}
_label_style = {'color': '#bbbbbb', 'fontFamily': 'sans-serif', 'fontSize': '0.9em', 'marginBottom': '4px'}

# ── Layout ─────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id='image-store'),
    dcc.Download(id='download-settings'),

    # Left Control Panel
    html.Div([
        html.H2("Microscopy Aligner", style={
            'color': '#ffffff', 'margin': '0 0 15px 0',
            'fontFamily': 'sans-serif', 'fontSize': '1.4em'
        }),

        # ── Upload ─────────────────────────────────────────────────
        html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag & Drop or ',
                    html.A('Select Image', style={
                        'color': '#00aaff', 'textDecoration': 'underline'
                    })
                ]),
                style={
                    'width': '100%', 'height': '50px', 'lineHeight': '50px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center',
                    'marginBottom': '15px', 'color': '#bbbbbb',
                    'borderColor': '#777', 'cursor': 'pointer', 'fontSize': '0.85em'
                },
                multiple=False
                # NOTE: deliberately no `accept=` filter here. dcc.Upload (react-dropzone)
                # enforces `accept` client-side by sniffing the file's MIME type, and browsers
                # frequently misdetect/blank the MIME type for double-extension files like
                # ".ome.tif" (instead of a clean "image/tiff") -- this silently dropped such
                # files with ZERO feedback (no server request, no error, no spinner) before the
                # Python callback ever ran. Server-side validation (filename-extension sniffing
                # in `update_image`, now with real error logging) is the actual gate.
            )
        ]),

        # ── TIF: load from local path (server-side, avoids the browser) ──
        html.Div([
            html.Label("Load TIF from local path", style={**_label_style, 'fontWeight': 'bold', 'color': '#fff'}),
            html.Div(
                "Reads the file directly on this machine instead of uploading it through the "
                "browser -- avoids the browser running out of memory on large (1-2+ GiB) "
                "whole-slide TIFs. Only works when the app and the TIF are on the same machine.",
                style={'color': '#888', 'fontFamily': 'sans-serif', 'fontSize': '0.75em', 'marginBottom': '6px'}
            ),
            dcc.Input(
                id='tif-path-input', type='text', placeholder='/path/to/image.tif',
                style={'width': '100%', 'backgroundColor': '#333', 'color': '#ddd',
                       'border': '1px solid #555', 'borderRadius': '4px', 'padding': '6px',
                       'marginBottom': '6px', 'boxSizing': 'border-box'}
            ),
            html.Div([
                html.Label("Downsample factor", style=_label_style),
                dcc.Dropdown(
                    id='tif-downsample-dropdown',
                    options=[{'label': f'{n}x', 'value': n} for n in [1, 2, 3, 4, 5, 8, 10]],
                    value=4, clearable=False,
                    style={'backgroundColor': '#333', 'color': '#ddd'},
                    className='dark-dropdown'
                )
            ], style={'marginBottom': '6px'}),
            html.Button("📂  Load TIF", id='btn-load-tif-path', n_clicks=0, style=_btn_style),
        ], style={'marginBottom': '15px'}),

        # ── TIF channel selector (only relevant/shown for TIF uploads) ─
        html.Div(id='tif-channel-container', children=[
            html.Label("Channel", style=_label_style),
            dcc.Dropdown(
                id='tif-channel-dropdown', options=[], value=0, clearable=False,
                style={'backgroundColor': '#333', 'color': '#ddd'},
                className='dark-dropdown'
            ),
        ], style={'marginBottom': '15px', 'display': 'none'}),

        # ── Display range (vmin/vmax) for the current channel ──────
        html.Div(id='vmin-vmax-container', children=[
            html.Label("Display Range (vmin / vmax)", style=_label_style),
            dcc.RangeSlider(
                id='vmin-vmax-slider', min=0, max=255, step=1, value=[0, 255],
                updatemode='drag', allowCross=False,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'marginBottom': '15px', 'display': 'none'}),

        # ── Interactive Controls ───────────────────────────────────
        html.Div([
            dcc.Store(id='placement-mode', data=False),
            dcc.Store(id='center-point-store', data={'x': 0.0, 'y': 0.0}),
            dcc.Store(id='keypress-store'),
            html.Div(id='dummy-listener'),
            html.Button("🎯 Place Center Point (W)", id='btn-place-center', n_clicks=0, style=_btn_style),
            html.Div(id='center-point-display', children='Center Point: (0.0, 0.0)', style={'color': '#00ffff', 'fontFamily': 'monospace', 'fontSize': '0.9em', 'marginBottom': '5px', 'textAlign': 'center'}),
            html.Div(id='placement-status', style={'color': '#ffaa00', 'fontFamily': 'sans-serif', 'fontSize': '0.85em', 'marginBottom': '15px', 'textAlign': 'center', 'fontWeight': 'bold'})
        ]),

        # ── Sliders ────────────────────────────────────────────────
        html.Div([
            html.Label("Image Rotation (°)", style=_label_style),
            dcc.Slider(id='rotation-slider', min=-180, max=180, step=0.01, value=0,
                       updatemode='mouseup',
                       marks={i: {'label': str(i), 'style': {'color': '#777'}}
                              for i in range(-180, 181, 90)},
                       tooltip={"placement": "bottom", "always_visible": True})
        ], style={'marginBottom': '15px'}),

        html.Div([
            dcc.Checklist(
                id='link-spacing-check',
                options=[{'label': ' Link X and Y Spacing', 'value': 'link'}],
                value=['link'],
                style={'color': '#bbbbbb', 'fontFamily': 'sans-serif', 'fontSize': '0.9em', 'marginBottom': '10px'}
            )
        ]),

        html.Div([
            html.Label("Grid X Spacing (px)", style=_label_style),
            dcc.Slider(id='grid-spacing-slider', min=10, max=2000, step=0.001, value=229,
                       updatemode='drag',
                       marks={i: {'label': str(i), 'style': {'color': '#777'}}
                              for i in range(500, 2001, 500)},
                       tooltip={"placement": "bottom", "always_visible": True})
        ], style={'marginBottom': '15px'}),

        html.Div(id='grid-y-spacing-container', children=[
            html.Label("Grid Y Spacing (px)", style=_label_style),
            dcc.Slider(id='grid-y-spacing-slider', min=10, max=2000, step=0.001, value=229,
                       updatemode='drag',
                       marks={i: {'label': str(i), 'style': {'color': '#777'}}
                              for i in range(500, 2001, 500)},
                       tooltip={"placement": "bottom", "always_visible": True})
        ], style={'marginBottom': '15px', 'display': 'none'}),

        html.Div([
            html.Label("Grid X Offset (px)", style=_label_style),
            dcc.Slider(id='grid-x-offset-slider', min=-2000, max=2000, step=0.01, value=0,
                       updatemode='drag',
                       marks={i: {'label': str(i), 'style': {'color': '#777'}}
                              for i in range(-2000, 2001, 1000)},
                       tooltip={"placement": "bottom", "always_visible": True})
        ], style={'marginBottom': '15px'}),

        html.Div([
            html.Label("Grid Y Offset (px)", style=_label_style),
            dcc.Slider(id='grid-y-offset-slider', min=-2000, max=2000, step=0.01, value=0,
                       updatemode='drag',
                       marks={i: {'label': str(i), 'style': {'color': '#777'}}
                              for i in range(-2000, 2001, 1000)},
                       tooltip={"placement": "bottom", "always_visible": True})
        ], style={'marginBottom': '15px'}),

        html.Div([
            html.Label("Grid Opacity", style=_label_style),
            dcc.Slider(id='grid-opacity-slider', min=0, max=1, step=0.1, value=0.7,
                       updatemode='drag',
                       marks={0: {'label': '0', 'style': {'color': '#777'}},
                              1: {'label': '1', 'style': {'color': '#777'}}},
                       tooltip={"placement": "bottom", "always_visible": False})
        ], style={'marginBottom': '15px'}),
        
        # ── Image Cropping ──────────────────────────────────────────
        html.Hr(style={'borderColor': '#444', 'margin': '12px 0'}),
        html.Label("Section Crop (%)", style={
            'color': '#ffffff', 'fontFamily': 'sans-serif',
            'fontWeight': 'bold', 'marginBottom': '4px', 'display': 'block'
        }),
        html.Div(
            "Restricts the grid to this region. Does not modify "
            "the underlying image — use this to isolate one tissue section at a time and "
            "save each section as its own settings file.",
            style={'color': '#888', 'fontFamily': 'sans-serif', 'fontSize': '0.75em', 'marginBottom': '8px'}
        ),

        html.Div([
            html.Label("Crop Top", style=_label_style),
            dcc.Slider(id='crop-top-slider', min=0, max=99, step=0.1, value=0, updatemode='drag', tooltip={"placement": "bottom", "always_visible": False})
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Label("Crop Bottom", style=_label_style),
            dcc.Slider(id='crop-bottom-slider', min=0, max=99, step=0.1, value=0, updatemode='drag', tooltip={"placement": "bottom", "always_visible": False})
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Label("Crop Left", style=_label_style),
            dcc.Slider(id='crop-left-slider', min=0, max=99, step=0.1, value=0, updatemode='drag', tooltip={"placement": "bottom", "always_visible": False})
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Label("Crop Right", style=_label_style),
            dcc.Slider(id='crop-right-slider', min=0, max=99, step=0.1, value=0, updatemode='drag', tooltip={"placement": "bottom", "always_visible": False})
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("Show Well Labels", style=_label_style),
            dcc.Checklist(
                id='show-labels-check',
                options=[{'label': ' Show A1, A2, B1... labels', 'value': 'show'}],
                value=['show'],
                labelStyle={'color': '#bbbbbb', 'display': 'inline-block', 'marginLeft': '5px'},
                style={'color': '#bbbbbb', 'fontFamily': 'sans-serif', 'marginTop': '3px'}
            )
        ], style={'marginBottom': '15px'}),

        # ── Export & Settings ──────────────────────────────────────
        html.Hr(style={'borderColor': '#444', 'margin': '12px 0'}),
        html.Label("Export & Settings", style={
            'color': '#ffffff', 'fontFamily': 'sans-serif',
            'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'
        }),

        html.Hr(style={'borderColor': '#444', 'margin': '12px 0'}),

        html.Div([
            html.Label("Settings Filename", style=_label_style),
            dcc.Input(
                id='save-settings-filename',
                type='text',
                placeholder='grid_settings.json',
                style={'width': '100%', 'backgroundColor': '#333', 'color': '#fff', 'border': '1px solid #444', 'padding': '5px', 'marginBottom': '8px', 'boxSizing': 'border-box'}
            )
        ]),
        html.Button("⬇️  Save Grid Settings", id='btn-save-settings', n_clicks=0, style=_btn_style),
        dcc.Upload(
            id='upload-settings',
            children=html.Button("⬆️  Load Grid Settings", style=_btn_style),
            style={'width': '100%'},
            multiple=False
        ),

        html.Div(id='status-text', style={
            'color': '#4CAF50', 'fontFamily': 'sans-serif',
            'fontSize': '0.85em', 'marginTop': '8px', 'minHeight': '20px'
        }),

    ], style={
        'width': '28%', 'height': '100vh', 'padding': '20px',
        'boxSizing': 'border-box', 'display': 'inline-block',
        'verticalAlign': 'top', 'backgroundColor': '#1e1e1e',
        'borderRight': '1px solid #333', 'overflowY': 'auto'
    }),

    # Right Main Display Panel
    html.Div([
        dcc.Graph(
            id='image-graph',
            style={'height': '100vh', 'width': '100%'},
            config={
                'scrollZoom': True,
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            }
        )
    ], style={
        'width': '72%', 'height': '100vh',
        'display': 'inline-block', 'verticalAlign': 'top',
        'backgroundColor': '#000000'
    })
], style={'margin': '0', 'padding': '0', 'display': 'flex'})


# ── Server callback: image processor (upload, local-path TIF load, rotate, channel, vmin/vmax) ──
@app.callback(
    [Output('image-store', 'data'),
     Output('status-text', 'children', allow_duplicate=True)],
    [Input('upload-image', 'contents'),
     Input('btn-load-tif-path', 'n_clicks'),
     Input('rotation-slider', 'value'),
     Input('tif-channel-dropdown', 'value'),
     Input('vmin-vmax-slider', 'value')],
    [State('upload-image', 'filename'),
     State('tif-path-input', 'value'),
     State('tif-downsample-dropdown', 'value')],
    prevent_initial_call='initial_duplicate'
)
def update_image(upload_contents, load_tif_clicks, rotation, channel, vmin_vmax,
                  upload_filename, tif_path, downsample_factor):
    global _uploaded_image, _uploaded_tif_array

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else '.'
    print(f"[update_image] triggered by {trigger_id!r} filename={upload_filename!r} "
          f"rotation={rotation} channel={channel} vmin_vmax={vmin_vmax} "
          f"contents_received={upload_contents is not None}", flush=True)

    status = dash.no_update

    if trigger_id == 'upload-image.contents':
        if upload_contents is not None:
            try:
                _, content_string = upload_contents.split(',')
                is_tif = bool(upload_filename) and upload_filename.lower().endswith(('.tif', '.tiff'))
                if is_tif:
                    _uploaded_tif_array = _decode_tif_from_b64(content_string)
                    _uploaded_image = None
                else:
                    decoded = base64.b64decode(content_string)
                    _uploaded_image = ImageOps.exif_transpose(Image.open(BytesIO(decoded))).convert('RGB')
                    _uploaded_tif_array = None
                _tif_rotated_cache.clear()
                print(f"[update_image] upload decoded OK (is_tif={is_tif})", flush=True)
                status = f"✅ Loaded {upload_filename}"
            except Exception:
                print(f"[update_image] FAILED to decode upload {upload_filename!r}:", flush=True)
                traceback.print_exc()
                status = f"❌ Failed to load {upload_filename}"
        else:
            print("[update_image] upload cleared", flush=True)
            _uploaded_image = None
            _uploaded_tif_array = None
            _tif_rotated_cache.clear()

    elif trigger_id == 'btn-load-tif-path.n_clicks':
        if not tif_path or not tif_path.strip():
            status = "❌ Enter a TIF path first"
        else:
            tif_path = tif_path.strip()
            try:
                t0 = time.time()
                step = max(int(downsample_factor or 1), 1)
                array = _load_tif_path_cached(tif_path, step)
                _uploaded_tif_array = array
                _uploaded_image = None
                _tif_rotated_cache.clear()
                print(f"[update_image] loaded TIF from path {tif_path!r}: shape={array.shape} "
                      f"dtype={array.dtype} downsample={step}x ({time.time() - t0:.2f}s)", flush=True)
                status = (f"✅ Loaded {tif_path} ({array.shape[1]}x{array.shape[0]}, "
                          f"{array.shape[2]} ch, {step}x downsampled)")
            except Exception:
                print(f"[update_image] FAILED to load TIF from path {tif_path!r}:", flush=True)
                traceback.print_exc()
                status = f"❌ Failed to load {tif_path} (see server console)"

    # TIFs may hold more channels than can be shown at once -- render whichever one the
    # "Channel" control currently selects (clamped to a valid index), stretched to the
    # vmin/vmax slider's current range, as a grayscale image. Store it in `_uploaded_image` so
    # every other callback in this app (which all just read that global directly) keeps
    # working unchanged. The (possibly large) rotation itself is cached separately by angle
    # (`_get_rotated_tif_array`) so only the cheap per-pixel vmin/vmax stretch re-runs when the
    # slider is dragged or the channel is switched.
    n_channels = 1
    source = 'jpg'
    if _uploaded_tif_array is not None:
        n_channels = _uploaded_tif_array.shape[2]
        ch = min(max(int(channel or 0), 0), n_channels - 1)
        rotated_array = _get_rotated_tif_array(rotation)
        plane = rotated_array[:, :, ch]
        if vmin_vmax and len(vmin_vmax) == 2 and vmin_vmax[1] > vmin_vmax[0]:
            img8 = _normalize_to_uint8_range(plane, vmin_vmax[0], vmin_vmax[1])
        else:
            img8 = _normalize_to_uint8(plane)
        _uploaded_image = Image.fromarray(img8, mode='L')
        source = 'tif'
        print(f"[update_image] displaying TIF channel {ch}/{n_channels - 1}, "
              f"vmin_vmax={vmin_vmax}, rotated array shape={rotated_array.shape}", flush=True)
        # Already rotated above (via `_get_rotated_tif_array`) -- pass rotation=0 here so
        # `_get_rotated_data` only thumbnails/caches the preview, without rotating again.
        data = _get_rotated_data(_uploaded_image, 0)
    else:
        current = _uploaded_image if _uploaded_image is not None else original_image
        data = _get_rotated_data(current, rotation)

    print(f"[update_image] returning image-store data: source={source} w={data['w']} "
          f"h={data['h']}", flush=True)
    return {'b64': data['b64'], 'w': data['w'], 'h': data['h'], 'pw': data['pw'], 'ph': data['ph'],
            'source': source, 'n_channels': n_channels}, status


# ── One-directional: populate the Channel dropdown's options based on the active upload's
# source. Deliberately never touches `tif-channel-dropdown.value` (see `reset_channel_on_new_load`
# below) -- doing so here (driven by `image-store.data`, which `update_image` itself produces
# FROM `tif-channel-dropdown.value`) would create a circular callback dependency.
@app.callback(
    [Output('tif-channel-container', 'style'),
     Output('tif-channel-dropdown', 'options')],
    Input('image-store', 'data')
)
def update_channel_ui(img_data):
    if not img_data or img_data.get('source') != 'tif':
        return {'marginBottom': '15px', 'display': 'none'}, []
    n_channels = img_data.get('n_channels', 1)
    options = [{'label': f'Channel {i}', 'value': i} for i in range(n_channels)]
    return {'marginBottom': '15px', 'display': 'block'}, options


# ── Reset the selected channel to 0 whenever a NEW file is loaded (browser upload or the
# local-path loader) -- any previously-selected channel index may not be valid for the new
# file. Driven only by the load events themselves, never by `image-store.data`, so this can't
# form a cycle with `update_image` (which reads `tif-channel-dropdown.value`).
@app.callback(
    Output('tif-channel-dropdown', 'value'),
    [Input('upload-image', 'contents'),
     Input('btn-load-tif-path', 'n_clicks')],
    prevent_initial_call=True
)
def reset_channel_on_new_load(contents, n_clicks):
    return 0


# ── Reset the vmin/vmax slider to a sensible default range whenever the channel changes OR a
# new file loads -- using the channel's own percentile stats, same spirit as
# `_normalize_to_uint8`'s default stretch. Also driven only by non-`image-store.data` inputs
# (the channel dropdown's value, plus the two load events directly, in case the dropdown's
# value doesn't actually change e.g. it's already 0) to avoid the same cycle risk.
#
# IMPORTANT: this callback and `update_image` are both triggered directly by
# `upload-image.contents`/`btn-load-tif-path.n_clicks`, with no Output/Input relationship
# between them -- Dash does NOT guarantee which one runs first. So on a fresh load, this
# callback must NOT simply read the `_uploaded_tif_array` global (which `update_image` writes)
# -- it might run BEFORE `update_image` has written it, silently seeing stale/None data. Instead
# it independently re-resolves the array via the same cached helpers `update_image` uses
# (`_decode_tif_from_b64` / `_load_tif_path_cached`), so whichever callback happens to run first
# does the actual work and the other just hits the cache.
@app.callback(
    [Output('vmin-vmax-container', 'style'),
     Output('vmin-vmax-slider', 'min'),
     Output('vmin-vmax-slider', 'max'),
     Output('vmin-vmax-slider', 'value')],
    [Input('tif-channel-dropdown', 'value'),
     Input('upload-image', 'contents'),
     Input('btn-load-tif-path', 'n_clicks')],
    [State('upload-image', 'filename'),
     State('tif-path-input', 'value'),
     State('tif-downsample-dropdown', 'value')],
    prevent_initial_call=True
)
def reset_vmin_vmax_for_channel(channel, upload_contents, load_clicks,
                                 upload_filename, tif_path, downsample_factor):
    hidden = {'marginBottom': '15px', 'display': 'none'}, dash.no_update, dash.no_update, dash.no_update

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else '.'

    array = None
    if trigger_id == 'upload-image.contents':
        is_tif = bool(upload_filename) and upload_filename.lower().endswith(('.tif', '.tiff'))
        if upload_contents is not None and is_tif:
            try:
                _, content_string = upload_contents.split(',')
                array = _decode_tif_from_b64(content_string)
            except Exception:
                return hidden
    elif trigger_id == 'btn-load-tif-path.n_clicks':
        if tif_path and tif_path.strip():
            try:
                step = max(int(downsample_factor or 1), 1)
                array = _load_tif_path_cached(tif_path.strip(), step)
            except Exception:
                return hidden
    else:
        # Channel dropdown changed on an already-loaded file -- by the time the user can
        # select a different channel, the dropdown's options were already populated from
        # `image-store.data` (which only happens after `update_image` ran), so the global is
        # guaranteed to be populated here; no race in this branch.
        array = _uploaded_tif_array

    if array is None:
        return hidden

    n_channels = array.shape[2]
    ch = min(max(int(channel or 0), 0), n_channels - 1)
    plane = array[:, :, ch].astype(np.float32)
    lo, hi = float(plane.min()), float(plane.max())
    if hi <= lo:
        hi = lo + 1.0
    p_lo, p_hi = (float(v) for v in np.percentile(plane, [1.0, 99.5]))
    return {'marginBottom': '15px', 'display': 'block'}, lo, hi, [p_lo, p_hi]



# ── Clientside callback: figure with grid + well labels + section bounds ──
app.clientside_callback(
    """
    function(imgData, centerPoint, gridSpacingX, gridSpacingY, linkSpacing, offsetX, offsetY, gridOpacity, showLabels, cropTop, cropBottom, cropLeft, cropRight, relayoutData) {
        if (!imgData) {
            return window.dash_clientside.no_update;
        }

        var cx = centerPoint ? centerPoint.x : 0;
        var cy = centerPoint ? centerPoint.y : 0;
        
        var trueOffsetX = cx + offsetX;
        var trueOffsetY = cy + offsetY;

        var b64 = imgData.b64;
        var imgW = imgData.w;
        var imgH = imgData.h;
        var previewW = imgData.pw || imgW;
        var previewH = imgData.ph || imgH;
        var dx = imgW / previewW;
        var dy = imgH / previewH;
        
        var gridColor = 'rgba(0, 255, 255, ' + gridOpacity + ')';
        var isLinked = (linkSpacing || []).includes('link');
        var spacingX = Math.max(gridSpacingX, 1);
        var spacingY = isLinked ? spacingX : Math.max(gridSpacingY, 1);
        var doLabels = showLabels && showLabels.indexOf('show') !== -1;

        // Section crop bounds: restricts grid/wells to this box (never modifies the image)
        var boundX0 = imgW * (cropLeft / 100);
        var boundX1 = imgW * (1 - cropRight / 100);
        var boundY0 = imgH * (cropTop / 100);
        var boundY1 = imgH * (1 - cropBottom / 100);

        var shapes = [];
        var dimColor = 'rgba(0, 0, 0, 0.7)';
        if (boundY0 > 0) {
            shapes.push({type: 'rect', x0: 0, y0: 0, x1: imgW, y1: boundY0, fillcolor: dimColor, line: {width: 0}});
        }
        if (boundY1 < imgH) {
            shapes.push({type: 'rect', x0: 0, y0: boundY1, x1: imgW, y1: imgH, fillcolor: dimColor, line: {width: 0}});
        }
        if (boundX0 > 0) {
            shapes.push({type: 'rect', x0: 0, y0: boundY0, x1: boundX0, y1: boundY1, fillcolor: dimColor, line: {width: 0}});
        }
        if (boundX1 < imgW) {
            shapes.push({type: 'rect', x0: boundX1, y0: boundY0, x1: imgW, y1: boundY1, fillcolor: dimColor, line: {width: 0}});
        }
        // Outline of the active section bounds
        shapes.push({
            type: 'rect', x0: boundX0, y0: boundY0, x1: boundX1, y1: boundY1,
            line: {color: 'rgba(255, 255, 0, 0.6)', width: 1.5}, fillcolor: 'rgba(0,0,0,0)'
        });

        // Compute grid line positions, clipped to the section bounds
        var startX = ((trueOffsetX % spacingX) + spacingX) % spacingX;
        var xPositions = [];
        for (var x = startX; x < imgW; x += spacingX) {
            if (x >= boundX0 && x <= boundX1) {
                xPositions.push(x);
                shapes.push({
                    type: 'line', x0: x, x1: x, y0: boundY0, y1: boundY1,
                    line: {color: gridColor, width: 1.5},
                    editable: false
                });
            }
        }

        var startY = ((trueOffsetY % spacingY) + spacingY) % spacingY;
        var yPositions = [];
        for (var y = startY; y < imgH; y += spacingY) {
            if (y >= boundY0 && y <= boundY1) {
                yPositions.push(y);
                shapes.push({
                    type: 'line', x0: boundX0, x1: boundX1, y0: y, y1: y,
                    line: {color: gridColor, width: 1.5},
                    editable: false
                });
            }
        }

        // Add a visible center point shape
        shapes.push({
            type: 'circle',
            x0: trueOffsetX - 8, y0: trueOffsetY - 8,
            x1: trueOffsetX + 8, y1: trueOffsetY + 8,
            line: {color: 'rgba(255, 50, 50, 0.9)', width: 2},
            fillcolor: 'rgba(255, 255, 255, 0.5)',
            name: 'center-point'
        });

        // Build axis tick labels centered in each box
        var xTickVals = [];
        var xTickText = [];
        var yTickVals = [];
        var yTickText = [];

        function rowLabel(idx) {
            var label = '';
            var i = idx;
            do {
                label = String.fromCharCode(65 + (i % 26)) + label;
                i = Math.floor(i / 26) - 1;
            } while (i >= 0);
            return label;
        }

        if (doLabels) {
            for (var ci = 0; ci < xPositions.length - 1; ci++) {
                xTickVals.push((xPositions[ci] + xPositions[ci + 1]) / 2);
                xTickText.push(String(ci + 1));
            }
            for (var ri = 0; ri < yPositions.length - 1; ri++) {
                yTickVals.push((yPositions[ri] + yPositions[ri + 1]) / 2);
                yTickText.push(rowLabel(ri));
            }
        }

        // No per-well value annotations (fluorescence analysis was removed) -- the figure
        // never draws any.
        var annotations = [];

        // Determine axis ranges (preserve zoom/pan if present)
        var xRange = [0, imgW];
        var yRange = [imgH, 0];
        if (relayoutData) {
            if (relayoutData['xaxis.range[0]'] !== undefined) {
                xRange = [relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']];
                yRange = [relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']];
            }
        }

        var leftMargin = doLabels ? 35 : 0;
        var topMargin = doLabels ? 25 : 0;

        return {
            data: [{
                type: 'image',
                source: b64,
                x0: dx / 2,
                y0: dy / 2,
                dx: dx,
                dy: dy,
                hoverinfo: 'none'
            }],
            layout: {
                shapes: shapes,
                annotations: annotations,
                xaxis: {
                    range: xRange,
                    showgrid: false, zeroline: false,
                    title: '',
                    scaleanchor: 'y',
                    side: 'top',
                    showticklabels: doLabels,
                    tickvals: xTickVals,
                    ticktext: xTickText,
                    tickfont: {color: 'rgba(0,255,255,0.85)', size: 12, family: 'monospace'},
                    ticks: ''
                },
                yaxis: {
                    range: yRange,
                    showgrid: false, zeroline: false,
                    title: '',
                    side: 'left',
                    showticklabels: doLabels,
                    tickvals: yTickVals,
                    ticktext: yTickText,
                    tickfont: {color: 'rgba(0,255,255,0.85)', size: 12, family: 'monospace'},
                    ticks: ''
                },
                margin: {l: leftMargin, r: 0, t: topMargin, b: 0},
                plot_bgcolor: '#000000',
                paper_bgcolor: '#000000',
                uirevision: 'constant',
                dragmode: 'pan'
            }
        };
    }
    """,
    Output('image-graph', 'figure'),
    [Input('image-store', 'data'),
     Input('center-point-store', 'data'),
     Input('grid-spacing-slider', 'value'),
     Input('grid-y-spacing-slider', 'value'),
     Input('link-spacing-check', 'value'),
     Input('grid-x-offset-slider', 'value'),
     Input('grid-y-offset-slider', 'value'),
     Input('grid-opacity-slider', 'value'),
     Input('show-labels-check', 'value'),
     Input('crop-top-slider', 'value'),
     Input('crop-bottom-slider', 'value'),
     Input('crop-left-slider', 'value'),
     Input('crop-right-slider', 'value')],
    [State('image-graph', 'relayoutData')]
)

# ── Server callback: Toggle Y Spacing Visibility ───────────────────────
@app.callback(
    Output('grid-y-spacing-container', 'style'),
    Input('link-spacing-check', 'value')
)
def toggle_y_spacing(link_val):
    base_style = {'marginBottom': '15px'}
    if 'link' in (link_val or []):
        return {**base_style, 'display': 'none'}
    return {**base_style, 'display': 'block'}


# ── Clientside callback: Global Keypress Listener ──────────────────────
app.clientside_callback(
    """
    function(id) {
        if (!window._keydown_listener_v2_added) {
            window._keydown_listener_v2_added = true;
            document.addEventListener('keydown', function(e) {
                if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
                var key = e.key;
                if (['w', 'W', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-'].includes(key)) {
                    if (key.startsWith('Arrow') || key === '+' || key === '-' || key === '=') {
                        e.preventDefault();
                        e.stopPropagation();
                    }
                    
                    var activeId = null;
                    if (document.activeElement) {
                        var sliderParent = document.activeElement.closest('[id$="-slider"]');
                        if (sliderParent) {
                            activeId = sliderParent.id;
                        }
                    }
                    window.dash_clientside.set_props('keypress-store', {data: {key: key, ts: Date.now(), active_id: activeId}});
                }
            }, true);
            
            document.addEventListener('mousedown', function(e) {
                if (e.target && e.target.closest('#image-graph')) {
                    if (document.activeElement && document.activeElement !== document.body) {
                        document.activeElement.blur();
                    }
                }
            }, true);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('dummy-listener', 'children'),
    Input('dummy-listener', 'id')
)

# ── Save grid settings ────────────────────────────────────────────────
# `crop_top/bottom/left/right` here are real, functional section bounds (not cosmetic) —
# they restrict which wells/grid lines are computed everywhere in this app. To align
# multiple tissue sections on one image, set the crop box + grid for one section, save
# settings with a distinct filename, then adjust the crop box + grid for the next
# section and save again (all sections share the same uploaded image/rotation).
@app.callback(
    Output('download-settings', 'data'),
    Input('btn-save-settings', 'n_clicks'),
    [State('rotation-slider', 'value'),
     State('grid-spacing-slider', 'value'),
     State('grid-y-spacing-slider', 'value'),
     State('link-spacing-check', 'value'),
     State('grid-x-offset-slider', 'value'),
     State('grid-y-offset-slider', 'value'),
     State('center-point-store', 'data'),
     State('grid-opacity-slider', 'value'),
     State('show-labels-check', 'value'),
     State('crop-top-slider', 'value'),
     State('crop-bottom-slider', 'value'),
     State('crop-left-slider', 'value'),
     State('crop-right-slider', 'value'),
     State('save-settings-filename', 'value')],
    prevent_initial_call=True
)
def save_settings(n_clicks, rotation, spacing_x, spacing_y, link_spacing, offset_x, offset_y, center_point, opacity, show_labels, c_top, c_bot, c_left, c_right, filename):
    settings = {
        'schema_version': SETTINGS_SCHEMA_VERSION,
        'source': 'tif' if _uploaded_tif_array is not None else 'jpg',
        'rotation': rotation,
        'grid_spacing': spacing_x,
        'grid_y_spacing': spacing_y,
        'link_spacing': link_spacing,
        'grid_x_offset': offset_x,
        'grid_y_offset': offset_y,
        'center_point': center_point or {'x': 0, 'y': 0},
        'grid_opacity': opacity,
        'show_labels': show_labels,
        'crop_top': c_top,
        'crop_bottom': c_bot,
        'crop_left': c_left,
        'crop_right': c_right
    }
    
    if not filename or not filename.strip():
        filename = 'grid_settings.json'
    else:
        filename = filename.strip()
        if not filename.endswith('.json'):
            filename += '.json'
            
    return dcc.send_string(json.dumps(settings, indent=2), filename)


# ── Load grid settings ────────────────────────────────────────────────
@app.callback(
    [Output('rotation-slider', 'value', allow_duplicate=True),
     Output('grid-spacing-slider', 'value', allow_duplicate=True),
     Output('grid-y-spacing-slider', 'value', allow_duplicate=True),
     Output('link-spacing-check', 'value', allow_duplicate=True),
     Output('grid-x-offset-slider', 'value', allow_duplicate=True),
     Output('grid-y-offset-slider', 'value', allow_duplicate=True),
     Output('center-point-store', 'data', allow_duplicate=True),
     Output('center-point-display', 'children', allow_duplicate=True),
     Output('grid-opacity-slider', 'value', allow_duplicate=True),
     Output('show-labels-check', 'value', allow_duplicate=True),
     Output('crop-top-slider', 'value', allow_duplicate=True),
     Output('crop-bottom-slider', 'value', allow_duplicate=True),
     Output('crop-left-slider', 'value', allow_duplicate=True),
     Output('crop-right-slider', 'value', allow_duplicate=True),
     Output('status-text', 'children', allow_duplicate=True)],
    Input('upload-settings', 'contents'),
    prevent_initial_call='initial_duplicate'
)
def load_settings(contents):
    if contents is None:
        raise dash.exceptions.PreventUpdate
    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        s = json.loads(decoded)
        cp = s.get('center_point', {'x': 0.0, 'y': 0.0})

        # Backwards compatibility: legacy files (no/old schema_version) had purely cosmetic
        # crop sliders, so ignore any stored crop values rather than treating them as real
        # bounds. To opt an old file into the new functional-crop behavior, manually add
        # `"schema_version": 2` to its JSON. (Compared against the fixed `CROP_SCHEMA_VERSION`
        # threshold, not `SETTINGS_SCHEMA_VERSION` -- the latter just tracks the current format
        # this app writes and has since moved on to 3 for the "source" field.)
        schema_version = s.get('schema_version', 1)
        if schema_version >= CROP_SCHEMA_VERSION:
            crop_top = s.get('crop_top', 0)
            crop_bottom = s.get('crop_bottom', 0)
            crop_left = s.get('crop_left', 0)
            crop_right = s.get('crop_right', 0)
            status = '✅ Settings loaded successfully'
        else:
            crop_top = crop_bottom = crop_left = crop_right = 0
            status = '✅ Settings loaded (legacy file — crop ignored; add "schema_version": 2 to the JSON to use its crop values as real bounds)'

        return (
            s.get('rotation', 0),
            s.get('grid_spacing', 229),
            s.get('grid_y_spacing', 229),
            s.get('link_spacing', ['link']),
            s.get('grid_x_offset', 0),
            s.get('grid_y_offset', 0),
            cp,
            f"Center Point: ({cp.get('x', 0)}, {cp.get('y', 0)})",
            s.get('grid_opacity', 0.7),
            s.get('show_labels', ['show']),
            crop_top,
            crop_bottom,
            crop_left,
            crop_right,
            status
        )
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, \
               dash.no_update, dash.no_update, dash.no_update, \
               dash.no_update, dash.no_update, \
               dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
               dash.no_update, dash.no_update, \
               f'❌ Error loading settings: {str(e)}'

# ── Server callback: Place Center Point ─────────────────────────────────
@app.callback(
    [Output('center-point-store', 'data', allow_duplicate=True),
     Output('center-point-display', 'children', allow_duplicate=True),
     Output('grid-x-offset-slider', 'value', allow_duplicate=True),
     Output('grid-y-offset-slider', 'value', allow_duplicate=True),
     Output('placement-mode', 'data', allow_duplicate=True),
     Output('placement-status', 'children')],
    [Input('image-graph', 'clickData'),
     Input('btn-place-center', 'n_clicks')],
    State('placement-mode', 'data'),
    prevent_initial_call=True
)
def update_offsets_from_click(clickData, btn_clicks, placement_mode):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
        
    trigger_id = ctx.triggered[0]['prop_id']
    
    # If the user clicked the "Place Center Point" button
    if 'btn-place-center' in trigger_id:
        if placement_mode:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, False, ''
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, True, 'Select a point on the image...'
    
    # If the user clicked somewhere on the image trace
    if 'clickData' in trigger_id and clickData:
        if placement_mode:
            try:
                pt = clickData['points'][0]
                cx, cy = round(pt['x'], 1), round(pt['y'], 1)
                return {'x': cx, 'y': cy}, f'Center Point: ({cx}, {cy})', 0, 0, False, ''
            except (KeyError, IndexError):
                pass

    raise dash.exceptions.PreventUpdate


# ── Server callback: Handle Keyboard Shortcuts ──────────────────────────
@app.callback(
    [Output('grid-x-offset-slider', 'value', allow_duplicate=True),
     Output('grid-y-offset-slider', 'value', allow_duplicate=True),
     Output('rotation-slider', 'value', allow_duplicate=True),
     Output('grid-spacing-slider', 'value', allow_duplicate=True),
     Output('grid-opacity-slider', 'value', allow_duplicate=True),
     Output('placement-mode', 'data', allow_duplicate=True),
     Output('placement-status', 'children', allow_duplicate=True),
     Output('grid-y-spacing-slider', 'value', allow_duplicate=True)],
    Input('keypress-store', 'data'),
    [State('grid-x-offset-slider', 'value'),
     State('grid-y-offset-slider', 'value'),
     State('rotation-slider', 'value'),
     State('grid-spacing-slider', 'value'),
     State('grid-opacity-slider', 'value'),
     State('placement-mode', 'data'),
     State('grid-y-spacing-slider', 'value')],
    prevent_initial_call=True
)
def handle_keypress(key_data, x_val, y_val, rot_val, space_val, op_val, placement_mode, y_space_val):
    if not key_data:
        raise dash.exceptions.PreventUpdate
        
    key = key_data.get('key')
    active_id = key_data.get('active_id')
    
    if key in ['w', 'W']:
        if placement_mode:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, False, '', dash.no_update
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, True, 'Select a point on the image...', dash.no_update
            
    if key.startswith('Arrow'):
        if active_id:
            # If a slider is focused, arrow keys adjust that slider
            direction = 1 if key in ['ArrowRight', 'ArrowUp'] else -1
            slider_step = 0.1 if active_id in ['grid-opacity-slider', 'rotation-slider'] else 1.0
            increment = slider_step * direction
            
            if active_id == 'grid-x-offset-slider':
                return x_val + increment, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            elif active_id == 'grid-y-offset-slider':
                return dash.no_update, y_val + increment, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            elif active_id == 'rotation-slider':
                return dash.no_update, dash.no_update, rot_val + increment, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            elif active_id == 'grid-spacing-slider':
                return dash.no_update, dash.no_update, dash.no_update, space_val + increment, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            elif active_id == 'grid-y-spacing-slider':
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, y_space_val + increment
            elif active_id == 'grid-opacity-slider':
                val = max(0.0, min(1.0, op_val + increment))
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, val, dash.no_update, dash.no_update, dash.no_update
        else:
            # If no slider is focused, arrow keys adjust the grid offset natively
            step = 1.0
            if key == 'ArrowLeft':
                return x_val - step, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            elif key == 'ArrowRight':
                return x_val + step, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            elif key == 'ArrowUp':
                return dash.no_update, y_val - step, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            elif key == 'ArrowDown':
                return dash.no_update, y_val + step, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
                
    elif key in ['+', '=', '-']:
        direction = 1 if key in ['+', '='] else -1
        increment = 0.1 * direction
        return dash.no_update, dash.no_update, rot_val + increment, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    raise dash.exceptions.PreventUpdate






if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    app.run(debug=True, port=port, dev_tools_hot_reload=False)
