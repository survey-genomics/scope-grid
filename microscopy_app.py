import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont, ImageOps
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import requests
import tifffile
import hashlib
import traceback
import time
from io import BytesIO
import base64
import json
import string

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


def _normalize_to_uint8(plane, low_pct=1.0, high_pct=99.5):
    """Percentile-stretch a single channel plane (any dtype) to a viewable uint8 grayscale
    image. Microscopy TIFs are commonly 16-bit with a handful of hot outlier pixels, so a
    plain min/max stretch tends to wash out everything else -- percentile clipping keeps the
    preview usable for alignment purposes.
    """
    plane = plane.astype(np.float32)
    lo, hi = np.percentile(plane, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros(plane.shape, dtype=np.uint8)
    stretched = np.clip((plane - lo) / (hi - lo), 0, 1) * 255
    return stretched.astype(np.uint8)


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


def _row_label(idx):
    """Convert row index to letter label: 0->A, 1->B, ..., 25->Z, 26->AA..."""
    label = ''
    i = idx
    while True:
        label = string.ascii_uppercase[i % 26] + label
        i = i // 26 - 1
        if i < 0:
            break
    return label


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
    dcc.Store(id='fluor-store'),
    dcc.Download(id='download-image'),
    dcc.Download(id='download-grid'),
    dcc.Download(id='download-merged'),
    dcc.Download(id='download-settings'),
    dcc.Download(id='download-crop'),
    dcc.Download(id='download-csv'),

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
                multiple=False,
                accept='image/*,.tif,.tiff,.jpg,.jpeg,.png'
            )
        ]),

        # ── TIF channel selector (only relevant/shown for TIF uploads) ─
        html.Div(id='tif-channel-container', children=[
            html.Label("Channel", style=_label_style),
            dcc.Input(
                id='tif-channel-input', type='number', min=0, step=1, value=0,
                style={'width': '100%', 'backgroundColor': '#333', 'color': '#ddd',
                       'border': '1px solid #555', 'borderRadius': '4px', 'padding': '6px'}
            ),
            html.Div(id='tif-channel-hint', style={'color': '#888', 'fontFamily': 'sans-serif',
                                                     'fontSize': '0.75em', 'marginTop': '3px'})
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
            "Restricts the grid/wells/fluorescence/exports to this region. Does not modify "
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

        # ── Fluorescence ───────────────────────────────────────────
        html.Hr(style={'borderColor': '#444', 'margin': '12px 0'}),
        html.Label("Fluorescence Analysis", style={
            'color': '#ffffff', 'fontFamily': 'sans-serif',
            'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'
        }),

        html.Div([
            html.Label("Channel", style=_label_style),
            dcc.Dropdown(
                id='fluor-channel',
                options=[
                    {'label': '── Raw Channels ──', 'value': '_sep1', 'disabled': True},
                    {'label': 'Grayscale (mean)', 'value': 'mean'},
                    {'label': 'Red (R)', 'value': 'r'},
                    {'label': 'Green (G)', 'value': 'g'},
                    {'label': 'Blue (B)', 'value': 'b'},
                    {'label': 'Cyan (G+B)', 'value': 'cyan'},
                    {'label': 'Magenta (R+B)', 'value': 'magenta'},
                    {'label': 'Yellow (R+G)', 'value': 'yellow'},
                    {'label': '── Fluorophores (Echo) ──', 'value': '_sep2', 'disabled': True},
                    {'label': 'DAPI (Blue)', 'value': 'dapi'},
                    {'label': 'FITC / GFP (Green)', 'value': 'fitc'},
                    {'label': 'TRITC / Texas Red (Red)', 'value': 'tritc'},
                    {'label': 'Cy5 (Far Red)', 'value': 'cy5'},
                    {'label': 'mCherry / RFP (Red)', 'value': 'mcherry'},
                    {'label': 'CFP (Cyan)', 'value': 'cfp'},
                    {'label': 'YFP (Yellow-Green)', 'value': 'yfp'},
                    {'label': '── Brightfield ──', 'value': '_sep3', 'disabled': True},
                    {'label': 'Brightfield (luminance)', 'value': 'brightfield'},
                ],
                value='mean',
                style={'backgroundColor': '#333', 'color': '#ddd', 'marginBottom': '8px'},
                className='dark-dropdown'
            ),
        ], style={'marginBottom': '8px'}),

        html.Button("🔬  Compute Fluorescence", id='btn-compute-fluor', n_clicks=0, style=_btn_style),

        html.Div([
            dcc.Checklist(
                id='show-fluor-check',
                options=[{'label': ' Show values on image', 'value': 'show'}],
                value=['show'],
                labelStyle={'color': '#bbbbbb', 'display': 'inline-block', 'marginLeft': '5px'},
                style={'color': '#bbbbbb', 'fontFamily': 'sans-serif', 'marginTop': '3px'}
            )
        ], style={'marginBottom': '8px'}),

        html.Button("📊  Export Matrix CSV", id='btn-save-csv', n_clicks=0, style=_btn_style),

        # ── Crop Well ──────────────────────────────────────────────
        html.Hr(style={'borderColor': '#444', 'margin': '12px 0'}),
        html.Label("Crop Well", style={
            'color': '#ffffff', 'fontFamily': 'sans-serif',
            'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'
        }),

        html.Div([
            html.Label("Select Well", style=_label_style),
            dcc.Dropdown(
                id='crop-well-dropdown',
                options=[],
                placeholder='Select a well (e.g. A1)',
                style={'backgroundColor': '#333', 'color': '#ddd', 'marginBottom': '8px'},
                className='dark-dropdown'
            ),
        ], style={'marginBottom': '8px'}),

        html.Button("✂️  Crop & Download Well", id='btn-crop-well', n_clicks=0, style=_btn_style),

        # ── Export & Settings ──────────────────────────────────────
        html.Hr(style={'borderColor': '#444', 'margin': '12px 0'}),
        html.Label("Export & Settings", style={
            'color': '#ffffff', 'fontFamily': 'sans-serif',
            'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'
        }),

        html.Button("💾  Save Image Only", id='btn-save-image', n_clicks=0, style=_btn_style),
        html.Button("🔲  Save Grid Only", id='btn-save-grid', n_clicks=0, style=_btn_style),
        html.Button("📸  Save Merged (Image + Grid)", id='btn-save-merged', n_clicks=0, style=_btn_style),

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

        # ── Matrix Heatmap Container ────────────────────────────────
        html.Hr(style={'borderColor': '#444', 'margin': '12px 0'}),
        html.Label("Fluorescence Matrix Heatmap", style={
            'color': '#ffffff', 'fontFamily': 'sans-serif',
            'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'
        }),
        dcc.Graph(
            id='matrix-graph',
            style={'height': '220px', 'width': '100%'},
            config={'displayModeBar': False}
        ),

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


# ── Server callback: image processor (upload, rotate) ──────────────────
@app.callback(
    Output('image-store', 'data'),
    [Input('upload-image', 'contents'),
     Input('rotation-slider', 'value'),
     Input('tif-channel-input', 'value')],
    State('upload-image', 'filename'),
    prevent_initial_call='initial_duplicate'
)
def update_image(upload_contents, rotation, channel, upload_filename):
    global _uploaded_image, _uploaded_tif_array

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else '.'
    print(f"[update_image] triggered by {trigger_id!r} filename={upload_filename!r} "
          f"rotation={rotation} channel={channel} "
          f"contents_received={upload_contents is not None}", flush=True)

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
                print(f"[update_image] upload decoded OK (is_tif={is_tif})", flush=True)
            except Exception:
                print(f"[update_image] FAILED to decode upload {upload_filename!r}:", flush=True)
                traceback.print_exc()
        else:
            print("[update_image] upload cleared", flush=True)
            _uploaded_image = None
            _uploaded_tif_array = None

    # TIFs may hold more channels than can be shown at once -- render whichever one the
    # "Channel" control currently selects (clamped to a valid index) as a grayscale image, and
    # store it in `_uploaded_image` so every other callback in this app (which all just read
    # that global directly) keeps working unchanged.
    n_channels = 1
    source = 'jpg'
    if _uploaded_tif_array is not None:
        n_channels = _uploaded_tif_array.shape[2]
        ch = min(max(int(channel or 0), 0), n_channels - 1)
        plane = _normalize_to_uint8(_uploaded_tif_array[:, :, ch])
        _uploaded_image = Image.fromarray(plane, mode='L')
        source = 'tif'
        print(f"[update_image] displaying TIF channel {ch}/{n_channels - 1}, "
              f"array shape={_uploaded_tif_array.shape}", flush=True)

    # Always generate rotated data
    current = _uploaded_image if _uploaded_image is not None else original_image
    data = _get_rotated_data(current, rotation)
    print(f"[update_image] returning image-store data: source={source} w={data['w']} "
          f"h={data['h']}", flush=True)
    return {'b64': data['b64'], 'w': data['w'], 'h': data['h'], 'pw': data['pw'], 'ph': data['ph'],
            'source': source, 'n_channels': n_channels}


# ── One-directional: toggle/label the Channel control based on the active upload's source ──
@app.callback(
    [Output('tif-channel-container', 'style'),
     Output('tif-channel-hint', 'children')],
    Input('image-store', 'data')
)
def update_channel_ui(img_data):
    if not img_data or img_data.get('source') != 'tif':
        return {'marginBottom': '15px', 'display': 'none'}, ''
    n_channels = img_data.get('n_channels', 1)
    return (
        {'marginBottom': '15px', 'display': 'block'},
        f'Valid range: 0-{n_channels - 1} ({n_channels} channel{"s" if n_channels != 1 else ""})'
    )



# ── Clientside callback: figure with grid + well labels + fluorescence + section bounds ──
app.clientside_callback(
    """
    function(imgData, centerPoint, gridSpacingX, gridSpacingY, linkSpacing, offsetX, offsetY, gridOpacity, showLabels, flourData, showFluor, cropTop, cropBottom, cropLeft, cropRight, relayoutData) {
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
        var doFluor = showFluor && showFluor.indexOf('show') !== -1 && flourData && flourData.values;

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

        // Fluorescence value annotations inside each well
        var annotations = [];
        if (doFluor) {
            var fVals = flourData.values;
            var fRows = flourData.n_rows;
            var fCols = flourData.n_cols;
            for (var frow = 0; frow < fRows; frow++) {
                for (var fcol = 0; fcol < fCols; fcol++) {
                    if (frow < yPositions.length - 1 && fcol < xPositions.length - 1) {
                        var fx = (xPositions[fcol] + xPositions[fcol + 1]) / 2;
                        var fy = (yPositions[frow] + yPositions[frow + 1]) / 2;
                        var val = fVals[frow][fcol];
                        var fontSize = Math.min(Math.max(Math.min(spacingX, spacingY) * 0.12, 8), 14);
                        annotations.push({
                            x: fx, y: fy,
                            text: val.toFixed(1),
                            showarrow: false,
                            font: {color: '#ffff00', size: fontSize, family: 'monospace'},
                            xref: 'x', yref: 'y',
                            opacity: 0.9,
                            bgcolor: 'rgba(0,0,0,0.5)',
                            borderpad: 2
                        });
                    }
                }
            }
        }

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
     Input('fluor-store', 'data'),
     Input('show-fluor-check', 'value'),
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

# ── Helper: compute grid positions ─────────────────────────────────────
def _crop_bounds(w, h, crop_top, crop_bottom, crop_left, crop_right):
    """Pixel-space (x0, y0, x1, y1) bounds for the section crop.

    Unlike the old "Apply Crop" button, this never modifies image pixels — it only
    restricts which grid lines/wells are computed/exported, so a single uploaded image
    can be divided into multiple independently-aligned sections (one settings file each).
    """
    x0 = w * (crop_left / 100.0)
    x1 = w * (1 - crop_right / 100.0)
    y0 = h * (crop_top / 100.0)
    y1 = h * (1 - crop_bottom / 100.0)
    return x0, y0, x1, y1


def _grid_positions(spacing, offset_x, offset_y, w, h, bounds=None):
    """Return lists of x and y grid line positions, optionally clipped to `bounds` (x0, y0, x1, y1)."""
    x_pos = []
    sx = offset_x % spacing
    x = sx
    while x < w:
        x_pos.append(x)
        x += spacing
    y_pos = []
    sy = offset_y % spacing
    y = sy
    while y < h:
        y_pos.append(y)
        y += spacing
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        x_pos = [x for x in x_pos if x0 <= x <= x1]
        y_pos = [y for y in y_pos if y0 <= y <= y1]
    return x_pos, y_pos


# ── Compute fluorescence per well ──────────────────────────────────────
@app.callback(
    Output('fluor-store', 'data'),
    Input('btn-compute-fluor', 'n_clicks'),
    [State('rotation-slider', 'value'),
     State('grid-spacing-slider', 'value'),
     State('grid-x-offset-slider', 'value'),
     State('grid-y-offset-slider', 'value'),
     State('center-point-store', 'data'),
     State('fluor-channel', 'value'),
     State('crop-top-slider', 'value'),
     State('crop-bottom-slider', 'value'),
     State('crop-left-slider', 'value'),
     State('crop-right-slider', 'value')],
    prevent_initial_call=True
)
def compute_fluorescence(n_clicks, rotation, spacing, offset_x, offset_y, center_point, channel,
                          c_top, c_bot, c_left, c_right):
    cx = center_point.get('x', 0) if center_point else 0
    cy = center_point.get('y', 0) if center_point else 0
    true_offset_x = cx + offset_x
    true_offset_y = cy + offset_y

    current = _uploaded_image if _uploaded_image is not None else original_image
    rotated = _get_rotated_pil(current, rotation)
    arr = np.array(rotated)
    w, h = rotated.size
    bounds = _crop_bounds(w, h, c_top, c_bot, c_left, c_right)
    x_pos, y_pos = _grid_positions(spacing, true_offset_x, true_offset_y, w, h, bounds=bounds)

    n_rows = max(0, len(y_pos) - 1)
    n_cols = max(0, len(x_pos) - 1)

    if arr.ndim == 2:
        # Grayscale (a single TIF channel, selected via the "Channel" control) -- there's only
        # one plane of data, so the RGB-fluorophore channel mapping below doesn't apply; just
        # use it directly regardless of the `channel` dropdown's selection.
        extracted_arr = arr.astype(float)
    else:
        # Channel mapping: map fluorophore names to RGB extraction logic
        channel_map = {
            'r': lambda c: c[:, :, 0].astype(float),
            'g': lambda c: c[:, :, 1].astype(float),
            'b': lambda c: c[:, :, 2].astype(float),
            'cyan': lambda c: (c[:, :, 1].astype(float) + c[:, :, 2].astype(float)) / 2,
            'magenta': lambda c: (c[:, :, 0].astype(float) + c[:, :, 2].astype(float)) / 2,
            'yellow': lambda c: (c[:, :, 0].astype(float) + c[:, :, 1].astype(float)) / 2,
            'dapi': lambda c: c[:, :, 2].astype(float),
            'fitc': lambda c: c[:, :, 1].astype(float),
            'tritc': lambda c: c[:, :, 0].astype(float),
            'cy5': lambda c: (c[:, :, 0].astype(float) * 0.8 + c[:, :, 2].astype(float) * 0.2),
            'mcherry': lambda c: c[:, :, 0].astype(float),
            'cfp': lambda c: (c[:, :, 1].astype(float) + c[:, :, 2].astype(float)) / 2,
            'yfp': lambda c: (c[:, :, 0].astype(float) * 0.3 + c[:, :, 1].astype(float) * 0.7),
            'brightfield': lambda c: (0.2126 * c[:, :, 0].astype(float) +
                                       0.7152 * c[:, :, 1].astype(float) +
                                       0.0722 * c[:, :, 2].astype(float)),
            'mean': lambda c: c.astype(float).mean(axis=2),
        }
        extract = channel_map.get(channel, channel_map['mean'])
        extracted_arr = extract(arr)

    values = []
    for r in range(n_rows):
        row_vals = []
        for c in range(n_cols):
            y0 = int(round(y_pos[r]))
            y1 = int(round(y_pos[r + 1]))
            x0 = int(round(x_pos[c]))
            x1 = int(round(x_pos[c + 1]))
            cell = extracted_arr[y0:y1, x0:x1]
            if cell.size == 0:
                row_vals.append(0.0)
            else:
                row_vals.append(float(cell.mean()))
        values.append(row_vals)

    return {'values': values, 'n_rows': n_rows, 'n_cols': n_cols, 'channel': channel}


# ── Render Matrix Heatmap ──────────────────────────────────────────────
@app.callback(
    Output('matrix-graph', 'figure'),
    Input('fluor-store', 'data')
)
def update_matrix_graph(fluor_data):
    if not fluor_data or not fluor_data.get('values'):
        # Empty placeholder figure
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e',
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[{
                'text': 'Click "Compute Fluorescence" to generate matrix',
                'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.5,
                'showarrow': False, 'font': {'color': '#777', 'size': 11}
            }],
            margin={'l': 10, 'r': 10, 't': 10, 'b': 10}
        )
        return fig

    values = np.array(fluor_data['values'])
    n_rows = fluor_data['n_rows']
    n_cols = fluor_data['n_cols']

    y_labels = [_row_label(r) for r in range(n_rows)]
    x_labels = [str(c + 1) for c in range(n_cols)]

    # Dynamic colorscale based on channel type
    ch = fluor_data.get('channel', 'mean')
    colorscale_map = {
        'r': 'Reds', 'tritc': 'Reds', 'mcherry': 'Reds',
        'g': 'Greens', 'fitc': 'Greens', 'yfp': 'YlGn',
        'b': 'Blues', 'dapi': 'Blues',
        'cyan': 'Ice', 'cfp': 'Ice',
        'magenta': 'Purples', 'cy5': 'Plasma'
    }
    cs = colorscale_map.get(ch, 'Viridis')

    text_vals = [[f"{v:.1f}" for v in row] for row in values]

    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=x_labels,
        y=y_labels,
        colorscale=cs,
        text=text_vals,
        texttemplate="%{text}",
        textfont={"size": 9, "color": "white"},
        hoverinfo="x+y+z",
        showscale=False
    ))

    fig.update_layout(
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        xaxis={'title': 'Column', 'side': 'top', 'tickfont': {'color': '#ccc', 'size': 9}, 'titlefont': {'color': '#aaa', 'size': 10}},
        yaxis={'title': 'Row', 'autorange': 'reversed', 'tickfont': {'color': '#ccc', 'size': 9}, 'titlefont': {'color': '#aaa', 'size': 10}},
        margin={'l': 30, 'r': 10, 't': 35, 'b': 20}
    )
    return fig


# ── Update well dropdown options ───────────────────────────────────────
@app.callback(
    Output('crop-well-dropdown', 'options'),
    [Input('image-store', 'data'),
     Input('grid-spacing-slider', 'value'),
     Input('grid-x-offset-slider', 'value'),
     Input('grid-y-offset-slider', 'value'),
     Input('center-point-store', 'data'),
     Input('crop-top-slider', 'value'),
     Input('crop-bottom-slider', 'value'),
     Input('crop-left-slider', 'value'),
     Input('crop-right-slider', 'value')]
)
def update_well_options(img_data, spacing, offset_x, offset_y, center_point, c_top, c_bot, c_left, c_right):
    if not img_data:
        return []
    w, h = img_data['w'], img_data['h']
    cx = center_point.get('x', 0) if center_point else 0
    cy = center_point.get('y', 0) if center_point else 0
    bounds = _crop_bounds(w, h, c_top, c_bot, c_left, c_right)
    x_pos, y_pos = _grid_positions(spacing, cx + offset_x, cy + offset_y, w, h, bounds=bounds)
    n_rows = max(0, len(y_pos) - 1)
    n_cols = max(0, len(x_pos) - 1)
    
    # Prevent OOM crashes by limiting the maximum number of generated options
    max_options = 1000
    options = []
    
    for r in range(n_rows):
        for c in range(n_cols):
            if len(options) >= max_options:
                return options
            label = _row_label(r) + str(c + 1)
            options.append({'label': label, 'value': f'{r},{c}'})
            
    return options


# ── Crop & download a single well ──────────────────────────────────────
@app.callback(
    Output('download-crop', 'data'),
    Input('btn-crop-well', 'n_clicks'),
    [State('crop-well-dropdown', 'value'),
     State('rotation-slider', 'value'),
     State('grid-spacing-slider', 'value'),
     State('grid-x-offset-slider', 'value'),
     State('grid-y-offset-slider', 'value'),
     State('center-point-store', 'data'),
     State('crop-top-slider', 'value'),
     State('crop-bottom-slider', 'value'),
     State('crop-left-slider', 'value'),
     State('crop-right-slider', 'value')],
    prevent_initial_call=True
)
def crop_well(n_clicks, well_value, rotation, spacing, offset_x, offset_y, center_point,
              c_top, c_bot, c_left, c_right):
    if not well_value:
        raise dash.exceptions.PreventUpdate
    r, c = [int(v) for v in well_value.split(',')]
    current = _uploaded_image if _uploaded_image is not None else original_image
    rotated = _get_rotated_pil(current, rotation)
    w, h = rotated.size
    cx = center_point.get('x', 0) if center_point else 0
    cy = center_point.get('y', 0) if center_point else 0
    bounds = _crop_bounds(w, h, c_top, c_bot, c_left, c_right)
    x_pos, y_pos = _grid_positions(spacing, cx + offset_x, cy + offset_y, w, h, bounds=bounds)

    x0 = int(round(x_pos[c]))
    x1 = int(round(x_pos[c + 1]))
    y0 = int(round(y_pos[r]))
    y1 = int(round(y_pos[r + 1]))
    cropped = rotated.crop((x0, y0, x1, y1))

    well_name = _row_label(r) + str(c + 1)
    buf = BytesIO()
    cropped.save(buf, format='PNG')
    buf.seek(0)
    return dcc.send_bytes(buf.getvalue(), f'well_{well_name}.png')


# ── Save image only ───────────────────────────────────────────────────
@app.callback(
    Output('download-image', 'data'),
    Input('btn-save-image', 'n_clicks'),
    [State('rotation-slider', 'value')],
    prevent_initial_call=True
)
def save_image(n_clicks, rotation):
    current = _uploaded_image if _uploaded_image is not None else original_image
    rotated = _get_rotated_pil(current, rotation)
    buf = BytesIO()
    rotated.save(buf, format='PNG')
    buf.seek(0)
    return dcc.send_bytes(buf.getvalue(), 'microscopy_image.png')


# ── Save grid only (transparent background) ───────────────────────────
@app.callback(
    Output('download-grid', 'data'),
    Input('btn-save-grid', 'n_clicks'),
    [State('rotation-slider', 'value'),
     State('grid-spacing-slider', 'value'),
     State('grid-x-offset-slider', 'value'),
     State('grid-y-offset-slider', 'value'),
     State('center-point-store', 'data'),
     State('grid-opacity-slider', 'value'),
     State('show-labels-check', 'value'),
     State('crop-top-slider', 'value'),
     State('crop-bottom-slider', 'value'),
     State('crop-left-slider', 'value'),
     State('crop-right-slider', 'value')],
    prevent_initial_call=True
)
def save_grid(n_clicks, rotation, spacing, offset_x, offset_y, center_point, opacity, show_labels,
              c_top, c_bot, c_left, c_right):
    current = _uploaded_image if _uploaded_image is not None else original_image
    rotated = _get_rotated_pil(current, rotation)
    w, h = rotated.size
    cx = center_point.get('x', 0) if center_point else 0
    cy = center_point.get('y', 0) if center_point else 0
    bounds = _crop_bounds(w, h, c_top, c_bot, c_left, c_right)
    grid_img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grid_img)
    alpha = int(opacity * 255)
    color = (0, 255, 255, alpha)

    col_positions, row_positions = _grid_positions(spacing, cx + offset_x, cy + offset_y, w, h, bounds=bounds)
    for x in col_positions:
        draw.line([(x, 0), (x, h)], fill=color, width=2)
    for y in row_positions:
        draw.line([(0, y), (w, y)], fill=color, width=2)

    if show_labels and 'show' in show_labels:
        font_size = min(max(int(spacing * 0.15), 8), 16)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
        for r in range(len(row_positions) - 1):
            for c in range(len(col_positions) - 1):
                cx = (col_positions[c] + col_positions[c + 1]) / 2
                cy = (row_positions[r] + row_positions[r + 1]) / 2
                label = _row_label(r) + str(c + 1)
                draw.text((cx, cy), label, fill=color, font=font, anchor='mm')

    buf = BytesIO()
    grid_img.save(buf, format='PNG')
    buf.seek(0)
    return dcc.send_bytes(buf.getvalue(), 'microscopy_grid.png')


# ── Save merged (image + grid overlay + fluorescence) ─────────────────
@app.callback(
    Output('download-merged', 'data'),
    Input('btn-save-merged', 'n_clicks'),
    [State('rotation-slider', 'value'),
     State('grid-spacing-slider', 'value'),
     State('grid-x-offset-slider', 'value'),
     State('grid-y-offset-slider', 'value'),
     State('center-point-store', 'data'),
     State('grid-opacity-slider', 'value'),
     State('show-labels-check', 'value'),
     State('fluor-store', 'data'),
     State('show-fluor-check', 'value'),
     State('crop-top-slider', 'value'),
     State('crop-bottom-slider', 'value'),
     State('crop-left-slider', 'value'),
     State('crop-right-slider', 'value')],
    prevent_initial_call=True
)
def save_merged(n_clicks, rotation, spacing, offset_x, offset_y, center_point, opacity, show_labels,
                 fluor_data, show_fluor, c_top, c_bot, c_left, c_right):
    current = _uploaded_image if _uploaded_image is not None else original_image
    rotated = _get_rotated_pil(current, rotation)
    w, h = rotated.size
    cx = center_point.get('x', 0) if center_point else 0
    cy = center_point.get('y', 0) if center_point else 0
    bounds = _crop_bounds(w, h, c_top, c_bot, c_left, c_right)

    # Draw grid on RGBA overlay
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = int(opacity * 255)
    color = (0, 255, 255, alpha)

    col_positions, row_positions = _grid_positions(spacing, cx + offset_x, cy + offset_y, w, h, bounds=bounds)
    for x in col_positions:
        draw.line([(x, 0), (x, h)], fill=color, width=2)
    for y in row_positions:
        draw.line([(0, y), (w, y)], fill=color, width=2)

    if show_labels and 'show' in show_labels:
        font_size = min(max(int(spacing * 0.15), 8), 16)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
        for r in range(len(row_positions) - 1):
            for c in range(len(col_positions) - 1):
                cx = (col_positions[c] + col_positions[c + 1]) / 2
                cy = (row_positions[r] + row_positions[r + 1]) / 2
                label = _row_label(r) + str(c + 1)
                draw.text((cx, cy), label, fill=color, font=font, anchor='mm')

    # Draw fluorescence values if computed and checked
    if show_fluor and 'show' in show_fluor and fluor_data and fluor_data.get('values'):
        vals = fluor_data['values']
        font_size = min(max(int(spacing * 0.14), 8), 14)
        try:
            f_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except Exception:
            try:
                f_font = ImageFont.truetype("Arial.ttf", font_size)
            except Exception:
                f_font = ImageFont.load_default()

        yellow_color = (255, 255, 0, 230)
        for r in range(min(len(vals), len(row_positions) - 1)):
            for c in range(min(len(vals[r]), len(col_positions) - 1)):
                cx = (col_positions[c] + col_positions[c + 1]) / 2
                cy = (row_positions[r] + row_positions[r + 1]) / 2 + (spacing * 0.15 if show_labels and 'show' in show_labels else 0)
                txt = f"{vals[r][c]:.1f}"
                draw.text((cx, cy), txt, fill=yellow_color, font=f_font, anchor='mm')

    # Composite
    merged = rotated.convert('RGBA')
    merged = Image.alpha_composite(merged, overlay)
    merged = merged.convert('RGB')

    buf = BytesIO()
    merged.save(buf, format='PNG')
    buf.seek(0)
    return dcc.send_bytes(buf.getvalue(), 'microscopy_merged.png')


# ── Save Fluorescence Matrix CSV ───────────────────────────────────────
@app.callback(
    Output('download-csv', 'data'),
    Input('btn-save-csv', 'n_clicks'),
    State('fluor-store', 'data'),
    prevent_initial_call=True
)
def save_csv(n_clicks, fluor_data):
    if not fluor_data or not fluor_data.get('values'):
        raise dash.exceptions.PreventUpdate

    values = fluor_data['values']
    n_rows = fluor_data['n_rows']
    n_cols = fluor_data['n_cols']
    channel = fluor_data.get('channel', 'mean')

    header = ['Row/Col'] + [str(c + 1) for c in range(n_cols)]
    rows = [header]
    for r in range(n_rows):
        row_str = [_row_label(r)] + [f"{values[r][c]:.3f}" for c in range(n_cols)]
        rows.append(row_str)

    content = '\n'.join([','.join(row) for row in rows])
    return dcc.send_string(content, f'fluorescence_matrix_{channel}.csv')


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
