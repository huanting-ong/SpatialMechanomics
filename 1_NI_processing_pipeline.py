"""
1_NI_processing_pipeline.py

Adapted from the AFM analysis code of Biswas et al., Nat Commun 16, 9578
(2025), Zenodo https://doi.org/10.5281/zenodo.17174104, modified for
Optics11 Chiaro nanoindentation input.

Fits raw nanoindentation force-displacement curves for one slice at a time and
returns the mechanical parameters for every indentation point.

Parameters
    mu              0.49        Poisson ratio, near-incompressible tissue
    rb              27.5        bead radius in um
    PMAX_NO_INDENT  -0.001      max load treated as a failed indentation
    GAIN_THRESHOLD  1.655       contact detection; ratio of a straight-line
                                fit to a baseline-plus-Hertz fit
    PERR_MAX        0.15        relative standard error of the fitted
                                amplitude. Deliberately looser than the
                                conventional 0.05, which scales inversely with
                                modulus and would discard soft tissue
                                preferentially
    VALID range     7.6 to 22,726 Pa   moduli outside this are set to NaN
    cover           'r'         'r' recompute all, 'o' skip points already
                                processed
    COORD_SOURCE    'filename'  where grid coordinates are read from

Actions required
    Place this script in the project root, with one folder per slice directly
    beneath it. The script takes its own location as the root.
    Run once per slice. The slice is chosen by index:
        py 1_NI_processing_pipeline.py 0
    Index 0 is the first slice folder in alphabetical order. Omit the argument
    to process index 0.

Input
    <root>/<slice>/            raw .txt files with time, load, indentation,
                               XYZ position and P[max] columns
Output
    <root>/<slice>/output/
        NI_<slice>_all_samples.csv      one row per indentation point, all
                                        mechanical parameters. The input to
                                        scripts 2, 3, 4, 5 and 6
        correlations.csv                pairwise parameter correlations
        elasticity_l_above_max.csv      points excluded as too stiff
        elasticity_l_below_min.csv      points excluded as too soft
    <root>/<slice>/output/<point>/
        result.csv                      per-point fitted values
        HertzFitting.jpg                fit over the first 2 um
        app-ret cycle.jpg               approach and retraction curves
        curve_holding_removed.jpg       curve after the hold segment is cut
        correction, log-log curve.jpg   baseline correction check
    <root>/<slice>/output/viscosity/
        <point>.jpg                     Maxwellian viscosity fit per point
"""

import re
from re import sub
import warnings
warnings.simplefilter("ignore")
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import os, tempfile, io
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), f"mpl_{os.getpid()}")

from scipy.optimize import curve_fit as fit
from scipy.signal import savgol_filter as svft
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import sys

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgba, Normalize
from matplotlib.cm import ScalarMappable

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid memory issues
import matplotlib.pyplot as plt


def shifted_colormap(base_cmap, new_min_color, n_colors=256, shift_fraction=0.1):
    base = plt.get_cmap(base_cmap)
    base_colors = base(np.linspace(shift_fraction, 1, n_colors))  # Skip lightest colors
    new_min_color_rgba = to_rgba(new_min_color)  # Convert hex to RGBA
    new_colors = np.vstack((new_min_color_rgba, base_colors))
    return LinearSegmentedColormap.from_list(f"{base_cmap}_shifted", new_colors)

mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 6,
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.titlesize': 6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

no = int(sys.argv[1]) if len(sys.argv) > 1 else 0
#no = 2 # individual file check
#cover = sys.argv[2]  # if you want to recalculate, set cover as "r", otherwise, set cover as "o"
cover='r'

# =============================================================================
mu = 0.49 # poisson ratio
rb = 27.5  # bead size

COORD_SOURCE = 'filename'
PMAX_LABEL = 'P[max] (uN)'
PMAX_NO_INDENT = -0.001
PMAX_TOL = 0.0
# Contact QC. gain = SSE(straight line) / SSE(flat baseline + Hertzian onset)
# on the approach segment. Validated on 39 hand-labelled curves, 6 tissues,
# young and aged: no contact 0.182-1.082, contact 2.530-394, no overlap.
# Threshold = geometric midpoint of that gap.
GAIN_THRESHOLD = 1.655
GAIN_GRID = 60

# Max fractional uncertainty on Elasticity_l. Perr_l = SE(A)/A for the fitted
# Hertz amplitude (line ~660), and E is proportional to A, so this is the
# 1-sigma relative error on the modulus. Perr_l scales as 1/E by construction,
# so a strict cut discriminates against soft tissue regardless of data quality.
# Measured on 12 slices: 0.05 -> 0.15 recovers 4% of points, moves the pooled
# median 1.6%, and cuts the aged-vs-young differential loss from 0.84 to
# 0.26 percentage points. Degenerate fits reach Perr_l ~1e11 and are removed by
# any threshold.
PERR_MAX = 0.15

#=========== ROOT is the folder you put this pipeline file and all data samples ===========
root = os.path.dirname(__file__)+'/' # the filefolder which this py file is located


#slice_name = '20240917_13w-02-ms03-young' # If you don't use sys arg input, nominate the slice name directly
allslices = sorted([
    f.name for f in os.scandir(root)
    if f.is_dir() and len(f.name) >= 8 and f.name[:8].isdigit()
])
print(allslices)
slice_name = allslices[no]
print(f"Processing folder {no+1} of {len(allslices)}: {slice_name}")
#slice_name = '20231025_14m-01a-ms02-aged' # If you don't use sys arg input, nominate the slice name directly

fileinput = root+'/'+slice_name # locate the folder where you put all your raw txt files
output = fileinput+'/output' # name the folder where you want to put all the graphical output and csv results for this slice
outputVsc = output+'/viscosity'
if os.path.exists(outputVsc)==False:
    os.makedirs(outputVsc)

# =============================================================================
def _parse_header(fullpath, max_lines=500):
    """Parse the tab-delimited metadata block of an Optics11 point file.

    Returns (header, data_text):
      header    dict mapping the label in column 0 to the list of remaining
                tab-separated fields on that line.
      data_text the 'Time (s)' row and everything after it, as one string.
    """
    header = {}
    with open(fullpath, 'r', errors='ignore') as f:
        lines = f.readlines()

    head_idx = None
    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        parts = [p.strip() for p in line.rstrip('\r\n').split('\t')]
        label = parts[0] if parts else ''
        if label.startswith('Time (s)'):
            head_idx = i
            break
        if label == '':
            continue
        header[label] = parts[1:]

    data_text = ''.join(lines[head_idx:]) if head_idx is not None else None
    return header, data_text


def _header_float(header, label, fullpath):
    """Fetch a single float value by its exact header label."""
    if label not in header:
        raise KeyError(f"header label '{label}' not found in {fullpath}")
    vals = header[label]
    if len(vals) == 0 or vals[0] == '':
        raise ValueError(f"header label '{label}' has no value in {fullpath}")
    return float(vals[0])


def _scan_indices(header, fullpath):
    """Return (x_index, y_index) from the 'Scan (#)' line.

    Preferred route is label-based: find the token 'X (#)' and take the field
    after it. Falls back to the original fixed positions (columns 3 and 5 of
    the raw line) if the labels are absent, and warns if the two disagree.
    """
    if 'Scan (#)' not in header:
        raise KeyError(f"'Scan (#)' line not found in {fullpath}")
    vals = header['Scan (#)']  # fields AFTER the 'Scan (#)' label

    def by_label(tag):
        for k, v in enumerate(vals):
            if v.replace(' ', '') == tag.replace(' ', ''):
                if k + 1 < len(vals) and vals[k + 1] != '':
                    return int(float(vals[k + 1]))
        return None

    x_lab, y_lab = by_label('X (#)'), by_label('Y (#)')

    # original positional route: raw column 3 -> vals[2], raw column 5 -> vals[4]
    x_pos = int(float(vals[2])) if len(vals) > 2 and vals[2] != '' else None
    y_pos = int(float(vals[4])) if len(vals) > 4 and vals[4] != '' else None

    x_idx = x_lab if x_lab is not None else x_pos
    y_idx = y_lab if y_lab is not None else y_pos

    if x_idx is None or y_idx is None:
        raise ValueError(f"could not resolve scan X/Y indices in {fullpath}")
    if x_lab is not None and x_pos is not None and x_lab != x_pos:
        print(f"[WARN] {os.path.basename(fullpath)}: scan X index label={x_lab} "
              f"positional={x_pos}; using label value.")
    if y_lab is not None and y_pos is not None and y_lab != y_pos:
        print(f"[WARN] {os.path.basename(fullpath)}: scan Y index label={y_lab} "
              f"positional={y_pos}; using label value.")
    return x_idx, y_idx

def contact_gain(z, L):
    """Does this force curve contain a contact onset at all?

    Compares a flat-baseline-then-Hertzian model against a plain straight line.
    Returns SSE(line) / SSE(contact model). Below ~1 means fitting an onset is
    WORSE than a straight line, i.e. there is nothing there. Measures curve
    SHAPE, so it also catches spurious contact points placed mid-stroke, which
    no stroke-position metric can detect.
    """
    n = len(z)
    if n < 50:
        return np.nan
    A = np.vstack([np.ones(n), z]).T
    try:
        sse_line = float(np.sum((L - A @ np.linalg.lstsq(A, L, rcond=None)[0]) ** 2))
    except np.linalg.LinAlgError:
        return np.nan
    zlo, zhi = float(np.min(z)), float(np.max(z))
    if zhi <= zlo:
        return np.nan
    best = np.inf
    for frac in np.linspace(0.02, 0.98, GAIN_GRID):
        z0 = zlo + frac * (zhi - zlo)
        B = np.vstack([np.ones(n), np.clip(z - z0, 0, None) ** 1.5]).T
        try:
            c = np.linalg.lstsq(B, L, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        sse = float(np.sum((L - B @ c) ** 2))
        if sse < best:
            best = sse
    if not np.isfinite(best) or best <= 0:
        return np.nan
    return sse_line / best

def _filename_indices(filepath):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    m = re.match(r'^(.*?)[\s_]S-\d+(?=$|[\s_])', stem)
    if m:
        slice_id = m.group(1)
    else:
        pieces = re.split(r'[\s_]+', stem)
        slice_id = pieces[0] if pieces else None
    xm = re.search(r'(?:^|[\s_])X-(\d+)(?=$|[\s_])', stem)
    ym = re.search(r'(?:^|[\s_])Y-(\d+)(?=$|[\s_])', stem)
    x = int(xm.group(1)) if xm else None
    y = int(ym.group(1)) if ym else None
    return slice_id, x, y

def ProcessFile(filepath):
    if filepath.find('XY position')>-1:
        return 'not this file'


    if (filepath.find('txt')==-1):
        return 'no such txt file'
    ## Create output folder
    outputsp = output+'/'+filepath[:-4]
    if os.path.exists(outputsp)==False:
        os.makedirs(outputsp)

    if cover=='o':   # go to judge whether we already have this point calculated 
        if os.path.exists(outputsp+'/result.csv')==True:
            print(filepath+' already calculated')
            return 



    # otherwise, we do the calculation from this point

    #=== read the metadata block by LABEL, not by row index ====================
    fullpath = fileinput + '/' + filepath
    hdr, data_text = _parse_header(fullpath)
    if data_text is None:
        raise ValueError(f"'Time (s)' data header not found in {fullpath}")
 
    # The data block is handed to pandas as text, so no line index crosses
    # between parsers. This also normalises the file's mixed line endings.
    df = pd.read_csv(io.StringIO(data_text), sep="\t", on_bad_lines='skip')
    heads = list(df.head())
 
    # Each value is now taken from its own labelled line. No +1 / +2 offsets.
    xpos = _header_float(hdr, 'X-position (um)', fullpath)
    ypos = _header_float(hdr, 'Y-position (um)', fullpath)
    zpos = _header_float(hdr, 'Z-position (um)', fullpath)

    #=== grid step indices from two independent sources, cross-checked ==========
    try:
        hdr_x, hdr_y = _scan_indices(hdr, fullpath)
    except (KeyError, ValueError, IndexError) as e:
        hdr_x, hdr_y = None, None
        print(f"[WARN] {filepath}: could not read scan indices from header ({e})")
 
    slice_id, fn_x, fn_y = _filename_indices(filepath)
 
    if COORD_SOURCE == 'filename':
        if fn_x is None or fn_y is None:
            raise ValueError(
                f"could not read 'X-<i>' / 'Y-<j>' step indices from filename '{filepath}'. "
                f"Expected '<sliceID> S-<n> X-<i> Y-<j> I-<k>.txt'")
        xx, yy = fn_x, fn_y
    elif COORD_SOURCE == 'header':
        if hdr_x is None or hdr_y is None:
            raise ValueError(f"could not read scan indices from header of {fullpath}")
        xx, yy = hdr_x, hdr_y
    else:
        raise ValueError(f"COORD_SOURCE must be 'filename' or 'header', got {COORD_SOURCE!r}")
 
    if fn_x is not None and hdr_x is not None and fn_x != hdr_x:
        print(f"[WARN] {filepath}: X index disagrees, filename={fn_x} header={hdr_x}, using {xx}")
    if fn_y is not None and hdr_y is not None and fn_y != hdr_y:
        print(f"[WARN] {filepath}: Y index disagrees, filename={fn_y} header={hdr_y}, using {yy}")
    if slice_id and slice_id not in slice_name:
        print(f"[WARN] {filepath}: slice ID '{slice_id}' not found in folder name '{slice_name}'")
    if slice_id and slice_id not in hdr:
        print(f"[WARN] {filepath}: slice ID '{slice_id}' not found in the file header")


    print(filepath,xpos,ypos,zpos,xx,yy)

    #=== contact QC: does this curve contain a contact onset at all? ===========
    gain_value = np.nan
    try:
        _c = {c.strip(): c for c in df.columns}
        _z = pd.to_numeric(df[_c['Piezo (nm)']], errors='coerce').values
        _l = pd.to_numeric(df[_c['Load (uN)']], errors='coerce').values
        _ap = slice(0, int(np.nanargmax(_z)) + 1)
        _ok = np.isfinite(_z[_ap]) & np.isfinite(_l[_ap])
        gain_value = contact_gain(_z[_ap][_ok], _l[_ap][_ok])
    except Exception as e:
        print(f"[WARN] contact gain failed for {filepath}: {e}")
    indentation_present = True
    reason = ''
    if np.isfinite(gain_value) and gain_value < GAIN_THRESHOLD:
        indentation_present = False
        reason = f'gain {gain_value:.3f} < {GAIN_THRESHOLD}'
    if PMAX_LABEL in hdr:
        try:
            if abs(float(hdr[PMAX_LABEL][0]) - PMAX_NO_INDENT) <= PMAX_TOL:
                indentation_present = False
                reason = (reason + '; ' if reason else '') + f'{PMAX_LABEL} sentinel'
        except (IndexError, ValueError):
            pass
 
    if not indentation_present:
        print(f"[SKIP] No contact in {filepath} ({reason}).")
        trtm = 'aged' if ('age' in slice_name) else 'young'
        # Coordinates are kept so the skipped point stays traceable to its
        # grid position instead of being written as NaN.
        res = pd.DataFrame([[slice_name, trtm, filepath, xpos, ypos, zpos, xx, yy,
                            np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False]],
                        columns=['Sample', 'Treatment', 'PointPath', 'X (um)', 'Y (um)', 'Z (um)',
                                    'X', 'Y', 'Energy_l (fJ)', 'Energy_r (fJ)', 'Energy_ad (fJ)', 'Loss', 'E_ad/E_l',
                                    'Elasticity_l (Pa)', 'Elasticity_r (Pa)', 'Perr_l',
                                    'Elasticity_eff (Pa)', 'Perr_eff', 'Viscosity (Ns/m)',
                                    'Power_l', 'Power_r', 'Diff (nm)', 'IndentationPresent'])
        res.to_csv(outputsp + '/result.csv')
        return 'skipped'

    if True:
    #try:
        sub_df = df[[heads[0],heads[1],heads[2]]] # heads[0]:Time, [1]:Load, [2]:Indentation
        fig,ax = plt.subplots(2,1,figsize=(7,6))
        ax[0].plot(sub_df[heads[0]],sub_df[heads[1]],color='blue')
        ax[0].set_xlabel(heads[0])
        ax[0].set_ylabel(heads[1])
        ax2 = ax[0].twinx()
        ax2.plot(sub_df[heads[0]],sub_df[heads[2]],color='red')
        ax2.set_ylabel(heads[2],rotation=270,labelpad=15)
        ax[0].spines['left'].set_color('blue')
        ax[0].tick_params(axis='y', colors='blue')
        ax[0].yaxis.label.set_color('blue')
        ax2.spines['right'].set_color('red')
        ax2.tick_params(axis='y', colors='red')
        ax2.yaxis.label.set_color('red')
        
        #====== Remove holding ======
        ts,force,ind=sub_df[heads[0]].to_numpy(),sub_df[heads[1]].to_numpy(),sub_df[heads[2]].to_numpy()
        force = force-force[:100].mean()
        ind = ind-ind[:100].mean()
        std = np.std(force[:100]-force[:100].mean())/force.max()

        w = int(101*np.sqrt(std/0.005)/2)*2+1
        force = svft(force, w, 1, mode='interp')
        ind   = svft(ind, w, 1, mode='interp')
        fg    = svft(np.diff(force), w, 1)


        pk = np.where(fg==fg.max())[0]
        vl = np.where(fg==fg.min())[0]-int(w/2)
        ax[1].plot(ts[1:],fg,color='black')
        ax3 = ax[1].twinx()
        ax3.plot(ts,force,color='blue')
        ax3.scatter(ts[pk],force[pk])
        ax3.scatter(ts[vl],force[vl])
        # Robust ceiling & split
        ceiling_arr = force[vl] if len(vl) > 0 else np.array([np.median(force)])
        ceiling = float(np.max(ceiling_arr))

        sel = np.where(force < ceiling)[0]
        unsel = np.where(force >= ceiling)[0]

        # If we can't find both approach and retraction segments, skip safely
        if len(unsel) < 2 or len(sel) == 0:
            print(f"[SKIP] {filepath}: unable to split approach/retraction (unsel={len(unsel)}, sel={len(sel)}).")
            trtm = 'aged' if ('age' in slice_name) else 'young'
            res = pd.DataFrame([[slice_name, trtm, filepath, np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False]],
                            columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                                        'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','E_ad/E_l',
                                        'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l',
                                        'Elasticity_eff (Pa)','Perr_eff','Viscosity (Ns/m)',
                                        'Power_l','Power_r','Diff (nm)','IndentationPresent'])
            res.to_csv(outputsp + '/result.csv')
            return 'skipped-no-split'

        lend, rstart = unsel[0], unsel[-1]
        time_l, time_r = ts[:lend], ts[rstart:]
        force_l, force_r = force[:lend], force[rstart:]
        ind_l, ind_r = ind[:lend], ind[rstart:]
        ax3.plot(time_l,force_l,color='red',lw=3,alpha=0.3)
        ax3.plot(time_r,force_r,color='lime',lw=3,alpha=0.3)

        ax[1].set_xlabel(heads[0])
        ax[1].set_ylabel('Force gradient')
        ax3.set_ylabel(heads[1],rotation=270,labelpad=15)
        ax3.spines['right'].set_color('blue')
        ax3.tick_params(axis='y',colors='blue')
        ax3.yaxis.label.set_color('blue')    
        plt.tight_layout()
        plt.savefig(outputsp+'/curve_holding_removed.jpg',dpi=300)
        plt.close('all')


        # If either side is empty, skip safely
        if len(ind_l) == 0 or len(ind_r) == 0:
            print(f"[SKIP] {filepath}: empty approach/retraction arrays (ind_l={len(ind_l)}, ind_r={len(ind_r)}).")
            trtm = 'aged' if ('age' in slice_name) else 'young'
            res = pd.DataFrame([[slice_name, trtm, filepath, np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False]],
                            columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                                        'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','E_ad/E_l',
                                        'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l',
                                        'Elasticity_eff (Pa)','Perr_eff','Viscosity (Ns/m)',
                                        'Power_l','Power_r','Diff (nm)','IndentationPresent'])
            res.to_csv(outputsp + '/result.csv')
            return 'skipped-empty'

        #======= correct slope ====================
        ind_r = ind_r-(ind_r[0]-ind_l[-1])
        time_r = time_r - (time_r[0]-time_l[-1])

        def _safe_savgol(arr, w=51, p=1):
            n = len(arr)
            # If too short to smooth, just return as-is
            if n <= p + 2:
                return arr
            # Largest odd ≤ n
            max_odd = n if (n % 2 == 1) else (n - 1)
            w_use = min(w, max_odd)
            if w_use <= p:
                return arr
            return svft(arr, w_use, p, mode='interp')


        force_ls = _safe_savgol(force_l, 51, 1)
        force_rs = _safe_savgol(force_r, 51, 1)

        slope = _safe_savgol(np.gradient(force_ls, time_l), 51, 1)
        sloslope = _safe_savgol(np.gradient(slope,   time_l), 51, 1)

        idx = np.where(sloslope > 1e-8)[0]
        start = int(idx[0]) if idx.size > 0 else 0   # fallback to 0 if no crossing found

        dx = np.diff(time_l)
        #sl_l = (force_ls[500]-force_ls[0])/(time_l[500]-time_l[0])

        # Robust pick for ind_start
        idx_gt3 = np.where(ind_l > 3)[0]

        if idx_gt3.size > 0:
            ind_start = int(idx_gt3[0] / 2) + 1  # cannot be 1
        else:
            # If approach segment is extremely short, skip this file
            if len(ind_l) < 5:
                print(f"[SKIP] {filepath}: approach segment too short (len(ind_l)={len(ind_l)}).")
                trtm = 'aged' if ('age' in slice_name) else 'young'
                res = pd.DataFrame([[slice_name, trtm, filepath, np.nan, np.nan, np.nan, np.nan, np.nan,
                                    np.nan, np.nan, np.nan, np.nan, np.nan,
                                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False]],
                                columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                                            'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','E_ad/E_l',
                                            'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l',
                                            'Elasticity_eff (Pa)','Perr_eff','Viscosity (Ns/m)',
                                            'Power_l','Power_r','Diff (nm)','IndentationPresent'])
                res.to_csv(outputsp + '/result.csv')
                return 'skipped-approach-too-short'
            # Otherwise, fall back to ~5% of the approach length (at least 2)
            ind_start = max(2, len(ind_l) // 20)

        sl_l = slope[:ind_start].mean()
        if sl_l>pow(10,-3): # meaning that his slope is too large, invalid
            sl_l = 0
        fluctuation = (force_l[:ind_start]-sl_l).std()
        #print('force_l slope',sl_l,'; slope fluctuation',fluctuation)
        #sl_l = (force_ls[500]-force_ls[0])/(time_l[500]-time_l[0])
        #print(sl_l)
        fl = time_l*sl_l+force_l[0]
        fr = -(time_r-time_r[0])*sl_l

        new_r = force_rs - fr - fl[-1]
        
        # use the last K points (K ≤ 50, but ≥ 5 to be meaningful)
        tail_len = min(50, new_r.size)
        if tail_len < 5:
            print(f"[SKIP] {filepath}: not enough points in new_r to estimate correction (len={new_r.size}).")
            trtm = 'aged' if ('age' in slice_name) else 'young'
            res = pd.DataFrame([[slice_name, trtm, filepath, np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False]],
                            columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                                        'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','E_ad/E_l',
                                        'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l',
                                        'Elasticity_eff (Pa)','Perr_eff','Viscosity (Ns/m)',
                                        'Power_l','Power_r','Diff (nm)','IndentationPresent'])
            res.to_csv(outputsp + '/result.csv')
            return 'skipped-short-new_r'

        corr = float(np.mean(new_r[-tail_len:]))


        sl_r = corr/(time_r.max()-time_r.min())
        #print(corr,sl_r)
        fr=  (time_r-time_r[0])*(-sl_l+sl_r)

        fig,ax = plt.subplots(1,2,figsize=(7,3))
        ax[0].plot(time_l,force_ls, color='blue',alpha=0.5)
        ax[0].plot(time_l, force_ls-fl,color='blue', linestyle='dashed')
        ax[0].plot(time_r,force_rs, color='orange',alpha=0.5,label='original')
        ax[0].plot(time_r, force_rs-fr-fl[-1],color='orange', linestyle='dashed',label='corrected')
        #find the offset time
        force_l, force_r = force_l-fl,force_r-fr-fl[-1]
        dforce = np.where(_safe_savgol(force_l, 51, 1) > min(3*fluctuation, force_l.max()/40))[0]

        #start = np.where(dforce==dforce.max())[0][0]
        if len(dforce)>0:
            start = dforce[0]
        else:
            start = ind_start
        time_s = time_l[start]    
        ax[0].plot([time_s,time_s],[-force_l.max()*0.1,force_l.max()*1.1],color='black')
        of = force_l[start]
        x1 = ind_l[start]
        idx_finish = np.where(force_r < of)[0]
        if idx_finish.size == 0:
            print(f"[SKIP] {filepath}: could not find finish index on retraction.")
            trtm = 'aged' if ('age' in slice_name) else 'young'
            res = pd.DataFrame([[slice_name, trtm, filepath, np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False]],
                            columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                                        'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','E_ad/E_l',
                                        'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l',
                                        'Elasticity_eff (Pa)','Perr_eff','Viscosity (Ns/m)',
                                        'Power_l','Power_r','Diff (nm)','IndentationPresent'])
            res.to_csv(outputsp + '/result.csv')
            return 'skipped-no-finish'
        finish = idx_finish[0]
        time_f = time_r[finish]

        #print(time_s,time_f)
        ax[0].plot([time_l[0],time_r[-1]],[0,0],color='black')
        # plt.scatter(time_r,force_r)
        ax[0].plot([time_f,time_f],[-force_l.max()*0.1,force_l.max()*1.1],color='grey')
        ax[0].set_ylabel(heads[1])
        ax[0].legend(loc=0,frameon=False)

        #============================== select range for fitting (robust) ===========================
        y0 = 3 * fluctuation

        # 1) Ensure there exists any approach point above threshold
        idx_bs = np.where(force_l > y0)[0]
        if idx_bs.size == 0:
            print(f"[SKIP] {filepath}: no approach points above threshold (force_l > y0).")
            trtm = 'aged' if ('age' in slice_name) else 'young'
            res = pd.DataFrame([[slice_name, trtm, filepath, np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False]],
                            columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                                        'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','E_ad/E_l',
                                        'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l',
                                        'Elasticity_eff (Pa)','Perr_eff','Viscosity (Ns/m)',
                                        'Power_l','Power_r','Diff (nm)','IndentationPresent'])
            res.to_csv(outputsp + '/result.csv')
            return 'skipped-no-approach-threshold'

        bs_st = idx_bs[0]

        # 2) Last zero-crossing on approach to get x0 (can be absent)
        temp_l = (force_l - y0)[1:] * (force_l - y0)[:-1]
        zc_l = np.where(temp_l < 0)[0]
        if zc_l.size > 0:
            x0 = ind_l[zc_l[-1]]
        else:
            x0 = 0

        # 3) First zero-crossing on retraction to get xf (can be absent)
        temp_r = (force_r - y0)[1:] * (force_r - y0)[:-1]
        zc_r = np.where(temp_r < 0)[0]
        if zc_r.size > 0:
            xf = ind_r[zc_r[0]]
        else:
            # fallbacks if no zero-crossing; ensure ind_r is non-empty
            xf = ind_r[-1] if len(ind_r) > 0 else 0


        #xf = np.max([xf,0]) # to avoid negative xf
        #print(bs_st,y0,x0,xf)
        #plt.plot([x0,x0],[force_l.max(),force_l.min()],color='black')
        
        ax[1].scatter(ind_l-x0,force_l-y0,color='blue',alpha=0.8,s=5,label='approach')
        ax[1].scatter(ind_r-xf,force_r-y0,color='orange',alpha=0.8,s=5,label='retraction')
        # plt.xlim(6*pow(10,-8),5*pow(10,-6))
        ax[1].set_ylim(pow(10,-4),2*force_l.max())
        #plt.plot([0.5*pow(10,-2),1*pow(10,2)],[0.00002*0.5*pow(10,-2),0.00002*1*pow(10,2)],color='black')
        ax[1].plot([0.5*pow(10,2),3*pow(10,3)],[0.0000002*pow(0.5*pow(10,2),1.5),0.0000002*pow(3*pow(10,3),1.5)],color='black',label='Hertz(1.5)')
        #plt.text(0.3*pow(10,3),0.0000002*pow(pow(10,3),1.5),s='1.5 \n (Hertz)',ha='center')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xlabel(heads[2])
        ax[1].set_ylabel(heads[1])
        ax[1].legend(loc=2,frameon=False)
        plt.tight_layout()
        plt.savefig(outputsp+'/correction, log-log curve.jpg',dpi=300)
        plt.close('all')

        #=============================  FITTING DATA ================================
        time_ll = time_l[start:]
        time_rr = time_r[:finish]
        force_ll, force_rr = force_l[start:],force_r[:finish]
        ind_ll ,ind_rr = ind_l[start:],ind_r[:finish]

        # Ensure we have enough points to proceed
        if (len(force_ll) < 5) or (len(force_rr) < 5):
            print(f"[SKIP] {filepath}: too few points after start/finish trimming (len={len(force_ll)}/{len(force_rr)}).")
            trtm = 'aged' if ('age' in slice_name) else 'young'
            res = pd.DataFrame([[slice_name, trtm, filepath, np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False]],
                            columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                                        'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','E_ad/E_l',
                                        'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l',
                                        'Elasticity_eff (Pa)','Perr_eff','Viscosity (Ns/m)',
                                        'Power_l','Power_r','Diff (nm)','IndentationPresent'])
            res.to_csv(outputsp + '/result.csv')
            return 'skipped-too-few-trimmed'

        #=============================   ENERGY ================================
        energy_l = np.abs(np.trapezoid(force_ll-of,x=ind_ll)) # x is ranging from max to 
        energy_r = np.abs(np.trapezoid(force_rr-of, x=ind_rr))
        loss = (energy_l-energy_r)/energy_l
        ind_rn = ind_r[finish:]
        force_rn = force_r[finish:]
        sel = np.where(force_rn<0)[0]
        #plt.plot(ind_rn[sel],force_rn[sel])
        if len(sel)>0:
            energy_a = np.abs(np.trapezoid(force_rn[sel],x=ind_rn[sel]))
        else:
            energy_a = 0
        print('Eng_a',energy_l,'fJ','Eng_r',energy_r,'fJ','WorkLoss',loss,'Adhesive energy:',energy_a, 'fJ', energy_a/energy_l) 
        #============================= ELASTIC MODULUS ================================
        plt.figure(figsize=(5,3))
        def simpleHertz(x,x0,y0,A):
            return A*pow((x-x0),1.5)+y0
        off = force_ll[:100].mean()
        den = max(ind_ll[-1] - ind_ll[0], 1e-9)
        x0, y0 = ind_ll[0], force_ll[0]
        A0 = (force_ll[-1] - force_ll[0]) / pow(den, 1.5)

        x0l,y0l=x0,y0
        #print(A0,off)
        sel = np.where((ind_ll[:int(len(ind_ll)*0.9)]<2000)&(ind_ll[:int(len(ind_ll)*0.9)]>500))[0]
        try:
            para,pcov = fit(simpleHertz,ind_ll[sel],force_ll[sel],p0 = [0,off,A0],bounds = ([-500,-np.inf,0],[500,np.inf,np.inf]))
            x0l,y0l,_=para
            Ea = 0.75*para[2]*(1-mu**2)*np.sqrt(1/rb) *pow(10,10.5)
            perr  = np.sqrt(np.abs(np.diag(pcov)))/para
            perr_l = perr[-1]
            print('app',para,perr_l,Ea)    
            newforce = para[2]*pow(ind_ll-para[0],1.5)+para[1]
            plt.plot(ind_ll-para[0],force_ll-para[1], color='blue')
            plt.plot(ind_ll-para[0],newforce-para[1], color='navy', linestyle='--',label='approach, '+str((round(Ea)))+" Pa")
        except:
            Ea = np.nan
            perr_l = np.nan
        
        

        ### for retraction curve (safe)
        if ind_rr.size < 2 or force_rr.size < 2:
            Er = np.nan
        else:
            base = ind_rr[-1]
            sel = np.where((ind_rr - base) > 100)[0]
            if sel.size < 3:
                Er = np.nan
            else:
                den = max(ind_rr[0] - ind_rr[-1], 1e-9)
                x0, y0 = base, force_rr[-1]
                A0 = (force_rr[0] - force_rr[-1]) / pow(den, 1.5)
                try:
                    para, pcov = fit(simpleHertz, ind_rr[sel], force_rr[sel], p0=[x0, y0, A0])
                    Er = round(0.75*para[2]*(1-mu**2)*np.sqrt(1/rb)*pow(10,10.5))
                    newforce = para[2]*pow(ind_rr-para[0],1.5)+para[1]
                    plt.plot(ind_rr-para[0], force_rr-para[1], color='red')
                    plt.plot(ind_rr-para[0], newforce-para[1], color='brown', linestyle='--', label='retraction, '+str((Er))+" Pa")
                except:
                    Er = np.nan

        
        

        plt.legend(loc=0, frameon=False, handlelength=0, labelcolor='linecolor')

        # Only apply log scaling when lower y-limit is positive
        ymin = float(off)
        ymax = float(force_ll.max() * 8)
        if (50 > 0) and (ymin > 0):
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(50, 15000)
            plt.ylim(ymin, ymax)
        else:
            print(f"[WARN] {filepath}: skipping log scale in Hertz plot (off={ymin:.3g}).")
            plt.xlim(0, max(15000, np.nanmax(ind_ll) * 1.1 if len(ind_ll) else 1000))
            plt.ylim(min(0, ymin), ymax)
    
        plt.xlabel(heads[2])
        plt.ylabel(heads[1])
        plt.tight_layout()
        plt.savefig(outputsp+'/HertzFitting.jpg',dpi=300)
        plt.close('all')


        #============= plot indentation cycle ==================
        plt.figure(figsize=(5,3))
        plt.scatter(ind_l,force_l,color='blue')
        plt.scatter(ind_r,force_r,color='orange')
        plt.scatter(sub_df[heads[2]],sub_df[heads[1]],color='grey',s=3,alpha=0.5)
        plt.plot([ind_rr.min(),ind_ll.max()],[0,0],color='black',linestyle='--')
        plt.xlabel(heads[2])
        plt.ylabel(heads[1])
        plt.tight_layout()
        plt.savefig(outputsp+'/app-ret cycle.jpg',dpi=300)
        plt.close('all')

        #print('Hertz Loading', Ea, 'Retraction', Er)
        #====================================== Stress-Hardening Model ====================================
        #=== Robust pick for ns (first index where ind_ll > x0l) ===
        idx_ns = np.where(ind_ll > x0l)[0]
        if idx_ns.size == 0:
            # If approach slice is too short, skip gracefully
            if len(ind_ll) < 5 or len(force_ll) < 5:
                print(f"[SKIP] {filepath}: no approach ind > x0l and segment too short (len(ind_ll)={len(ind_ll)}).")
                trtm = 'aged' if ('age' in slice_name) else 'young'
                res = pd.DataFrame([[slice_name, trtm, filepath, np.nan, np.nan, np.nan, np.nan, np.nan,
                                    np.nan, np.nan, np.nan, np.nan, np.nan,
                                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False]],
                                columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                                            'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','E_ad/E_l',
                                            'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l',
                                            'Elasticity_eff (Pa)','Perr_eff','Viscosity (Ns/m)',
                                            'Power_l','Power_r','Diff (nm)','IndentationPresent'])
                res.to_csv(outputsp + '/result.csv')
                return 'skipped-no-ind-gt-x0l'
            # Otherwise use a conservative fallback (~5% into approach; at least 1)
            ns = max(1, len(ind_ll) // 20)
        else:
            ns = int(idx_ns[0])

        off = float(force_ll[ns])

        #=== Robust pick for nf (last index where force_rr > off) ===
        idx_nf = np.where(force_rr > off)[0]
        if idx_nf.size == 0:
            print(f"[SKIP] {filepath}: no retraction force above offset (off={off:.3g}).")
            trtm = 'aged' if ('age' in slice_name) else 'young'
            res = pd.DataFrame([[slice_name, trtm, filepath, np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False]],
                            columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                                        'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','E_ad/E_l',
                                        'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l',
                                        'Elasticity_eff (Pa)','Perr_eff','Viscosity (Ns/m)',
                                        'Power_l','Power_r','Diff (nm)','IndentationPresent'])
            res.to_csv(outputsp + '/result.csv')
            return 'skipped-no-ret-gt-off'

        nf = int(idx_nf[-1])

        # Proceed as before
        nind_l, nforce_l = ind_ll - ind_ll[ns], force_ll - off
        nind_r, nforce_r = ind_rr - ind_rr[nf], force_rr - off

        #print('ns',ns,'nf',nf,nforce_r[-1], 'ind_nf',nind_r[nf],nind_r[-5:]-ind_r[nf])
        ntime_ll = time_ll[ns:]-time_ll[ns]
        ntime_rr = np.flip(time_rr[nf]-time_rr[:nf])
        nind_ll = nind_l[ns:]
        nind_rr = np.flip(nind_r[:nf])
        nforce_ll = nforce_l[ns:]
        nforce_rr = np.flip(nforce_r[:nf])
        #================== interpolate ===================
        ftl = interp1d(nforce_ll,ntime_ll,fill_value='extrapolate')
        ftr = interp1d(nforce_rr,ntime_rr,fill_value='extrapolate')
        fil = interp1d(nforce_ll,nind_ll,fill_value='extrapolate')
        fir = interp1d(nforce_rr,nind_rr,fill_value='extrapolate')
        newf = np.linspace(max(nforce_ll.min(),nforce_rr.min()),nforce_ll.max(),1000)
        newind_l = fil(newf)
        newind_r = fir(newf)
        newtime_l = ftl(newf)
        newtime_r = ftr(newf)
        #==== Robust lb_l / lb_r selection ====
        # Guard for trivially short sequences
        if (newind_l.size < 5) or (newind_r.size < 5):
            print(f"[SKIP] {filepath}: newind_l/newind_r too short (len={newind_l.size}/{newind_r.size}).")
            trtm = 'aged' if ('age' in slice_name) else 'young'
            res = pd.DataFrame([[slice_name, trtm, filepath, np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False]],
                            columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                                        'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','E_ad/E_l',
                                        'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l',
                                        'Elasticity_eff (Pa)','Perr_eff','Viscosity (Ns/m)',
                                        'Power_l','Power_r','Diff (nm)','IndentationPresent'])
            res.to_csv(outputsp + '/result.csv')
            return 'skipped-newind-too-short'

        # Threshold for lb_l: 90th-percentile index (clamped) but not above 1000 nm
        idx90 = max(0, min(int(0.9 * len(newind_l)) - 1, len(newind_l) - 1))
        thr_l = min(float(newind_l[idx90]), 1000.0)
        idx_l = np.where(newind_l > thr_l)[0]
        lb_l = int(idx_l[0]) if idx_l.size > 0 else max(0, len(newind_l) // 10)  # fallback ~10% into curve

        # Threshold for lb_r: similar logic; your original used 200
        # If you prefer the old cap, keep 200; or use min(max-1, 100) as earlier.
        thr_r = 200.0
        idx_r = np.where(newind_r > thr_r)[0]
        lb_r = int(idx_r[0]) if idx_r.size > 0 else max(0, len(newind_r) // 10)

        lb = max(lb_l, lb_r)

        # =============================================================================
        dind = newind_l-newind_r
        #print('lb',lb)
        sumFl, sumFr = [],[]
        for i in range(len(newf)):
            sumFl.append(np.trapezoid(newf[:i],x=newtime_l[:i]))
            sumFr.append(np.trapezoid(newf[:i],x=newtime_r[:i]))
        sumFl,sumFr = np.array(sumFl),np.array(sumFr)
        eta0 = (sumFl+sumFr)/dind
        etac = eta0[-50:].mean()
        ind_el = newind_l - sumFl/etac
        ind_er = newind_r + sumFr/etac 
        # =============================================================================
        eta0 = svft(eta0, 51, 1)

        # keep only finite & positive values; use only the tail half
        mask = np.isfinite(newf) & np.isfinite(eta0) & (newf > 0) & (eta0 > 0)
        start_ind = len(newf) // 2
        # zero out the first half so we only fit the tail (as before)
        if start_ind > 0:
            m = mask.copy()
            m[:start_ind] = False
        else:
            m = mask

        x = np.log(newf[m])
        y = np.log(eta0[m])

        if x.size >= 5 and (np.ptp(x) > 1e-6) and (np.ptp(y) > 1e-6):
            try:
                k_m, e_m = np.polyfit(x, y, deg=1)
            except np.linalg.LinAlgError:
                # fallback: robust slope/intercept from medians
                k_m = np.median(np.diff(y) / np.diff(x)) if x.size >= 3 else 0.0
                e_m = np.median(y - k_m * x) if x.size >= 2 else 0.0
        else:
            # not enough good points; default to a flat viscosity scaling
            k_m = 0.0
            e_m = np.log(np.nanmedian(eta0[m])) if np.any(m) else 0.0

        plt.figure()
        plt.plot(np.log(newf),np.log(sumFl),label= 'sumFl')
        plt.plot(np.log(newf),np.log(sumFr),label = 'sumFr')
        plt.plot(np.log(newf),np.log(newind_l),label='newind_l')
        plt.plot(np.log(newf),np.log(newind_r),label='newind_r')
        plt.plot(np.log(newf),np.log(dind),label='dind')
        plt.plot(np.log(newf),np.log(eta0),color='red',linestyle='--')
        plt.legend(loc=0,frameon=False,handlelength=0.3,labelcolor='linecolor')
        plt.xlabel('log(newF)')
        plt.ylabel('log(eta0)')
        plt.close()
        kl,kr = k_m,k_m

        def delta_ind(x):
            diff = []
            el,kl,kr = x[0],x[1],x[2]
            emax = el*pow(newf[-1],kl)
            er = emax/pow(newf[-1],kr)
            for i in range(len(newf)):
                diff.append((np.trapezoid(newf[:i]/(el*pow(newf[:i],kl)),x=newtime_l[:i])+np.trapezoid(newf[:i]/(er*pow(newf[:i],kr)),x=newtime_r[:i])))
            diff = np.array(diff)
            
            return np.abs(dind-diff)[lb:].mean()
        seed_el = float(np.exp(e_m)) if np.isfinite(e_m) else 1.0
        if k_m > 0:
            res = minimize(delta_ind, x0=(seed_el, k_m, k_m),
                        bounds=[(0, 2*seed_el), (0.5*k_m, 2*k_m), (0.5*k_m, 2*k_m)],
                        method='Nelder-Mead', tol=1e-6)
        else:
            res = minimize(delta_ind, x0=(seed_el, k_m, k_m),
                        method='Nelder-Mead', tol=1e-6)

        el,kl,kr= res.x
        #print('kl,kr',kl,kr)

        
        er = el*pow(newf[-1],kl)/pow(newf[-1],kr)
        sumFls, sumFrs = [],[]
        for i in range(len(newf)):
            sumFls.append(np.trapezoid(newf[:i]/(el*pow(newf[:i],kl)),x=newtime_l[:i]))
            sumFrs.append(np.trapezoid(newf[:i]/(er*pow(newf[:i],kr)),x=newtime_r[:i]))

        sumFls,sumFrs = np.array(sumFls),np.array(sumFrs)
        ind_els = newind_l - sumFls
        ind_ers = newind_r + sumFrs 
        diff = np.mean(np.abs(ind_els-ind_ers))
        #print('diff',diff)
        etal = el*pow(newf,kl)
        etar = er*pow(newf,kr)
        sel=np.where(((newf<0.01)&(newf>0.001)))[0]
        eta_f = 1000*etal[sel].mean()
        #print('viscosity',eta_f)
        ind_elav = (ind_els+ind_ers)/2
        ### fit Hertz for pure elastic components ####
        A0 = np.nanmean(newf/pow(ind_elav,1.5))
        #print('A0',A0)
        try:
            para,pcov = fit(simpleHertz,ind_elav[:int(len(ind_elav)*0.9)],newf[:int(len(ind_elav)*0.9)],p0 = [0,0,A0])
            #para,pcov = fit(simpleHertz,ind_elav[lb:int(len(ind_elav)*0.9)],newf[lb:int(len(ind_elav)*0.9)],p0 = [0,0,A0])
            x0,y0,A =para
            perr = np.sqrt(np.abs(np.diag(pcov)))/para
            perr_av = perr[-1]
            Eav = 0.75*para[2]*(1-mu**2)*np.sqrt(1/rb)*pow(10,10.5)
            fitf = A*pow((ind_elav-x0),1.5)+y0
        except:
            Eav,perr_av = np.nan,np.nan
            fitf = np.zeros(len(newf))
        #print('Eav',Eav)
        # =============================================================================
        fig,ax = plt.subplots(1,2,figsize=(6,3))
        # =============================================================================
        ax[0].plot(newind_l,newf,color='blue',label='app')
        ax[0].plot(newind_r,newf,color='red',label='ret')
        ax[0].plot(ind_er,newf,color='pink',linestyle='--',alpha=0.5)
        ax[0].plot(ind_el,newf,color='purple',linestyle='--',alpha=0.5)
        ax[0].plot(ind_elav,newf,color='cyan',alpha=0.8,label='K, '+str(round(diff,2)))

        # ax[0].set_xscale('log')
        # ax[0].set_yscale('log')
        try:
            ax[0].plot(ind_elav,fitf,color='black',label='H, '+str(round(Eav))+'Pa')
        except:
            print('Eav is not fitted')
        ax[0].set_xlim(0,nind_l.max()*1.2)
        ax[0].set_ylim(0,newf.max()*1.2)
        ax[0].legend(loc=4,ncol=1,frameon=False,handlelength=0.3,labelcolor='linecolor',columnspacing=0.2,)
        ax[0].set_xlabel(heads[2])
        ax[0].set_ylabel(heads[1])

        # =============================================================================
        ax[1].plot(newf,etal*1000,color='cyan',label='kl,'+str(round(kl,2)))
        ax[1].plot(newf,etar*1000,color='lime',label='kr,'+str(round(kr,2)))
        ax[1].plot(newf,eta0*1000,color='violet',label='v: '+str(round(eta_f*1000,4)))
        ax[1].set_xlabel(heads[1])
        ax[1].set_ylabel('Viscosity (Ns/m)')
        # Decide limits first
        xmin = float(newf.min() * 2)
        xmax = float(newf.max() * 1.2)
        y_all = np.r_[etal*1000, etar*1000, eta0*1000]
        ymin = float(np.nanmin(y_all)) if np.isfinite(y_all).any() else 0.0
        ymax = float(np.nanmax(y_all)) if np.isfinite(y_all).any() else 0.0

        # Only use log scales if both axes are strictly positive
        if (xmin > 0) and (xmax > 0) and (ymin > 0) and (ymax > 0):
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            ax[1].set_xlim(xmin, xmax)
        else:
            print(f"[WARN] {filepath}: skipping log scale in viscosity plot "
                f"(xmin={xmin:.3g}, ymin={ymin:.3g}).")
            # fall back to linear x; keep values visible
            ax[1].set_xlim(min(0, xmin), xmax)

        ax[1].legend(loc=4, frameon=False)
        plt.suptitle(filepath[:-4])
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(outputVsc + '/' + filepath[:-4] + '.jpg', dpi=300)
        plt.close('all') 


    #======== output the results to csv file ===========
        # =============================================================================

        trtm = 'young'
        if slice_name.find('age')>-1:
            trtm = 'aged'
        res = pd.DataFrame([[slice_name,trtm,filepath,xpos,ypos,zpos, xx,
                             yy, energy_l,energy_r,energy_a,loss,energy_a/energy_l,
                             Ea,Er,perr_l,Eav,perr_av,eta_f,kl,kr,diff, True]],
                   columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                            'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','E_ad/E_l',
                            'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l',
                            'Elasticity_eff (Pa)','Perr_eff','Viscosity (Ns/m)',
                            'Power_l','Power_r','Diff (nm)', 'IndentationPresent'])
        
        res.to_csv(outputsp+'/result.csv')
        return res

    # except Exception as e:
    #     print(f"[ERROR] Failed to process {filepath}: {e}")
    #     return 'error'



# These are for visual map colour scaling, not a validity filter!
def Generate_map(qtys):
    FEATURE_LIMITS = {
        'Z (um)': (3172.96, 3779.38),
        'Energy_l (fJ)': (9.76893, 60.3478),
        'Loss': (0.190971, 0.689299),
        'Energy_ad (fJ)': (0.501413, 35.8438),
        'Elasticity_l (Pa)': (89.0775, 2681.48),
        'Elasticity_eff (Pa)': (139.759, 4889.36),
        'Viscosity (Ns/m)': (0.00397344, 0.360795),
        'Power_l': (-0.577621, 1.13058),
    }

    FEATURE_DISPLAY_SETTINGS = {
        'Z (um)': {
            'range': (3100, 3800),
            'ticks': [3100, 3275, 3450, 3625, 3800],
            'ticklabels': ['3100', '3275', '3450', '3625', '3800'],
        },
        'Energy_l (fJ)': {
            'range': (0, 65),
            'ticks': [0, 16.25, 32.5, 48.75, 65],
            'ticklabels': ['0', '16', '33', '49', '65'],
        },
        'Loss': {
            'range': (0.1, 0.7),
            'ticks': [0.1, 0.25, 0.4, 0.55, 0.7],
            'ticklabels': ['0.1', '0.25', '0.4', '0.55', '0.7'],
        },
        'Energy_ad (fJ)': {
            'range': (0, 36),
            'ticks': [0, 9, 18, 27, 36],
            'ticklabels': ['0', '9', '18', '27', '36'],
        },
        'Elasticity_l (Pa)': {
            'range': (0, 3000),
            'ticks': [0, 750, 1500, 2250, 3000],
            'ticklabels': ['0', '750', '1500', '2250', '3000'],
        },
        'Elasticity_eff (Pa)': {
            'range': (0, 5000),
            'ticks': [0, 1250, 2500, 3750, 5000],
            'ticklabels': ['0', '1250', '2500', '3750', '5000'],
        },
        'Viscosity (Ns/m)': {
            'range': (0, 0.07),
            'ticks': [0, 0.0175, 0.035, 0.0525, 0.07],
            'ticklabels': ['0', '0.018', '0.035', '0.053', '0.07'],
        },
        'Power_l': {
            'range': (-0.6, 1.2),
            'ticks': [-0.6, -0.15, 0.3, 0.75, 1.2],
            'ticklabels': ['-0.6', '-0.15', '0.3', '0.75', '1.2'],
        },
    }
    
    parentfilename = os.path.basename(fileinput)
    all_samples_file = output + f'/NI_{parentfilename}_all_samples.csv'

    if os.path.exists(all_samples_file)==False:
        print('no NI_[parentfilename]_all_samples.csv file found')
        return 

    allres = pd.read_csv(all_samples_file)
    allres['El_r/El_l'] = allres['Elasticity_r (Pa)'] / allres['Elasticity_l (Pa)']

    # Grid indices are now 1-based (X1 Y1 is the origin), matching the raw files.
    GRID_ORIGIN = 1
    xmin, ymin = GRID_ORIGIN, GRID_ORIGIN
    xmax, ymax = allres['X'].max(), allres['Y'].max()
    nx, ny = int(xmax) - xmin + 1, int(ymax) - ymin + 1  # number of columns / rows
    newres = allres

    #newres = allres.dropna()
    heads = list(newres)
    if len(qtys)==0:
        qtys = [heads[6],heads[9],heads[12],heads[11],heads[14],'Elasticity_eff (Pa)','Viscosity (Ns/m)', 'Power_l'] # default['Z (um)', 'Energy_l (fJ)', 'Loss', 'Energy_ad (fJ)', 'Elasticity_l (Pa)', 'Power_l']

    fig, axes = plt.subplots(2, 4, figsize=((nx * 4 * 0.15 + 4), (ny + 3) * 0.3))
    plt.suptitle(newres['Sample'].unique()[0])
    cmps = {
        'Z (um)':'Greys',
        'Energy_l (fJ)':'GnBu',
        'Loss':'RdYlBu',
        'Energy_ad (fJ)':'Blues',
        'Elasticity_l (Pa)':'YlOrRd',
        'Elasticity_eff (Pa)': 'pink_r', 
        'Viscosity (Ns/m)':'YlGn',
        'Power_l':'PuBuGn'}
    # Create adjusted color maps with no white at the low end
    adjusted_cmps = {
        'Z (um)': shifted_colormap('Greys', '#bdbdbd'),
        'Energy_l (fJ)': shifted_colormap('YlGnBu', '#ccebc5'),
        'Loss': shifted_colormap('RdYlBu', '#f46d43'),
        'Energy_ad (fJ)': shifted_colormap('Blues', '#cde0f1'),
        'Elasticity_l (Pa)': shifted_colormap('YlOrRd', '#fee8c8'),
        'Elasticity_eff (Pa)': shifted_colormap('pink_r', '#f2f3d4'),
        'Viscosity (Ns/m)': shifted_colormap('YlGn', '#d9f0a3'),
        'Power_l': shifted_colormap('PuBuGn', '#d0d1e6'),
    }
    
    for i, ax in enumerate(axes.flatten()):
        qty = qtys[i]
        cmap = adjusted_cmps.get(qty, cmps[qty])

        if qty in FEATURE_DISPLAY_SETTINGS:
            vmn, vmx = FEATURE_DISPLAY_SETTINGS[qty]['range']
            cbar_min, cbar_max = vmn, vmx
            cbar_ticks = FEATURE_DISPLAY_SETTINGS[qty]['ticks']
            cbar_ticklabels = FEATURE_DISPLAY_SETTINGS[qty]['ticklabels']
        else:
            vals_all = pd.to_numeric(newres[qty], errors='coerce').to_numpy()
            vals_all = vals_all[np.isfinite(vals_all)]
            if vals_all.size > 0:
                vmn = float(np.nanmin(vals_all))
                vmx = float(np.nanmax(vals_all))
            else:
                vmn, vmx = 0.0, 1.0
            if vmn == vmx:
                vmx = vmn + 1.0
            cbar_min, cbar_max = vmn, vmx
            cbar_ticks = np.linspace(vmn, vmx, 5).tolist()
            cbar_ticklabels = [f"{t:g}" for t in cbar_ticks]

        norm = Normalize(vmin=vmn, vmax=vmx)


        # Determine marker sizes
        if qty == "Z (um)":
            # For Z (um), plot ALL points with fixed size 60 or 50; 20/45/60 or 15/30/50
            plotdata = newres.loc[~newres[qty].isna()]
            sizes = 60
        elif qty in ["Elasticity_eff (Pa)", "Viscosity (Ns/m)", "Power_l"]:
            plotdata = newres.loc[~newres[qty].isna()]
            diff = plotdata['Diff (nm)']
            sizes = diff.copy()
            sizes[diff > 150] = 20
            sizes[(diff > 50) & (diff <= 150)] = 45
            sizes[diff <= 50] = 60
        else:
            # For other maps, split by significance
            sig = newres.loc[(newres['Perr_l'] <= PERR_MAX) & (~newres[qty].isna())]
            nonsig = newres.loc[(newres['Perr_l'] > PERR_MAX) & (~newres[qty].isna())]


        # Plot for Z and Diff-scaled maps
        if qty in ["Z (um)", "Elasticity_eff (Pa)", "Viscosity (Ns/m)", "Power_l"]:
            plot_vals = pd.to_numeric(plotdata[qty], errors='coerce').to_numpy()
            plot_vals = np.clip(plot_vals, vmn, vmx)

            ax.scatter(
                plotdata['X'], plotdata['Y'],
                c=plot_vals, cmap=cmap, norm=norm,
                s=sizes, marker='s',
                edgecolors='none'
            )

        # Plot for maps using significance split
        else:
            # Significant points s=60 or 50
            if not sig.empty:
                sig_vals = pd.to_numeric(sig[qty], errors='coerce').to_numpy()
                sig_vals = np.clip(sig_vals, vmn, vmx)

                ax.scatter(
                    sig['X'], sig['Y'],
                    c=sig_vals, cmap=cmap, norm=norm,
                    s=60, marker='s',
                    edgecolors='none'
                )

            # Non-significant points 
            if not nonsig.empty:
                nonsig_vals = pd.to_numeric(nonsig[qty], errors='coerce').to_numpy()
                nonsig_vals = np.clip(nonsig_vals, vmn, vmx)

                ax.scatter(
                    nonsig['X'], nonsig['Y'],
                    c=nonsig_vals, cmap=cmap, norm=norm,
                    s=20, marker='s',
                    edgecolors='none'
                )
        
        # Axis labels and scaled ticks
        ax.set_xlabel('X (µm)', fontsize=6, fontname='Arial')
        ax.set_ylabel('Y (µm)', fontsize=6, fontname='Arial')
        ax.set_aspect('equal', adjustable='box')

        
        #==== Safe colorbar creation using the SAME normalization as the plot ====
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])

        c = plt.colorbar(mappable, ax=ax)
        c.set_ticks(cbar_ticks)
        c.set_ticklabels(cbar_ticklabels)
        c.ax.tick_params(labelsize=6, colors='black', length=2)

        for lab in c.ax.get_yticklabels():
            lab.set_fontname('Arial')
            lab.set_fontsize(6)

        #==== scale map axes in 100-um labels across the full map extent ====
        grid_step_um = 50
        half_step_idx = 0.5
        tick_step_um = 100

        max_x_um = int(nx * grid_step_um)
        max_y_um = int(ny * grid_step_um)

        tick_um_x = np.arange(0, max_x_um + 1, tick_step_um)
        tick_um_y = np.arange(0, max_y_um + 1, tick_step_um)

        tick_idx_x = (tick_um_x / grid_step_um) + xmin - half_step_idx
        tick_idx_y = (tick_um_y / grid_step_um) + ymin - half_step_idx

        ax.set_xlim(xmin - half_step_idx, xmax + half_step_idx)
        ax.set_ylim(ymin - half_step_idx, ymax + half_step_idx)

        ax.set_xticks(tick_idx_x)
        ax.set_yticks(tick_idx_y)

        ax.set_xticklabels([str(int(t)) for t in tick_um_x], fontname='Arial', fontsize=6, color='black')
        ax.set_yticklabels([str(int(t)) for t in tick_um_y], fontname='Arial', fontsize=6, color='black')

        for spine in ax.spines.values():
            spine.set_color('black')

        # Set title above the plot and bold
        ax.set_title(qty, fontweight='bold', loc='center', pad=5)
    f
    plt.subplots_adjust(hspace=0.55, wspace=0.4)

    try:
        plt.tight_layout(rect=[0, 0, 1, 0.97], pad=1.0, h_pad=1.2, w_pad=1.0)
    except Exception as e:
        print(f"[WARN] tight_layout failed: {e}; enabling constrained_layout fallback.")
        # Use the figure created earlier in Generate_map
        fig.set_constrained_layout(True)

    filename_base = f"{output}/{slice_name}_maps"
    plt.savefig(f"{filename_base}.tiff", dpi=600)  # TIFF for publication
    plt.savefig(f"{filename_base}.png", dpi=600)   # PNG high-res
    plt.savefig(f"{filename_base}.pdf")            # Vector PDF
    plt.savefig(f"{filename_base}.jpg", dpi=600)   # JPG as backup
    plt.close('all')


    # export map 
    # Change increments to 50 um
    increment = 50
    # xmax, ymax are index maxima; build index grids
    xs_idx = np.arange(xmin, int(xmax) + 1)   # 1,2,...,xmax
    ys_idx = np.arange(ymin, int(ymax) + 1)   # 1,2,...,ymax

    # Column/row labels in microns, but keep lookups in indices
    # Index xmin is the 0 um reference, so labels are unchanged from before.
    cols = [f"X-{int((i - xmin) * increment)}" for i in xs_idx]
    rows = [f"Y-{int((j - ymin) * increment)}" for j in np.flip(ys_idx)]

    with ExcelWriter(output+'/maps.xlsx') as writer:
        for qty in qtys:
            grid = np.ndarray(shape=(len(ys_idx),len(xs_idx)))
            for j, y_idx in enumerate(np.flip(ys_idx)):
                for i, x_idx in enumerate(xs_idx):
                    element = allres.loc[(allres['X']== x_idx) & (allres['Y']==y_idx)][qty].to_numpy()
                    grid[j][i] = element[0] if len(element) > 0 else np.nan
            
            df_grid = pd.DataFrame(grid, columns=cols, index=rows)
            sheetname = qty.replace('/', ' vs ')
            df_grid.to_excel(writer, sheetname)
    return newres


def Calculate_correlation(newres,qtys):
    corrs =[]
    heads = list(newres)
    newres=newres.replace([np.inf],np.nan)
    newres = newres.dropna()
    
    if len(qtys)==0: #by default 
        qtys = [heads[6],heads[9],heads[12],heads[11],heads[14],'Elasticity_eff (Pa)','Viscosity (Ns/m)', 'Power_l','Diff (nm)','Perr_eff'] 
    
    for i in range(len(qtys)):
        for j in range(len(qtys)):
            if j<=i:
                continue
            
            x = newres[qtys[i]]
            y = newres[qtys[j]]

            # Combine x and y, drop NaNs and infs
            xy = pd.concat([x, y], axis=1).dropna()
            xy = xy[~xy.isin([np.inf, -np.inf]).any(axis=1)]

            if len(xy) >= 2:
                corr, pval = pearsonr(xy.iloc[:, 0], xy.iloc[:, 1])
                corrs.append([qtys[i], qtys[j], corr, pval])


    corrs = sorted(corrs,key = lambda x:np.abs(x[2]),reverse=True)
    corrs = pd.DataFrame(corrs,columns=['Qty1','Qty2','Corr','Pval'])
    corrs.to_csv(output+'/correlations.csv')


def export_tissue_summary_stats(allres, root, slice_name):
    summary_features = [
        'Z (um)',
        'Energy_l (fJ)',
        'Energy_r (fJ)',
        'Energy_ad (fJ)',
        'Loss',
        'E_ad/E_l',
        'Elasticity_l (Pa)',
        'Elasticity_r (Pa)',
        'Elasticity_eff (Pa)',
        'Viscosity (Ns/m)',
        'Power_l',
        'Power_r',
        'Diff (nm)',
        'Perr_l',
        'Perr_eff',
    ]

    row = {'Tissue': slice_name}

    for col in summary_features:
        vals = pd.to_numeric(allres[col], errors='coerce')
        row[f'{col} Mean'] = np.nanmean(vals)
        row[f'{col} SD'] = np.nanstd(vals)
        row[f'{col} Median'] = np.nanmedian(vals)

    summary_path = os.path.join(root, 'NI_tissue_summary_stats.csv')
    row_df = pd.DataFrame([row])

    if os.path.exists(summary_path):
        old = pd.read_csv(summary_path)
        old = old[old['Tissue'] != slice_name]
        new = pd.concat([old, row_df], ignore_index=True)
    else:
        new = row_df

    new.to_csv(summary_path, index=False)
    print(f"[OK] Wrote tissue summary stats: {summary_path}")


# =============================================================================
# MAIN MAIN MAIN MAIN 
# =============================================================================

import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
import multiprocessing
if __name__ == '__main__':

    allres = []
    error = []
    files = sorted(os.listdir(fileinput))
    # keep only real .txt files
    files = [f for f in files
            if f.lower().endswith('.txt')
            and not (f.startswith('._') or f.lower().startswith('.ds_store'))]

    # Pre-create all output folders to avoid race conditions
    for f in files:
        outputsp = os.path.join(output, f[:-4])
        if not os.path.exists(outputsp):
            os.makedirs(outputsp, exist_ok=True)

    for n, file in enumerate(files):
        print(n,str(file))
    print('all points no.',len(files))



    #ProcessFile(files[0])  # test the first file to see if the code works
    # ProcessFile(files[64])  # test the second file to see if the code works
    # # ===== If you don't have paralleled computational power, then use the loop =====

    # # for f in files[:10]:
    # #     t0 = time.time()
    # #     r = ProcessFile(f)
    # #     if type(r)!=str:
    # #         allres.append(r)
    # #     else:
    # #         error.append([f,r])
    # #     ts = time.time()
    # #     print(f, 'takes', ts-t0, 'sec to process')

    # =============================================================================
    # Parallel
    # =============================================================================
    t0 = time.time()
    freeze_support() # if you use windows to run the code, unmute this line
    core = multiprocessing.cpu_count() 
    
    print('Start Parallellization, core no.',str(core))
    with ProcessPoolExecutor(max_workers=min(len(files), 56)) as executor:
        for f, res in zip(files, executor.map(ProcessFile, files)):
            if isinstance(res, str) and 'error' in res.lower():
                print(f"[ERROR] Failed for file: {f} -> {res}")
            else:
                print(f"[OK] Processed {f}")
    ts = time.time()
    print('All takes', ts-t0, 'sec to process')

    # =============================================================================
    # Integrate the point results
    # =============================================================================
    import pathlib
    allresfile = list(pathlib.Path(output).rglob('*result.csv'))
    print(len(allresfile))

    for filename in allresfile:
        resdf = pd.read_csv(filename,index_col=0, encoding='latin1')
        allres.append(resdf)

    if len(allres) == 0:
        print("[FATAL] No successful outputs found. Check for errors earlier in processing.")
        sys.exit(1)
    allres = pd.concat(allres, ignore_index=True)
    print(allres)

    # =============================================================================
    # Clean values before saving all samples
    # =============================================================================
    min_val, max_val = 7.6, 22726

    allres['Elasticity_l_raw_before_limits'] = allres['Elasticity_l (Pa)']
    allres['Elasticity_eff_raw_before_limits'] = allres['Elasticity_eff (Pa)']

    high_elasticity = allres.loc[
        allres['Elasticity_l_raw_before_limits'] > max_val
    ].copy()
    if len(high_elasticity) > 0:
        high_elasticity.to_csv(output + '/elasticity_l_above_max.csv', index=False)

    low_elasticity = allres.loc[
        allres['Elasticity_l_raw_before_limits'] < min_val
    ].copy()
    if len(low_elasticity) > 0:
        low_elasticity.to_csv(output + '/elasticity_l_below_min.csv', index=False)

    # significance masking
    allres.loc[allres['Perr_l'] > PERR_MAX, 'Elasticity_l (Pa)'] = np.nan

    # physical-range masking
    allres.loc[
        (allres['Elasticity_l (Pa)'] < min_val) |
        (allres['Elasticity_l (Pa)'] > max_val),
        'Elasticity_l (Pa)'
    ] = np.nan

    allres.loc[
        (allres['Elasticity_eff (Pa)'] < min_val) |
        (allres['Elasticity_eff (Pa)'] > max_val),
        'Elasticity_eff (Pa)'
    ] = np.nan

    # Z masking
    if 'IndentationPresent' in allres.columns:
        allres['Z (um)'] = allres['Z (um)'].where(allres['IndentationPresent'] == True, np.nan)

    allres.loc[allres['Z (um)'] > 5000, 'Z (um)'] = np.nan

    # Save
    parentfilename = os.path.basename(fileinput)
    allres.to_csv(output + f'/NI_{parentfilename}_all_samples.csv')
    
    export_tissue_summary_stats(allres, root, slice_name)



    # =============================================================================
    # Generating Maps
    # =============================================================================
    ## All quantitys to be selected are listed in the column names of the "all samples"
    selected_qty = [] # if you don't want to customize the selection, use the default one

    newres = Generate_map(selected_qty) # return the cleaned all-samples table used for plotting

    # =============================================================================
    # Calculate Correlations
    # =============================================================================
    Calculate_correlation(newres,selected_qty)