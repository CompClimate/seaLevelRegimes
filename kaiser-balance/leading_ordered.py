#!/usr/bin/env python3
import os
import sys
import string

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, to_hex
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

plt.rcParams["font.family"] = "DejaVu Sans"


def add_map_features(ax, draw_countries=False, draw_labels=False, right=False):
    # Add coastlines
    ax.coastlines(color="lightgrey", linewidth=0.1125)

    # Add land and states
    ax.add_feature(cfeature.LAND, facecolor='darkgrey', edgecolor='lightgrey', linewidth=0.15, zorder=-1)
    if draw_countries:
        ax.add_feature(cfeature.BORDERS, linewidth=0.125, alpha=0.25, edgecolor='black')

    # Create gridlines and specify the coordinate system
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=draw_labels,
                      linewidth=0.5, color='black', alpha=0.25, linestyle='--', x_inline=False, y_inline=False)

    if draw_labels:
        # Customize the labels format
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        # Optional: Customize label appearance
        gl.xlabel_style = {'size': 8, 'color': 'black'}
        gl.ylabel_style = {'size': 8, 'color': 'black'}

        # Remove default labels on the top and right sides
        gl.top_labels = False
        gl.right_labels = right


# Add local module to the path
source = os.path.abspath('/home/djeutsch/Projects/seaLevelRegimes')
if source not in sys.path:
    sys.path.append(source)

# Import local modules
from kaiserbalance import kaiser_balance as kbal

# Set up the base directory
BASE_DIR = "/group/maikesgrp/laique/SLVP"


# Load the data: CM4X-p25 global BVB fields, labels, and entropy
N = 10
data_res = "p25"
path_bvb = f"{BASE_DIR}/inputs/global_CM4X_{data_res}_BVB_tm_fields.zarr"
path_lbs = f"{BASE_DIR}/inputs/nn_labels_HAC10_emb_ID17_CM4X_{data_res}_tm.zarr"
path_ent = f"{BASE_DIR}/inputs/entropy_HAC10_emb_ID17_CM4X_{data_res}_tm.zarr"
ds_bvb_p25 = xr.open_zarr(path_bvb, chunks=None)
da_lbl_p25 = xr.open_zarr(path_lbs, chunks=None).bvb_regime.copy()
da_ent_p25 = xr.open_zarr(path_ent, chunks=None).entropy.copy()

# Load the data: CM4X-p125 global BVB fields, labels, and entropy
data_res = "p125"
path_bvb = f"{BASE_DIR}/inputs/global_CM4X_{data_res}_BVB_tm_fields.zarr"
path_lbs = f"{BASE_DIR}/inputs/nn_labels_HAC10_emb_ID19_CM4X_{data_res}_tm.zarr"
path_ent = f"{BASE_DIR}/inputs/entropy_HAC10_emb_ID19_CM4X_{data_res}_tm.zarr"
ds_bvb_p125 = xr.open_zarr(path_bvb, chunks=None)
da_lbl_p125 = xr.open_zarr(path_lbs, chunks=None).bvb_regime.copy()
da_ent_p125 = xr.open_zarr(path_ent, chunks=None).entropy.copy()

nf = 9
quantile = 1
percent_cutoff = None
df_magnitudes_p25, df_dominant_balance_p25 = kbal.dominant_balance(ds_bvb_p25, da_lbl_p25, da_ent_p25, nf=nf,
                                                                   qt=quantile, percent_cutoff=percent_cutoff, bias_flag='unbiased')
df_magnitudes_p125, df_dominant_balance_p125 = kbal.dominant_balance(ds_bvb_p125, da_lbl_p125, da_ent_p125, nf=nf,
                                                                     qt=quantile, percent_cutoff=percent_cutoff, bias_flag='unbiased')

#  Configuration for creating the figure
lw = 0.125
colors = ["#abdc19", "#ced2ff", "#D24165", "#CD2FE2", "#e5c0f9ff",
          "#ff8200", "#0F8979", "#CC9739", "#999933", "#44CE31",]
cmap = ListedColormap(colors, name="cluster_cmap")
label_bounds = np.arange(0.5, N + 1.5, 1)
label_norm = mcolors.BoundaryNorm(label_bounds, cmap.N, clip=True)

# Prepare data
for data_res, (da, da_ent, df) in zip(("p25", "p125"), ((da_lbl_p25, da_ent_p25, df_dominant_balance_p25), (da_lbl_p125, da_ent_p125, df_dominant_balance_p125))):
    # Ensure the balance table has exactly N rows
    df = df.reindex(range(N))
    mask = df.to_numpy()
    matrix = np.where(mask == 1, np.arange(1, N + 1)[:, None], 0)

    matrix_cmap = ListedColormap(["white"] + [to_hex(c) for c in cmap.colors])
    matrix_bounds = np.arange(-0.5, N + 1.5, 1)
    matrix_norm = mcolors.BoundaryNorm(matrix_bounds, matrix_cmap.N)

    # Figure and panels 
    fig = plt.figure(figsize=(18, 8), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.6])

    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
    ax3 = fig.add_subplot(gs[0, 2])

    # Dynamical regimes
    cf1 = da.plot(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap, norm=label_norm, add_colorbar=False,)

    # Entropy
    cf2 = da_ent.plot(ax=ax2, vmin=0., vmax=100., transform=ccrs.PlateCarree(), cmap=plt.cm.Reds_r, add_colorbar=False,)

    # Leading-order balance matrix
    ax3.pcolor(matrix, cmap=matrix_cmap, norm=matrix_norm, edgecolors="k", linewidths=lw,)

    # Highlight a selected cell
    r, c = 3, 0
    rect = patches.Rectangle((c, r), 1, 1, facecolor=colors[r], alpha=0.7, edgecolor="none", zorder=5,)
    ax3.add_patch(rect)

    # Highlight a range of cells with hatching
    rect = patches.Rectangle((0, 1), 9, 1, fill=False, hatch="////", edgecolor="k", linewidth=0, alpha=0.2, zorder=3,)
    ax3.add_patch(rect)

    num_rows, num_cols = df.shape
    ax3.set_yticks(np.arange(0.5, num_rows, 1))
    ax3.set_yticklabels(range(1, num_rows + 1), size=10)
    ax3.set_xticks(np.arange(0.5, num_cols + 0.5))
    ax3.set_xticklabels(df.columns, size=9, rotation=75)

    # Geographic gridlines
    for ax in (ax1, ax2):
        add_map_features(ax=ax, draw_countries=True, right=False)

    # Titles and panel labels
    row_label = f"CM4X-{data_res}"

    for ax, title, letter in zip((ax1, ax2, ax3), ("Dynamical regimes", "Entropy (%)", "Leading-order balance"), string.ascii_lowercase,):
        ax.set_title(f"{title}", fontsize=13)

    ax1.text(-0.12, 0.5, row_label, transform=ax1.transAxes, ha="right", va="center", fontsize=13, weight="bold", rotation=90,)

    # Colorbars 
    fig.subplots_adjust(hspace=0.1)
    cbar_ticks = np.arange(1, num_rows + 1)
    cax1 = fig.add_axes([0.125, 0.52, 0.23, 0.03])
    cbar1 = fig.colorbar(cf1, cax=cax1, orientation="horizontal", 
                        ticks=cbar_ticks, boundaries=label_bounds, extend="neither",)
    cbar1.ax.tick_params(labelsize=8, which="minor", direction="in", length=0)
    cbar1.set_ticklabels(list(range(1, num_rows + 1)), fontsize=8)

    cax2 = fig.add_axes([0.4, 0.52, 0.23, 0.03])
    cbar2 = fig.colorbar(cf2, cax=cax2, orientation="horizontal", extend="neither")
    cbar2.ax.tick_params(labelsize=8)

    # Save the figure
    fig.savefig(f'{BASE_DIR}/kaiser2022/figures/lead_order_balance_HAC10_emb_ID17_CM4X_{data_res}_tm.png', bbox_inches='tight', dpi=300)
