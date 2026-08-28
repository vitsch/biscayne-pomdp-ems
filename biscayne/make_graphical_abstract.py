#!/usr/bin/env python3
"""
Revised Graphical Abstract — Jakaite & Schetinin (2026)
"Integrating Robust Satisficing, Directed Learning, and Real Options
 for Resilient Environmental Decisions under Deep Uncertainty"

Three-column layout:
  Left   — Deep Uncertainty  (aquifer schematic + challenge bullets)
  Centre — Unified POMDP     (three-component diagram + equation)
  Right  — Empirical Results (6-strategy bar chart, Biscayne Aquifer)

Outputs:
  latex/els_GA_revised.png  (150 dpi raster for preview)
  latex/els_GA_revised.pdf  (vector, for LaTeX inclusion)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size':   9,
    'pdf.fonttype': 42,
})

# ── Palette ──────────────────────────────────────────────────────────────────
C = dict(
    bdt='#90A4AE', robust='#81C784', myopic='#FFB74D',
    dapp='#42A5F5', rdm='#AB47BC',  unified='#E53935',
    caution='#1565C0', learning='#2E7D32', option='#E65100',
    sky='#B3E5FC', ground='#795548', freshw='#BBDEFB',
    saltw='#FF8A80', rock='#BDBDBD', sea='#1E88E5',
)

# ── Canvas  (wide landscape, good for EMS graphical abstract) ────────────────
FW, FH = 15.6, 5.6
fig = plt.figure(figsize=(FW, FH), facecolor='white', dpi=150)

# Column boundaries in figure-fraction units  [left-edges ... right-edge]
COL = [0.010, 0.338, 0.660, 0.998]
TOP, BOT = 0.968, 0.040

# Thin separator lines
for xc in COL[1:3]:
    fig.add_artist(plt.Line2D([xc, xc], [BOT, TOP],
                              transform=fig.transFigure,
                              color='#BBBBBB', linewidth=1.0))


# ════════════════════════════════════════════════════════════════════════════
#  PANEL 1 — Deep Uncertainty
# ════════════════════════════════════════════════════════════════════════════
L1 = COL[0] + 0.004
W1 = COL[1] - COL[0] - 0.008
ax1 = fig.add_axes([L1, BOT, W1, TOP - BOT])
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis('off')

ax1.text(0.5, 0.985, 'Deep Uncertainty',
         ha='center', va='top', fontsize=13, fontweight='bold', color='#212121')

# ── Aquifer cross-section ────────────────────────────────────────────────────
X0, X1 = 0.04, 0.90   # schematic left / right (rightmost column = sea)
YB, YT  = 0.372, 0.908

ax1.add_patch(mpatches.FancyBboxPatch(    # sky
    (X0, 0.775), X1 - X0, YT - 0.775,
    boxstyle='square,pad=0', facecolor=C['sky'], edgecolor='none'))
ax1.add_patch(plt.Circle(                 # sun
    (0.13, 0.858), 0.040,
    facecolor='#FDD835', edgecolor='#F9A825', linewidth=1.2))
for cx, cy, cr in [                       # cloud
    (0.54, 0.848, 0.038), (0.60, 0.862, 0.050), (0.66, 0.848, 0.038)]:
    ax1.add_patch(plt.Circle(
        (cx, cy), cr, facecolor='#B0BEC5', edgecolor='none', alpha=0.85))
ax1.add_patch(mpatches.FancyBboxPatch(    # ground
    (X0, 0.722), X1 - X0, 0.053,
    boxstyle='square,pad=0', facecolor=C['ground'], edgecolor='none'))
ax1.add_patch(mpatches.FancyBboxPatch(    # freshwater
    (X0, 0.558), X1 - X0, 0.164,
    boxstyle='square,pad=0', facecolor=C['freshw'], edgecolor='none'))
ax1.text(0.28, 0.640, 'Freshwater', ha='center', va='center',
         fontsize=8, color='#0D47A1', style='italic')
ax1.add_patch(plt.Polygon(               # saltwater wedge
    [[X1, YB], [X1, 0.558], [0.30, 0.558]],
    facecolor=C['saltw'], edgecolor='#C62828', linewidth=0.8, alpha=0.82))
ax1.text(0.70, 0.498, 'Saltwater\nintrusion', ha='center', va='center',
         fontsize=7.5, color='#B71C1C', style='italic', multialignment='center')
ax1.add_patch(mpatches.FancyBboxPatch(   # rock base
    (X0, YB), X1 - X0, 0.040,
    boxstyle='square,pad=0', facecolor=C['rock'], edgecolor='none', alpha=0.8))
ax1.add_patch(mpatches.FancyBboxPatch(   # sea column
    (X1, YB), 1.0 - X1, YT - YB,
    boxstyle='square,pad=0', facecolor=C['sea'], edgecolor='none', alpha=0.35))
ax1.text(0.945, 0.634, 'Sea', ha='center', va='center',
         fontsize=7.5, color='#0D47A1', fontweight='bold', rotation=90)
xw = np.linspace(X1, 1.0, 40)           # wavy sea-level line
ax1.plot(xw, 0.658 + 0.011*np.sin(np.linspace(0, 5*np.pi, 40)),
         color=C['sea'], linewidth=1.5)
ax1.add_patch(mpatches.FancyBboxPatch(   # schematic border
    (X0, YB), 1.0 - X0, YT - YB,
    boxstyle='square,pad=0', facecolor='none',
    edgecolor='#555555', linewidth=1.2))
px = 0.30                                # pump T-shape
ax1.plot([px, px], [0.558, 0.808], color='#212121', linewidth=2.5)
ax1.plot([px-0.09, px+0.09], [0.808, 0.808], color='#212121', linewidth=3.5)
ax1.text(px, 0.840, 'Q', ha='center', va='bottom',
         fontsize=10, fontweight='bold', color='#212121')
for lbl, bx, by in [                     # competing model boxes
    ('M₁', 0.155, 0.490), ('M₂', 0.320, 0.438), ('M₃', 0.490, 0.393)]:
    ax1.add_patch(mpatches.FancyBboxPatch(
        (bx-0.060, by-0.031), 0.120, 0.062,
        boxstyle='square,pad=0.01', facecolor='white',
        edgecolor='#666666', linewidth=0.8, linestyle='--', alpha=0.90))
    ax1.text(bx, by, lbl, ha='center', va='center', fontsize=8.5, color='#333333')
for xq, yq in [(0.615, 0.460), (0.740, 0.416)]:  # ? marks
    ax1.text(xq, yq, '?', ha='center', va='center',
             fontsize=15, color='#C62828', fontweight='bold', alpha=0.65)

# challenge bullets
for k, txt in enumerate([
    'Non-stationary climate regimes (DRY / STAT / WET)',
    'Competing intrusion models (M₁ sharp · M₂ dispersive · M₃ tipping)',
    'Noisy salinity proxy observations',
    'Irreversible aquifer salinisation',
]):
    ax1.text(0.04, 0.322 - k*0.079, f'• {txt}',
             ha='left', va='top', fontsize=8.2, color='#333333')


# ════════════════════════════════════════════════════════════════════════════
#  PANEL 2 — Unified Framework
# ════════════════════════════════════════════════════════════════════════════
L2 = COL[1] + 0.006
W2 = COL[2] - COL[1] - 0.012
ax2 = fig.add_axes([L2, BOT, W2, TOP - BOT])
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis('off')

ax2.text(0.5, 0.985, 'Unified Robust-Adaptive POMDP',
         ha='center', va='top', fontsize=12.5, fontweight='bold', color='#212121')

# hexagon
CX, CY, CR = 0.50, 0.530, 0.138
ang = np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/6
ax2.add_patch(plt.Polygon(
    list(zip(CX + CR*np.cos(ang), CY + CR*np.sin(ang))),
    facecolor='#ECEFF1', edgecolor='#455A64', linewidth=2.2))
ax2.text(CX, CY+0.022, 'UNIFIED', ha='center', va='center',
         fontsize=9, fontweight='bold', color='#37474F')
ax2.text(CX, CY-0.022, 'POLICY',  ha='center', va='center',
         fontsize=9, fontweight='bold', color='#37474F')

# component boxes
def comp_box(ax, cx, cy, title, subtitle, col):
    bw, bh = 0.272, 0.158
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx-bw/2, cy-bh/2), bw, bh,
        boxstyle='round,pad=0.018',
        facecolor=col+'28', edgecolor=col, linewidth=2.0))
    ax.text(cx, cy+0.024, title, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=col, multialignment='center')
    ax.text(cx, cy-0.046, subtitle, ha='center', va='center',
            fontsize=7.8, color='#555555', style='italic', multialignment='center')

comp_box(ax2, 0.50, 0.845, 'Robust\nSatisficing', 'Viability ≥ Uₘᵢₙ', C['caution'])
comp_box(ax2, 0.14, 0.415, 'Directed\nLearning',   'Non-myopic VoI',      C['learning'])
comp_box(ax2, 0.86, 0.415, 'Real\nOptions',         'Delay & Flexibility', C['option'])

# arrows
def arrow2(ax, x0, y0, x1, y1, col):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=col, linewidth=1.8,
                                connectionstyle='arc3,rad=0'))

arrow2(ax2, 0.50, 0.766, 0.50, CY+CR*1.05, C['caution'])
arrow2(ax2, 0.268, 0.440, CX-CR*0.87, CY-CR*0.50, C['learning'])
arrow2(ax2, 0.732, 0.440, CX+CR*0.87, CY-CR*0.50, C['option'])

# input / output arrows on hex flanks
ax2.annotate('', xy=(CX-CR*1.05, CY), xytext=(0.04, CY),
             arrowprops=dict(arrowstyle='->', color='#757575', linewidth=1.6))
ax2.text(0.02, CY+0.050, 'Input\nθ', ha='center', va='bottom',
         fontsize=8, color='#616161')
ax2.annotate('', xy=(0.96, CY), xytext=(CX+CR*1.05, CY),
             arrowprops=dict(arrowstyle='->', color='#757575', linewidth=1.6))
ax2.text(0.98, CY+0.050, 'Policy\nπ*', ha='center', va='bottom',
         fontsize=8, color='#616161')

# objective equation
ax2.text(0.5, 0.213,
         r'$\max_{a}\;\mathbb{E}[R]+\lambda\,\mathrm{VoI}+\mu\,\mathrm{OV}$'
         '\n'
         r'$\mathrm{s.t.}\;\mathbb{P}(U\!\geq\!U_{\min})\geq 1-\delta$',
         ha='center', va='center', fontsize=9.5, color='#333333',
         multialignment='center',
         bbox=dict(boxstyle='round,pad=0.30', facecolor='#FFF9C4',
                   edgecolor='#F9A825', linewidth=1.4))

ax2.text(0.5, 0.050,
         'From fragile optimisation to safe, adaptive,\nand flexible environmental management',
         ha='center', va='bottom', fontsize=8.5, color='#616161',
         style='italic', multialignment='center')


# ════════════════════════════════════════════════════════════════════════════
#  PANEL 3 — Empirical Results
#  Strategy names are rendered as text LEFT of ax_bar; ax_bar is inset
#  (shifted right to leave room for labels inside the panel).
# ════════════════════════════════════════════════════════════════════════════
L3 = COL[2] + 0.006
W3 = COL[3] - COL[2] - 0.008

# Panel title + subtitle as figure text
fig.text(L3 + W3/2, 0.972, 'Empirical Validation',
         ha='center', va='top', fontsize=12.5, fontweight='bold', color='#212121',
         transform=fig.transFigure)
fig.text(L3 + W3/2, 0.912, 'Biscayne Aquifer  ·  ERA5 2000–2023  ·  n = 1,000 MC trials',
         ha='center', va='top', fontsize=8.5, color='#555555', style='italic',
         transform=fig.transFigure)

# Callout strip at the very bottom of the right panel
CALL_H = 0.120                             # height fraction of figure
callout_b = BOT
callout_ax = fig.add_axes([L3, callout_b, W3, CALL_H])
callout_ax.set_xlim(0, 1); callout_ax.set_ylim(0, 1); callout_ax.axis('off')
callout_ax.add_patch(mpatches.FancyBboxPatch(
    (0.01, 0.07), 0.98, 0.88,
    boxstyle='round,pad=0.02',
    facecolor='#FFEBEE', edgecolor=C['unified'], linewidth=1.8))
callout_ax.text(0.5, 0.67,
                '0.0% failure  ·  86% lower cost vs BDT',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=C['unified'])
callout_ax.text(0.5, 0.26,
                'Proactive real-options commitment eliminates DAPP reactive lag',
                ha='center', va='center', fontsize=8, color='#B71C1C')

# Bar chart axes — indented from L3 to give room for y-tick labels
LABEL_FRAC = 0.395        # fraction of W3 reserved for y-tick labels
bar_left   = L3 + LABEL_FRAC * W3
bar_width  = W3 * (1.0 - LABEL_FRAC) - 0.006
bar_bottom = callout_b + CALL_H + 0.038
bar_top    = 0.900
ax_bar = fig.add_axes([bar_left, bar_bottom, bar_width, bar_top - bar_bottom])

# Data (worst → best, displayed bottom → top)
labels = [
    'BDT Baseline',
    'Robust Satisficing',
    'Myopic POMDP',
    'DAPP  (Haasnoot 2013)',
    'RDM-Cons.  (Lempert 2006)',
    'Unified Framework',
]
rates  = [58.6, 43.5, 54.3,  9.8,   0.0,   0.0]
colors = [C['bdt'], C['robust'], C['myopic'], C['dapp'], C['rdm'], C['unified']]

y = np.arange(len(labels))
bars = ax_bar.barh(y, rates, color=colors, height=0.60,
                   edgecolor='white', linewidth=0.6, zorder=3)

# Highlight row: Unified Framework (top bar, y=5)
ub = bars[5]
ax_bar.add_patch(mpatches.FancyBboxPatch(
    (-0.5, ub.get_y()-0.07), 74, ub.get_height()+0.14,
    boxstyle='round,pad=0.02',
    facecolor='#FFEBEE', edgecolor=C['unified'], linewidth=1.5, zorder=1))

# Value labels right of each bar
for bar, val, col in zip(bars, rates, colors):
    xlab = val + 0.7 if val > 0 else 0.7
    ax_bar.text(xlab, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', ha='left',
                fontsize=9, fontweight='bold', color=col, zorder=4)

# Dashed line between EMS benchmarks (DAPP/RDM, y=3&4) and others (y=0-2)
ax_bar.axhline(y=2.5, color='#AAAAAA', linewidth=0.9, linestyle='--', zorder=2)

# Strategy name labels rendered manually to the LEFT of ax_bar, inside panel
# Using figure text with ha='right' placed just left of bar_left
for i, (lbl, col) in enumerate(zip(labels, colors)):
    # Convert data y-coord → figure fraction
    # ax_bar y-axis: ylim set to [-0.5, 5.5] below, total span 6
    # bar center at y=i; map to figure coords
    bar_h_fig = bar_top - bar_bottom          # height in figure fraction
    ylim_span = 6.0                           # set explicitly below
    y_fig = bar_bottom + bar_h_fig * ((i + 0.5) / ylim_span)
    fig.text(bar_left - 0.006, y_fig, lbl,
             ha='right', va='center', fontsize=8.5, color=col,
             fontweight='bold' if lbl == 'Unified Framework' else 'normal',
             transform=fig.transFigure)

# Hide y-axis tick labels (using our manual labels above)
ax_bar.set_yticks([])
ax_bar.set_xlabel('Saltwater intrusion failure rate (%)', fontsize=9)
ax_bar.set_xlim(0, 72)
ax_bar.set_ylim(-0.5, 5.5)          # ylim_span = 6.0 above
ax_bar.tick_params(axis='x', labelsize=8.5, width=0.8)
ax_bar.spines[['top', 'right', 'left']].set_visible(False)
ax_bar.spines['bottom'].set_linewidth(0.8)
ax_bar.grid(axis='x', linewidth=0.4, color='#E0E0E0', zorder=0)

# EMS benchmarks brace (small annotation inside bars)
ax_bar.annotate('', xy=(67, 3.45), xytext=(67, 4.55),
                arrowprops=dict(arrowstyle=']-[', color='#888888', linewidth=0.9))
ax_bar.text(68.5, 4.0, 'EMS\nbenchmarks', ha='left', va='center',
            fontsize=7.5, color='#666666', multialignment='center',
            style='italic')

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.001,
         'Jakaite & Schetinin  ·  University of Bedfordshire  ·  '
         'Environmental Modelling & Software  ·  2026',
         ha='center', va='bottom', fontsize=7.5, color='#9E9E9E',
         transform=fig.transFigure)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), '..', 'latex')
for ext in ('png', 'pdf'):
    path = os.path.join(out_dir, f'els_GA_revised.{ext}')
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {path}')

plt.close(fig)
