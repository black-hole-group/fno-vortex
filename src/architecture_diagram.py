import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 20)
ax.axis('off')
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')

# Color palette
C_INPUT   = '#4fc3f7'
C_OP      = '#81c784'
C_FOURIER = '#ce93d8'
C_CONV    = '#ffb74d'
C_PROJ    = '#ef9a9a'
C_SHAPE   = '#b0bec5'
C_ARROW   = '#ffffff'
C_TEXT    = '#ffffff'
C_DARK    = '#1a1a2e'


def box(ax, x, y, w, h, label, sublabel=None, color='#4fc3f7', fontsize=18):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08", linewidth=1.5,
                          edgecolor=color, facecolor=C_DARK)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.13, label, ha='center', va='center',
                color=color, fontsize=fontsize, fontweight='bold', fontfamily='monospace')
        ax.text(x, y - 0.22, sublabel, ha='center', va='center',
                color=C_SHAPE, fontsize=15, fontfamily='monospace')
    else:
        ax.text(x, y, label, ha='center', va='center',
                color=color, fontsize=fontsize, fontweight='bold', fontfamily='monospace')


def arrow(ax, x, y1, y2, label=None):
    ax.annotate('', xy=(x, y2 + 0.05), xytext=(x, y1 - 0.05),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
    if label:
        ax.text(x + 0.25, (y1 + y2) / 2, label, ha='left', va='center',
                color=C_SHAPE, fontsize=15, fontfamily='monospace')


cx = 7.0  # centre x

# ── INPUT ──────────────────────────────────────────────────────────────────
box(ax, cx, 19.2, 6.5, 0.65,
    'INPUT: 5 frames + ν, μ',
    '(batch, 128, 128, 10, 7)',
    color=C_INPUT, fontsize=18)

arrow(ax, cx, 18.87, 18.27, 'append x,y,t grid')

box(ax, cx, 18.0, 6.5, 0.45,
    '(batch, 128, 128, 10, 10)',
    color=C_SHAPE, fontsize=16)

arrow(ax, cx, 17.77, 17.17)

# ── LIFT ───────────────────────────────────────────────────────────────────
box(ax, cx, 16.9, 6.5, 0.65,
    'LIFT   fc0: Linear(10 → 30)',
    '(batch, 128, 128, 10, 30)',
    color=C_OP, fontsize=18)

arrow(ax, cx, 16.57, 16.0, 'permute → channels first')

box(ax, cx, 15.73, 6.5, 0.45,
    '(batch, 30, 128, 128, 10)',
    color=C_SHAPE, fontsize=16)

arrow(ax, cx, 15.5, 14.93, 'pad time +6 (aperiodic)')

box(ax, cx, 14.67, 6.5, 0.45,
    '(batch, 30, 128, 128, 16)',
    color=C_SHAPE, fontsize=16)

arrow(ax, cx, 14.44, 13.7)

# ── FOURIER LAYERS ─────────────────────────────────────────────────────────
fl_top    = 13.55
fl_bottom = 9.85
fl_h      = fl_top - fl_bottom

rect_outer = FancyBboxPatch((1.5, fl_bottom), 11, fl_h,
                             boxstyle="round,pad=0.12", linewidth=2,
                             edgecolor=C_FOURIER, facecolor='#16162a', linestyle='--')
ax.add_patch(rect_outer)
ax.text(cx, fl_top + 0.05, '× 5   FOURIER LAYERS', ha='center', va='bottom',
        color=C_FOURIER, fontsize=18, fontweight='bold')

# left path: SpectralConv3d
lx = 4.5
box(ax, lx, 12.9, 4.6, 0.55, 'rfftn  (x, y, t)', color=C_FOURIER, fontsize=16)
ax.annotate('', xy=(lx, 12.27), xytext=(lx, 12.62),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2))
box(ax, lx, 12.0, 4.6, 0.55,
    'truncate modes',
    '(64, 64, 5)',
    color=C_FOURIER, fontsize=16)
ax.annotate('', xy=(lx, 11.37), xytext=(lx, 11.72),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2))
box(ax, lx, 11.1, 4.6, 0.55,
    'complex multiply  R ∈ ℂ',
    '4 weight tensors (kx/ky quadrants)',
    color=C_FOURIER, fontsize=16)
ax.annotate('', xy=(lx, 10.47), xytext=(lx, 10.82),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2))
box(ax, lx, 10.2, 4.6, 0.55, 'irfftn → physical space', color=C_FOURIER, fontsize=16)

# right path: 1×1×1 Conv
rx = 10.0
box(ax, rx, 11.55, 3.6, 0.55,
    'Conv3d  1×1×1',
    '(pointwise, local)',
    color=C_CONV, fontsize=16)

# lines from input split to both paths
ax.plot([cx, lx], [13.42, 13.18], color=C_ARROW, lw=1.3)
ax.plot([cx, rx], [13.42, 11.83], color=C_ARROW, lw=1.3)

# convergence point
conv_y = 9.55
ax.plot([lx, cx], [9.93, conv_y + 0.22], color=C_ARROW, lw=1.3)
ax.plot([rx, cx], [11.28, conv_y + 0.22], color=C_ARROW, lw=1.3)

ax.text(cx, conv_y + 0.05, '+', ha='center', va='center',
        color=C_ARROW, fontsize=24, fontweight='bold')
ax.text(cx - 0.5, conv_y - 0.25, 'GELU', ha='center', va='center',
        color=C_OP, fontsize=16, fontweight='bold')

# ── after fourier layers ────────────────────────────────────────────────
arrow(ax, cx, 9.32, 8.73, 'unpad time (remove 6)')

box(ax, cx, 8.47, 6.5, 0.45,
    '(batch, 30, 128, 128, 10)',
    color=C_SHAPE, fontsize=16)

arrow(ax, cx, 8.24, 7.67, 'permute → channels last')

box(ax, cx, 7.4, 6.5, 0.45,
    '(batch, 128, 128, 10, 30)',
    color=C_SHAPE, fontsize=16)

arrow(ax, cx, 7.17, 6.6)

# ── PROJECTION ─────────────────────────────────────────────────────────────
box(ax, cx, 6.33, 6.5, 0.65,
    'PROJECT  fc1: Linear(30 → 128) + GELU',
    '(batch, 128, 128, 10, 128)',
    color=C_PROJ, fontsize=18)

arrow(ax, cx, 6.0, 5.43)

box(ax, cx, 5.17, 6.5, 0.65,
    'PROJECT  fc2: Linear(128 → 1)',
    '(batch, 128, 128, 10, 1)',
    color=C_PROJ, fontsize=18)

arrow(ax, cx, 4.84, 4.27, 'reshape / squeeze')

box(ax, cx, 4.0, 6.5, 0.65,
    'OUTPUT: predicted frames',
    '(batch, 128, 128, 10)',
    color=C_INPUT, fontsize=18)

# ── legend ─────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_INPUT,   label='I/O tensors'),
    mpatches.Patch(color=C_OP,      label='Lift / activation'),
    mpatches.Patch(color=C_FOURIER, label='Spectral path (FNO)'),
    mpatches.Patch(color=C_CONV,    label='Pointwise conv path'),
    mpatches.Patch(color=C_PROJ,    label='Projection MLP'),
    mpatches.Patch(color=C_SHAPE,   label='Tensor shape'),
]
ax.legend(handles=legend_items, loc='lower left', bbox_to_anchor=(0.01, 0.0),
          fontsize=14, framealpha=0.3, labelcolor='white',
          facecolor='#1a1a2e', edgecolor='#444')

plt.title('FNO3d Architecture & Data Flow\n(Orszag–Tang MHD surrogate)',
          color=C_TEXT, fontsize=20, fontweight='bold', pad=6)
plt.tight_layout()
plt.savefig('architecture.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved architecture.png")
