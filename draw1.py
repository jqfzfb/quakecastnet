import os
import numpy as np
import pandas as pd

from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as Polygon2

from shapely.geometry import Polygon, MultiPolygon

import cartopy.crs as ccrs  
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from cartopy.geodesic import Geodesic
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

import contextily as ctx
ctx.set_cache_dir(".contextily_cache") 

import seaborn as sns
sns.reset_defaults()

def plot_cumulative_fmd(bins, log_counts, 
                        xlabel="Magnitude (M0)",
                        ylabel=r"log$_{10}$(1 + N(M ≥ M0))",
                        figsize=(6, 4),
                        title=None,
                        save_path=None,
                        show=True,
                        nl=0,
                       ):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    bins = np.asarray(bins)
    log_counts = np.asarray(log_counts)

    assert bins.shape == log_counts.shape, \
        f"bins and log_counts must have the SAME shape, got {bins.shape} vs {log_counts.shape}"

    x = bins
    y = log_counts

    if len(x) > 1:
        bar_width = np.median(np.diff(x))
    else:
        bar_width = 0.1

    plt.figure(figsize=figsize)

    if nl > 0 and nl < len(x):
        plt.bar(x[:-nl], y[:-nl], width=bar_width, 
                color="lightgray", edgecolor="none", alpha=0.6)
        plt.bar(x[-nl:], y[-nl:], width=bar_width, 
                color="tab:blue", edgecolor="none", alpha=0.6)
    else:
        plt.bar(x, y, width=bar_width, 
                color="lightgray", edgecolor="none", alpha=0.6)

    if nl > 0 and nl < len(x):
        plt.plot(x[:-nl], y[:-nl], "--o", linewidth=1.5, markersize=4, color="tab:red")
        plt.plot(x[-nl:], y[-nl:], "--o", linewidth=1.5, markersize=4, color="tab:blue")
    else:
        plt.plot(x, y, "--o", linewidth=1.5, markersize=4, color="tab:red")

    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    if title:
        plt.title(title, fontsize=12, pad=8)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_3d_clusters_with_ellipsoids(coordinates_km, 
                                     cluster_indices, 
                                     centers_km, 
                                     ellipsoids=None, 
                                     figsize=(6, 6), 
                                     elev=30, 
                                     azim=210,
                                     alpha=0.1,
                                     cluster_mark=False,
                                     cmap_name="tab10"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    event_x, event_y, event_z = coordinates_km[:, 0], coordinates_km[:, 1], coordinates_km[:, 2]
    cmap = plt.get_cmap(cmap_name)

    for i, indices in enumerate(cluster_indices):
        ax.scatter(
            event_x[indices],
            event_y[indices],
            event_z[indices],
            color=cmap(i % 10),
            label=f"Cluster {i}",
            s=1,
            alpha=0.6,
            zorder=11,
        )

    ax.scatter(
        centers_km[:, 0],
        centers_km[:, 1],
        centers_km[:, 2],
        color="black",
        s=100,
        marker="*",
        label="Cluster centers",
        zorder=12,
    )
    if cluster_mark:
        for i, (x, y, z) in enumerate(centers_km):
            ax.text(
                x+ 0.3, y+ 0.3, z+ 0.3,
                f"{i}",
                fontsize=16,
                color='black',
                ha='center',
                va='center',
                zorder=12,
            )

    if ellipsoids is not None:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        uu, vv = np.meshgrid(u, v)
        x_sphere = np.cos(uu) * np.sin(vv)
        y_sphere = np.sin(uu) * np.sin(vv)
        z_sphere = np.cos(vv)
        sphere_points = np.stack([x_sphere, y_sphere, z_sphere], axis=-1)

        for i, ellipsoid in enumerate(ellipsoids):
            center = ellipsoid["center"]
            axes = ellipsoid["axes"]
            R = ellipsoid["rotation"]

            ellipsoid_points = sphere_points @ np.diag(axes) @ R.T + center
            x_ellip = ellipsoid_points[..., 0]
            y_ellip = ellipsoid_points[..., 1]
            z_ellip = ellipsoid_points[..., 2]

            ax.plot_surface(
                x_ellip,
                y_ellip,
                z_ellip,
                color=cmap(i % 10),
                alpha=alpha,
                edgecolor='none',
                zorder=10,
            )

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_zlabel("Depth (km)")
    ax.invert_zaxis()
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


def plot_prob_comparison(P_set1, bin_edges, t, P_set2=None, label1='ground truth', label2='Prediction'):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, P_set1[t], color='black', linewidth=2, label=label1)
    plt.fill_between(bin_centers, P_set1[t], color='gray', alpha=0.3)
    if P_set2 is not None:
        plt.plot(bin_centers, P_set2[t], color='darkblue', linewidth=2, linestyle='--', label=label2)
        plt.fill_between(bin_centers, P_set2[t], color='blue', alpha=0.2)

    plt.xlabel('Magnitude', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Magnitude Distribution Comparison at Time Step {}'.format(t), fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_magnitude_distribution(P_true, magnitude_bins, title="True Magnitude Distribution", color="royalblue"):
    assert len(P_true) == len(magnitude_bins) - 1, "P_true 与 magnitude_bins 不匹配"

    bin_centers = (magnitude_bins[:-1] + magnitude_bins[1:]) / 2
    bin_width = magnitude_bins[1] - magnitude_bins[0]

    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, P_true, width=bin_width * 0.9, color=color, edgecolor="black", alpha=0.8)
    plt.xlabel("Magnitude", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.title(title, fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_single_series(data_array, title="Time Series Plot"):
    data_array = np.ravel(data_array) 
    x = np.arange(len(data_array))

    plt.figure(figsize=(12, 5))
    plt.plot(x, data_array, color='b', linewidth=2, label="Data Series")

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()


def gradient_fill_between_values_limited(x, y1, y2, values, cmap, ax=None, **kwargs):
    colormap = plt.get_cmap(cmap)
    if ax is None:
        ax = plt.gca()

    y1 = np.array(y1)
    y2 = np.array(y2)
    values = np.array(values)

    norm = plt.Normalize(values.min(), values.max()*1.5)

    verts = []
    colors = []
    for i in range(len(x) - 1):
        verts.append([(x[i], y1[i]), (x[i], y2[i]), (x[i+1], y2[i+1]), (x[i+1], y1[i+1])])
        color_value = np.mean(values[i:i+2])
        colors.append(colormap(norm(color_value)))

    poly_collection = PolyCollection(verts, facecolors=colors, **kwargs)
    ax.add_collection(poly_collection)

def draw_wave(xs, cft=None, msk=None, trg=None, ann=None, dt=None, title=None,
              cft_colors=['r', 'b', 'g', 'y', 'c', 'k'] , 
              trg_colors=['r', 'b'], 
              ann_colors=['r', 'b'], 
              ann_styles=['-', '-'],
              clean=False, cft_visble=True, 
              figsize=(8, 2),
              tlim=None,
              save_file=None, dpi=300,
             ):

    npts = len(xs[0]) 
    dt = 1 if dt is None else dt
    t = np.linspace(0, dt*(npts-1), npts)  

    n = len(xs)
    fig, axs = plt.subplots(n, 1, figsize=figsize)
    if n == 1:
        axs = [axs]
    for i, x in enumerate(xs):
        axs[i].plot(t, x, color='k', linewidth=0.5)
        if title is not None and i == 0:
            axs[0].set_title(title)
        axs[i].grid(False)
        axs[i].set_xlim(t[0], t[-1])
        if i < n - 1:
            axs[i].set_xticks([])
        else:
             axs[i].set_xlabel("Time (s)")

        axs[i].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        axs[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))     

        if clean:
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xlabel('')  
            axs[i].set_ylabel('')  
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['left'].set_visible(False)

        if i > 0:
            axs[i].yaxis.get_offset_text().set_visible(False)   

        if msk is not None:
            y1, y2 = axs[i].get_ylim()
            gradient_fill_between_values_limited(t, np.ones(npts) * y1 , np.ones(npts) * y2, 
                                                 msk[i], 'BuPu', ax=axs[i], alpha=0.8)

        if cft is not None:
            y1, y2 = x.min(), x.max()
            for k, c in enumerate(cft):
                m = c.mean()
                if np.abs(c - m).sum() > 0:
                    on_off = np.array(trigger_onset(c, 0.8, 0.2))
                else:
                    if m > 0:
                        on_off = np.array([t[0], t[-1]])[np.newaxis, :]
                    else:
                        continue


                if len(on_off.shape) < 2:
                    continue
                t0 = np.linspace(y1, y2, npts)
                for j, (t1, t2) in enumerate(zip(on_off[:, 0], on_off[:, 1])):
                    axs[i].fill_betweenx(t0, t[t1], t[t2], 
                                      facecolor=cft_colors[k], 
                                      edgecolor=None, 
                                      alpha=0.25)

        if trg is not None:
            y1, y2 = x.min(), x.max()
            for k, c in enumerate(trg):
                m = c.mean()
                if np.abs(c - m).sum() > 0:
                    on_off = np.array(trigger_onset(c, 0.8, 0.2))
                else:
                    if m > 0:
                        on_off = np.array([t[0], t[-1]])[np.newaxis, :]
                    else:
                        continue
                if len(on_off.shape) < 2:
                    continue 
                for j, (t1, t2) in enumerate(zip(on_off[:, 0], on_off[:, 1])):
                    axs[i].vlines(t[t1], y1, y2, color=trg_colors[0], lw=1.5)
                    axs[i].vlines(t[t2], y1, y2, color=trg_colors[1], lw=1.5)

        if ann is not None:
            y1, y2 = x.min(), x.max()
            for j in range(len(ann)):
                axs[i].vlines(ann[j], y1, y2, color=ann_colors[j], lw=1.5, linestyles=ann_styles[j])

        if tlim is not None:
            axs[i].set_xlim(tlim)

        if save_file is not None:
            fig.savefig(save_file)
            plt.close()  

def get_colormap_colors(colormap_name, num_colors):
    cmap = plt.get_cmap(colormap_name)  
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)] 
    return colors

def _pcolormesh_same_dim(ax, x, y, v, **kwargs):
    try:
        return ax.pcolormesh(x, y, v, shading='nearest', **kwargs)
    except TypeError:
        return ax.pcolormesh(x, y, v[:-1, :-1], **kwargs)

def draw_sequence_same(time_data_list, title=None, fill_datetime=None, figsize=(10,6), 
                       lcolor=None, label="Value", lstyle=None, date_axis=True,
                       markersize=2):

    fig, ax1 = plt.subplots(figsize=figsize)

    num_data = len(time_data_list)
    if lcolor is None:
        lcolor = plt.cm.get_cmap('tab10').colors[:num_data]  
    if lstyle is None:
        lstyle = ['-'] * len(time_data_list)

    for j, (t1, d1) in enumerate(time_data_list):
        ax1.plot(t1, d1, marker='o', label="Time Series 1", color=lcolor[j], linestyle=lstyle[j], markersize=markersize)
    if date_axis:
        ax1.set_xlabel("Datetime", fontsize=14)
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d \n %H:%M:%S"))
    else:
        ax1.set_xlabel("Index", fontsize=14)

    ax1.set_ylabel(label, fontsize=14, color="blue")
    ax1.tick_params(axis="x", labelsize=12, rotation=45)
    ax1.tick_params(axis="y", labelsize=12, colors="blue")

    if date_axis and fill_datetime is not None:
        for x in fill_datetime:
            highlight_start, highlight_end = x
            ax1.axvspan(highlight_start, highlight_end, color='grey', alpha=0.3)

    if title is not None:
         plt.title(title, fontsize=16, y=1.05) 
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.grid(True)

    plt.show()

from matplotlib.colors import to_rgb
def generate_colors(n_colors, excluded_colors=["blue", "orange"], colormap='Greens'):
    excluded_colors_rgb = [to_rgb(color) for color in excluded_colors]

    cmap = plt.cm.get_cmap(colormap, n_colors * 2)
    all_colors = [cmap(i) for i in range(cmap.N)]

    filtered_colors = [color for color in all_colors if to_rgb(color[:3]) not in excluded_colors_rgb]

    if len(filtered_colors) < n_colors:
        raise ValueError(f"{len(filtered_colors)}")
    return filtered_colors[:n_colors]

def plot_two_timeseries(data1, data2, labels=("Series 1", "Series 2"), title=None, figsize=(10, 6), save_path=None):
    x = range(len(data1))

    plt.figure(figsize=figsize)

    plt.plot(x, data1, linestyle='-', marker='o')

    plt.plot(x, data2, linestyle='--', marker='x')

    plt.xlabel("Index", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    if title:
        plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_quantile_calibration_multi(
    results,
    labels=None,
    figsize=(6, 4),
    save_path=None,
    show_ks_point=False,
    cmap="viridis",
):

    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "lines.linewidth": 2.0,
    })

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Identity")

    if labels is None:
        labels = [f"Empirical {i+1}" for i in range(len(results))]

    timesteps = [r[3] for r in results]
    norm = plt.Normalize(min(timesteps), max(timesteps))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, (alphas_sorted, emp, ks, timestep) in enumerate(results):
        alphas_sorted = np.asarray(alphas_sorted, dtype=float).ravel()
        emp = np.asarray(emp, dtype=float).ravel()
        assert alphas_sorted.shape == emp.shape, "alphas_sorted 与 emp 长度必须一致"

        diff = np.abs(emp - alphas_sorted)
        if ks is None:
            ks = float(np.max(diff))
        ks_idx = int(np.argmax(diff))

        color = sm.to_rgba(timestep)

        ax.plot(alphas_sorted, emp, marker="o", color=color, label=f"{labels[i]} (KS={ks:.3f}, t={timestep})", alpha=0.6, linewidth=1.5)

        if show_ks_point and len(alphas_sorted) > 0:
            ax.scatter([alphas_sorted[ks_idx]], [emp[ks_idx]], s=80, color="red", zorder=3, marker="+")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02) 
    ax.set_xlabel("Nominal quantile level (α)")
    ax.set_ylabel(r"Empirical coverage  $\hat{p}(\alpha)$")
    ax.grid(True, linestyle=":", linewidth=1, alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def draw_colorbar(cmap="jet", 
                  orientation="horizontal", 
                  cmin=0, cmax=1,
                  font_size=18,
                  num_ticks=5,
                  figsize=(6,0.8),
                  decimal_places=1):
    import matplotlib as mpl
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.5)
    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        fraction=0.023,
        pad=0.02,
        orientation=orientation,
    )

    ticks = np.linspace(cmin, cmax, num_ticks)
    ticks = np.round(ticks, decimal_places)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=font_size)

    plt.show()


def draw_sequence_diff(time_data_list, uncertainty=None, title=None, figsize=(12,6), types=None, dt=None, outward=60,
                       legend=True, fill_datetime=None, mark_datetime=None, linestyles=None, nax=None, nay=None,
                       fontsize_ticks=18, xlabel=None, fontsize=14, ylims=None, mcolor='gray', bbox_to_anchor=(0.5, 1.12),
                       labels=None, lcolor=None, ucolor=None, markersize=3, markersize2=10, save_path=None, 
                       q=0.10, u_levels=None, errcapsize=3, elinewidth=1.2, ealpha=0.8, err_stride=4, error_bar=False,
                       time_format='ymd', align_zero=False, clip_unc_zero=False):
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    def _plot_unc_as_errorbar(ax, t, u_band, color,
                              q=q, u_levels=u_levels,
                              errcapsize=errcapsize, elinewidth=elinewidth, ealpha=ealpha,
                              err_stride=None, clip_zero=False):
        k = len(u_band) // 2
        if k == 0:
            return

        if u_levels is None:
            u_lv = np.linspace(0.5 / k, 0.5 - 0.5 / k, k)
        else:
            u_lv = np.asarray(u_levels)
            assert len(u_lv) == k, "u_levels 长度必须等于下侧带数量 k"

        idx = int(np.argmin(np.abs(u_lv - q)))
        lower = np.asarray(u_band[idx], dtype=float)
        upper = np.asarray(u_band[-idx - 1], dtype=float)
        center = 0.5 * (lower + upper)

        t_arr = np.asarray(t)
        n = len(t_arr)
        if err_stride is None or err_stride < 1:
            sel_idx = np.arange(n)
        else:
            sel_idx = np.arange(0, n, int(err_stride))

        t_sel = t_arr[sel_idx]
        c_sel = center[sel_idx]
        lower_sel = lower[sel_idx]
        upper_sel = upper[sel_idx]

        if clip_zero:
            keep = upper_sel > 0
            t_sel, c_sel = t_sel[keep], c_sel[keep]
            lower_sel = np.maximum(lower_sel[keep], 0)
            upper_sel = upper_sel[keep]

        yerr = np.vstack([c_sel - lower_sel, upper_sel - c_sel])

        ax.errorbar(t_sel, c_sel, yerr=yerr, fmt='none',
                    ecolor=color, elinewidth=elinewidth,
                    capsize=errcapsize, alpha=ealpha, zorder=3)

    def _fill_unc_band(ax, t, u_band, color, clip_zero=False):
        n_u = len(u_band) // 2
        for u in range(n_u):
            lower = np.asarray(u_band[u], dtype=float)
            upper = np.asarray(u_band[-u - 1], dtype=float)
            if clip_zero:
                lower = np.clip(lower, 0, None)
                upper = np.maximum(upper, 0)
            ax.fill_between(t, lower, upper,
                            facecolor=color, alpha=0.2, edgecolor='none')

    def _align_zero_axes(axes):
        axes = [ax for ax in axes if ax is not None]
        if len(axes) < 2:
            return
        y_lims = np.array([ax.get_ylim() for ax in axes], dtype=float)
        y_ext = y_lims[:, 1] - y_lims[:, 0]
        valid = y_ext > 0
        if valid.sum() < 2:
            return
        y_lims = y_lims[valid]
        y_lims[:, 0] = np.clip(y_lims[:, 0], None, 0.0)
        y_lims[:, 1] = np.clip(y_lims[:, 1], 0.0, None)
        y_ext = y_lims[:, 1] - y_lims[:, 0]
        y_reset = y_lims / y_ext[:, None]
        y_min = y_reset[:, 0].min()
        y_max = y_reset[:, 1].max()
        new_lims = np.column_stack([y_min * y_ext, y_max * y_ext])
        for ax, lim in zip([a for a, v in zip(axes, valid) if v], new_lims):
            ax.set_ylim(lim)

    fig, ax1 = plt.subplots(figsize=figsize)
    dt = pd.Timedelta(days=8) if dt is None else dt

    num_data = len(time_data_list)
    if types is None:
        types = ['line'] * num_data
    if linestyles is None:
        linestyles = [['-']*3] * num_data
    if ylims is None:
        ylims = [None] * num_data

    if lcolor is None:
        lcolor = [["#1f77b4"]*3, ["#2ca02c"]*3, ["#ff7f0e"]*3]
    if labels is None:
        labels = ["Value 1", "Value 2", "Value 3"]
    if ucolor is None:
        ucolor = [["#ff7f0e"], ["#d62728"], ["#4c72b0"]]  

    time1, data1 = time_data_list[0]
    if types[0] == 'bar':
        if isinstance(time1, list):  
            for j, (t1, d1) in enumerate(zip(time1, data1)):
                ax1.bar(t1, d1, width=dt, label=labels[0][0], facecolor=lcolor[0][0], alpha=0.1, edgecolor='none')
        else:
            ax1.bar(time1, data1, width=dt, label=labels[0][0], facecolor=lcolor[0][0], alpha=0.1, edgecolor='none')
    else:      
        if isinstance(time1, list):  
            for j, (t1, d1) in enumerate(zip(time1, data1)):
                ax1.plot(t1, d1, marker='none', linestyle=linestyles[0][j], label=labels[0], color=lcolor[0][j], markersize=markersize, alpha=0.85)
        else:
            ax1.plot(time1, data1, marker='none', linestyle=linestyles[0][0], label=labels[0], color=lcolor[0][0], markersize=markersize, alpha=0.85)

    if uncertainty is not None:
        t1s, u1s = uncertainty[0]
        for i1, (t1, u1) in enumerate(zip(t1s, u1s)):
            if error_bar:
                _plot_unc_as_errorbar(ax1, t1, u1, color=ucolor[0][i1],
                                      err_stride=err_stride, clip_zero=clip_unc_zero)
            else:
                _fill_unc_band(ax1, t1, u1, color=ucolor[0][i1], clip_zero=clip_unc_zero)

    if xlabel is not None:
        ax1.set_xlabel(xlabel, fontsize=fontsize, color='k')
    ax1.set_ylabel(labels[0], fontsize=fontsize, color=lcolor[0][0])
    ax1.tick_params(axis="y", labelsize=12, colors=lcolor[0][0])

    if ylims[0] is not None:
         ax1.set_ylim(*ylims[0])
    print("1st Y-axis range:", ax1.get_ylim())

    if num_data > 1 and len(time_data_list[1]) > 0:
        ax2 = ax1.twinx()
        time2, data2 = time_data_list[1]
        if types[1] == 'scatter':
            ax2.scatter(time2, data2, marker='D', s=markersize2, label=labels[1], color=lcolor[1][0], 
                        facecolors='none', edgecolors=lcolor[1][0], linewidth=1, alpha=0.7)
        elif types[1] == 'bar':
            if isinstance(time2, list):  
                for j, (t2, d2) in enumerate(zip(time2, data2)):
                    ax2.bar(t2, d2, width=dt, label=labels[1][0], facecolor=lcolor[1][0], alpha=0.1, edgecolor='none')
            else:
                ax2.bar(time2, data2, width=dt, label=labels[1][0], facecolor=lcolor[1][0], alpha=0.1, edgecolor='none')
        else:
            if isinstance(time2, list):  
                for j, (t2, d2) in enumerate(zip(time2, data2)):
                    ax2.plot(t2, d2, marker='none', linestyle=linestyles[1][j], linewidth=2, label=labels[1], color=lcolor[1][j], markersize=markersize, alpha=0.85)
            else:
                ax2.plot(time2, data2, linestyle=linestyles[1][0], linewidth=2, label=labels[1], color=lcolor[1][0], markersize=markersize, alpha=0.85)

        if uncertainty is not None and len(uncertainty) > 1:
            t2s, u2s = uncertainty[1]
            for i2, (t2, u2) in enumerate(zip(t2s, u2s)):
                if error_bar:
                    _plot_unc_as_errorbar(ax2, t2, u2, color=ucolor[1][i2],
                                          err_stride=err_stride, clip_zero=clip_unc_zero)
                else:
                    _fill_unc_band(ax2, t2, u2, color=ucolor[1][i2], clip_zero=clip_unc_zero)

        ax2.set_ylabel(labels[1], fontsize=fontsize, color=lcolor[1][0])
        ax2.tick_params(axis="y", labelsize=12, colors=lcolor[1][0])

        if ylims[1] is not None:
             ax2.set_ylim(*ylims[1])
        print("2nd Y-axis range:", ax2.get_ylim())

    if num_data > 2 and len(time_data_list[2]) > 0:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', outward+20))  
        time3, data3 = time_data_list[2]
        if types[2] == 'scatter':
            if isinstance(time3, list):
                for j, (t3, d3) in enumerate(zip(time3, data3)):
                    ax3.scatter(t3, d3, marker='D', s=markersize2, label=labels[2], color=lcolor[2][j], 
                                facecolors='none', edgecolors=lcolor[2][j], linewidth=1, alpha=0.7)
            else:
                ax3.scatter(time3, data3, marker='D', s=markersize2, label=labels[2], color=lcolor[2][0], 
                            facecolors='none', edgecolors=lcolor[2][0], linewidth=1, alpha=0.7)
        elif types[2] == 'bar':
            if isinstance(time3, list):  
                for j, (t3, d3) in enumerate(zip(time3, data3)):
                    ax3.bar(t3, d3, width=dt, label=labels[2][0], facecolor=lcolor[2][0], alpha=0.4, edgecolor='none')
            else:
                ax3.bar(time3, data3, width=dt, label=labels[2][0], facecolor=lcolor[2][0], alpha=0.4, edgecolor='none')

        else:
            if isinstance(time3, list):
                for j, (t3, d3) in enumerate(zip(time3, data3)):
                    ax3.plot(t3, d3, linestyle=linestyles[2][j], linewidth=2, label=labels[2], color=lcolor[2][j], markersize=markersize, alpha=0.85)
            else:
                ax3.plot(time3, data3, linestyle=linestyles[2][0], linewidth=2, label=labels[2], color=lcolor[2][0], markersize=markersize, alpha=0.85)

        if uncertainty is not None and len(uncertainty) > 2:
            t3s, u3s = uncertainty[2]
            for i3, (t3, u3) in enumerate(zip(t3s, u3s)):
                if error_bar:
                    _plot_unc_as_errorbar(ax3, t3, u3, color=ucolor[2][i3],
                                          err_stride=err_stride, clip_zero=clip_unc_zero)
                else:
                    _fill_unc_band(ax3, t3, u3, color=ucolor[2][i3], clip_zero=clip_unc_zero)


        ax3.set_ylabel(labels[2], fontsize=fontsize, color=lcolor[2][0])
        ax3.tick_params(axis="y", labelsize=12, colors=lcolor[2][0])

        if ylims[2] is not None:
             ax3.set_ylim(*ylims[2])
        print("3rd Y-axis range:", ax3.get_ylim())

    if time_format == 'hms':
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="center", fontsize=12)
    elif time_format == 'hm':
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax1.xaxis.get_majorticklabels(), ha="center", fontsize=12) 
    elif time_format == 'mdhm':
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=12)
    elif time_format == 'dhm':
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%H:%M"))
        plt.setp(ax1.xaxis.get_majorticklabels(), ha="center", fontsize=12)
    elif time_format == 'md':
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.setp(ax1.xaxis.get_majorticklabels(), ha="center", fontsize=12)
    elif time_format == 'ymd':
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="center", fontsize=12)
    elif time_format == 'ym':
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax1.xaxis.get_majorticklabels(), ha="center", fontsize=12)

    if fill_datetime is not None:
        colors = ['#9467bd', 
                  '#1f77b4', 
                  '#2ca02c'] 
        for i, (highlight_start, highlight_end) in enumerate(fill_datetime):
            color = colors[i % len(colors)]
            ax1.axvspan(
                highlight_start, highlight_end,
                facecolor=color, alpha=0.2,
                edgecolor=color, linewidth=1.5
            )

    if mark_datetime is not None:
        for mark_start in mark_datetime:
            ax1.axvline(mark_start, color=mcolor, linestyle='--', linewidth=1.5, alpha=0.7)

    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)


    if title is not None:
        plt.title(title, fontsize=fontsize, fontweight="bold", y=1.05) 

    if legend:
        handles, labels = [], []
        for ax in [ax1, ax2 if num_data > 1 else None, ax3 if num_data > 2 else None]:
            if ax is not None:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=bbox_to_anchor, ncol=3, fontsize=fontsize, frameon=False)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    ax1.tick_params(axis="x", labelsize=fontsize_ticks)
    if nax is not None:
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=nax))
    ax1.tick_params(axis="y", labelsize=fontsize_ticks)
    if nay is not None:
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=nay[0]))
    if num_data > 1:
        if nay is not None: 
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=nay[1]))
        ax2.tick_params(axis="y", labelsize=fontsize_ticks)
    if num_data > 2:
        if nay is not None:
            ax3.yaxis.set_major_locator(MaxNLocator(nbins=nay[2]))
        ax3.tick_params(axis="y", labelsize=fontsize_ticks)

    if align_zero:
        _align_zero_axes([
            ax1,
            ax2 if num_data > 1 else None,
            ax3 if num_data > 2 else None,
        ])

    for spine in ax1.spines.values():
        spine.set_edgecolor('black')

    if num_data > 1:
        for spine in ax2.spines.values():
            spine.set_edgecolor('black')
    if num_data > 2:
        for spine in ax3.spines.values():
            spine.set_edgecolor('black')

    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')  
        plt.close()
    else:
        plt.show()


def draw_event_station(region, 
                       image=None,
                       points=None,
                       lines=None,
                       events=None, 
                       stations=None, 
                       faults=None,
                       polygon=None, 
                       circle=None,
                       slab=None,
                       cbar=None,
                       elim=None,
                       graph=None,
                       voronoi=None,
                       number=None,
                       triangle=None,
                       title=None, 
                       scale_factor=None,
                       scalebar=None,
                       zoom=14,
                       scalebar_height = 3e-4,
                       num_colors=8,
                       clean=False,
                       gridline=True,
                       save_path=None,
                      ):

    def get_colors(n, cmap='jet'):
        colmap = matplotlib.cm.get_cmap(cmap)
        colors = [colmap(i / n) for i in range(n)]
        return colors

    projection = ccrs.PlateCarree()
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection=projection)
    ax.set_extent([region[i] for i in [2,3,0,1]])

    if image is None:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=ccrs.PlateCarree(), zoom=zoom)

    for text in ax.texts:
        if "Esri" in text.get_text():
            text.set_visible(False)


    if gridline:
        gl = ax.gridlines(crs=projection, draw_labels=True, linewidth=0.25)
        gl.xlocator = mticker.FixedLocator(np.linspace(region[2], region[3], 5, endpoint=True)[1:-1])
        gl.ylocator = mticker.FixedLocator(np.linspace(region[0], region[1], 5, endpoint=True)[1:-1])
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.ylabel_style = {'size': 10, 'weight': 'bold', 'rotation': 90}
        gl.xlabel_style = {'size': 10, 'weight': 'bold'}    


    legend_dict = {}
    if points is not None:
        lats = np.array([x['lat']for x in points])
        lons = np.array([x['lon'] for x in points])
        if 'val' in points[0].keys():
            pcmap = plt.get_cmap('turbo') 
            values = np.array([x['val'] for x in points])
            norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
        else:
            values = None
            pcmap = 'blue'
            norm=None
        xs, ys, _ = projection.transform_points(ccrs.Geodetic(), lons, lats).T
        legend_dict["Wells"] = ax.scatter(xs, ys, s=72, c=values, cmap=pcmap, norm=norm, marker="*", 
                             edgecolor="none", zorder=6, alpha=1.0)

    if stations is not None:
        networks = stations.keys()
        cmap = plt.get_cmap('tab20') 
        norm = mcolors.Normalize(vmin=0, vmax=len(networks)-1)

        for j, network in enumerate(networks):
            _stations = stations[network]
            lats = np.array([x['sta_lat']for x in _stations])
            lons = np.array([x['sta_lon'] for x in _stations])
            xs, ys, _ = projection.transform_points(ccrs.Geodetic(), lons, lats).T
            vs = [j]*len(xs)
            legend_dict[network.title()] = ax.scatter(xs, ys, s=36, c=vs, cmap=cmap, norm=norm, marker="v", 
                             edgecolor="none", zorder=5, alpha=1.0)

            if triangle is not None and triangle[j]:
                ax.triplot(xs, ys, triangle[j], color='blue', linewidth=0.5)


            if number and number[j]:
                for i, (x, y) in enumerate(zip(xs, ys)):
                    ax.text(x, y, str(i), fontsize=9, ha='right', va='bottom', color='black', zorder=6)


    if events is not None:
        emin, emax = None, None
        if elim is not None:
            emin, emax = elim

        _mags = np.array([x['mag'] for x in events])
        _events = [events[i] for i in np.argsort(_mags)]
        lats = np.array([x['src_lat'] for x in _events])
        lons = np.array([x['src_lon'] for x in _events])
        mags = np.array([x['mag'] for x in _events])
        xs, ys, _ = projection.transform_points(ccrs.Geodetic(), lons, lats).T
        if len(events) == 1:
            _event = ax.scatter(xs, ys, 64, c='r',
                             vmin=emin, vmax=emax, 
                   edgecolor="none", marker="*", zorder=5, alpha=0.6)     
            legend_dict['Source'] = _event

            legend = ax.legend(*_event.legend_elements(),
                               title="Magnitude",
                               loc='lower right',
                               prop={'size': 8},
                              )
            legend.get_title().set_fontsize(8)
            ax.add_artist(legend)
        else:

            if scale_factor is None:
                min_size, max_size = 20, 50 
            else:
                min_size, max_size = scale_factor
            sizes = np.interp(mags, (mags.min(), mags.max()), (min_size, max_size))

            _event = ax.scatter(lons, lats, s=sizes, c=mags, cmap='Reds', edgecolor="none",
                                vmin=emin, vmax=emax,
                            marker="o", zorder=5, alpha=0.6)

            if not clean and cbar:
                cbar = fig.colorbar(_event, ax=ax, orientation='horizontal', 
                                    fraction=0.04, pad=0.1, aspect=30, shrink=0.5)
                cbar.set_label("Magnitude (Mw)")     

    if polygon is not None:
        from matplotlib.patches import Polygon as PolygonMat 
        _polygon = PolygonMat(polygon, closed=True, fill=True, zorder=8, linestyle='dashed',
                          facecolor='none', edgecolor='black', alpha=1.0, linewidth=0.5,
                         )
        ax.add_patch(_polygon)
        legend_dict['Reservoir'] = _polygon

    if circle is not None:
        ax.add_patch(Circle(circle, 0.1, fill=False, linestyle='dashed', linewidth=0.6, color='g'))

    if lines is not None:
        for line in lines:
            if 'lat' in line and 'lon' in line and len(line['lat']) > 0 and len(line['lon']) > 0:
                lats, lons = np.array(line['lat']), np.array(line['lon'])
                ax.plot(lons, lats, transform=ccrs.PlateCarree(), linewidth=2, color='blue', zorder=5, label="Well Trajectory", alpha=0.5)

    if faults is not None:
        for fault in faults:
            if 'lat' in fault and 'lon' in fault and len(fault['lat']) > 0 and len(fault['lon']) > 0:
                lats, lons = np.array(fault['lat']), np.array(fault['lon'])
                ax.plot(lons, lats, transform=ccrs.PlateCarree(), linewidth=0.5, color='black', zorder=5, label="Faults", alpha=1.0)

    if slab is not None:
        import matplotlib.tri as tri
        lon = slab["lon"]
        lat = slab["lat"]
        depth = slab["depth"]

        triang = tri.Triangulation(lon, lat)

        triangles = triang.triangles
        pts = np.column_stack((lon, lat))
        max_len = np.max(np.linalg.norm(pts[triangles[:, [0, 1]]] - pts[triangles[:, [1, 2]]], axis=2), axis=1)
        mask = max_len > 0.5
        triang.set_mask(mask)

        tpc = ax.tripcolor(triang, depth, cmap="viridis", shading='flat', alpha=0.4)

    from shapely.geometry import Polygon, MultiPolygon
    if voronoi is not None:
        voronoi_poly_group, hull_points = voronoi['voronoi_poly_group'], voronoi['hull_points']
        cluster_color, polygon_color = voronoi['cluster_color'], voronoi['polygon_color']
        for clipped_poly_dict in voronoi_poly_group:
            clipped_poly = clipped_poly_dict['poly']

            if clipped_poly.geom_type == 'Polygon':
                x, y = clipped_poly.exterior.xy
                ax.fill(x, y, alpha=0.4, facecolor=polygon_color, edgecolor='black', transform=ccrs.PlateCarree())

            elif clipped_poly.geom_type == 'MultiPolygon':
                for poly in clipped_poly.geoms:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, alpha=0.4, facecolor=polygon_color, edgecolor='black', transform=ccrs.PlateCarree())

        cluster_centers = np.array([x['cluster'] for x in voronoi_poly_group])
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color=cluster_color, s=40, marker='*', edgecolors='none',
                   label='Voronoi Centers', transform=ccrs.PlateCarree(), zorder=11)

        x, y = hull_points[:, 0], hull_points[:, 1]
        ax.plot(np.append(x, x[0]), np.append(y, y[0]), color='black', linewidth=1.0, linestyle='dashed', label="Convex Hull")

    if graph is not None:
        positions = np.array(graph['positions'])
        edge_index = graph['edge_index']
        edge_attr = graph.get('edge_attr', None)

        ax.scatter(positions[:, 0], positions[:, 1], s=40, marker='*', color='blue', edgecolors='none',
                   transform=ccrs.PlateCarree(), zorder=12)

        edges = edge_index.t().tolist()
        weights = edge_attr.tolist() if edge_attr is not None else [1.0] * len(edges)
        max_weight = max(weights) if weights else 1.0
        min_weight = min(weights) if weights else 0.0

        cmap = plt.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=min_weight, vmax=max_weight)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        for (src, dst), w in zip(edges, weights):
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            ax.plot([x0, x1], [y0, y1],
                    linewidth=1.5 * w / max_weight,
                    color=cmap(norm(w)),
                    alpha=1.0,
                    transform=ccrs.PlateCarree(),
                    zorder=11)

    if title is not None and not clean:    
        ax.set_title(title) 

    if scalebar is not None:
        lon0 = region[2] + 0.2 * (region[3] - region[2])
        lat0 = region[0] + 0.1 * (region[1] - region[0])

        nblocks = 4
        block_len_km = scalebar / nblocks
        geod = Geodesic()

        for i in range(nblocks):
            direction = 90
            start = geod.direct((lon0, lat0), direction, i * block_len_km * 1000)
            end = geod.direct((lon0, lat0), direction, (i + 1) * block_len_km * 1000)

            lon1, lat1 = start[0, 0], start[0, 1]
            lon2, lat2 = end[0, 0], end[0, 1]

            ax.add_patch(plt.Polygon(
                [[lon1, lat1], [lon2, lat2], [lon2, lat2 + scalebar_height], [lon1, lat1 + scalebar_height]],
                facecolor='black' if i % 2 == 0 else 'white',
                edgecolor='black',
                transform=ccrs.PlateCarree()
            ))

        mid = geod.direct((lon0, lat0), 90, scalebar * 500)
        mid_lon, mid_lat = mid[0, 0], mid[0, 1]
        ax.text(mid_lon, lat0 - 0.04 * (region[1] - region[0]), f'{scalebar} km',
                ha='center', va='bottom',
                transform=ccrs.PlateCarree(), fontsize=8, weight='bold')

    if (legend_dict.keys()) and not clean:
        ax.legend(legend_dict.values(), legend_dict.keys(), loc='upper left', prop={'size': 8})

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def draw_spatiotemporal_map(region, 
                       points=None,
                       lines=None,
                       faults=None,
                       events=None, 
                       stations=None, 
                       polygon=None, 
                       circle=None,
                       magnitudes=None,
                       slab=None,
                       elim=None,
                       graph=None,
                       voronoi=None,
                       field=None,
                       number=None,
                       triangle=None,
                       has_injection=None,
                       title=None, 
                       scale_factor=None,
                       scalebar=None,
                       scalebar_height = 3e-4,
                       num_colors=8,
                       clean=False,
                       gridline=True,
                       save_path=None,
                       current_time=None,
                       time_window=None,
                       figsize=None,
                       cbar=False,
                      ):
    import matplotlib.tri as tri
    from matplotlib.path import Path
    from shapely.geometry import Polygon, MultiPolygon
    from scipy.interpolate import RBFInterpolator
    import matplotlib.gridspec as gridspec

    def get_colors(n, cmap='jet'):
        colmap = matplotlib.cm.get_cmap(cmap)
        colors = [colmap(i / n) for i in range(n)]
        return colors

    projection = ccrs.PlateCarree()

    if has_injection:
        fig = plt.figure(figsize=figsize, dpi=150)
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.1)
    else:
        fig = plt.figure(figsize=figsize, dpi=150)
        gs = gridspec.GridSpec(1, 1) 

    ax = fig.add_subplot(gs[0], projection=projection)

    if gridline:
        gl = ax.gridlines(crs=projection, draw_labels=True, linewidth=0.25)
        gl.xlocator = mticker.FixedLocator(np.linspace(region[2], region[3], 5, endpoint=True)[1:-1])
        gl.ylocator = mticker.FixedLocator(np.linspace(region[0], region[1], 5, endpoint=True)[1:-1])
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        gl.ylabel_style = {'size': 10, 'rotation': 90}
        gl.xlabel_style = {'size': 10}    

    legend_dict = {}
    if points is not None:
        lats = np.array([x['lat']for x in points])
        lons = np.array([x['lon'] for x in points])
        if 'val' in points[0].keys():
            pcmap = plt.get_cmap('turbo') 
            values = np.array([x['val'] for x in points])
            norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
        else:
            values = 'black'
            pcmap = None
            norm=None
        xs, ys, _ = projection.transform_points(ccrs.Geodetic(), lons, lats).T
        legend_dict["Cluster Centers"] = ax.scatter(xs, ys, s=72, c=values, cmap=pcmap, norm=norm, marker="*", 
                             edgecolor="none", zorder=10, alpha=1.0)

    if stations is not None:
        networks = stations.keys()
        cmap = plt.get_cmap('tab20') 
        norm = mcolors.Normalize(vmin=0, vmax=len(networks)-1)

        for j, network in enumerate(networks):
            _stations = stations[network]
            lats = np.array([x['sta_lat']for x in _stations])
            lons = np.array([x['sta_lon'] for x in _stations])
            xs, ys, _ = projection.transform_points(ccrs.Geodetic(), lons, lats).T
            vs = [j]*len(xs)

            legend_dict[network.title()] = ax.scatter(xs, ys, s=36, c=vs, cmap=cmap, norm=norm, marker="v", 
                             edgecolor="none", zorder=5, alpha=1.0)

            if triangle is not None and triangle[j]:
                ax.triplot(xs, ys, triangle[j], color='blue', linewidth=0.5)


            if number and number[j]:
                for i, (x, y) in enumerate(zip(xs, ys)):
                    ax.text(x, y, str(i), fontsize=9, ha='right', va='bottom', color='black', zorder=6)

    if events is not None:
        emin, emax = None, None
        if elim is not None:
            emin, emax = elim

        _mags = np.array([x['mag'] for x in events])
        _events = [events[i] for i in np.argsort(_mags)]
        lats = np.array([x['src_lat'] for x in _events])
        lons = np.array([x['src_lon'] for x in _events])
        mags = np.array([x['mag'] for x in _events])
        xs, ys, _ = projection.transform_points(ccrs.Geodetic(), lons, lats).T
        if len(events) == 1:
            _event = ax.scatter(xs, ys, 64, c='r',
                             vmin=emin, vmax=emax, 
                   edgecolor="none", marker="*", zorder=5, alpha=0.6)     
            legend_dict['Source'] = _event

            legend = ax.legend(*_event.legend_elements(),
                               title="Magnitude",
                               loc='lower right',
                               prop={'size': 8},
                              )
            legend.get_title().set_fontsize(8)
            ax.add_artist(legend)
        else:
            if scale_factor is None:
                min_size, max_size = 20, 50 
            else:
                min_size, max_size = scale_factor
            sizes = np.interp(mags, (mags.min(), mags.max()), (min_size, max_size))

            _event = ax.scatter(lons, lats, s=sizes, c=mags, cmap='Reds', edgecolor="none",
                                vmin=emin, vmax=emax,
                            marker="o", zorder=5, alpha=0.6)

            if not clean and cbar:
                cbar = fig.colorbar(_event, ax=ax, orientation='horizontal', 
                                    fraction=0.04, pad=0.1, aspect=30, shrink=0.5)

    if polygon is not None:
        from matplotlib.patches import Polygon as PolygonMat 
        _polygon = PolygonMat(polygon, closed=True, fill=True, zorder=10, linestyle='dashed',
                          facecolor='none', edgecolor='black', alpha=1.0, linewidth=0.5,
                         )
        ax.add_patch(_polygon)
        legend_dict['Reservoir'] = _polygon

    if circle is not None:
        ax.add_patch(Circle(circle, 0.1, fill=False, linestyle='dashed', linewidth=0.6, color='g'))

    if lines is not None:
        for line in lines:
            if 'lat' in line and 'lon' in line and len(line['lat']) > 0 and len(line['lon']) > 0:
                lats, lons = np.array(line['lat']), np.array(line['lon'])
                ax.plot(lons, lats, transform=ccrs.PlateCarree(), linewidth=2, color='blue', zorder=5, label="Well Trajectory", alpha=0.5)

    if faults is not None:
        lat_min, lat_max = region[0], region[1]
        lon_min, lon_max = region[2], region[3]

        for fault in faults:
            if 'lat' in fault and 'lon' in fault:
                lats = np.array(fault['lat'])
                lons = np.array(fault['lon'])

                in_bounds = (
                    (lats >= lat_min) & (lats <= lat_max) &
                    (lons >= lon_min) & (lons <= lon_max)
                )

                if np.sum(in_bounds) >= 2:
                    ax.plot(
                        lons[in_bounds], lats[in_bounds],
                        transform=ccrs.PlateCarree(),
                        linewidth=0.5, color='black', zorder=10,
                        label="Faults", alpha=1.0
                    )

    if slab is not None:
        lon = slab["lon"]
        lat = slab["lat"]
        depth = slab["depth"]

        triang = tri.Triangulation(lon, lat)

        triangles = triang.triangles
        pts = np.column_stack((lon, lat))
        max_len = np.max(np.linalg.norm(pts[triangles[:, [0, 1]]] - pts[triangles[:, [1, 2]]], axis=2), axis=1)
        mask = max_len > 0.5
        triang.set_mask(mask)

        tpc = ax.tripcolor(triang, depth, cmap="viridis", shading='flat', alpha=0.4)

    if voronoi is not None:
        voronoi_poly_group, hull_points = voronoi['voronoi_poly_group'], voronoi['hull_points']
        cluster_color, polygon_color = voronoi['cluster_color'], voronoi['polygon_color']
        for clipped_poly_dict in voronoi_poly_group:
            clipped_poly = clipped_poly_dict['poly']

            if clipped_poly.geom_type == 'Polygon':
                x, y = clipped_poly.exterior.xy
                ax.fill(x, y, alpha=0.4, facecolor=polygon_color, edgecolor='black', transform=ccrs.PlateCarree())

            elif clipped_poly.geom_type == 'MultiPolygon':
                for poly in clipped_poly.geoms:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, alpha=0.4, facecolor=polygon_color, edgecolor='black', transform=ccrs.PlateCarree())

        x, y = hull_points[:, 0], hull_points[:, 1]
        ax.plot(np.append(x, x[0]), np.append(y, y[0]), color='black', linewidth=1.0, linestyle='dashed', label="Convex Hull")

    if graph is not None:
        positions = np.array(graph['positions'])
        edge_index = graph['edge_index']
        edge_attr = graph.get('edge_attr', None)

        ax.scatter(positions[:, 0], positions[:, 1], s=40, marker='*', color='blue', edgecolors='none',
                   transform=ccrs.PlateCarree(), zorder=12)

        edges = edge_index.t().tolist()
        weights = edge_attr.tolist() if edge_attr is not None else [1.0] * len(edges)
        max_weight = max(weights) if weights else 1.0
        min_weight = min(weights) if weights else 0.0

        cmap = plt.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=min_weight, vmax=max_weight)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 

        for (src, dst), w in zip(edges, weights):
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            ax.plot([x0, x1], [y0, y1],
                    linewidth=1.5 * w / max_weight,
                    color=cmap(norm(w)),
                    alpha=1.0,
                    transform=ccrs.PlateCarree(),
                    zorder=11)

    if magnitudes is not None:
        emin, emax = None, None
        if elim is not None:
            emin, emax = elim

        if scale_factor is None:
            min_size, max_size = 20, 50 
        else:
            min_size, max_size = scale_factor

        _magnitudes = magnitudes
        lats = np.array([x['src_lat'] for x in _magnitudes])
        lons = np.array([x['src_lon'] for x in _magnitudes])
        mags = np.array([x['mag'] for x in _magnitudes])
        times = np.array([x['time'] for x in _magnitudes])
        xs, ys, _ = projection.transform_points(ccrs.Geodetic(), lons, lats).T

        last_time = current_time
        first_time = current_time - time_window
        indices_in_period = np.where( (times>=first_time) & (times<last_time) )[0]
        indices_left_period = np.where( times<first_time )[0]
        indices_right_period = np.where( times>=last_time )[0]

        lats_in, lons_in = lats[indices_in_period], lons[indices_in_period]
        lats_left, lons_left = lats[indices_left_period], lons[indices_left_period]
        lats_right, lons_right = lats[indices_right_period], lons[indices_right_period]

        mag_sizes = np.interp(mags[indices_in_period], (mags.min(), mags.max()), (min_size, max_size))

        timestamps = np.array([dt.timestamp() for dt in times])
        timestamps -= timestamps.min() 
        timestamps = timestamps[indices_in_period]

        cmap = plt.get_cmap('Reds_r')
        first_color = cmap(0.0)

        _mag0 = ax.scatter(
            lons_left, lats_left,
            s=min_size,
            facecolors=first_color,        
            edgecolors='none',      
            linewidths=1.0,
            marker='o',
            zorder=6,
            alpha=1.0,
        )

        _mag1 = ax.scatter(
            lons_right, lats_right,
            s=min_size,
            facecolors='gray',        
            edgecolors='none',      
            linewidths=1.0,
            marker='o',
            zorder=5,
            alpha=1.0,
        )

        if len(timestamps) > 0:
            time_norm = plt.Normalize(vmin=timestamps.min(), vmax=timestamps.max())
            time_colors = cmap(time_norm(timestamps)) 

            _mag2 = ax.scatter(
                lons_in, lats_in,
                s=mag_sizes,
                facecolors='none',        
                edgecolors=time_colors,      
                linewidths=1.0,
                marker='o',
                zorder=9,
                alpha=0.6,
            )

            _mag3 = ax.scatter(
                lons_in, lats_in,
                s=min_size,   
                facecolors=time_colors,
                edgecolors='none',      
                linewidths=1.0,
                marker='o',
                zorder=9,
                alpha=1.0,
            )


    if field is not None:
        interp_points = [x[:2] for x in field['points']]
        interp_values = field['values']
        hull_points = field.get('hull_points', None) 

        grid_lon = np.linspace(region[2], region[3], 800)
        grid_lat = np.linspace(region[0], region[1], 800)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

        flat_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        rbf = RBFInterpolator(interp_points, interp_values, kernel='linear') 
        interp_grid = rbf(flat_grid).reshape(grid_x.shape)

        if hull_points is not None:
            poly_path = Path(hull_points)
            inside_mask = poly_path.contains_points(flat_grid)
            mask_grid = np.full(grid_x.shape, np.nan)
            mask_grid[inside_mask.reshape(grid_x.shape)] = interp_grid[inside_mask.reshape(grid_x.shape)]
        else:
            mask_grid = interp_grid

        mesh = ax.imshow(
            mask_grid,
            extent=[region[2], region[3], region[0], region[1]],
            origin='lower',
            cmap='turbo',
            transform=ccrs.PlateCarree(),
            zorder=1,
        )

        if not clean and cbar:
            cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.5, pad=0.06)
            cbar.set_label('Seismicity Rate (events/day)', fontsize=8)

    if has_injection:
        time_arr = np.array(has_injection['time_axis'])
        rate_dict = has_injection['injection']          
        current_time = has_injection.get('current_time', None)

        icolors = plt.get_cmap('tab10', len(rate_dict))

        ax_inj = fig.add_subplot(gs[1], sharex=None)
        for i, (well_name, rate_series) in enumerate(rate_dict.items()):
            rate_arr = np.array(rate_series)
            ax_inj.plot(time_arr, rate_arr, label=well_name, linewidth=0.8, color=icolors(i)) 

        ax_inj.set_ylabel('Injection Rate', fontsize=8)
        ax_inj.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_inj.tick_params(axis='x', labelsize=8)
        ax_inj.tick_params(axis='y', labelsize=8)
        ax_inj.grid(True, linestyle='--', alpha=0.5)

        if current_time is not None:
            ax_inj.axvline(current_time, color='red', linestyle='--', linewidth=0.5, label='Current Time')
        ax_inj.legend(
            loc='upper left',
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0.,
            fontsize=8,
        )
        plt.setp(ax.get_xticklabels(), visible=False)

    if title is not None:    
        ax.set_title(title) 

    if scalebar is not None:
        lon0 = region[2] + 0.20 * (region[3] - region[2])
        lat0 = region[0] + 0.05 * (region[1] - region[0])

        nblocks = 4
        block_len_km = scalebar / nblocks
        geod = Geodesic()

        for i in range(nblocks):
            direction = 90
            start = geod.direct((lon0, lat0), direction, i * block_len_km * 1000)
            end = geod.direct((lon0, lat0), direction, (i + 1) * block_len_km * 1000)

            lon1, lat1 = start[0, 0], start[0, 1]
            lon2, lat2 = end[0, 0], end[0, 1]

            ax.add_patch(plt.Polygon(
                [[lon1, lat1], [lon2, lat2], [lon2, lat2 + scalebar_height], [lon1, lat1 + scalebar_height]],
                facecolor='black' if i % 2 == 0 else 'white',
                edgecolor='black',
                transform=ccrs.PlateCarree()
            ))

        mid = geod.direct((lon0, lat0), 90, scalebar * 500)
        mid_lon, mid_lat = mid[0, 0], mid[0, 1]
        ax.text(mid_lon, lat0 - 0.04 * (region[1] - region[0]), f'{scalebar} km',
                ha='center', va='bottom',
                transform=ccrs.PlateCarree(), fontsize=8, weight='bold')

    if (legend_dict.keys()) and not clean:
        ax.legend(legend_dict.values(), legend_dict.keys(), loc='upper left', prop={'size': 8})

    ax.set_extent([region[2], region[3], region[0], region[1]], crs=ccrs.PlateCarree())

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_etas_rolling_results(
    hist_rate,
    lambda_recon,
    lambda_future,
    seismic_rate_target,
    etas_params,
    n_future,
    n_hist,
    future_params=None,
    window_stride_plot=1,
    figsize=(9, 7),
    title=None,
    fontsize=16,
):

    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize * 0.75,
        "ytick.labelsize": fontsize * 0.75,
        "legend.fontsize": fontsize * 0.65,
    })

    hist_rate = np.asarray(hist_rate)
    lambda_recon = np.asarray(lambda_recon)

    T = len(hist_rate)
    lambda_target = seismic_rate_target[:n_future]

    t_plot_start = max(0, T - n_hist)

    t_center = np.array([p["t_center"] for p in etas_params])
    mu_t = np.array([p["mu"] for p in etas_params])
    K_t  = np.array([p["K"]  for p in etas_params])
    p_t  = np.array([p["p"]  for p in etas_params])

    mask = t_center >= t_plot_start

    if future_params is not None:
        future_mu = np.array([p["mu"] for p in future_params])
        future_K  = np.array([p["K"]  for p in future_params])
        future_p  = np.array([p["p"]  for p in future_params])

        t_future = np.arange(T, T + n_future)

    fig, axes = plt.subplots(
        4, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw=dict(height_ratios=[2.2, 1, 1, 1])
    )

    ax0, ax1, ax2, ax3 = axes

    ax0.plot(
        np.arange(t_plot_start, T),
        hist_rate[t_plot_start:T],
        color="k",
        alpha=0.4,
        label="Observed (history)"
    )

    ax0.plot(
        np.arange(t_plot_start, T),
        lambda_recon[t_plot_start:T],
        color="red",
        lw=2,
        label="ETAS reconstruction"
    )

    ax0.plot(
        np.arange(T, T + n_future),
        lambda_future,
        "--",
        color="red",
        lw=2,
        label="ETAS forecast"
    )

    ax0.plot(
        np.arange(T, T + n_future),
        lambda_target,
        color="k",
        lw=2,
        alpha=0.8,
        label="Observed (future)"
    )

    ax0.axvline(T, color="gray", ls=":")
    ax0.set_ylabel("Seismicity rate")
    ax0.legend(frameon=False)

    ax1.plot(
        t_center[mask],
        mu_t[mask],
        color="tab:blue",
        lw=1.8,
        marker="o",
        ms=4,
        label="History"
    )

    if future_params is not None:
        ax1.plot(
            t_future,
            future_mu,
            "--",
            color="tab:blue",
            marker="s",
            ms=5,
            lw=1.5,
            label="Future refit"
        )

    ax1.axvline(T, color="gray", ls=":")
    ax1.set_ylabel(r"$\mu(t)$")
    ax1.legend(frameon=False)

    ax2.plot(
        t_center[mask],
        K_t[mask],
        color="tab:orange",
        lw=1.8,
        marker="o",
        ms=4,
        label="History"
    )

    if future_params is not None:
        ax2.plot(
            t_future,
            future_K,
            "--",
            color="tab:orange",
            marker="s",
            ms=5,
            lw=1.5,
            label="Future refit"
        )

    ax2.axvline(T, color="gray", ls=":")
    ax2.set_ylabel(r"$K(t)$")
    ax2.legend(frameon=False)

    ax3.plot(
        t_center[mask],
        p_t[mask],
        color="tab:green",
        lw=1.8,
        marker="o",
        ms=4,
        label="History"
    )

    if future_params is not None:
        ax3.plot(
            t_future,
            future_p,
            "--",
            color="tab:green",
            marker="s",
            ms=5,
            lw=1.5,
            label="Future refit"
        )

    ax3.axvline(T, color="gray", ls=":")
    ax3.set_ylabel(r"$p(t)$")
    ax3.set_xlabel("Time step")
    ax3.legend(frameon=False)

    ax3.set_xlim(t_plot_start, T + n_future)

    for ax in axes:
        ax.yaxis.set_label_coords(-0.05, 0.5)

    if title is not None:
        fig.suptitle(title, y=0.98)

    plt.tight_layout()
    plt.show()
