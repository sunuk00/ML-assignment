import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from collections import Counter
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf as _compute_acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ──────────────────────────────────────────────────────────────────────────────
# Shared config
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR           = "data"
COLORS             = {"continuous": "#4C72B0", "discrete": "#DD8452"}
DISCRETE_THRESHOLD = 20       # channels with fewer unique values are treated as discrete
MISSING_METHOD     = "ffill"  # options: 'linear' | 'ffill' | 'spline'

# Korean font (Malgun Gothic is bundled with Windows)
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False  # prevent minus sign from breaking


def get_channels(df, exclude="t"):
    return [c for c in df.columns if c != exclude]


# ──────────────────────────────────────────────────────────────────────────────
# Missing value handling
# ──────────────────────────────────────────────────────────────────────────────

def handle_missing(df, method=MISSING_METHOD):
    """
    Detect and fill missing values in channel columns only (t is left untouched).
    method:
        'linear' — linear interpolation between neighbors
        'ffill'  — forward-fill (then back-fill for leading NaNs)
        'spline' — cubic spline interpolation (order=3)
    """
    channels     = get_channels(df)
    miss_counts  = df[channels].isnull().sum()
    total_before = miss_counts.sum()

    if total_before == 0:
        print(f"  [missing] no missing values — skipping ({method})")
        return df

    print(f"  [missing] {total_before} NaN(s) detected before filling:")
    print(miss_counts[miss_counts > 0].to_string())

    df = df.copy()
    if method == "linear":
        df[channels] = df[channels].interpolate(method="linear")
    elif method == "ffill":
        df[channels] = df[channels].ffill().bfill()
    elif method == "spline":
        df[channels] = df[channels].interpolate(method="spline", order=3)
    else:
        raise ValueError(f"Unknown missing method: {method!r}. Choose 'linear', 'ffill', or 'spline'.")

    total_after = df[channels].isnull().sum().sum()
    print(f"  [missing] after {method!r}: {total_after} NaN(s) remaining")
    return df


def load_data(missing_method=MISSING_METHOD):
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    val   = pd.read_csv(f"{DATA_DIR}/val.csv")
    test  = pd.read_csv(f"{DATA_DIR}/test_public.csv")

    print("── Missing value check ──────────────────────────────────")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        print(f"  [{name}]", end=" ")
        miss = df.isnull().sum().sum()
        if miss:
            print(f"{miss} NaN(s) -> applying '{missing_method}'")
        else:
            print("no missing values")

    train = handle_missing(train, missing_method)
    val   = handle_missing(val,   missing_method)
    test  = handle_missing(test,  missing_method)
    print()
    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# Step 1. 채널별 속성 파악 (연속형 vs 이산형 분류)
# ──────────────────────────────────────────────────────────────────────────────

def classify_channels(df, threshold=DISCRETE_THRESHOLD):
    """Returns {channel: 'continuous' | 'discrete'} based on unique-value count."""
    return {
        ch: ("discrete" if df[ch].nunique() <= threshold else "continuous")
        for ch in get_channels(df)
    }


def _channel_stats(df, channels):
    rows = []
    for ch in channels:
        rows.append({
            "channel":  ch,
            "n_unique": df[ch].nunique(),
            "zero_pct": (df[ch] == 0).mean() * 100,
            "min":      df[ch].min(),
            "max":      df[ch].max(),
            "mean":     df[ch].mean(),
            "std":      df[ch].std(),
        })
    return pd.DataFrame(rows).set_index("channel")


def _waveform_label(ch, ch_type, df):
    """Short Korean label describing the waveform shape."""
    if ch_type == "discrete":
        return "이진 계단형"
    zero_pct = (df[ch] == 0).mean() * 100
    if zero_pct > 10:
        return f"영점 급등형  (0값: {zero_pct:.0f}%)"
    return "연속 파형"



def step1_channel_properties(df):
    classification = classify_channels(df)
    channels       = get_channels(df)
    stats_df       = _channel_stats(df, channels)

    discrete_chs   = [ch for ch, t in classification.items() if t == "discrete"]
    continuous_chs = [ch for ch, t in classification.items() if t == "continuous"]

    print("=" * 60)
    print("Step 1. Channel Property Analysis")
    print("=" * 60)
    print(f"  Discrete   ({len(discrete_chs)}): {discrete_chs}")
    print(f"  Continuous ({len(continuous_chs)}): {continuous_chs}")
    print()
    print(stats_df.to_string())
    print()

    t    = df["t"].values
    n_ch = len(channels)

    fig, axes = plt.subplots(
        n_ch, 1, sharex=True,
        figsize=(16, 2.0 * n_ch),
        gridspec_kw={"hspace": 0.12},
    )

    for i, ch in enumerate(channels):
        ax      = axes[i]
        ch_type = classification[ch]
        color   = COLORS[ch_type]
        values  = df[ch].values

        if ch_type == "discrete":
            ax.step(t, values, where="post", color=color, lw=0.9, alpha=0.85)
            ax.set_ylim(-0.3, 1.3)
            ax.set_yticks([0, 1])
        else:
            ax.plot(t, values, color=color, lw=0.5, alpha=0.8)

        # channel name on y-axis, wave shape annotation inside plot
        ax.set_ylabel(ch, rotation=0, ha="right", va="center",
                      fontsize=10, fontweight="bold", color=color)
        shape = _waveform_label(ch, ch_type, df)
        ax.text(0.995, 0.90, shape, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=color, alpha=0.85)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f5f8ff")
        ax.tick_params(labelsize=8)
        if i < n_ch - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("t (timestep)", fontsize=11)

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["continuous"], label="Continuous"),
        mpatches.Patch(facecolor=COLORS["discrete"],   label="Discrete"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=10,
               bbox_to_anchor=(0.995, 0.995))

    fig.suptitle(
        f"Step 1. Channel Properties  —  train.csv  |  {len(df):,} timesteps",
        fontsize=13, fontweight="bold", y=1.002,
    )

    out_path = "./eda/outputs/step1_channel_properties.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()

    return classification



# ──────────────────────────────────────────────────────────────────────────────
# Step 2. 정상 데이터의 Seasonality/Trend 분석
# ──────────────────────────────────────────────────────────────────────────────

def _detect_period(series, min_period=50, max_period=2000):
    """FFT로 지배적 주기(dominant period) 검출."""
    s = np.asarray(series, dtype=float) - np.mean(series)
    fft_mag = np.abs(rfft(s))
    freqs   = rfftfreq(len(s))
    with np.errstate(divide="ignore", invalid="ignore"):
        periods_arr = np.where(freqs > 0, 1.0 / freqs, 0.0)
    valid = (freqs > 0) & (periods_arr >= min_period) & (periods_arr <= max_period)
    if not valid.any():
        return 100
    return int(round(periods_arr[valid][np.argmax(fft_mag[valid])]))


def _decomp_strengths(result):
    """
    Hyndman & Athanasopoulos (2021) 방식의 계절성·추세 강도.
    F_s, F_t ∈ [0, 1] — 높을수록 강함.
    """
    R   = result.resid.dropna()
    idx = R.index
    S, T = result.seasonal.loc[idx], result.trend.loc[idx]
    var_R = float(R.var())
    def strength(comp):
        denom = float((comp + R).var())
        return max(0.0, 1.0 - var_R / denom) if denom > 0 else 0.0
    return round(strength(S), 3), round(strength(T), 3)


def _channel_class(F_s, F_t, s_thr=0.6, t_thr=0.6):
    if F_s >= s_thr:
        return "강한 주기성"
    if F_t >= t_thr:
        return "Trend 위주"
    return "복합"


def _adf_pvalue(series, maxlag=50):
    """ADF 검정 p-value. p < 0.05 -> 정상(stationary)."""
    _, pvalue, *_ = adfuller(series, maxlag=maxlag, regression="c", autolag=None)
    return round(float(pvalue), 4)


def _acf_dominant_lag(series, max_lag=1500, height=0.3, distance=20):
    """ACF 첫 번째 유의미한 피크 lag (= 반복 주기 확인)."""
    acf_vals = _compute_acf(series, nlags=max_lag, fft=True)
    peaks, _ = find_peaks(acf_vals[1:], height=height, distance=distance)
    return int(peaks[0] + 1) if len(peaks) > 0 else None


# ── Figure 2A: Seasonal Decomposition ────────────────────────────────────────

def _plot_step2_decomposition(df, continuous_chs, decomp_map, metrics):
    n_ch = len(continuous_chs)
    COMP_COLORS = {
        "observed": COLORS["continuous"],
        "trend":    "#E05A2B",
        "seasonal": "#2E8B57",
        "residual": "#888888",
    }
    comp_keys  = ["observed", "trend", "seasonal", "residual"]
    comp_label = {
        "observed": "Observed (원본)",
        "trend":    "Trend (추세)",
        "seasonal": "Seasonal (계절성)",
        "residual": "Residual (잔차)",
    }
    out_files  = {
        "observed": "./eda/outputs/step2a_observed.png",
        "trend":    "./eda/outputs/step2a_trend.png",
        "seasonal": "./eda/outputs/step2a_seasonal.png",
        "residual": "./eda/outputs/step2a_residual.png",
    }
    cls_colors = {"강한 주기성": "#2E8B57", "Trend 위주": "#E05A2B", "복합": "#888888"}

    t = df["t"].values

    for key in comp_keys:
        color = COMP_COLORS[key]
        fig, axes = plt.subplots(
            n_ch, 1, sharex=True,
            figsize=(16, 2.2 * n_ch),
            gridspec_kw={"hspace": 0.12},
        )
        if n_ch == 1:
            axes = [axes]

        for i, ch in enumerate(continuous_chs):
            ax     = axes[i]
            result = decomp_map[ch]
            m      = metrics[ch]
            comp   = {
                "observed": result.observed,
                "trend":    result.trend,
                "seasonal": result.seasonal,
                "residual": result.resid,
            }[key]

            ax.plot(t, comp.values, color=color, lw=0.6, alpha=0.85)

            # 수정: 잔차 y축을 잔차 자체 스케일로 설정 (원본 스케일 강제 고정 제거)
            if key == "residual":
                res_vals   = comp.dropna()
                res_max    = float(res_vals.abs().max())
                res_mean   = float(res_vals.mean())
                # 잔차 중심 ± 여유(10%) 로 y축 지정
                margin     = res_max * 1.1
                ax.set_ylim(res_mean - margin, res_mean + margin)
                # 0 기준선 추가 (잔차는 0 중심이어야 정상)
                ax.axhline(0, color="#cccccc", lw=0.8, ls="--", zorder=0)
                ax.text(0.005, 0.90,
                        f"max |residual| = {res_max:.2e}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=8, color="#888888")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_facecolor("#f5f8ff")
            ax.tick_params(labelsize=8)

            ch_color = cls_colors.get(m["class"], "#333")
            ax.set_ylabel(ch, rotation=0, ha="right", va="center",
                          fontsize=10, fontweight="bold", color=ch_color)
            ax.text(0.995, 0.90,
                    f"period={m['period']}  F_s={m['F_s']:.2f}  [{m['class']}]",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=ch_color, alpha=0.85)

            if i < n_ch - 1:
                ax.tick_params(labelbottom=False)

        axes[-1].set_xlabel("t (timestep)", fontsize=11)
        fig.text(
            0.01, 0.5, f"{comp_label[key]}  [sensor value]",
            va="center", ha="center", rotation=90,
            fontsize=10, color="#555555",
        )
        fig.suptitle(
            f"Step 2-A.  {comp_label[key]}  —  train.csv",
            fontsize=13, fontweight="bold", y=1.002,
        )

        out = out_files[key]
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        plt.show()
        plt.close(fig)


# ── Figure 2B: ACF / PACF ─────────────────────────────────────────────────────

def _plot_step2_acf_pacf(df, continuous_chs, metrics):
    n_ch  = len(continuous_chs)
    color = COLORS["continuous"]

    fig, axes = plt.subplots(
        n_ch, 2,
        figsize=(16, 2.8 * n_ch),
        gridspec_kw={"hspace": 0.55, "wspace": 0.30},
    )
    if n_ch == 1:
        axes = axes.reshape(1, 2)

    for i, ch in enumerate(continuous_chs):
        m       = metrics[ch]
        period  = m["period"]
        acf_lag = m["acf_lag"]
        n_lags  = min(2 * period, 1000, len(df) // 4)

        ax_acf, ax_pacf = axes[i, 0], axes[i, 1]
        try:
            plot_acf( df[ch], lags=n_lags, ax=ax_acf,
                      color=color, alpha=0.05, zero=False, title="")
            plot_pacf(df[ch], lags=n_lags, ax=ax_pacf,
                      color=color, alpha=0.05, zero=False, method="ywm", title="")
        except Exception as e:
            for ax in (ax_acf, ax_pacf):
                ax.text(0.5, 0.5, f"N/A\n{e}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8)
            continue

        # 검출된 주기 lag에 수직선 표시
        if acf_lag is not None and acf_lag <= n_lags:
            ax_acf.axvline(acf_lag, color="#E05A2B", lw=1.4, ls="--", alpha=0.85,
                           label=f"ACF peak lag={acf_lag}")
            ax_acf.legend(fontsize=8, loc="upper right")

        for ax, label in [(ax_acf, "ACF"), (ax_pacf, "PACF")]:
            ax.set_title(
                f"{ch}  —  {label}  "
                f"(period={period}, F_s={m['F_s']:.2f}, [{m['class']}])",
                fontsize=9, color=color, fontweight="bold",
            )
            ax.set_xlabel("lag (timestep)", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_facecolor("#f5f8ff")
            ax.tick_params(labelsize=7.5)

    fig.suptitle(
        "Step 2-B.  ACF / PACF  —  train.csv  |  연속형 채널",
        fontsize=13, fontweight="bold", y=1.002,
    )
    out = "./eda/outputs/step2b_acf_pacf.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.show()


# ── 전처리 파라미터 제안 출력 ──────────────────────────────────────────────────

def _print_step2_recommendations(metrics, common_period):
    bar = "=" * 68
    print(f"\n{bar}")
    print("  Step 2. 전처리 파라미터 제안")
    print(bar)

    print(f"\n  {'채널':<8} {'주기':>6} {'F_s':>6} {'F_t':>6}  {'분류':<12} {'ADF_p':>8}  정상성")
    print("  " + "─" * 60)
    nonstat = []
    for ch, m in metrics.items():
        stat = "정상" if m["adf_p"] < 0.05 else "비정상(!)"
        if m["adf_p"] >= 0.05:
            nonstat.append(ch)
        acf_str = f"acf_lag={m['acf_lag']}" if m["acf_lag"] else "acf_lag=미검출"
        print(f"  {ch:<8} {m['period']:>6} {m['F_s']:>6.3f} {m['F_t']:>6.3f}"
              f"  {m['class']:<12} {m['adf_p']:>8.4f}  {stat}  ({acf_str})")

    print(f"\n  [Sliding Window 크기 제안]")
    print(f"    공통 주기 : {common_period} timesteps  (FFT + ACF 일치 확인)")
    print(f"    권장 크기 : {common_period}  (1 사이클)  ~  {2 * common_period}  (2 사이클)")
    print(f"    -> 모델이 정상 1 사이클을 온전히 학습하려면 최소 {common_period} 이상 권장")

    print(f"\n  [차분(Differencing) 필요 여부]")
    if nonstat:
        for ch in nonstat:
            m = metrics[ch]
            print(f"    {ch}: ADF p={m['adf_p']:.4f} -> 비정상  "
                  f"-> 1차 차분(d=1) 또는 추세 제거 권장  (F_t={m['F_t']:.3f})")
    else:
        print("    전체 채널 정상성 확인 — 차분 불필요")
    print()


# ── 메인 Step 2 함수 ──────────────────────────────────────────────────────────

def step2_seasonality_trend(df):
    classification = classify_channels(df)
    channels       = get_channels(df)
    continuous_chs = [ch for ch, t in classification.items() if t == "continuous"]

    print("=" * 60)
    print("Step 2. Seasonality / Trend Analysis")
    print("=" * 60)

    decomp_map = {}
    metrics    = {}

    for ch in continuous_chs:
        period = _detect_period(df[ch])
        try:
            result = seasonal_decompose(
                df[ch], model="additive", period=period, extrapolate_trend="freq"
            )
            F_s, F_t = _decomp_strengths(result)
        except Exception as e:
            print(f"  [decompose] {ch} 실패: {e}")
            result, F_s, F_t = None, 0.0, 0.0

        adf_p   = _adf_pvalue(df[ch])
        acf_lag = _acf_dominant_lag(df[ch], max_lag=min(2 * period, 1500))
        cls     = _channel_class(F_s, F_t)

        decomp_map[ch] = result
        metrics[ch]    = {
            "period":  period,
            "F_s":     F_s,
            "F_t":     F_t,
            "class":   cls,
            "adf_p":   adf_p,
            "acf_lag": acf_lag,
        }
        print(f"  {ch}: period={period}  F_s={F_s:.3f}  F_t={F_t:.3f}"
              f"  [{cls}]  ADF_p={adf_p:.4f}  acf_lag={acf_lag}")

    # 강한 주기성 채널의 최빈 주기를 공통 주기로 사용
    periodic_periods = [
        metrics[ch]["period"] for ch in continuous_chs
        if metrics[ch]["class"] == "강한 주기성"
    ] or [metrics[ch]["period"] for ch in continuous_chs]
    common_period = Counter(periodic_periods).most_common(1)[0][0]
    print(f"\n  공통 주기(most common): {common_period} timesteps\n")

    valid_chs = [ch for ch in continuous_chs if decomp_map[ch] is not None]
    _plot_step2_decomposition(df, valid_chs, decomp_map, metrics)
    _plot_step2_acf_pacf(df, valid_chs, metrics)
    _print_step2_recommendations(metrics, common_period)

    return metrics, common_period


# ──────────────────────────────────────────────────────────────────────────────
# Step 3. 다변량 채널 간의 상관관계 분석
# ──────────────────────────────────────────────────────────────────────────────

def _plot_step3_heatmap(corr_ordered, ordered_chs, classification, metrics):
    from matplotlib.lines import Line2D
    n   = len(ordered_chs)
    fig, ax = plt.subplots(figsize=(n * 1.15 + 1.5, n * 1.05 + 1.5))

    im = ax.imshow(corr_ordered.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            val   = corr_ordered.iloc[i, j]
            color = "white" if abs(val) > 0.65 else "#222"
            weight = "bold" if abs(val) > 0.65 else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8.5, color=color, fontweight=weight)

    def _tag(ch):
        if classification[ch] == "discrete":
            return "D"
        if metrics and ch in metrics:
            cls = metrics[ch]["class"]
            if cls == "강한 주기성":
                return "P"
            if cls == "Trend 위주":
                return "T"
        return "C"

    labels       = [f"{ch} [{_tag(ch)}]" for ch in ordered_chs]
    label_colors = [COLORS["discrete"] if classification[ch] == "discrete"
                    else COLORS["continuous"] for ch in ordered_chs]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for tick, c in zip(ax.get_xticklabels(), label_colors):
        tick.set_color(c)
    for tick, c in zip(ax.get_yticklabels(), label_colors):
        tick.set_color(c)

    cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.04)
    cb.set_label("Pearson r", fontsize=9)

    legend_items = [
        Line2D([0], [0], color=COLORS["continuous"], lw=3, label="Continuous"),
        Line2D([0], [0], color=COLORS["discrete"],   lw=3, label="Discrete"),
        mpatches.Patch(facecolor="none", edgecolor="none",
                       label="[P]=강한 주기성  [T]=Trend 위주  [D]=이산  [C]=복합"),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, -0.25), ncol=2, frameon=False)

    fig.suptitle("Step 3-A.  Pearson Correlation Matrix  —  train.csv\n"
                 "(계층적 클러스터링 순 정렬  |  [P]=주기성  [T]=Trend  [D]=이산  [C]=복합)",
                 fontsize=12, fontweight="bold")

    out = "./eda/outputs/step3a_correlation_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.show()
    plt.close(fig)


def _plot_step3_crosscorr(df, channels, corr, n_lags=600, threshold=0.5, max_pairs=10):
    pairs = sorted(
        [(channels[i], channels[j], float(corr.loc[channels[i], channels[j]]))
         for i in range(len(channels))
         for j in range(i + 1, len(channels))
         if abs(float(corr.loc[channels[i], channels[j]])) >= threshold],
        key=lambda x: -abs(x[2]),
    )[:max_pairs]

    if not pairs:
        print(f"  [step3b] |r| >= {threshold} 인 채널 쌍 없음 — 시각화 생략")
        return

    n_pairs = len(pairs)
    fig, axes = plt.subplots(
        n_pairs, 1, figsize=(16, 2.8 * n_pairs),
        gridspec_kw={"hspace": 0.55},
    )
    if n_pairs == 1:
        axes = [axes]

    lags  = np.arange(-n_lags, n_lags + 1)
    color = COLORS["continuous"]

    for ax, (ch1, ch2, r) in zip(axes, pairs):
        s1 = df[ch1].values.astype(float)
        s2 = df[ch2].values.astype(float)
        s1 = (s1 - s1.mean()) / (s1.std() + 1e-12)
        s2 = (s2 - s2.mean()) / (s2.std() + 1e-12)

        ccf_full  = np.correlate(s1, s2, mode="full") / len(s1)
        mid       = len(ccf_full) // 2
        ccf_slice = ccf_full[mid - n_lags: mid + n_lags + 1]

        ax.plot(lags, ccf_slice, color=color, lw=0.7, alpha=0.85)
        ax.axhline(0, color="gray",    lw=0.5, ls="--")
        ax.axvline(0, color="#888888", lw=1.0, ls="--", alpha=0.6)

        peak_idx = int(np.argmax(np.abs(ccf_slice)))
        peak_lag = int(lags[peak_idx])
        peak_val = float(ccf_slice[peak_idx])
        label = f"peak lag={peak_lag}  (CCF={peak_val:.3f})"
        if peak_lag != 0:
            ax.axvline(peak_lag, color="#E05A2B", lw=1.2, ls=":", label=label)
        else:
            ax.text(0.5, 0.93, label, transform=ax.transAxes,
                    ha="center", va="top", fontsize=8, color="#E05A2B")

        ax.set_title(f"{ch1}  x  {ch2}   (Pearson r = {r:+.3f})",
                     fontsize=9, fontweight="bold", color=color)
        ax.set_xlabel("lag (timestep)", fontsize=8)
        ax.set_ylabel("normalized CCF", fontsize=8)
        if peak_lag != 0:
            ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f5f8ff")
        ax.tick_params(labelsize=7.5)

    fig.suptitle(
        f"Step 3-B.  Cross-Correlation Function  —  train.csv  (|r| >= {threshold})",
        fontsize=13, fontweight="bold", y=1.002,
    )
    out = "./eda/outputs/step3b_cross_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.show()
    plt.close(fig)


def step3_correlation(df, metrics=None):
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    classification = classify_channels(df)
    channels       = get_channels(df)

    print("=" * 60)
    print("Step 3. Multivariate Channel Correlation")
    print("=" * 60)

    corr = df[channels].corr()

    # 계층적 클러스터링으로 채널 순서 재정렬
    dist_mat = 1 - corr.abs().values
    np.fill_diagonal(dist_mat, 0)
    dist_mat  = np.clip((dist_mat + dist_mat.T) / 2, 0, None)
    condensed = squareform(dist_mat, checks=False)
    Z         = linkage(condensed, method="average")
    order     = leaves_list(Z)
    ordered_chs  = [channels[i] for i in order]
    corr_ordered = corr.loc[ordered_chs, ordered_chs]

    # 콘솔 출력
    print("\n  강한 상관관계 쌍 (|r| >= 0.5):")
    found = False
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            r    = float(corr.loc[channels[i], channels[j]])
            if abs(r) >= 0.5:
                sign = "양(+)" if r > 0 else "음(-)"
                print(f"    {channels[i]} -- {channels[j]}: r={r:+.3f}  {sign}")
                found = True
    if not found:
        print("    없음")
    print()

    _plot_step3_heatmap(corr_ordered, ordered_chs, classification, metrics)
    _plot_step3_crosscorr(df, channels, corr, threshold=0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Step 4. Validation 이상징후 유형 및 형태 파악
# ──────────────────────────────────────────────────────────────────────────────

_ANOM_PALETTE = {
    "Point":      "#E05A2B",
    "Contextual": "#FFA040",
    "Collective": "#C0392B",
}


def _get_anomaly_segments(label_arr, pt_max=10, coll_min=300):
    segs, in_seg, start = [], False, 0
    for i, v in enumerate(label_arr):
        if v == 1 and not in_seg:
            start = i; in_seg = True
        elif v == 0 and in_seg:
            length = i - start
            segs.append({"start": start, "end": i - 1, "length": length,
                          "type": "Point" if length <= pt_max
                                  else ("Collective" if length >= coll_min else "Contextual")})
            in_seg = False
    if in_seg:
        length = len(label_arr) - start
        segs.append({"start": start, "end": len(label_arr) - 1, "length": length,
                      "type": "Point" if length <= pt_max
                              else ("Collective" if length >= coll_min else "Contextual")})
    return segs


def _draw_anomaly_overlays(ax, segs, t_arr):
    for seg in segs:
        t0    = float(t_arr[seg["start"]])
        t1    = float(t_arr[seg["end"]])
        color = _ANOM_PALETTE[seg["type"]]
        if seg["type"] == "Point":
            ax.axvline((t0 + t1) / 2, color=color, lw=1.2, alpha=0.9, zorder=5)
        elif seg["type"] == "Contextual":
            ax.axvspan(t0, t1 + 1, color=color, alpha=0.30, zorder=3)
        else:
            ax.axvspan(t0, t1 + 1, color=color, alpha=0.18, zorder=3)


def _plot_step4_overview(val_df, channels, classification, segs):
    from matplotlib.lines import Line2D
    n_ch = len(channels)
    fig, axes = plt.subplots(
        n_ch, 1, sharex=True,
        figsize=(18, 2.0 * n_ch),
        gridspec_kw={"hspace": 0.12},
    )
    if n_ch == 1:
        axes = [axes]

    t = val_df["t"].values

    for i, ch in enumerate(channels):
        ax      = axes[i]
        ch_type = classification[ch]
        color   = COLORS[ch_type]
        vals    = val_df[ch].values

        if ch_type == "discrete":
            ax.step(t, vals, where="post", color=color, lw=0.9, alpha=0.7)
            ax.set_ylim(-0.3, 1.3)
            ax.set_yticks([0, 1])
        else:
            ax.plot(t, vals, color=color, lw=0.5, alpha=0.8)

        _draw_anomaly_overlays(ax, segs, t)

        ax.set_ylabel(ch, rotation=0, ha="right", va="center",
                      fontsize=10, fontweight="bold", color=color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f5f8ff")
        ax.tick_params(labelsize=8)
        if i < n_ch - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("t (timestep)", fontsize=11)

    n_p = sum(1 for s in segs if s["type"] == "Point")
    n_c = sum(1 for s in segs if s["type"] == "Contextual")
    n_k = sum(1 for s in segs if s["type"] == "Collective")
    legend_handles = [
        Line2D([0], [0], color=_ANOM_PALETTE["Point"],
               lw=1.5, label=f"Point Anomaly ({n_p})"),
        mpatches.Patch(facecolor=_ANOM_PALETTE["Contextual"],
                       alpha=0.4, label=f"Contextual Anomaly ({n_c})"),
        mpatches.Patch(facecolor=_ANOM_PALETTE["Collective"],
                       alpha=0.4, label=f"Collective Anomaly ({n_k})"),
        mpatches.Patch(facecolor=COLORS["continuous"], label="Continuous"),
        mpatches.Patch(facecolor=COLORS["discrete"],   label="Discrete"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9,
               bbox_to_anchor=(0.99, 0.999), ncol=5)
    fig.suptitle(
        f"Step 4-A.  Anomaly Overview  —  val.csv  |  {len(val_df):,} timesteps",
        fontsize=13, fontweight="bold", y=1.002,
    )
    out = "./eda/outputs/step4a_anomaly_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.show()
    plt.close(fig)


def _plot_step4_zoom(val_df, channels, classification, segs, context=300):
    from matplotlib.lines import Line2D
    t    = val_df["t"].values
    n_ch = len(channels)
    type_info = {
        "Point":      ("Step 4-B.  Point Anomaly (튀는 값)",           "./eda/outputs/step4b_point_anomaly.png"),
        "Contextual": ("Step 4-B.  Contextual Anomaly (맥락적 이상)",   "./eda/outputs/step4b_contextual_anomaly.png"),
        "Collective": ("Step 4-B.  Collective Anomaly (구간 패턴 이상)", "./eda/outputs/step4b_collective_anomaly.png"),
    }

    for anom_type, (title, out) in type_info.items():
        type_segs = [s for s in segs if s["type"] == anom_type]
        if not type_segs:
            continue
        n_inst = len(type_segs)
        fig, axes = plt.subplots(
            n_ch, n_inst,
            figsize=(n_inst * 5.0, n_ch * 1.8),
            gridspec_kw={"hspace": 0.15, "wspace": 0.08},
        )
        if n_ch == 1:
            axes = axes.reshape(1, -1)
        if n_inst == 1:
            axes = axes.reshape(-1, 1)

        for col, seg in enumerate(type_segs):
            i0    = max(0, seg["start"] - context)
            i1    = min(len(t) - 1, seg["end"] + context)
            t_win = t[i0: i1 + 1]

            for row, ch in enumerate(channels):
                ax      = axes[row, col]
                ch_type = classification[ch]
                color   = COLORS[ch_type]
                vals    = val_df[ch].values[i0: i1 + 1]

                if ch_type == "discrete":
                    ax.step(t_win, vals, where="post", color=color, lw=0.9, alpha=0.8)
                    ax.set_ylim(-0.3, 1.3)
                    ax.set_yticks([0, 1])
                else:
                    ax.plot(t_win, vals, color=color, lw=0.6, alpha=0.85)

                _draw_anomaly_overlays(ax, [seg], t)

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_facecolor("#f5f8ff")
                ax.tick_params(labelsize=7)
                if col == 0:
                    ax.set_ylabel(ch, rotation=0, ha="right", va="center",
                                  fontsize=9, fontweight="bold", color=color)
                else:
                    ax.set_yticklabels([])
                if row < n_ch - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("t", fontsize=8)

            axes[0, col].set_title(
                f"t={seg['start']}~{seg['end']}  (len={seg['length']})",
                fontsize=9, fontweight="bold",
                color=_ANOM_PALETTE[anom_type],
            )

        fig.suptitle(
            f"{title}  —  val.csv   (context ±{context} timesteps)",
            fontsize=12, fontweight="bold",
        )
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        plt.show()
        plt.close(fig)


def step4_anomaly_analysis(val_df):
    label_col      = "label"
    all_chs        = get_channels(val_df)
    channels       = [c for c in all_chs if c != label_col]
    classification = {c: v for c, v in classify_channels(val_df).items() if c != label_col}
    label_arr      = val_df[label_col].values
    segs           = _get_anomaly_segments(label_arr)

    print("=" * 60)
    print("Step 4. Validation Anomaly Analysis")
    print("=" * 60)
    print(f"\n  총 이상치 타임스텝: {label_arr.sum()} / {len(label_arr)}"
          f"  ({label_arr.mean()*100:.1f}%)")
    print(f"  이상치 구간 수: {len(segs)}\n")
    print(f"  {'#':<3} {'type':<12} {'start':>6} {'end':>6} {'length':>7}")
    print("  " + "-" * 40)
    for i, seg in enumerate(segs, 1):
        print(f"  {i:<3} {seg['type']:<12} {seg['start']:>6} {seg['end']:>6} {seg['length']:>7}")
    print()

    _plot_step4_overview(val_df, channels, classification, segs)
    _plot_step4_zoom(val_df, channels, classification, segs)


# ──────────────────────────────────────────────────────────────────────────────
# Step 5. Rolling Mean (이동 평균) 시각화
# ──────────────────────────────────────────────────────────────────────────────

def step5_rolling_mean(df, window_sizes=(30, 200), label_col="label"):
    """
    각 연속형 채널의 원본 신호 위에 rolling mean을 오버레이합니다.

    Parameters
    ----------
    df           : 시각화할 DataFrame (train 또는 val)
    window_sizes : 표시할 윈도우 크기 목록 (기본: 30, 200)
    label_col    : val 데이터의 경우 이상치 레이블 컬럼명 (없으면 None)

    저장 파일:
      eda/outputs/step5_rolling_mean_w{W}.png — 윈도우 크기별 1장씩
    """
    classification = classify_channels(df)
    all_chs        = get_channels(df)
    channels       = [c for c in all_chs if c != label_col and classification.get(c) == "continuous"]

    has_label = label_col in df.columns
    segs      = _get_anomaly_segments(df[label_col].values) if has_label else []

    print("=" * 60)
    print("Step 5. Rolling Mean Visualization")
    print("=" * 60)
    print(f"  연속형 채널: {channels}")
    print(f"  윈도우 크기: {list(window_sizes)}\n")

    ROLL_COLORS = ["#E05A2B", "#2E8B57", "#9B59B6", "#F39C12"]
    t = df["t"].values if "t" in df.columns else np.arange(len(df))

    for W in window_sizes:
        n_ch = len(channels)
        fig, axes = plt.subplots(
            n_ch, 1, sharex=True,
            figsize=(18, 2.2 * n_ch),
            gridspec_kw={"hspace": 0.15},
        )
        if n_ch == 1:
            axes = [axes]

        for i, ch in enumerate(channels):
            ax     = axes[i]
            values = df[ch].values
            rm     = pd.Series(values).rolling(window=W, min_periods=1).mean().to_numpy()

            # 원본 신호 (연한 배경)
            ax.plot(t, values, color=COLORS["continuous"], lw=0.5, alpha=0.4, label="원본")
            # rolling mean 오버레이
            ax.plot(t, rm, color=ROLL_COLORS[0], lw=1.2, alpha=0.9, label=f"Rolling Mean (W={W})")

            # 이상 구간 표시 (val 데이터일 경우)
            if segs:
                _draw_anomaly_overlays(ax, segs, t)

            ax.set_ylabel(ch, rotation=0, ha="right", va="center",
                          fontsize=10, fontweight="bold", color=COLORS["continuous"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_facecolor("#f5f8ff")
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.legend(fontsize=8, loc="upper right")
            if i < n_ch - 1:
                ax.tick_params(labelbottom=False)

        axes[-1].set_xlabel("t (timestep)", fontsize=11)
        split = "val" if has_label else "train"
        fig.suptitle(
            f"Step 5.  Rolling Mean (W={W})  —  {split}.csv  |  {len(df):,} timesteps",
            fontsize=13, fontweight="bold", y=1.002,
        )

        out = f"./eda/outputs/step5_rolling_mean_w{W}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        plt.show()
        plt.close(fig)


def step5_rolling_mean_compare(df, channel, window_sizes=(30, 200), label_col="label"):
    """
    단일 채널에 대해 여러 윈도우 크기의 rolling mean을 한 장에 비교합니다.

    Parameters
    ----------
    df           : 시각화할 DataFrame
    channel      : 비교할 채널명 (예: 'x_f8')
    window_sizes : 비교할 윈도우 크기 목록
    label_col    : 이상치 레이블 컬럼명 (없으면 None)
    """
    has_label = label_col in df.columns
    segs      = _get_anomaly_segments(df[label_col].values) if has_label else []
    t         = df["t"].values if "t" in df.columns else np.arange(len(df))
    values    = df[channel].values

    ROLL_COLORS = ["#E05A2B", "#2E8B57", "#9B59B6", "#F39C12"]

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(t, values, color=COLORS["continuous"], lw=0.5, alpha=0.35, label="원본")

    for idx, W in enumerate(window_sizes):
        rm = pd.Series(values).rolling(window=W, min_periods=1).mean().to_numpy()
        ax.plot(t, rm, color=ROLL_COLORS[idx % len(ROLL_COLORS)],
                lw=1.3, alpha=0.9, label=f"W={W}")

    if segs:
        _draw_anomaly_overlays(ax, segs, t)

    ax.set_xlabel("t (timestep)", fontsize=11)
    ax.set_ylabel(channel, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f5f8ff")

    split = "val" if has_label else "train"
    fig.suptitle(
        f"Step 5.  Rolling Mean 비교 — {channel}  ({split}.csv)",
        fontsize=12, fontweight="bold",
    )

    out = f"./eda/outputs/step5_rolling_mean_compare_{channel}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Step 6. Validation Residual Visualization
# ---------------------------------------------------------------------------

def step6_validation_residual(val_df, metrics=None, label_col="label"):
    """
    Visualize seasonal-decomposition residuals for validation continuous channels.

    The period detected from train.csv in Step 2 is reused when available so the
    residual definition stays consistent between train and validation.
    """
    classification = classify_channels(val_df)
    continuous_chs = [
        ch for ch, ch_type in classification.items()
        if ch != label_col and ch_type == "continuous"
    ]
    if not continuous_chs:
        print("Step 6. Validation Residual Visualization - no continuous channels")
        return

    t         = val_df["t"].values if "t" in val_df.columns else np.arange(len(val_df))
    has_label = label_col in val_df.columns
    segs      = _get_anomaly_segments(val_df[label_col].values) if has_label else []

    print("=" * 60)
    print("Step 6. Validation Residual Visualization")
    print("=" * 60)

    fig, axes = plt.subplots(
        len(continuous_chs), 1, sharex=True,
        figsize=(18, 2.2 * len(continuous_chs)),
        gridspec_kw={"hspace": 0.12},
    )
    if len(continuous_chs) == 1:
        axes = [axes]

    for i, ch in enumerate(continuous_chs):
        ax = axes[i]
        period = None
        if metrics is not None and ch in metrics:
            period = metrics[ch].get("period")
        if period is None:
            period = _detect_period(val_df[ch])

        try:
            result = seasonal_decompose(
                val_df[ch], model="additive", period=period, extrapolate_trend="freq"
            )
            residual = result.resid.to_numpy()
        except Exception as e:
            print(f"  [validation residual] {ch} failed: {e}")
            residual = np.full(len(val_df), np.nan)

        ax.plot(t, residual, color="#666666", lw=0.6, alpha=0.9, label="Residual")
        ax.axhline(0, color="#cccccc", lw=0.8, ls="--", zorder=0)
        if segs:
            _draw_anomaly_overlays(ax, segs, t)

        finite = residual[np.isfinite(residual)]
        if len(finite):
            res_max  = float(np.max(np.abs(finite)))
            res_mean = float(np.mean(finite))
            margin   = res_max * 1.1 if res_max > 0 else 1.0
            ax.set_ylim(res_mean - margin, res_mean + margin)
            ax.text(
                0.005, 0.90,
                f"period={period}  max |residual|={res_max:.2e}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="#666666",
            )

        ax.set_ylabel(ch, rotation=0, ha="right", va="center",
                      fontsize=10, fontweight="bold", color="#666666")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f5f8ff")
        ax.tick_params(labelsize=8)
        if i < len(continuous_chs) - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("t (timestep)", fontsize=11)
    fig.suptitle(
        f"Step 6.  Validation Residual  -  val.csv  |  {len(val_df):,} timesteps",
        fontsize=13, fontweight="bold", y=1.002,
    )
    out = "./eda/outputs/step6_validation_residual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.show()
    plt.close(fig)


def main():
    train, val, test = load_data(missing_method=MISSING_METHOD)

    step1_channel_properties(train)
    metrics, _ = step2_seasonality_trend(train)
    step3_correlation(train, metrics=metrics)
    step4_anomaly_analysis(val)
    step5_rolling_mean(train, window_sizes=(30, 200))
    step5_rolling_mean(val,   window_sizes=(30, 200))
    step6_validation_residual(val, metrics=metrics)


if __name__ == "__main__":
    main()
