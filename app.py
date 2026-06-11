import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
from dotenv import load_dotenv
import os
import threading
import datetime

load_dotenv()

# ── Model & data ───────────────────────────────────────────────────────────────
model    = joblib.load("electricity_model.pkl")
features = joblib.load("features.pkl")          # plain list from X.columns.tolist()

explainer = shap.TreeExplainer(model)

df         = pd.read_csv("electricity_bill_dataset.csv")
BILL_MEAN  = df["ElectricityBill"].mean()
BILL_STD   = df["ElectricityBill"].std()

APPLIANCES = ["Fan", "Refrigerator", "AirConditioner", "Television", "Monitor", "MotorPump"]

RANGES = {
    "Fan": (0, 10), "Refrigerator": (0, 3), "AirConditioner": (0, 3),
    "Television": (0, 4), "Monitor": (0, 3), "MotorPump": (0, 2),
    "Month": (1, 12), "DailyHours": (4, 18), "TariffRate": (5, 25),
}

# ── Theme ──────────────────────────────────────────────────────────────────────
BG     = "#0f172a"
CARD   = "#1e293b"
ACCENT = "#22c55e"
RED    = "#ef4444"
TEXT   = "white"
MUTED  = "#94a3b8"


# ── Helpers ────────────────────────────────────────────────────────────────────
def make_sample(data):
    return pd.DataFrame([data], columns=features)

def predict(data):
    return max(0.0, model.predict(make_sample(data))[0])

def lbl(parent, text, font=("Arial", 11), fg=TEXT, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg,
                    bg=parent.cget("bg"), **kw)

def card(parent, **kw):
    return tk.Frame(parent, bg=CARD, padx=14, pady=10, **kw)

def btn(parent, text, cmd, **kw):
    return tk.Button(parent, text=text, command=cmd,
                     font=("Arial", 11, "bold"), bg=ACCENT, fg="black",
                     relief="flat", cursor="hand2", padx=12, pady=6, **kw)

def dark_axes(fig, ax):
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")


# ── Root window ────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Smart Electricity Bill Predictor AI")
root.configure(bg=BG)
root.resizable(True, True)
root.state("zoomed")   # maximized on Windows

lbl(root, "SMART ELECTRICITY BILL PREDICTOR  |  AI",
    font=("Helvetica", 20, "bold")).pack(pady=(14, 4))

# ── Notebook + widget styles ───────────────────────────────────────────────────
style = ttk.Style()
style.theme_use("clam")
style.configure("TNotebook",     background=BG, borderwidth=0)
style.configure("TNotebook.Tab", background=CARD, foreground=TEXT,
                font=("Arial", 11, "bold"), padding=[14, 7])
style.map("TNotebook.Tab",
          background=[("selected", ACCENT)],
          foreground=[("selected", "black")])
style.configure("Field.TEntry",
                fieldbackground="#0f172a", foreground="white",
                insertcolor="white", borderwidth=0, relief="flat",
                font=("Arial", 13))
style.configure("TCombobox",
                fieldbackground="#0f172a", foreground="white",
                selectbackground=ACCENT, selectforeground="black",
                borderwidth=0)
style.map("TCombobox", fieldbackground=[("readonly", "#0f172a")],
          foreground=[("readonly", "white")])

nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True, padx=15, pady=(0, 10))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
t1 = tk.Frame(nb, bg=BG)
nb.add(t1, text="  Predictor  ")

MONTH_NAMES = {
    "1 - January": 1,  "2 - February": 2,  "3 - March": 3,
    "4 - April": 4,    "5 - May": 5,        "6 - June": 6,
    "7 - July": 7,     "8 - August": 8,     "9 - September": 9,
    "10 - October": 10, "11 - November": 11, "12 - December": 12,
}

FIELD_HINTS = {
    "Fan":            ("Number of fans", "e.g.  3"),
    "Refrigerator":   ("Number of fridges", "e.g.  1"),
    "AirConditioner": ("Number of ACs", "e.g.  1"),
    "Television":     ("Number of TVs", "e.g.  2"),
    "Monitor":        ("Number of monitors", "e.g.  1"),
    "MotorPump":      ("Motor pumps", "e.g.  0"),
    "Month":          ("Current month", "Select from list"),
    "DailyHours":     ("Avg hours/day appliances run", "e.g.  8"),
    "TariffRate":     ("Electricity rate (PKR/kWh)", "e.g.  15"),
}

# ── section header ──
hdr = tk.Frame(t1, bg=BG)
hdr.pack(fill="x", padx=24, pady=(14, 4))
lbl(hdr, "Enter Your Usage Details",
    font=("Helvetica", 15, "bold")).pack(side="left")
lbl(hdr, "Fill in all fields then click Predict",
    font=("Arial", 10), fg=MUTED).pack(side="left", padx=14)

# ── 3-column input grid ──
grid_outer = tk.Frame(t1, bg=BG)
grid_outer.pack(fill="x", padx=18, pady=4)

pred_sliders = {}   # shared with Tips & AI tabs

for col in range(3):
    grid_outer.columnconfigure(col, weight=1, uniform="col")

for i, feat in enumerate(features):
    row_num = i // 3
    col_num = i %  3

    # field card
    fc = tk.Frame(grid_outer, bg=CARD,
                  highlightbackground="#334155", highlightthickness=1)
    fc.grid(row=row_num, column=col_num, padx=8, pady=8, sticky="nsew")

    inner = tk.Frame(fc, bg=CARD)
    inner.pack(fill="both", expand=True, padx=16, pady=12)

    title_txt, hint_txt = FIELD_HINTS.get(feat, (feat, ""))
    lbl(inner, feat, font=("Arial", 11, "bold"), fg=TEXT).pack(anchor="w")
    lbl(inner, title_txt, font=("Arial", 9), fg=MUTED).pack(anchor="w")

    sep = tk.Frame(inner, bg="#334155", height=1)
    sep.pack(fill="x", pady=(6, 8))

    if feat == "Month":
        cb = ttk.Combobox(inner, values=list(MONTH_NAMES.keys()),
                          state="readonly", font=("Arial", 12), width=20)
        current_month = datetime.datetime.now().month
        default_key   = [k for k, v in MONTH_NAMES.items() if v == current_month][0]
        cb.set(default_key)
        cb.pack(fill="x")

        class _MonthProxy:
            def __init__(self, c): self._c = c
            def get(self): return str(MONTH_NAMES.get(self._c.get(), 1))

        pred_sliders[feat] = _MonthProxy(cb)
    else:
        lo, hi  = RANGES.get(feat, (0, 100))
        default = int((lo + hi) / 2)
        e = tk.Entry(inner, font=("Arial", 14), bg="#0f172a", fg="white",
                     insertbackground="white", relief="flat",
                     highlightthickness=1, highlightbackground="#475569",
                     highlightcolor=ACCENT, width=14)
        e.insert(0, str(default))
        e.pack(fill="x", ipady=6)
        lbl(inner, hint_txt, font=("Arial", 8), fg="#475569").pack(anchor="w", pady=(3, 0))
        pred_sliders[feat] = e

# ── predict button + error label ──
btn_bar = tk.Frame(t1, bg=BG)
btn_bar.pack(pady=(4, 6))

err_lbl = lbl(btn_bar, "", font=("Arial", 9), fg=RED)
err_lbl.pack()

predict_btn = tk.Button(btn_bar, text="  PREDICT  &  EXPLAIN  ",
                        command=lambda: _do_predict(),
                        font=("Arial", 14, "bold"), bg=ACCENT, fg="#0f172a",
                        relief="flat", cursor="hand2",
                        padx=30, pady=10, bd=0,
                        activebackground="#16a34a", activeforeground="black")
predict_btn.pack()

# ── result bar ──
res_bar = tk.Frame(t1, bg="#1a2744",
                   highlightbackground="#334155", highlightthickness=1)
res_bar.pack(fill="x", padx=18, pady=(2, 6))

res_inner = tk.Frame(res_bar, bg="#1a2744")
res_inner.pack(fill="x", padx=20, pady=12)

pred_result  = lbl(res_inner, "Enter values above and click Predict",
                   font=("Helvetica", 22, "bold"), fg=MUTED)
pred_result.pack(side="left")

pred_anomaly = lbl(res_inner, "", font=("Arial", 11, "bold"), fg=MUTED)
pred_anomaly.pack(side="right", padx=10)

# ── SHAP chart area ──
shap_holder  = tk.Frame(t1, bg=BG)
shap_holder.pack(fill="both", expand=True, padx=18, pady=(0, 8))

_shap_canvas = [None]

def _do_predict():
    try:
        data = [float(pred_sliders[f].get()) for f in features]
        err_lbl.config(text="")
    except ValueError:
        err_lbl.config(text="Please enter valid numbers in all fields.")
        return

    p = predict(data)
    pred_result.config(text=f"Rs. {p:,.2f} / month", fg=ACCENT)

    if p > BILL_MEAN + 1.5 * BILL_STD:
        pred_anomaly.config(text="  ⚠  HIGH bill detected!", fg=RED)
    elif p < BILL_MEAN - 1.5 * BILL_STD:
        pred_anomaly.config(text="  ✓  Below average!", fg=ACCENT)
    else:
        pred_anomaly.config(text="  Normal range", fg=MUTED)

    sv = explainer.shap_values(make_sample(data))[0]

    if _shap_canvas[0]:
        _shap_canvas[0].get_tk_widget().destroy()

    fig, ax = plt.subplots(figsize=(13, 3.4))
    dark_axes(fig, ax)
    colors = [ACCENT if v >= 0 else RED for v in sv]
    ax.barh(list(features), sv, color=colors, height=0.55)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("PKR Contribution to Bill", color=TEXT, fontsize=9)
    ax.set_title("Why this bill?   (SHAP explanation — green = raises bill, red = lowers it)",
                 color=TEXT, fontsize=10, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    c = FigureCanvasTkAgg(fig, master=shap_holder)
    c.draw()
    c.get_tk_widget().pack(fill="both", expand=True)
    _shap_canvas[0] = c
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
t2 = tk.Frame(nb, bg=BG)
nb.add(t2, text="  Insights  ")

# ── page header ──
t2_hdr = tk.Frame(t2, bg=BG)
t2_hdr.pack(fill="x", padx=20, pady=(14, 6))
lbl(t2_hdr, "Model Insights  &  What-If Simulator",
    font=("Helvetica", 15, "bold")).pack(side="left")

# ── two-panel row ──
t2_row = tk.Frame(t2, bg=BG)
t2_row.pack(fill="both", expand=True, padx=18, pady=(0, 12))
t2_row.columnconfigure(0, weight=5)
t2_row.columnconfigure(1, weight=4)
t2_row.rowconfigure(0, weight=1)

# ─── LEFT: Feature Importance ───────────────────────────────────────────────
fi_card = tk.Frame(t2_row, bg=CARD,
                   highlightbackground="#334155", highlightthickness=1)
fi_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

fi_hdr = tk.Frame(fi_card, bg=CARD)
fi_hdr.pack(fill="x", padx=18, pady=(14, 4))
lbl(fi_hdr, "Feature Importance", font=("Arial", 13, "bold")).pack(side="left")
lbl(fi_hdr, "Which inputs drive your bill the most",
    font=("Arial", 9), fg=MUTED).pack(side="left", padx=10)

sep2 = tk.Frame(fi_card, bg="#334155", height=1)
sep2.pack(fill="x", padx=18)

importances = model.feature_importances_
idx_sorted  = np.argsort(importances)
feat_names  = [features[i] for i in idx_sorted]
imp_vals    = importances[idx_sorted]

# gradient colours: low importance = muted, high = ACCENT
bar_colors  = [ACCENT if v >= np.median(imp_vals) else "#16a34a" for v in imp_vals]

fig_fi, ax_fi = plt.subplots(figsize=(7, 4.8))
dark_axes(fig_fi, ax_fi)
bars = ax_fi.barh(feat_names, imp_vals, color=bar_colors, height=0.55)
ax_fi.set_xlabel("Importance Score", color=MUTED, fontsize=9)
ax_fi.set_title("Feature Importance  (Random Forest)",
                color=TEXT, fontsize=11, fontweight="bold", pad=10)
for bar, val in zip(bars, imp_vals):
    ax_fi.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
               f"{val:.3f}", va="center", ha="left", color=TEXT, fontsize=8)
ax_fi.set_xlim(0, max(imp_vals) * 1.18)
for sp in ax_fi.spines.values():
    sp.set_visible(False)
ax_fi.xaxis.label.set_color(MUTED)
fig_fi.tight_layout(pad=1.5)

c_fi = FigureCanvasTkAgg(fig_fi, master=fi_card)
c_fi.draw()
c_fi.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
plt.close(fig_fi)

# ─── RIGHT: Savings Analysis ────────────────────────────────────────────────
sim_card = tk.Frame(t2_row, bg=CARD,
                    highlightbackground="#334155", highlightthickness=1)
sim_card.grid(row=0, column=1, sticky="nsew")

sim_hdr = tk.Frame(sim_card, bg=CARD)
sim_hdr.pack(fill="x", padx=18, pady=(14, 4))
lbl(sim_hdr, "Bill Breakdown", font=("Arial", 13, "bold")).pack(side="left")
lbl(sim_hdr, "How much each feature contributes to your bill",
    font=("Arial", 9), fg=MUTED).pack(side="left", padx=10)

sep3 = tk.Frame(sim_card, bg="#334155", height=1)
sep3.pack(fill="x", padx=18)

bill_row = tk.Frame(sim_card, bg=CARD)
bill_row.pack(fill="x", padx=18, pady=(10, 2))
lbl(bill_row, "Current Bill:", font=("Arial", 11), fg=MUTED).pack(side="left")
savings_base_lbl = lbl(bill_row, "  Rs. ---",
                       font=("Arial", 14, "bold"), fg=ACCENT)
savings_base_lbl.pack(side="left")

lbl(sim_card,
    "Enter values in Predictor tab first, then click below",
    font=("Arial", 9), fg=MUTED).pack(pady=(0, 4))

savings_chart_frame = tk.Frame(sim_card, bg=CARD)
savings_chart_frame.pack(fill="both", expand=True, padx=10, pady=4)

_savings_canvas = [None]


def _refresh_savings():
    try:
        data = [float(pred_sliders[f].get()) for f in features]
    except ValueError:
        return
    base = predict(data)
    savings_base_lbl.config(text=f"  Rs. {base:,.2f}")

    # SHAP absolute contribution — always non-zero for every feature
    sv_raw = explainer.shap_values(make_sample(data))[0]
    sv_abs = np.abs(sv_raw)

    idx          = np.argsort(sv_abs)           # ascending → biggest at top
    sorted_names = [features[i] for i in idx]
    sorted_vals  = sv_abs[idx]

    if _savings_canvas[0]:
        _savings_canvas[0].get_tk_widget().destroy()

    mx = max(sorted_vals) if max(sorted_vals) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    dark_axes(fig, ax)
    colors = [ACCENT if v > mx * 0.3 else "#16a34a" if v > mx * 0.05 else "#475569"
              for v in sorted_vals]
    bars = ax.barh(sorted_names, sorted_vals, color=colors, height=0.55)

    for bar, val in zip(bars, sorted_vals):
        label = f"Rs.{val:,.0f}" if val >= 0.5 else "< Rs.1"
        ax.text(val + mx * 0.02,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center", ha="left", color=TEXT, fontsize=8)

    ax.set_xlabel("PKR contribution to your bill (SHAP)", color=MUTED, fontsize=9)
    ax.set_title("How much does each feature add to your bill?",
                 color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlim(0, mx * 1.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=1.5)

    c = FigureCanvasTkAgg(fig, master=savings_chart_frame)
    c.draw()
    c.get_tk_widget().pack(fill="both", expand=True)
    _savings_canvas[0] = c
    plt.close(fig)


btn(sim_card, "Refresh from Predictor Tab", _refresh_savings).pack(pady=(6, 10))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TIPS & FORECAST
# ══════════════════════════════════════════════════════════════════════════════
t3 = tk.Frame(nb, bg=BG)
nb.add(t3, text="  Tips & Forecast  ")

t3l = tk.Frame(t3, bg=BG)
t3l.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

t3r = tk.Frame(t3, bg=BG)
t3r.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

# ── smart saving tips ──
tips_card = card(t3l)
tips_card.pack(fill="both", expand=True)

lbl(tips_card, "Smart Saving Tips", font=("Arial", 12, "bold")).pack()
lbl(tips_card, "Based on Predictor tab sliders",
    font=("Arial", 9), fg=MUTED).pack()

tips_box = tk.Text(tips_card, bg="#0a1628", fg=TEXT, font=("Consolas", 10),
                   relief="flat", height=10, wrap="word", state="disabled")
tips_box.pack(fill="both", expand=True, pady=6)

def _show_tips():
    data = [float(pred_sliders[f].get()) for f in features]
    base = predict(data)
    rows = [f"Current Bill:  Rs.{base:,.2f}\n\n20% reduction per appliance:\n"]
    savings = []

    for app in APPLIANCES:
        if app not in features:
            continue
        i    = features.index(app)
        d2   = data.copy()
        d2[i] = max(0, d2[i] * 0.8)
        s    = base - predict(d2)
        if s > 1:
            savings.append((app, s))

    savings.sort(key=lambda x: x[1], reverse=True)

    if savings:
        for app, s in savings:
            rows.append(f"  Reduce {app:<18}  →  save Rs.{s:,.2f}\n")
    else:
        rows.append("  Usage already looks optimised!")

    tips_box.config(state="normal")
    tips_box.delete("1.0", "end")
    tips_box.insert("end", "".join(rows))
    tips_box.config(state="disabled")

btn(tips_card, "Generate Saving Tips", _show_tips).pack(pady=4)

# ── bill forecast ──
fc_card = card(t3r)
fc_card.pack(fill="both", expand=True)

lbl(fc_card, "Bill Forecast (Next Month)", font=("Arial", 12, "bold")).pack()
lbl(fc_card, "Enter your last 3 months' actual bills",
    font=("Arial", 9), fg=MUTED).pack()

fc_entries = []
fc_row = tk.Frame(fc_card, bg=CARD)
fc_row.pack(pady=8)
for i in range(3):
    col = tk.Frame(fc_row, bg=CARD)
    col.pack(side="left", padx=10)
    lbl(col, f"Month {i+1}  (PKR)", font=("Arial", 9), fg=MUTED).pack()
    e = ttk.Entry(col, width=9, font=("Arial", 10))
    e.pack()
    fc_entries.append(e)

fc_result = lbl(fc_card, "", font=("Arial", 14, "bold"), fg=ACCENT)
fc_result.pack(pady=3)

fc_note = lbl(fc_card, "", font=("Arial", 9), fg=MUTED)
fc_note.pack()

fc_chart_frame = tk.Frame(fc_card, bg=CARD)
fc_chart_frame.pack(fill="both", expand=True, pady=4)

_fc_canvas = [None]

def _forecast():
    try:
        bills = [float(e.get()) for e in fc_entries]
    except ValueError:
        fc_result.config(text="Enter valid amounts in all 3 fields", fg=RED)
        fc_note.config(text="")
        return

    slope, intercept = np.polyfit([1, 2, 3], bills, 1)
    nxt   = max(0.0, slope * 4 + intercept)
    trend = ("Rising trend" if slope > 50 else
             "Falling trend" if slope < -50 else "Stable trend")

    fc_result.config(text=f"Next Month:  Rs.{nxt:,.2f}",
                     fg=RED if slope > 50 else ACCENT)
    fc_note.config(text=trend)

    # Draw trend chart
    if _fc_canvas[0]:
        _fc_canvas[0].get_tk_widget().destroy()

    fig, ax = plt.subplots(figsize=(4, 2.4))
    dark_axes(fig, ax)
    ax.plot([1, 2, 3], bills,         "o-",  color=ACCENT, label="Actual",   linewidth=2)
    ax.plot([3, 4],    [bills[-1], nxt], "o--", color=RED,   label="Forecast", linewidth=2)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Month 1", "Month 2", "Month 3", "Next"], fontsize=7)
    ax.tick_params(colors=TEXT)
    ax.legend(fontsize=7, facecolor=CARD, edgecolor="#334155", labelcolor=TEXT)
    fig.tight_layout()

    c = FigureCanvasTkAgg(fig, master=fc_chart_frame)
    c.draw()
    c.get_tk_widget().pack(fill="both", expand=True)
    _fc_canvas[0] = c
    plt.close(fig)

btn(fc_card, "Forecast Next Month", _forecast).pack(pady=4)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
t4 = tk.Frame(nb, bg=BG)
nb.add(t4, text="  AI Assistant  ")

# ── header bar ──────────────────────────────────────────────────────────────
t4_head = tk.Frame(t4, bg=CARD, highlightbackground="#334155", highlightthickness=1)
t4_head.pack(fill="x")
t4_hi = tk.Frame(t4_head, bg=CARD)
t4_hi.pack(fill="x", padx=22, pady=12)
lbl(t4_hi, "⚡  AI Bill Assistant",
    font=("Helvetica", 15, "bold"), fg=ACCENT).pack(side="left")
lbl(t4_hi, "Powered by Free ChatGPT API  •  Uses your Predictor values as context",
    font=("Arial", 9), fg=MUTED).pack(side="left", padx=18)

# ── scrollable chat messages area ──────────────────────────────────────────
chat_outer = tk.Frame(t4, bg=BG)
chat_outer.pack(fill="both", expand=True)

chat_canvas = tk.Canvas(chat_outer, bg=BG, highlightthickness=0, bd=0)
chat_scroll = tk.Scrollbar(chat_outer, orient="vertical", command=chat_canvas.yview)
chat_canvas.configure(yscrollcommand=chat_scroll.set)
chat_scroll.pack(side="right", fill="y")
chat_canvas.pack(side="left", fill="both", expand=True)

msgs_frame = tk.Frame(chat_canvas, bg=BG)
_cwin = chat_canvas.create_window((0, 0), window=msgs_frame, anchor="nw")


def _on_msgs_resize(e):
    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))

def _on_canvas_resize(e):
    chat_canvas.itemconfig(_cwin, width=e.width)

def _mousewheel(e):
    chat_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

def _scroll_to_bottom():
    msgs_frame.update_idletasks()
    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
    chat_canvas.yview_moveto(1.0)

msgs_frame.bind("<Configure>", _on_msgs_resize)
chat_canvas.bind("<Configure>",  _on_canvas_resize)
chat_canvas.bind("<MouseWheel>", _mousewheel)
msgs_frame.bind("<MouseWheel>",  _mousewheel)

# ── input bar ───────────────────────────────────────────────────────────────
inp_card = tk.Frame(t4, bg="#0d1f38",
                    highlightbackground="#334155", highlightthickness=1)
inp_card.pack(fill="x", padx=20, pady=(4, 16))

inp_inner = tk.Frame(inp_card, bg="#1a2e4a",
                     highlightbackground="#475569", highlightthickness=1)
inp_inner.pack(fill="x", padx=3, pady=3)

chat_text = tk.Text(
    inp_inner, height=3, font=("Arial", 13), bg="#1a2e4a", fg=TEXT,
    insertbackground=TEXT, relief="flat", wrap="word",
    highlightthickness=0, padx=16, pady=12, bd=0,
)
chat_text.pack(side="left", fill="x", expand=True)

PLACEHOLDER = "Ask about your electricity bill...  (Press Enter to send, Ctrl+Enter for new line)"
chat_text.insert("1.0", PLACEHOLDER)
chat_text.config(fg="#64748b")

send_btn = tk.Button(
    inp_inner, text="Send  ▶",
    font=("Arial", 12, "bold"), bg=ACCENT, fg="#0f172a",
    relief="flat", cursor="hand2", padx=22, pady=0,
    activebackground="#16a34a", activeforeground="black",
)
send_btn.pack(side="right", padx=12, pady=12)

status_lbl = lbl(t4, "", font=("Arial", 9), fg=MUTED)
status_lbl.pack(pady=(0, 4))


def _focus_in(e):
    if chat_text.get("1.0", "end-1c") == PLACEHOLDER:
        chat_text.delete("1.0", "end")
        chat_text.config(fg=TEXT)

def _focus_out(e):
    if not chat_text.get("1.0", "end-1c").strip():
        chat_text.insert("1.0", PLACEHOLDER)
        chat_text.config(fg="#64748b")

chat_text.bind("<FocusIn>",  _focus_in)
chat_text.bind("<FocusOut>", _focus_out)


def _append(role, text):
    row = tk.Frame(msgs_frame, bg=BG)
    row.pack(fill="x", padx=18, pady=6)

    if role == "user":
        tk.Frame(row, bg=BG).pack(side="left", fill="x", expand=True)
        bubble = tk.Frame(row, bg="#1e3a5f",
                          highlightbackground="#2d5a8e", highlightthickness=1)
        bubble.pack(side="right")
        lbl(bubble, "You", font=("Arial", 8, "bold"),
            fg="#7db3e0").pack(anchor="e", padx=14, pady=(8, 2))
        tk.Label(bubble, text=text, font=("Arial", 11), bg="#1e3a5f",
                 fg=TEXT, wraplength=760, justify="right",
                 padx=14, pady=8).pack(anchor="e")
    else:
        icon = tk.Label(row, text="⚡", font=("Arial", 13),
                        bg=ACCENT, fg="#0f172a", width=3)
        icon.pack(side="left", anchor="n", padx=(0, 12), pady=4)
        bubble = tk.Frame(row, bg=CARD,
                          highlightbackground="#334155", highlightthickness=1)
        bubble.pack(side="left", fill="x", expand=True)
        lbl(bubble, "AI Assistant", font=("Arial", 8, "bold"),
            fg=ACCENT).pack(anchor="w", padx=14, pady=(8, 2))
        tk.Label(bubble, text=text, font=("Arial", 11), bg=CARD,
                 fg=TEXT, wraplength=1000, justify="left",
                 padx=14, pady=8).pack(anchor="w", fill="x")

    root.after(60, _scroll_to_bottom)


def _send():
    msg = chat_text.get("1.0", "end-1c").strip()
    if not msg or msg == PLACEHOLDER:
        return
    chat_text.delete("1.0", "end")
    chat_text.config(fg=TEXT)
    _append("user", msg)
    send_btn.config(state="disabled")
    status_lbl.config(text="  AI is thinking...")

    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        _append("ai", "RapidAPI key not found. Add RAPIDAPI_KEY=your_key to the .env file.")
        send_btn.config(state="normal")
        status_lbl.config(text="")
        return

    data  = [float(pred_sliders[f].get()) for f in features]
    p     = predict(data)
    usage = {f: f"{v:.0f}" for f, v in zip(features, data)}

    def _call():
        try:
            full_prompt = (
                f"You are an electricity bill advisor. "
                f"User bill: Rs.{p:.2f}/month. "
                f"Appliances: Fan={usage.get('Fan','?')}, "
                f"AC={usage.get('AirConditioner','?')}, "
                f"Fridge={usage.get('Refrigerator','?')}, "
                f"DailyHours={usage.get('DailyHours','?')}h/day, "
                f"Tariff=Rs.{usage.get('TariffRate','?')}/kWh. "
                f"Answer in 2-3 sentences. Question: {msg}"
            )
            resp = requests.get(
                "https://free-chatgpt-api.p.rapidapi.com/chat-completion-one",
                params={"prompt": full_prompt},
                headers={
                    "x-rapidapi-key":  api_key,
                    "x-rapidapi-host": "free-chatgpt-api.p.rapidapi.com",
                },
                timeout=20,
            )
            if resp.status_code != 200:
                reply = f"API Error {resp.status_code}: {resp.text[:200]}"
            else:
                data_r = resp.json()
                reply = (data_r.get("response")
                         or data_r.get("text")
                         or data_r.get("choices", [{}])[0].get("message", {}).get("content")
                         or str(data_r))
        except Exception as ex:
            reply = f"Could not reach AI: {ex}"
        root.after(0, lambda: [
            _append("ai", reply),
            send_btn.config(state="normal"),
            status_lbl.config(text=""),
        ])

    threading.Thread(target=_call, daemon=True).start()


def _on_enter_key(e):
    if not (e.state & 0x4):   # plain Enter → send; Ctrl+Enter → newline
        _send()
        return "break"

chat_text.bind("<Return>", _on_enter_key)
send_btn.config(command=_send)

_append("ai", "Hello! Set your appliance values in the Predictor tab, then ask me anything — e.g. 'Why is my bill high?' or 'How can I save money?'")


root.mainloop()
