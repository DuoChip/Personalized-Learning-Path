# File: demo/app.py
# Purpose: Interactive Modern MIS demo: learner profile, knowledge gaps, Top-N path and live update.

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

import core

st.set_page_config(page_title="MOOC-Radar · Lộ trình học cá nhân hóa", page_icon="🧭",
                   layout="wide", initial_sidebar_state="expanded")

INDIGO, SKY, TEAL, AMBER, ROSE, SLATE = "#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#E11D48", "#94A3B8"
MODEL_VI = {"smoothed_concept": "Smoothed concept (chọn)", "bkt": "BKT (đối chứng)"}
MODEL_COLOR = {"smoothed_concept": INDIGO, "bkt": AMBER}
SPLIT_VI = {"train": "Huấn luyện", "validation": "Kiểm định", "test": "Kiểm tra"}
PLOT_CONFIG = {"displayModeBar": False}
RADAR_AXES = dict(radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1], angle=90, tickangle=90,
                                  tickfont=dict(size=10, color="#94A3B8")),
                  angularaxis=dict(tickfont=dict(size=11)))

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea, select { font-family: 'Be Vietnam Pro', sans-serif !important; }
.block-container { padding-top: 1.4rem; max-width: 1320px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%); }
section[data-testid="stSidebar"] * { color: #E2E8F0; }
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * { color: #0F172A !important;
  -webkit-text-fill-color: #0F172A !important; font-weight: 600; }
section[data-testid="stSidebar"] code { background: rgba(255,255,255,.12); color: #C7D2FE; }
header[data-testid="stHeader"] { background: transparent; }
.hero { background: linear-gradient(120deg, #1E1B4B 0%, #4F46E5 50%, #0EA5E9 100%); color: #fff;
        border-radius: 22px; padding: 30px 34px; margin-bottom: 22px; position: relative; overflow: hidden;
        box-shadow: 0 18px 40px rgba(79, 70, 229, .28); }
.hero:after { content: ""; position: absolute; right: -80px; top: -80px; width: 280px; height: 280px;
              border-radius: 50%; background: rgba(255,255,255,.08); }
.hero h1 { color: #fff; font-size: 2.05rem; font-weight: 800; margin: 0 0 6px 0; line-height: 1.2; }
.hero p { color: rgba(255,255,255,.88); margin: 0 0 14px 0; font-size: 1.02rem; }
.chip { display: inline-block; padding: 5px 13px; border-radius: 999px; background: rgba(255,255,255,.16);
        border: 1px solid rgba(255,255,255,.25); margin: 4px 8px 0 0; font-size: .8rem; font-weight: 500; }
.kpi { background: #fff; border: 1px solid #E2E8F0; border-radius: 18px; padding: 16px 18px; height: 100%;
       box-shadow: 0 1px 3px rgba(15,23,42,.05); border-top: 4px solid var(--accent); }
.kpi .label { color: #64748B; font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; font-weight: 700; }
.kpi .value { font-size: 1.75rem; font-weight: 800; color: #0F172A; margin-top: 2px; }
.kpi .sub { color: #64748B; font-size: .8rem; }
.sec { margin: 26px 0 10px 0; }
.sec h3 { margin: 0; font-weight: 800; color: #0F172A; font-size: 1.3rem; }
.sec p { margin: 2px 0 0 0; color: #64748B; font-size: .92rem; }
.card { background: #fff; border: 1px solid #E2E8F0; border-radius: 18px; padding: 18px 20px; height: 100%;
        box-shadow: 0 1px 3px rgba(15,23,42,.05); }
.card h4 { margin: 0 0 6px 0; font-weight: 700; color: #0F172A; font-size: 1.05rem; padding: 0; }
.card p, .card li { color: #475569; font-size: .9rem; }
.path { display: flex; gap: 16px; align-items: flex-start; background: #fff; border: 1px solid #E2E8F0;
        border-radius: 16px; padding: 14px 16px; margin-bottom: 10px; transition: transform .15s, box-shadow .15s; }
.path:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(15,23,42,.08); }
.rank { width: 46px; height: 46px; border-radius: 13px; flex-shrink: 0; display: flex; align-items: center;
        justify-content: center; font-weight: 800; font-size: 1.25rem; color: #fff;
        background: linear-gradient(135deg, #4F46E5, #0EA5E9); }
.path .title { font-weight: 700; color: #0F172A; }
.path .meta { color: #64748B; font-size: .82rem; margin-top: 2px; }
.score { margin-left: auto; text-align: right; flex-shrink: 0; }
.score b { font-size: 1.2rem; color: #4F46E5; }
.score span { display: block; color: #94A3B8; font-size: .72rem; }
.tag { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .74rem; font-weight: 600;
       margin: 4px 5px 0 0; background: #EEF2FF; color: #3730A3; }
.tag.gap { background: #FEF3C7; color: #92400E; } .tag.unobs { background: #F1F5F9; color: #475569; }
.tag.ok { background: #D1FAE5; color: #065F46; } .tag.weak { background: #FFE4E6; color: #9F1239; }
.move { font-size: .72rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; margin-left: 6px; }
.move.up { background: #D1FAE5; color: #065F46; } .move.down { background: #FFE4E6; color: #9F1239; }
.move.new { background: #E0E7FF; color: #3730A3; }
.callout { border-radius: 14px; padding: 14px 18px; margin: 8px 0; font-size: .92rem;
           background: #EEF2FF; border-left: 5px solid #4F46E5; color: #1E1B4B; }
.callout.warn { background: #FFFBEB; border-left-color: #F59E0B; color: #78350F; }
.callout.ok { background: #ECFDF5; border-left-color: #10B981; color: #064E3B; }
.flow { display: flex; gap: 10px; align-items: stretch; flex-wrap: wrap; }
.flow .step { flex: 1; min-width: 150px; background: #fff; border: 1px solid #E2E8F0; border-radius: 16px; padding: 14px; }
.flow .step b { display: block; color: #4F46E5; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }
.flow .step div { font-weight: 700; color: #0F172A; margin: 2px 0; }
.flow .step small { color: #64748B; }
.flow .arrow { display: flex; align-items: center; color: #94A3B8; font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)


# ---------- data ----------
@st.cache_resource(show_spinner="Đang chạy pipeline train lần đầu (seed 42)…")
def get_bundle() -> dict:
    processed = core.processed_dir()
    return core.load_bundle(processed, core.ensure_run(processed))


@st.cache_data(show_spinner=False)
def cached_infer(user_id: str, course_id: str | None, problem_id: str | None = None,
                 is_correct: int | None = None) -> dict:
    return core.run_infer(get_bundle(), user_id, course_id, problem_id, is_correct)


B = get_bundle()
NAMES = core.concept_names(B)
TARGET = 0.7


# ---------- ui helpers ----------
def kpi(col, label: str, value: str, sub: str = "", accent: str = INDIGO) -> None:
    col.markdown(f'<div class="kpi" style="--accent:{accent}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="sec"><h3>{title}</h3><p>{subtitle}</p></div>', unsafe_allow_html=True)


def callout(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", ".")


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def style(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                      font=dict(family="Be Vietnam Pro, sans-serif", size=13, color="#334155"),
                      title_font=dict(size=15, color="#0F172A"), showlegend=legend,
                      legend=dict(orientation="h", y=-0.18), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(font_family="Be Vietnam Pro"))
    return fig


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


def status_of(row) -> str:
    if row["insufficient_evidence"] in (1, True, "True"):
        return "Mới tiếp xúc"
    return "Đã vững" if row["mastery"] >= TARGET else "Cần củng cố"


STATUS_COLOR = {"Đã vững": TEAL, "Cần củng cố": AMBER, "Mới tiếp xúc": SLATE}


def mastery_frame(result: dict) -> pd.DataFrame:
    frame = pd.DataFrame(result["mastery"])
    if frame.empty:
        return frame
    frame["concept"] = frame["concept_id"].map(NAMES)
    frame["status"] = frame.apply(status_of, axis=1)
    return frame.sort_values("mastery")


def path_cards(rows: list[dict], gaps: list[dict], moves: dict | None = None) -> None:
    gap_status = {g["concept_id"]: g["evidence_status"] for g in gaps}
    for row in rows:
        tags = "".join(
            f'<span class="tag {"gap" if gap_status.get(c) == "observed_gap" else "unobs"}">{NAMES.get(c, c)}</span>'
            for c in str(row["concept_ids"]).split("|"))
        move = (moves or {}).get(row["problem_id"], "")
        bloom = int(B["problems"].set_index("problem_id").loc[row["problem_id"], "cognitive_dimension"])
        st.markdown(
            f'<div class="path"><div class="rank">{row["rank"]}</div><div>'
            f'<div class="title">{core.problem_label(B, row["problem_id"])}{move}</div>'
            f'<div class="meta">{row["problem_id"]} · Bloom {bloom} – {core.BLOOM_VI[bloom]} · '
            f'{core.REASON_VI.get(row["reason"], row["reason"])}</div><div>{tags}</div></div>'
            f'<div class="score"><b>{float(row["score"]):.3f}</b><span>điểm xếp hạng</span></div></div>',
            unsafe_allow_html=True)


# ---------- sidebar ----------
students = sorted(B["interactions"]["user_id"].unique())
with st.sidebar:
    st.markdown("## 🧭 MOOC-Radar DSS")
    st.caption("Hệ thống gợi ý lộ trình học cá nhân hóa · Modern MIS")
    page = st.radio("Điều hướng", [
        "🏠 Tổng quan hệ thống", "👤 Hồ sơ người học", "🧭 Lộ trình & mô phỏng",
        "⚖️ So sánh hai sinh viên", "📊 Đánh giá mô hình", "🏛️ Kiến trúc hệ thống thông tin"],
        label_visibility="collapsed")
    st.divider()
    user = st.selectbox("Sinh viên đang xem", students, key="user")
    st.divider()
    mode = B["quality"].get("mode", "")
    st.markdown(f"**Nguồn dữ liệu:** {'mẫu (sample)' if mode == 'sample' else 'đầy đủ (full)'}")
    st.caption(f"Run `{B['manifest']['run_id']}` · trạng thái `{B['manifest']['status']}` · "
               f"mô hình `{B['manifest']['selected_decision_model']}`")
    with st.expander("🎤 Kịch bản trình bày 7 phút"):
        st.markdown("""
1. **Tổng quan** – bài toán, dữ liệu, các vấn đề chất lượng.
2. **Hồ sơ người học** – mastery theo concept, độ tin cậy, Bloom.
3. **Lộ trình & mô phỏng** – Top-5 kèm lý do, rồi *nộp thử một câu trả lời* để thấy hồ sơ cập nhật tức thì.
4. **So sánh** – hai sinh viên cùng khóa nhận lộ trình khác nhau.
5. **Đánh giá** – vì sao chọn smoothed concept thay cho BKT.
6. **Kiến trúc IS** – 4 thành phần, quy trình vận hành, giới hạn.
""")


def hero(title: str, subtitle: str, chips: list[str]) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>'
                + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>",
                unsafe_allow_html=True)


# ====================== PAGE 1: OVERVIEW ======================
def page_overview() -> None:
    q, F = B["quality"], core.FULL_DATA
    hero("Hệ thống gợi ý lộ trình học tập cá nhân hóa",
         "Từ nhật ký làm bài MOOC đến hồ sơ năng lực, lỗ hổng kiến thức và Top-5 bài luyện tập có giải thích.",
         ["📚 THU-KEG / MOOC-Radar", "🧠 Smoothed concept + BKT", "🎯 Ngưỡng mastery 0,7",
          f"🗂️ Run {B['manifest']['run_id'][:15]}…"])

    section("Bộ dữ liệu MOOC-Radar đầy đủ", f"Số liệu kiểm toán trên toàn bộ file fine · nguồn: docs/RESULTS.md (run {F['run_id']})")
    cols = st.columns(5)
    kpi(cols[0], "Sinh viên", fmt(F["students"]), "chuỗi tương tác", INDIGO)
    kpi(cols[1], "Lượt trả lời", fmt(F["interactions"]), "train 69,5% · val 14,3% · test 16,2%", SKY)
    kpi(cols[2], "Bài tập", fmt(F["problems"]), f"{fmt(F['multi_concept_problems'])} bài gắn nhiều concept", TEAL)
    kpi(cols[3], "Concept", fmt(F["concepts"]), "tên khái niệm riêng biệt", AMBER)
    kpi(cols[4], "Khóa học", fmt(F["courses"]), "phạm vi gợi ý theo khóa", ROSE)

    left, right = st.columns([1.15, 1])
    with left:
        section("Thách thức dữ liệu chính", "Phát hiện khi quét dữ liệu gốc và cách pipeline xử lý")
        issues = pd.DataFrame([
            ["Thiếu điểm `score`", F["missing_score"] / F["interactions"], "Giữ nguyên; dùng `is_correct` (đủ 100%) làm nhãn"],
            ["Chuỗi bị đảo thứ tự thời gian", F["out_of_order_sequences"] / F["students"], "Sắp xếp ổn định theo submit_time trước khi chia 70/15/15"],
            ["Bài gắn nhiều concept", F["multi_concept_problems"] / F["problems"], "Bảng cầu nối problem_concepts, 1 dòng/lượt trả lời"],
            ["Nhãn 'đúng' trên tập test", F["positive_rate_test"], "Đọc ROC-AUC + Brier + ma trận nhầm lẫn, không chỉ Accuracy"],
        ], columns=["Vấn đề", "Tỷ lệ", "Giải pháp trong pipeline"])
        fig = go.Figure(go.Bar(x=issues["Tỷ lệ"], y=issues["Vấn đề"], orientation="h",
                               marker=dict(color=[ROSE, AMBER, SKY, INDIGO]),
                               text=[pct(v) for v in issues["Tỷ lệ"]], textposition="outside",
                               hovertext=issues["Giải pháp trong pipeline"], hoverinfo="text"))
        fig.update_xaxes(range=[0, 1.05], tickformat=".0%")
        fig.update_yaxes(autorange="reversed")
        plot(style(fig, 280, legend=False))
        st.dataframe(issues.assign(**{"Tỷ lệ": issues["Tỷ lệ"].map(pct)}), hide_index=True, width="stretch")
    with right:
        section("Luồng xử lý dữ liệu → quyết định", "Mô hình DSS: Intelligence → Design → Choice")
        st.markdown("""
<div class="flow" style="flex-direction:column">
 <div class="step"><b>1 · Intelligence</b><div>Kiểm toán & chuẩn hóa</div><small>eda.py · preparation.py — đọc luồng 1 MiB, khử trùng lặp concept, sắp xếp thời gian</small></div>
 <div class="arrow" style="justify-content:center">↓</div>
 <div class="step"><b>2 · Design</b><div>Ước lượng mastery & tìm gap</div><small>modeling.py · recommendation.py — Empirical Bayes 2 tầng, observed_gap vs unobserved</small></div>
 <div class="arrow" style="justify-content:center">↓</div>
 <div class="step"><b>3 · Choice</b><div>Top-5 bài luyện tập có lý do</div><small>learning_paths.csv · inference.py — cập nhật ngay khi có câu trả lời mới</small></div>
</div>""", unsafe_allow_html=True)

    section("Dữ liệu đang chạy trong bản demo",
            "Bộ mẫu cùng cấu trúc MOOC-Radar trong demo/sample_data — đủ nhỏ để chạy pipeline thật trong vài giây")
    cols = st.columns(5)
    kpi(cols[0], "Sinh viên", fmt(q["distinct_students"]), f"độ dài chuỗi {q['student_sequence_length']['min']}–{q['student_sequence_length']['max']}", INDIGO)
    kpi(cols[1], "Lượt trả lời", fmt(q["interactions"]), f"tỷ lệ đúng {pct(q['correct_rate'])}", SKY)
    kpi(cols[2], "Bài tập", fmt(q["problems"]), f"{q['exercises']} exercise", TEAL)
    kpi(cols[3], "Concept", fmt(q["concepts"]), f"{q['problem_concept_links']} liên kết bài–concept", AMBER)
    kpi(cols[4], "Khóa học", fmt(q["courses"]), "mỗi khóa 20 bài", ROSE)

    a, b = st.columns([1.35, 1])
    with a:
        m = B["mastery"].copy()
        m["concept"] = m["concept_id"].map(NAMES)
        grid = m.pivot_table(index="user_id", columns="concept", values="mastery")
        grid = grid[grid.mean().sort_values().index]
        fig = go.Figure(go.Heatmap(
            z=grid.values, x=grid.columns, y=grid.index, zmin=0, zmax=1, zmid=TARGET,
            colorscale=[[0, "#E11D48"], [0.45, "#FDE68A"], [0.7, "#A7F3D0"], [1, "#047857"]],
            hovertemplate="%{y}<br>%{x}<br>mastery = %{z:.2f}<extra></extra>",
            colorbar=dict(title="mastery", thickness=12)))
        fig.update_layout(title="Bản đồ năng lực cả lớp (ô trống = chưa có bằng chứng)")
        fig.update_xaxes(tickangle=-35)
        plot(style(fig, 520, legend=False))
    with b:
        inter = B["interactions"].merge(B["links"], on="problem_id")
        acc = inter.groupby("concept_id")["is_correct"].agg(["mean", "count"]).reset_index()
        acc["concept"] = acc["concept_id"].map(NAMES)
        acc = acc.sort_values("mean")
        fig = go.Figure(go.Bar(x=acc["mean"], y=acc["concept"], orientation="h",
                               marker=dict(color=acc["mean"], colorscale="RdYlGn", cmin=0.3, cmax=0.8),
                               text=[f"{v:.0%} · n={n}" for v, n in zip(acc["mean"], acc["count"])],
                               textposition="outside"))
        fig.add_vline(x=TARGET, line_dash="dash", line_color=INDIGO, annotation_text="mục tiêu 0,7")
        fig.update_layout(title="Concept cả lớp đang làm sai nhiều nhất")
        fig.update_xaxes(range=[0, 1.1], tickformat=".0%")
        plot(style(fig, 520, legend=False))
    callout("👩‍🏫 <b>Góc nhìn giảng viên:</b> bản đồ năng lực và biểu đồ tỷ lệ đúng giúp phát hiện concept "
            "cả lớp còn yếu để điều chỉnh bài giảng, trước khi đi sâu vào từng sinh viên.")


# ====================== PAGE 2: LEARNER PROFILE ======================
def page_profile() -> None:
    res = cached_infer(user, None)
    mf = mastery_frame(res)
    hist = B["interactions"].query("user_id == @user").sort_values("sequence_position")
    hero(f"Hồ sơ năng lực · {user}",
         "Mức thành thạo theo concept đi kèm số lần quan sát, độ tin cậy n/(n+5) và cờ thiếu bằng chứng.",
         [f"📍 Khóa đích {res['course_id']}", f"🧾 {len(hist)} lượt làm bài",
          "🔒 Chỉ dùng mã ẩn danh"])
    cols = st.columns(5)
    kpi(cols[0], "Lượt làm bài", fmt(len(hist)), f"{hist['problem_id'].nunique()} bài khác nhau", INDIGO)
    kpi(cols[1], "Tỷ lệ đúng", pct(hist["is_correct"].mean()), "trên toàn bộ lịch sử", SKY)
    kpi(cols[2], "Concept đã gặp", fmt(len(mf)), f"trên {len(NAMES)} concept", TEAL)
    kpi(cols[3], "Đã vững (≥0,7)", fmt((mf["status"] == "Đã vững").sum()), "đủ bằng chứng", TEAL)
    kpi(cols[4], "Cần củng cố", fmt((mf["status"] == "Cần củng cố").sum()),
        f"+ {(mf['status'] == 'Mới tiếp xúc').sum()} concept mới tiếp xúc", AMBER)

    a, b = st.columns([1, 1.2])
    with a:
        radar = mf.sort_values("concept")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[TARGET] * len(radar) + [TARGET], theta=list(radar["concept"]) + [radar["concept"].iloc[0]],
                                      name="Mục tiêu 0,7", line=dict(color=ROSE, dash="dash"), hoverinfo="skip"))
        fig.add_trace(go.Scatterpolar(r=list(radar["mastery"]) + [radar["mastery"].iloc[0]],
                                      theta=list(radar["concept"]) + [radar["concept"].iloc[0]], fill="toself",
                                      name="Mastery hiện tại", line=dict(color=INDIGO, width=2),
                                      fillcolor="rgba(79,70,229,.25)"))
        fig.update_layout(title="Mạng năng lực theo concept", polar=RADAR_AXES)
        plot(style(fig, 440).update_layout(margin=dict(l=70, r=70, t=50, b=10)))
    with b:
        fig = go.Figure()
        for status, color in STATUS_COLOR.items():
            part = mf[mf["status"] == status]
            fig.add_trace(go.Bar(x=part["mastery"], y=part["concept"], orientation="h", name=status,
                                 marker_color=color, text=[f"n={n}" for n in part["observations"]],
                                 textposition="inside", customdata=part[["observations", "confidence"]],
                                 hovertemplate="%{y}<br>mastery %{x:.2f}<br>n=%{customdata[0]} · "
                                               "tin cậy %{customdata[1]:.2f}<extra></extra>"))
        fig.add_vline(x=TARGET, line_dash="dash", line_color=ROSE)
        fig.update_layout(title="Mastery theo concept (xám = mới làm < 3 lần)", barmode="stack")
        fig.update_yaxes(categoryorder="array", categoryarray=list(mf["concept"]))
        fig.update_xaxes(range=[0, 1])
        plot(style(fig, 440))

    a, b = st.columns([1, 1.4])
    with a:
        bloom = pd.DataFrame(res["bloom_profile"])
        bloom["level"] = bloom["cognitive_dimension"].map(lambda v: f"{v}. {core.BLOOM_VI[int(v)]}")
        fig = go.Figure(go.Bar(x=bloom["level"], y=bloom["accuracy"],
                               marker=dict(color=bloom["accuracy"], colorscale="Tealgrn", cmin=0, cmax=1),
                               text=[f"{a:.0%}<br>n={n}" for a, n in zip(bloom["accuracy"], bloom["observations"])],
                               textposition="outside"))
        fig.update_layout(title="Hồ sơ nhận thức theo thang Bloom")
        fig.update_yaxes(range=[0, 1.2], tickformat=".0%")
        plot(style(fig, 380, legend=False))
    with b:
        h = hist.reset_index(drop=True)
        h["attempt"] = np.arange(1, len(h) + 1)
        h["cum_acc"] = h["is_correct"].expanding().mean()
        fig = go.Figure()
        for split, color, pos in (("train", "#EEF2FF", "top left"), ("validation", "#FEF3C7", "bottom left"),
                                  ("test", "#FFE4E6", "top right")):
            part = h[h["split"] == split]
            if not part.empty:
                fig.add_vrect(x0=part["attempt"].min() - 0.5, x1=part["attempt"].max() + 0.5, fillcolor=color,
                              line_width=0, layer="below", annotation_text=SPLIT_VI[split],
                              annotation_position=pos, annotation_font_size=11)
        fig.add_trace(go.Scatter(x=h["attempt"], y=h["cum_acc"], mode="lines", name="Tỷ lệ đúng lũy kế",
                                 line=dict(color=INDIGO, width=3, shape="spline")))
        fig.add_trace(go.Scatter(x=h["attempt"], y=h["is_correct"], mode="markers", name="Kết quả từng lượt",
                                 marker=dict(size=9, color=[TEAL if v else ROSE for v in h["is_correct"]],
                                             line=dict(color="white", width=1)),
                                 customdata=h[["problem_id", "submit_time"]].astype(str),
                                 hovertemplate="Lượt %{x}: %{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
        fig.update_layout(title="Dòng thời gian làm bài và cách chia 70/15/15 theo thời gian")
        fig.update_yaxes(range=[-0.1, 1.15], tickformat=".0%")
        plot(style(fig, 380))

    section("Bảng student_mastery (theo OUTPUT_CONTRACT.md)", "Dữ liệu giảng viên tra cứu khi tư vấn học tập")
    table = mf[["concept", "mastery", "observations", "confidence", "insufficient_evidence", "status"]].sort_values("mastery")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "concept": "Concept",
        "mastery": st.column_config.ProgressColumn("Mastery", min_value=0, max_value=1, format="%.2f"),
        "observations": st.column_config.NumberColumn("Số lần quan sát (n)"),
        "confidence": st.column_config.ProgressColumn("Độ tin cậy n/(n+5)", min_value=0, max_value=1, format="%.2f"),
        "insufficient_evidence": st.column_config.CheckboxColumn("Thiếu bằng chứng (n<3)"),
        "status": "Trạng thái"})
    callout("Concept chưa từng làm <b>không bị gán mastery = 0</b>: chúng không xuất hiện ở bảng này mà được đánh dấu "
            "<code>unobserved</code> trong bảng lỗ hổng kiến thức — tránh đánh giá oan sinh viên.", "ok")


# ====================== PAGE 3: PATH & SIMULATION ======================
def page_path() -> None:
    base = cached_infer(user, None)
    courses = sorted(B["problems"]["course_id"].unique())
    hero(f"Lộ trình luyện tập · {user}",
         "Top-5 bài tập được xếp hạng từ lỗ hổng kiến thức; mỗi điểm số tách được thành từng thành phần để giải thích.",
         ["🎯 observed_gap trước unobserved", "🧩 ưu tiên bài phủ nhiều concept yếu",
          "🪜 khớp bậc Bloom (ZPD)", "✨ ưu tiên bài chưa làm"])
    course = st.selectbox("Khóa học đích", courses, index=courses.index(base["course_id"]),
                          help="Mặc định là khóa của lượt làm bài gần nhất")
    res = cached_infer(user, course)
    seen = set(B["interactions"].query("user_id == @user")["problem_id"])

    a, b = st.columns([1, 1.3])
    with a:
        section("Lỗ hổng kiến thức", "knowledge_gaps.csv · tối đa 20 concept/khóa")
        gaps = pd.DataFrame(res["knowledge_gaps"])
        if gaps.empty:
            st.info("Không còn concept nào dưới ngưỡng trong khóa này.")
        else:
            gaps["concept"] = gaps["concept_id"].map(NAMES)
            gaps["Trạng thái"] = gaps["evidence_status"].map({"observed_gap": "Đã có bằng chứng yếu",
                                                              "unobserved": "Chưa quan sát"})
            obs = gaps[gaps["evidence_status"] == "observed_gap"]
            fig = go.Figure(go.Bar(x=obs["priority"], y=obs["concept"], orientation="h",
                                   marker=dict(color=obs["priority"], colorscale=[[0, "#FDE68A"], [1, ROSE]]),
                                   text=[f"m={float(m):.2f} · n={n}" for m, n in zip(obs["mastery"], obs["observations"])],
                                   textposition="inside"))
            fig.update_layout(title="Mức ưu tiên = (0,7 − mastery) × (1 + 0,12·min(n,5))")
            fig.update_yaxes(autorange="reversed")
            plot(style(fig, 70 + 42 * max(len(obs), 1), legend=False))
            unobs = gaps[gaps["evidence_status"] == "unobserved"]["concept"].tolist()
            if unobs:
                st.markdown("**Chưa có bằng chứng (mastery để trống, ưu tiên sau):** " + "".join(
                    f'<span class="tag unobs">{c}</span>' for c in unobs), unsafe_allow_html=True)
    with b:
        section("Top-5 bài nên làm tiếp", "learning_paths.csv · nhãn vàng = concept có bằng chứng yếu")
        path_cards(res["learning_path"], res["knowledge_gaps"])
        if res["learning_path"]:
            with st.expander("🔍 Vì sao bài #1 đứng đầu? (tách điểm xếp hạng)"):
                parts = core.explain_score(B, res["learning_path"][0], res["knowledge_gaps"], seen)
                names = ["Mức cấp thiết của gap", "Độ phủ nhiều concept", "Khớp bậc Bloom (ZPD)", "Bài chưa làm"]
                fig = go.Figure()
                for name, color in zip(names, [ROSE, SKY, TEAL, INDIGO]):
                    fig.add_trace(go.Bar(x=[parts[name]], y=["Điểm"], orientation="h", name=name,
                                         marker_color=color, text=f"{parts[name]:.3f}", textposition="inside"))
                fig.update_layout(barmode="stack", title=f"Tổng = {parts['total']:.3f}")
                plot(style(fig, 200))
                st.caption(f"Bậc Bloom mục tiêu = 1 + 4 × mastery TB = {parts['target_bloom']:.2f}; bài này ở bậc "
                           f"{parts['bloom']}. Tách điểm khớp với pipeline: {'✅' if parts['consistent'] else '⚠️'}")

    section("⚡ Mô phỏng: sinh viên vừa nộp một câu trả lời",
            "Gọi inference.infer() với lượt làm mới — không huấn luyện lại, hồ sơ và lộ trình cập nhật ngay")
    course_probs = B["problems"].query("course_id == @course")["problem_id"].tolist()
    default = res["learning_path"][0]["problem_id"] if res["learning_path"] else course_probs[0]
    c1, c2, c3 = st.columns([2.2, 1, 1])
    pid = c1.selectbox("Bài vừa làm", course_probs, index=course_probs.index(default),
                       format_func=lambda p: f"{p} — {core.problem_label(B, p)}")
    outcome = c2.radio("Kết quả", ["✅ Đúng", "❌ Sai"], horizontal=True)
    c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    go_btn = c3.button("🚀 Nộp bài & cập nhật", type="primary", width="stretch")
    if go_btn:
        correct = 1 if outcome.startswith("✅") else 0
        start = time.perf_counter()
        after = core.run_infer(B, user, course, pid, correct)
        elapsed = (time.perf_counter() - start) * 1000
        st.session_state["sim"] = {"user": user, "course": course, "pid": pid, "correct": correct,
                                   "after": after, "ms": elapsed}
    sim = st.session_state.get("sim")
    if sim and sim["user"] == user and sim["course"] == course:
        before_m = {m["concept_id"]: m["mastery"] for m in res["mastery"]}
        after_m = {m["concept_id"]: m["mastery"] for m in sim["after"]["mastery"]}
        touched = B["links"].query("problem_id == @sim['pid']")["concept_id"].tolist()
        callout(f"Đã ghi nhận <b>{sim['pid']}</b> = {'ĐÚNG' if sim['correct'] else 'SAI'} · "
                f"cập nhật {len(touched)} concept liên quan và xếp lại lộ trình trong <b>{sim['ms']:.0f} ms</b> "
                "(phát lại lịch sử + cộng bộ đếm, không train lại).", "ok" if sim["correct"] else "warn")
        a, b = st.columns([1, 1.3])
        with a:
            delta = pd.DataFrame([{"concept": NAMES[c], "before": before_m.get(c), "after": after_m.get(c)}
                                  for c in touched])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=delta["concept"], y=delta["before"].fillna(0), name="Trước", marker_color=SLATE))
            fig.add_trace(go.Bar(x=delta["concept"], y=delta["after"], name="Sau",
                                 marker_color=TEAL if sim["correct"] else ROSE,
                                 text=["mới" if pd.isna(b) else f"{a - b:+.3f}"
                                       for a, b in zip(delta["after"], delta["before"])],
                                 textposition="outside"))
            fig.add_hline(y=TARGET, line_dash="dash", line_color=INDIGO)
            fig.update_layout(title="Mastery các concept của bài vừa làm", barmode="group")
            fig.update_yaxes(range=[0, 1.1])
            plot(style(fig, 380))
        with b:
            old_rank = {r["problem_id"]: r["rank"] for r in res["learning_path"]}
            moves = {}
            for r in sim["after"]["learning_path"]:
                prev = old_rank.get(r["problem_id"])
                if prev is None:
                    moves[r["problem_id"]] = '<span class="move new">MỚI</span>'
                elif prev > r["rank"]:
                    moves[r["problem_id"]] = f'<span class="move up">▲ {prev - r["rank"]}</span>'
                elif prev < r["rank"]:
                    moves[r["problem_id"]] = f'<span class="move down">▼ {r["rank"] - prev}</span>'
            st.markdown("**Lộ trình mới sau khi cập nhật**")
            path_cards(sim["after"]["learning_path"], sim["after"]["knowledge_gaps"], moves)


# ====================== PAGE 4: COMPARE ======================
def page_compare() -> None:
    hero("Cùng khóa học — lộ trình khác nhau",
         "Bằng chứng cá nhân hóa: hai sinh viên cùng khóa đích có hồ sơ khác nhau nên nhận Top-5 khác nhau.",
         ["⚖️ So sánh song song", "🧬 Hồ sơ riêng từng người"])
    others = [s for s in students if s != user]
    c1, c2, c3 = st.columns(3)
    s1 = c1.selectbox("Sinh viên A", students, index=students.index(user))
    s2 = c2.selectbox("Sinh viên B", others, index=0)
    course = c3.selectbox("Khóa học đích chung", sorted(B["problems"]["course_id"].unique()),
                          index=sorted(B["problems"]["course_id"].unique()).index(cached_infer(s1, None)["course_id"]))
    r1, r2 = cached_infer(s1, course), cached_infer(s2, course)
    m1, m2 = mastery_frame(r1), mastery_frame(r2)
    concepts = sorted(set(m1["concept"]) | set(m2["concept"]))
    a, b = st.columns([1, 1])
    with a:
        fig = go.Figure()
        for frame, name, color, fill in ((m1, s1, INDIGO, "rgba(79,70,229,.22)"), (m2, s2, AMBER, "rgba(245,158,11,.22)")):
            vals = frame.set_index("concept")["mastery"].reindex(concepts).fillna(0).tolist()
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=concepts + concepts[:1], fill="toself",
                                          name=name, line=dict(color=color, width=2), fillcolor=fill))
        fig.update_layout(title="Mạng năng lực chồng lớp (0 = chưa gặp)", polar=RADAR_AXES)
        plot(style(fig, 460).update_layout(margin=dict(l=80, r=80, t=50, b=10)))
    with b:
        p1 = [r["problem_id"] for r in r1["learning_path"]]
        p2 = [r["problem_id"] for r in r2["learning_path"]]
        overlap = len(set(p1) & set(p2))
        cols = st.columns(3)
        kpi(cols[0], "Bài trùng trong Top-5", f"{overlap}/{max(len(p1), len(p2), 1)}", "càng ít càng cá nhân hóa", INDIGO)
        kpi(cols[1], f"Gap của {s1}", fmt(sum(g['evidence_status'] == 'observed_gap' for g in r1['knowledge_gaps'])), "có bằng chứng", AMBER)
        kpi(cols[2], f"Gap của {s2}", fmt(sum(g['evidence_status'] == 'observed_gap' for g in r2['knowledge_gaps'])), "có bằng chứng", AMBER)
        lists = [tuple(r["problem_id"] for r in cached_infer(s, course)["learning_path"]) for s in students]
        callout(f"Trên khóa <b>{course}</b>, {len(students)} sinh viên nhận <b>{len(set(lists))}</b> lộ trình Top-5 "
                "khác nhau — lộ trình phụ thuộc hồ sơ từng người chứ không phải một danh sách chung cho cả lớp.")
    a, b = st.columns(2)
    with a:
        st.markdown(f"#### 🟣 {s1}")
        path_cards(r1["learning_path"], r1["knowledge_gaps"])
    with b:
        st.markdown(f"#### 🟠 {s2}")
        path_cards(r2["learning_path"], r2["knowledge_gaps"])


# ====================== PAGE 5: EVALUATION ======================
def confusion_fig(tp: int, fp: int, tn: int, fn: int, title: str) -> go.Figure:
    z = [[tn, fp], [fn, tp]]
    text = [[f"TN<br>{fmt(tn)}", f"FP<br>{fmt(fp)}"], [f"FN<br>{fmt(fn)}", f"TP<br>{fmt(tp)}"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Dự đoán: sai", "Dự đoán: đúng"], y=["Thực tế: sai", "Thực tế: đúng"],
                               text=text, texttemplate="%{text}", colorscale="Purples", showscale=False))
    fig.update_layout(title=title)
    return style(fig, 300, legend=False)


def page_evaluation() -> None:
    F = core.FULL_DATA
    hero("Đánh giá & lựa chọn mô hình",
         "Dữ liệu lệch 80% nhãn 'đúng' nên không thể chỉ nhìn Accuracy — cần ROC-AUC, Brier và ma trận nhầm lẫn.",
         ["📈 ROC-AUC: khả năng xếp hạng", "🎯 Brier: độ chuẩn xác suất", "🧮 TP/TN/FP/FN"])
    tab_full, tab_demo = st.tabs(["🌐 Toàn bộ MOOC-Radar (docs/RESULTS.md)", "🧪 Run demo trên dữ liệu mẫu"])
    with tab_full:
        met = pd.DataFrame(F["metrics"])
        met["Mô hình"] = met["model"].map(MODEL_VI)
        met["Tập"] = met["split"].map(SPLIT_VI)
        cols = st.columns(4)
        sm = met.query("model == 'smoothed_concept' and split == 'test'").iloc[0]
        bk = met.query("model == 'bkt' and split == 'test'").iloc[0]
        kpi(cols[0], "ROC-AUC test · smoothed", f"{sm.roc_auc:.4f}", f"+{(sm.roc_auc - bk.roc_auc) * 100:.2f} điểm so với BKT", INDIGO)
        kpi(cols[1], "Brier test · smoothed", f"{sm.brier:.4f}", f"BKT: {bk.brier:.4f} (thấp hơn là tốt)", TEAL)
        kpi(cols[2], "Accuracy test · BKT", pct(bk.accuracy, 2), f"= tỷ lệ lớp 'đúng' {pct(F['positive_rate_test'])}", AMBER)
        kpi(cols[3], "Câu sai BKT phát hiện", "0", f"trên {fmt(F['bkt_test_confusion']['fp'])} câu sai", ROSE)
        a, b = st.columns([1.3, 1])
        with a:
            long = met.melt(id_vars=["Mô hình", "Tập"], value_vars=["roc_auc", "brier", "accuracy", "f1"],
                            var_name="metric", value_name="value")
            long["metric"] = long["metric"].map({"roc_auc": "ROC-AUC ↑", "brier": "Brier ↓", "accuracy": "Accuracy ↑", "f1": "F1 ↑"})
            fig = px.bar(long, x="metric", y="value", color="Mô hình", barmode="group", facet_col="Tập",
                         color_discrete_map={MODEL_VI[k]: v for k, v in MODEL_COLOR.items()}, text_auto=".3f")
            fig.for_each_annotation(lambda ann: ann.update(text=ann.text.split("=")[-1]))
            fig.update_traces(textangle=0, textfont_size=10, textposition="outside", cliponaxis=False)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None, range=[0, 1.05])
            fig.update_layout(title=f"Kết quả run {F['run_id']}")
            plot(style(fig, 400))
        with b:
            c = F["bkt_test_confusion"]
            plot(confusion_fig(c["tp"], c["fp"], c["tn"], c["fn"], "BKT trên test (145.307 lượt, ngưỡng 0,5)"))
        callout("<b>Quyết định:</b> chọn <code>smoothed_concept</code> làm mô hình ra quyết định — ROC-AUC cao hơn, "
                "Brier thấp hơn, không suy biến về một lớp như BKT, giải thích được bằng số lần đúng/tổng và cập nhật "
                "tức thì. BKT được giữ làm đối chứng trong mọi lần train.")
        st.dataframe(met[["Mô hình", "Tập", "roc_auc", "accuracy", "f1", "brier"]], hide_index=True, width="stretch")
    with tab_demo:
        callout(f"Run <code>{B['manifest']['run_id']}</code> trên bộ mẫu {B['quality']['distinct_students']} sinh viên "
                f"(test = {B['manifest']['counts']['test']} lượt). Mẫu nhỏ nên số liệu dao động mạnh — dùng để minh họa "
                "quy trình đánh giá, <b>không</b> dùng làm kết luận.", "warn")
        met = B["metrics"].copy()
        met["Mô hình"] = met["model"].map(MODEL_VI)
        met["Tập"] = met["split"].map(SPLIT_VI)
        st.dataframe(met[["Mô hình", "Tập", "n", "roc_auc", "accuracy", "f1", "brier", "tp", "tn", "fp", "fn"]],
                     hide_index=True, width="stretch")
        pred = B["predictions"].query("split == 'test'")
        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=SLATE), name="Ngẫu nhiên"))
            for model, color in MODEL_COLOR.items():
                part = pred[pred["model"] == model]
                fpr, tpr, _ = roc_curve(part["is_correct"], part["probability_correct"])
                auc = met.query("model == @model and split == 'test'")["roc_auc"].iloc[0]
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{MODEL_VI[model]} · AUC {auc:.3f}",
                                         line=dict(color=color, width=3)))
            fig.update_layout(title="Đường ROC trên tập kiểm tra", xaxis_title="FPR", yaxis_title="TPR")
            plot(style(fig, 400))
        with b:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=SLATE), name="Hoàn hảo"))
            for model, color in MODEL_COLOR.items():
                part = pred[pred["model"] == model].copy()
                part["bin"] = pd.cut(part["probability_correct"], np.linspace(0, 1, 6), include_lowest=True)
                cal = part.groupby("bin", observed=True).agg(p=("probability_correct", "mean"),
                                                             y=("is_correct", "mean"), n=("is_correct", "size"))
                fig.add_trace(go.Scatter(x=cal["p"], y=cal["y"], mode="lines+markers", name=MODEL_VI[model],
                                         marker=dict(size=8 + cal["n"] / cal["n"].max() * 12, color=color),
                                         line=dict(color=color, width=2)))
            fig.update_layout(title="Biểu đồ hiệu chuẩn xác suất (Brier)", xaxis_title="Xác suất dự đoán",
                              yaxis_title="Tỷ lệ đúng thực tế")
            plot(style(fig, 400))
        a, b = st.columns(2)
        for col, model in zip((a, b), MODEL_COLOR):
            row = met.query("model == @model and split == 'test'").iloc[0]
            with col:
                plot(confusion_fig(int(row.tp), int(row.fp), int(row.tn), int(row.fn), f"{MODEL_VI[model]} · test"))


# ====================== PAGE 6: IS ARCHITECTURE ======================
def page_architecture() -> None:
    hero("Kiến trúc hệ thống thông tin", "Lõi dự đoán được bao quanh bởi 4 thành phần: phần cứng, phần mềm, dữ liệu, con người.",
         ["🖥️ CPU-only", "🐍 Gói Python moocradar", "🗃️ raw → processed → artifacts", "👥 RBAC 4 vai trò"])
    cards = [
        ("🖥️ Phần cứng", "Chạy thử trên Google Colab CPU; đọc luồng JSON theo khối 1 MiB nên RAM gần như không đổi theo số sinh viên. Không cần GPU cho mô hình đã chọn."),
        ("🧩 Phần mềm", "CLI <code>python -m moocradar</code>: eda → prepare → train → demo → translation. Demo này gọi trực tiếp <code>train()</code> và <code>infer()</code>."),
        ("🗃️ Dữ liệu", "data/raw (chỉ đọc) → data/processed (4 bảng chuẩn hóa) → artifacts/&lt;run_id&gt; bất biến, có manifest.json trạng thái complete."),
        ("👥 Con người", "Sinh viên xem lộ trình; giảng viên đọc lý do & độ tin cậy; chuyên viên dịch metadata; kỹ sư vận hành và chuyển run."),
    ]
    cols = st.columns(4)
    for col, (title, body) in zip(cols, cards):
        col.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    section("Quy trình vận hành", "Kết hợp cập nhật tức thì trong ngày và huấn luyện lại định kỳ")
    st.markdown("""
<div class="flow">
 <div class="step"><b>LMS</b><div>Sinh viên nộp bài</div><small>POST sự kiện log mới</small></div><div class="arrow">→</div>
 <div class="step"><b>inference.py</b><div>Cập nhật tức thì</div><small>Phát lại lịch sử + 1 lượt mới, không train lại</small></div><div class="arrow">→</div>
 <div class="step"><b>Dashboard</b><div>Top-5 mới</div><small>Kèm lý do, độ tin cậy</small></div><div class="arrow">→</div>
 <div class="step"><b>Hàng tuần</b><div>eda → prepare → train</div><small>Run mới, đối chiếu metrics rồi mới chuyển con trỏ; rollback được</small></div>
</div>""", unsafe_allow_html=True)

    a, b = st.columns([1.3, 1])
    with a:
        section("Yêu cầu hệ thống & mức hiện thực", "FR/NFR xác lập ở Chương 3")
        req = pd.DataFrame([
            ["FR-01", "Kiểm toán chất lượng dữ liệu & EDA", "eda.py, eda_plots.py", "✅"],
            ["FR-02", "Chuẩn hóa quan hệ & sắp xếp thời gian", "preparation.py, reading.py", "✅"],
            ["FR-03", "Huấn luyện & đánh giá theo thời gian", "modeling.py, evaluation.py", "✅"],
            ["FR-04", "Hồ sơ mastery + Bloom", "pipeline.py", "✅"],
            ["FR-05", "Gap có bằng chứng & Top-N", "recommendation.py", "✅"],
            ["FR-06", "Suy luận tức thì & dịch ngoại tuyến", "inference.py, translation.py", "✅"],
            ["NFR-01", "Bộ nhớ giới hạn, chạy CPU", "reading.py", "✅"],
            ["NFR-02", "Tái lập, bất biến, chống rò rỉ nhãn", "tests/", "✅"],
            ["NFR-03", "Minh bạch sư phạm & quyền riêng tư", "toàn bộ kiến trúc", "✅"],
        ], columns=["Mã", "Yêu cầu", "Module", "Trạng thái"])
        st.dataframe(req, hide_index=True, width="stretch")
    with b:
        section("Hợp đồng đầu ra cho Dashboard", "Số dòng trong run demo và run toàn bộ")
        F = core.FULL_DATA["outputs"]
        out = pd.DataFrame([
            ["student_mastery.csv", len(B["mastery"]), F["student_mastery"]],
            ["cognitive_profile.csv", len(B["bloom"]), F["cognitive_profile"]],
            ["knowledge_gaps.csv", len(B["gaps"]), F["knowledge_gaps"]],
            ["learning_paths.csv", len(B["paths"]), F["learning_paths"]],
        ], columns=["Tệp", "Run demo", "Run toàn bộ"])
        fig = go.Figure(go.Bar(x=out["Run toàn bộ"], y=out["Tệp"], orientation="h",
                               marker_color=[INDIGO, SKY, AMBER, TEAL],
                               text=[fmt(v) for v in out["Run toàn bộ"]], textposition="outside", cliponaxis=False))
        fig.update_xaxes(range=[0, out["Run toàn bộ"].max() * 1.3], showticklabels=False, showgrid=False)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Số dòng đầu ra trên dữ liệu đầy đủ")
        plot(style(fig, 280, legend=False))
        st.dataframe(out.assign(**{c: out[c].map(fmt) for c in ("Run demo", "Run toàn bộ")}),
                     hide_index=True, width="stretch")

    section("Giới hạn hiện tại & hướng phát triển", "Nêu rõ để tránh diễn giải quá mức")
    cols = st.columns(3)
    cols[0].markdown('<div class="card"><h4>⏳ Chưa triển khai</h4><ul><li>Lịch học theo deadline (vd. thi sau 3 tháng)</li>'
                     '<li>Mục tiêu fast_track / exam_preparation</li><li>Đồ thị tiên quyết & liên kết video (dữ liệu gốc không có)</li></ul></div>',
                     unsafe_allow_html=True)
    cols[1].markdown('<div class="card"><h4>🧪 Chưa kiểm chứng</h4><ul><li>Lộ trình có làm tăng kết quả học tập — cần A/B test với người học</li>'
                     '<li>Ngưỡng 0,7 và trọng số xếp hạng là heuristic</li></ul></div>', unsafe_allow_html=True)
    cols[2].markdown('<div class="card"><h4>🚀 Giai đoạn tiếp theo</h4><ul><li>REST API bọc infer() cho LMS</li>'
                     '<li>PostgreSQL cho 8 bảng</li><li>Thử nghiệm NCDM (ncdm.py) qua A/B test</li></ul></div>',
                     unsafe_allow_html=True)


{"🏠 Tổng quan hệ thống": page_overview, "👤 Hồ sơ người học": page_profile,
 "🧭 Lộ trình & mô phỏng": page_path, "⚖️ So sánh hai sinh viên": page_compare,
 "📊 Đánh giá mô hình": page_evaluation, "🏛️ Kiến trúc hệ thống thông tin": page_architecture}[page]()
