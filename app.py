import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업의 기준 기간 끝
MIN_DAYS = 300        # 관측일이 이보다 적은 해는 제외

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")
st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온으로 회귀 직선을 만들고, 선택한 연도의 기온을 예측합니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])      # 관측값이 있는 날만 사용
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]                 # 2025년 이후 제외

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS]  # 300일 미만인 해 제외
    return yearly.reset_index(drop=True)


try:
    yearly = load_yearly(DATA_URL)
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 만들 수 있는 연도가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# ---- 직선 정보 ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{r:.3f}")
st.write(
    f"회귀 직선: **기온 = {slope:.4f} × 연도 + {intercept:.2f}** "
    f"(10년에 약 {slope * 10:+.2f}℃)"
)

# ---- 연도 선택 & 예측 ----
year = st.slider("예측할 연도", min_value=1900, max_value=2100, value=2030, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:1.2rem; opacity:0.7;">{year}년 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.1;">{pred:.1f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if year < start_year or year > end_year:
    st.caption(
        f"※ {year}년은 직선을 만든 기간({start_year}~{end_year}년) 밖이라, 직선을 연장해서 구한 값입니다."
    )

# ---- 그래프 ----
line_x = np.array([1900, 2100])
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8),
        customdata=yearly["관측일수"],
        hovertemplate="%{x}년<br>%{y:.2f}℃<br>관측일수 %{customdata}일<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=slope * line_x + intercept,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, dash="solid"),
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예측",
        marker=dict(size=16, symbol="star", line=dict(width=1, color="black")),
        hovertemplate="%{x}년 예측<br>%{y:.2f}℃<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온 (r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(range=[1900, 2100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=520,
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("연도별 평균기온 표 보기"):
    st.dataframe(yearly, use_container_width=True, hide_index=True)
