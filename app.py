import pandas as pd
import numpy as np
import streamlit as st
import requests

# ▶️ Use the URL you provided
SAMPLE_FILE_URL = "https://huggingface.co/spaces/Em4e/seo-b2b-saas-forecasting-tool/resolve/main/sample_gsc_data.csv"

st.set_page_config(page_title="SEO ROI Forecasting Tool for B2B SaaS", layout="wide")
st.title("📈 SEO ROI Forecasting Tool for B2B SaaS")

st.markdown("""
This app helps you estimate the **financial upside** of ranking improvements for your SEO keywords,
and compare that to what it would cost you in paid ads.
<br>

👉 **Make sure your CSV has a `CPC` column** (cost per click in $).
If you don’t, we’ll simulate one for you.
<br>

Developed by: [Emilija Gjorgjevska](https://www.linkedin.com/in/emilijagjorgjevska/)
""", unsafe_allow_html=True)

# ———————
# Download button for the sample file
sample_bytes = requests.get(SAMPLE_FILE_URL).content
st.download_button(
    label="📥 Download sample CSV",
    data=sample_bytes,
    file_name="sample_gsc_data.csv",
    mime="text/csv",
)
# ———————

# === Sidebar inputs ===
with st.sidebar:
    st.header("🔧 Assumptions & Inputs")
    uploaded_file = st.file_uploader("Upload Google Search Console CSV", type="csv")
    target_position = st.slider("Target SERP Position", 1.0, 10.0, 4.0, 0.5)
    conversion_rate = st.slider("Conversion Rate (% → signup)", 0.1, 10.0, 2.0, 0.1)
    close_rate = st.slider("Close Rate (% → customer)", 1.0, 100.0, 20.0, 1.0)
    mrr_per_customer = st.slider("MRR per Customer ($)", 10, 1000, 200, 10)
    seo_cost = st.slider("Total SEO Investment ($)", 1_000, 100_000, 10_000, 1_000)

# === Load & prep data ===
def load_csv():
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(SAMPLE_FILE_URL)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # If CPC is missing, simulate it
    if 'CPC' not in df.columns:
        st.warning("No `CPC` column found—simulating CPC values between $0.50–$3.00.")
        df['CPC'] = np.round(np.random.uniform(0.5, 3.0, size=len(df)), 2)

    return df

# === ROI + Savings calculator ===
def calculate_roi(df):
    # map and rename
    cols = {c.lower(): c for c in df.columns}
    need = {
        'query': ['query','keyword','keywords','queries'],
        'impressions': ['impressions'],
        'position': ['position','avg. position','average position'],
        'cpc': ['cpc']
    }
    found = {}
    for k, opts in need.items():
        for o in opts:
            if o in cols:
                found[k] = cols[o]
                break
        if k not in found:
            st.error(f"Missing required column: {k}")
            return None, pd.DataFrame()
    df.rename(columns={found[k]: k for k in found}, inplace=True)

    # CTR benchmarks
    ctr = {i:v for i,v in zip(range(1,11), [0.25,0.15,0.10,0.08,0.06,0.04,0.03,0.02,0.015,0.01])}
    ctr.update({i: 0.005 for i in range(11,21)})
    get_ctr = lambda p: ctr.get(int(round(p)), 0.005)

    # filter for positions 5–20
    df = df[df.position.between(5,20)].copy()
    if df.empty:
        st.warning("No keywords in positions 5–20.")
        return None, pd.DataFrame()

    # compute clicks
    df['Current_CTR'] = df.position.apply(get_ctr)
    df['Target_CTR'] = get_ctr(target_position)
    df['Current_Clicks'] = df.impressions * df.Current_CTR
    df['Projected_Clicks'] = df.impressions * df.Target_CTR
    df['Incremental_Clicks'] = df.Projected_Clicks - df.Current_Clicks
    df = df[df.Incremental_Clicks > 0]
    if df.empty:
        st.warning("No positive incremental clicks projected.")
        return None, pd.DataFrame()

    # monetize
    conv = conversion_rate / 100
    close = close_rate / 100
    df['Signups'] = df.Incremental_Clicks * conv
    df['Customers'] = df.Signups * close
    df['MRR'] = df.Customers * mrr_per_customer

    # paid-ads cost & savings
    df['Paid_Cost'] = df.Incremental_Clicks * df.cpc
    total_paid_cost = df.Paid_Cost.sum()
    savings_vs_paid_ads = total_paid_cost - seo_cost

    # totals & ROI
    tot_clicks = df.Incremental_Clicks.sum()
    tot_signups = df.Signups.sum()
    tot_customers = df.Customers.sum()
    tot_mrr = df.MRR.sum()
    seo_roi_pct = float('inf') if seo_cost == 0 else ((tot_mrr - seo_cost) / seo_cost) * 100

    summary = {
        "clicks": f"{tot_clicks:,.0f}",
        "signups": f"{tot_signups:,.1f}",
        "customers": f"{tot_customers:,.1f}",
        "mrr": f"${tot_mrr:,.2f}",
        "roi": f"{seo_roi_pct:,.2f}%",
        "paid_cost": f"${total_paid_cost:,.2f}",
        "savings": f"${savings_vs_paid_ads:,.2f}"
    }

    # table
    out = df[['query','MRR','Paid_Cost']].copy()
    out.rename(columns={
        'query': 'Keyword',
        'MRR': 'Projected Incremental MRR ($)',
        'Paid_Cost': 'Equivalent Paid Ads Cost ($)'
    }, inplace=True)
    out['Impact'] = pd.cut(
        out['Projected Incremental MRR ($)'],
        bins=[-1, 500, 2000, float('inf')],
        labels=['Low Priority','Moderate ROI','High ROI']
    )
    out.sort_values(['Impact','Projected Incremental MRR ($)'], ascending=[True,False], inplace=True)

    return summary, out

# === Run ===
if st.button("Run Forecast"):
    df = load_csv()
    if df is not None:
        summary, table = calculate_roi(df)
        if summary:
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("Incremental Clicks", summary['clicks'])
            c2.metric("Projected Signups", summary['signups'])
            c3.metric("New Customers", summary['customers'])
            c4.metric("Incremental MRR", summary['mrr'])
            c5.metric("SEO ROI", summary['roi'])
            c6.metric("Paid Ads Cost", summary['paid_cost'])
            c7.metric("Savings vs Paid Ads", summary['savings'])

            st.subheader("📊 Opportunity Keywords")
            st.dataframe(table, use_container_width=True)