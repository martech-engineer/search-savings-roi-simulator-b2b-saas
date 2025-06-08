import pandas as pd
import numpy as np
import streamlit as st
import requests

# ▶️ Source CSV (with or without CPC column)
SAMPLE_FILE_URL = "https://huggingface.co/spaces/Em4e/seo-b2b-saas-forecasting-tool/resolve/main/sample_gsc_data.csv"

st.set_page_config(page_title="SEO ROI + Savings Tool", layout="wide")
st.title("📈 SEO ROI & Savings Forecasting for B2B SaaS")

# — Info section on how it works
with st.expander("ℹ️ How the app works", expanded=True):
    st.markdown("""
1. **Load your GSC data** (must include `Impressions`, `Position`, and `CPC`; if CPC is missing we simulate \$0.50–\$3.00).  
2. **CTR benchmarks** by position map average CTR for positions 1–20.  
3. **Incremental clicks** =  
   &nbsp;&nbsp;Projected_Clicks – Current_Clicks  
   &nbsp;&nbsp;• Current_Clicks = Impressions×Current_CTR  
   &nbsp;&nbsp;• Projected_Clicks = Impressions×Target_CTR  
4. **Financials**  
   &nbsp;&nbsp;• Avoided Paid Spend = Incremental_Clicks×CPC  
   &nbsp;&nbsp;• Net Savings vs Paid = Avoided Paid Spend – SEO Investment  
   &nbsp;&nbsp;• Incremental MRR = Customers×MRR_per_Customer  
   &nbsp;&nbsp;• SEO ROI = (Incremental MRR – SEO Investment) ÷ SEO Investment  
5. **Results**  
   Top-line metrics + keyword-level table with Impact labels.
    """, unsafe_allow_html=True)

# — Sidebar Inputs
with st.sidebar:
    st.header("🔧 Assumptions & Inputs")
    uploaded_file    = st.file_uploader("Upload GSC CSV", type="csv")
    target_position  = st.slider("Target SERP Position",        1.0, 10.0, 4.0, 0.5)
    conversion_rate  = st.slider("Conversion Rate (Visitor→Signup %)", 0.1, 10.0, 2.0, 0.1)
    close_rate       = st.slider("Close Rate (Signup→Customer %)",      1.0, 100.0, 20.0, 1.0)
    mrr_per_customer = st.slider("MRR per Customer ($)",             10, 1000, 200, 10)
    seo_cost         = st.slider("Total SEO Investment ($)",       1_000, 100_000, 10_000, 1_000)

# — Download sample button
sample_bytes = requests.get(SAMPLE_FILE_URL).content
st.download_button(
    label="📥 Download sample CSV",
    data=sample_bytes,
    file_name="sample_gsc_data.csv",
    mime="text/csv",
)

# === Load & prep data ===
def load_csv():
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(SAMPLE_FILE_URL)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # simulate CPC if missing
    if 'CPC' not in df.columns:
        st.warning("No `CPC` column found—simulating CPC values between $0.50–$3.00.")
        df['CPC'] = np.round(np.random.uniform(0.5, 3.0, len(df)), 2)
    return df

# === Core calculation ===
def calculate(df):
    # map required cols (including CPC)
    cols = {c.lower(): c for c in df.columns}
    required = {
        'query':       ['query','keyword','queries'],
        'impressions': ['impressions'],
        'position':    ['position','avg. position'],
        'cpc':         ['cpc']  # we’ll end up with a lowercase 'cpc' column
    }
    found = {}
    for k, opts in required.items():
        for o in opts:
            if o in cols:
                found[k] = cols[o]
                break
        if k not in found:
            st.error(f"Missing column: {k}")
            return None, pd.DataFrame()

    # rename to our standard lowercase names
    df = df.rename(columns={found[k]: k for k in found})

    # CTR benchmarks & filtering omitted for brevity...

    # after you’ve computed Incremental_Clicks:
    # use direct key access to CPC:
    df['Avoided_Paid_Spend'] = df['Incremental_Clicks'] * df['cpc']

    total_avoided = df['Avoided_Paid_Spend'].sum()
    net_savings   = total_avoided - seo_cost

    # … build summary & table …

    # when building your table, again reference the column by key:
    out = df[['query', 'MRR', 'Avoided_Paid_Spend']].copy()
    out.columns = ['Keyword', 'Projected Incremental MRR ($)', 'Avoided Paid Spend ($)']

    return summary, out


# === Run & display ===
if st.button("Run Forecast"):
    df = load_csv()
    if df is not None:
        summary, table = calculate(df)
        if summary:
            c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
            c1.metric("Incremental Clicks", summary['clicks'])
            c2.metric("Projected Signups",  summary['signups'])
            c3.metric("New Customers",      summary['customers'])
            c4.metric("Incremental MRR",    summary['mrr'])
            c5.metric("SEO ROI",            summary['roi'])
            c6.metric("Avoided Paid Spend", summary['avoid'])
            c7.metric("Net Savings vs Paid",summary['net'])

            st.subheader("📊 Opportunity Keywords")
            st.dataframe(table, use_container_width=True)
