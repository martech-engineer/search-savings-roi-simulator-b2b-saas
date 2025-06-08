import pandas as pd
import numpy as np
import streamlit as st
import requests

# ▶️ URL for your sample GSC CSV
SAMPLE_FILE_URL = (
    "https://huggingface.co/spaces/Em4e/seo-b2b-saas-forecasting-tool/"
    "resolve/main/sample_gsc_data.csv"
)

st.set_page_config(page_title="SEO ROI & Savings Forecasting", layout="wide")
st.title("📈 SEO ROI & Savings Forecasting Tool for B2B SaaS")

# ---
# ℹ️ How the app works
with st.expander("ℹ️ How the app works", expanded=True):
    st.markdown("""
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
    <p>1. <b>Load your GSC data</b> (we lowercase all column names on load). If no file is uploaded, we use the default sample data. If no <code>cpc</code> column is present, we simulate values between \\$0.50–\\$3.00.</p>
    <p>2. <b>CTR benchmarks</b> by position map an expected click-through rate for positions 1–20.</p>
    <p>3. <b>Incremental Clicks</b> = Projected_Clicks – Current_Clicks</p>
    <p>&nbsp;&nbsp;&nbsp;&nbsp;• Current_Clicks = Impressions × Current_CTR</p>
    <p>&nbsp;&nbsp;&nbsp;&nbsp;• Projected_Clicks = Impressions × Target_CTR</p>
    <p>4. <b>Financials</b></p>
    <p>&nbsp;&nbsp;&nbsp;&nbsp;• <b>Avoided Paid Spend</b> = Incremental_Clicks × CPC. This represents the money you <b>don't</b> have to spend on paid ads because your organic SEO efforts are now bringing in those clicks and conversions.</p>
    <p>&nbsp;&nbsp;&nbsp;&nbsp;• <b>Net Savings vs Paid</b> = Avoided Paid Spend – SEO Investment</p>
    <p>&nbsp;&nbsp;&nbsp;&nbsp;• <b>Incremental MRR</b> = Customers × MRR_per_Customer</p>
    <p>&nbsp;&nbsp;&nbsp;&nbsp;• <b>SEO ROI</b> = (Incremental MRR – SEO Investment) ÷ SEO Investment</p>
    <p><b>Understanding "Additional Ad Spend"</b></p>
    <p>The "Additional Ad Spend" input in the sidebar is a <b>hypothetical budget figure you provide for comparison</b>. It's <b>not</b> calculated from your GSC data or CPC. Instead, it allows you to:</p>
    <ul>
        <li><b>Compare SEO's revenue generation directly against a specific paid ad budget.</b> For instance, if you're considering spending an extra \\$X on Google Ads, you can see if your SEO's projected incremental MRR is higher or lower than that \\$X.</li>
        <li><b>Visualize the efficiency of your SEO investment.</b> If your SEO investment generates significantly more incremental MRR than a comparable "Additional Ad Spend," it highlights SEO as a potentially more effective use of marketing funds.</li>
    </ul>
    <p>The "Ad Spend" metric will show <span style="color: green; font-weight: bold;">green</span> if your projected Incremental MRR from SEO <b>trumps</b> (is greater than) this additional ad spend, and <span style="color: red; font-weight: bold;">red</span> if it does not.</p>
    <p>5. <b>Results</b></p>
    <p>Top-line metrics + keyword-level table with impact labels.</p>
</div>
""", unsafe_allow_html=True) # Ensure this is within the st.expander context

# — Sidebar inputs
with st.sidebar:
    st.header("🔧 Assumptions & Inputs") # Moved st.header inside with block
    uploaded_file    = st.file_uploader("Upload GSC CSV", type="csv")
    target_position  = st.slider("Target SERP Position",
        1.0, 10.0, 4.0, 0.5)
    conversion_rate  = st.slider("Conversion Rate (% → signup)", 0.1, 10.0, 2.0, 0.1)
    close_rate       = st.slider("Close Rate (% → customer)",
        1.0, 100.0, 20.0, 1.0)
    mrr_per_customer = st.slider("MRR per Customer ($)",
        10,    1000, 200, 10)
    seo_cost         = st.slider("Total SEO Investment ($)",
        1_000, 100_000, 10_000, 1_000)
    add_spend        = st.slider("Additional Ad Spend ($)",
        0, 50_000, 0, 1_000)


# — Download sample CSV button
sample_bytes = requests.get(SAMPLE_FILE_URL).content
st.download_button(
    label="📥 Download sample CSV",
    data=sample_bytes,
    file_name="sample_gsc_data.csv",
    mime="text/csv",
)

# === Load & normalize CSV ===
def load_csv():
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(SAMPLE_FILE_URL)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # lowercase all column names
    df.columns = [col.lower() for col in df.columns]

    # ensure a 'cpc' column
    if 'cpc' not in df.columns:
        st.warning("No `cpc` column found—simulating CPC values between $0.50–$3.00.")
        df['cpc'] = np.round(np.random.uniform(0.5, 3.0, size=len(df)), 2)
    return df

# === Core calculation ===
def calculate(df, target_position, conversion_rate, close_rate, mrr_per_customer, seo_cost, add_spend):
    # required columns mapping
    cols = {c: c for c in df.columns}
    required = {
        'query':       ['query', 'keyword', 'queries'],
        'impressions': ['impressions'],
        'position':    ['position', 'avg. position', 'average position'],
        'cpc':         ['cpc']
    }
    found = {}
    for key, opts in required.items():
        for opt in opts:
            if opt in df.columns:
                found[key] = opt
                break
        if key not in found:
            st.error(f"Missing required column: {key}")
            return None, pd.DataFrame()

    # rename to our standard keys
    df = df.rename(columns={found[k]: k for k in found})

    # CTR benchmarks
    ctr = {i: v for i, v in zip(range(1, 11), [0.25,0.15,0.10,0.08,0.06,0.04,0.03,0.02,0.015,0.01])}
    ctr.update({i: 0.005 for i in range(11,21)})
    get_ctr = lambda p: ctr.get(int(round(p)), 0.005)

    # filter positions 5–20
    df = df[df['position'].between(5, 20)].copy()
    if df.empty:
        st.warning("No keywords in positions 5–20.")
        return None, pd.DataFrame()

    # clicks projections
    df['current_ctr']     = df['position'].map(get_ctr)
    df['target_ctr']      = get_ctr(target_position)
    df['current_clicks']  = df['impressions'] * df['current_ctr']
    df['projected_clicks'] = df['impressions'] * df['target_ctr']
    df['incremental_clicks'] = df['projected_clicks'] - df['current_clicks']
    df = df[df['incremental_clicks'] > 0]
    if df.empty:
        st.warning("No positive incremental clicks projected.")
        return None, pd.DataFrame()

    # conversions → MRR
    conv  = conversion_rate / 100
    close = close_rate       / 100
    df['signups']    = df['incremental_clicks'] * conv
    df['customers']  = df['signups']            * close
    df['mrr']        = df['customers']          * mrr_per_customer

    # financials: avoided spend & net savings
    df['avoided_paid_spend'] = df['incremental_clicks'] * df['cpc']
    total_avoided            = df['avoided_paid_spend'].sum()
    net_savings              = total_avoided - seo_cost

    # totals & ROI
    tot_clicks    = df['incremental_clicks'].sum()
    tot_signups   = df['signups'].sum()
    tot_customers = df['customers'].sum()
    tot_mrr       = df['mrr'].sum()
    seo_roi_pct   = float('inf') if seo_cost == 0 else ((tot_mrr - seo_cost) / seo_cost) * 100

    # Calculate if SEO investment trumps add spend (using Incremental MRR for comparison)
    seo_trumps_add_spend = tot_mrr > add_spend

    summary = {
        "clicks":    f"{tot_clicks:,.0f}",
        "signups":   f"{tot_signups:,.1f}",
        "customers": f"{tot_customers:,.1f}",
        "mrr":       f"${tot_mrr:,.2f}",
        "roi":       f"{seo_roi_pct:,.2f}%",
        "avoid":     f"${total_avoided:,.2f}",
        "net":       f"${net_savings:,.2f}",
        "add_spend": f"${add_spend:,.2f}", # Include add_spend in summary
        "seo_trumps_add_spend": seo_trumps_add_spend # Boolean for coloring
    }

    # keyword-level table
    out = df[['query', 'mrr', 'avoided_paid_spend']].copy()
    out.columns = [
        'Keyword',
        'Projected Incremental MRR ($)',
        'Avoided Paid Spend ($)'
    ]
    out['Impact'] = pd.cut(
        out['Projected Incremental MRR ($)'],
        bins=[-1, 500, 2000, float('inf')],
        labels=['Low Priority','Moderate ROI','High ROI']
    )
    out = out.sort_values(['Impact','Projected Incremental MRR ($)'],
                          ascending=[True, False])
    return summary, out

# === Run forecast & display ===
if st.button("Run Forecast"):
    df = load_csv()
    if df is not None:
        # Pass all relevant inputs to the calculate function
        summary, table = calculate(df, target_position, conversion_rate, close_rate, mrr_per_customer, seo_cost, add_spend)
        if summary:
            c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8) # Added one more column for add_spend
            c1.metric("Incremental Clicks",     summary['clicks'])
            c2.metric("Projected Signups",      summary['signups'])
            c3.metric("New Customers",          summary['customers'])
            c4.metric("Incremental MRR",        summary['mrr'])
            c5.metric("SEO ROI",                summary['roi'])
            c6.metric("Avoided Paid Spend",     summary['avoid'])
            c7.metric("Net Savings vs Paid",    summary['net'])

            # Ad Spend section with conditional coloring
            with c8:
                st.markdown(
                    f"**Ad Spend**<br>"
                    f"<span style='color: {'green' if summary['seo_trumps_add_spend'] else 'red'}; font-size: 24px; font-weight: bold;'>{summary['add_spend']}</span>",
                    unsafe_allow_html=True
                )

            st.subheader("📊 Opportunity Keywords")
            st.dataframe(table, use_container_width=True)