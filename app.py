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
    st.markdown("""<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
    <p>1. <b>Load your GSC data</b> (we lowercase all column names on load). If no file is uploaded, we use the default sample data. If no <code>cpc</code> column is present, we simulate values between 0.50–3.00 dollars.</p>
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
        <li><b>Compare SEO's revenue generation directly against a specific paid ad budget.</b> For instance, if you're considering spending an extra \$X on Google Ads, you can see if your SEO's projected incremental MRR is higher or lower than that \$X.</li>
        <li><b>Visualize the efficiency of your SEO investment.</b> If your SEO investment generates significantly more incremental MRR than a comparable "Additional Ad Spend," it highlights SEO as a potentially more effective use of marketing funds.</li>
    </ul>
    <p>The "Ad Spend" metric will show <span style="color: green; font-weight: bold;">green</span> if your projected Incremental MRR from SEO <b>trumps</b> (is greater than) this additional ad spend, and <span style="color: red; font-weight: bold;">red</span> if it does not.</p>
    <p>5. <b>Results</b></p>
    <p>Top-line metrics + keyword-level table with impact labels.</p></div>""", unsafe_allow_html=True) # Ensure this is within the st.expander context

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

    df['current_ctr'] = df['position'].apply(get_ctr)
    df['target_ctr'] = df['position'].apply(lambda x: ctr.get(int(round(target_position)), 0.005))
    
    df['current_clicks'] = df['impressions'] * df['current_ctr']
    df['projected_clicks'] = df['impressions'] * df['target_ctr']
    df['incremental_clicks'] = df['projected_clicks'] - df['current_clicks']
    
    df['avoided_paid_spend'] = df['incremental_clicks'] * df['cpc']
    
    # Financial calculations
    total_avoided_paid_spend = df['avoided_paid_spend'].sum()
    net_savings_vs_paid = total_avoided_paid_spend - seo_cost
    
    total_incremental_conversions = df['incremental_clicks'].sum() * (conversion_rate / 100)
    total_incremental_customers = total_incremental_conversions * (close_rate / 100)
    
    incremental_mrr = total_incremental_customers * mrr_per_customer
    
    # SEO ROI calculation
    if seo_cost > 0:
        seo_roi = (incremental_mrr - seo_cost) / seo_cost
    else:
        seo_roi = np.inf # Undefined or very high if no SEO cost

    # Categorize impact for each query
    def categorize_impact(row):
        if row['position'] > target_position:
            return '🚀 Improvement'
        elif row['position'] <= target_position and row['incremental_clicks'] > 0:
            return '✅ Maintain & Grow'
        else:
            return '🎯 Reached Target'
            
    df['impact_category'] = df.apply(categorize_impact, axis=1)

    return {
        'total_avoided_paid_spend': total_avoided_paid_spend,
        'net_savings_vs_paid': net_savings_vs_paid,
        'total_incremental_conversions': total_incremental_conversions,
        'total_incremental_customers': total_incremental_customers,
        'incremental_mrr': incremental_mrr,
        'seo_roi': seo_roi
    }, df

# ---
# Main app logic
df = load_csv()

if df is not None:
    metrics, df_results = calculate(df.copy(), target_position, conversion_rate, close_rate, mrr_per_customer, seo_cost, add_spend)

    if metrics is not None:
        st.write("---")
        st.header("📊 SEO Performance Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Avoided Paid Spend 💰", value=f"${metrics['total_avoided_paid_spend']:,.2f}")
        with col2:
            st.metric(label="Net Savings vs Paid 📈", value=f"${metrics['net_savings_vs_paid']:,.2f}")
        with col3:
            st.metric(label="Incremental MRR  recurrent revenue 🚀", value=f"${metrics['incremental_mrr']:,.2f}")
        
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric(label="Total Incremental Conversions 🎯", value=f"{metrics['total_incremental_conversions']:,.0f}")
        with col5:
            st.metric(label="Total Incremental Customers 🤝", value=f"{metrics['total_incremental_customers']:,.0f}")
        with col6:
            st.metric(label="SEO ROI (Return on Investment) 💰", value=f"{metrics['seo_roi']:.2%}")
        
        st.write("---")
        st.header("Detailed Keyword Performance") # Kept as st.header for prominence
        st.dataframe(df_results[[
            'query', 'impressions', 'position', 'current_ctr', 'target_ctr', 
            'current_clicks', 'projected_clicks', 'incremental_clicks', 
            'cpc', 'avoided_paid_spend', 'impact_category'
        ]].sort_values(by='incremental_clicks', ascending=False), use_container_width=True)

        st.write("---")
        st.header("Hypothetical Comparison: SEO vs. Additional Ad Spend")
        col_ad1, col_ad2 = st.columns(2)
        with col_ad1:
            st.metric(label="Incremental MRR from SEO", value=f"${metrics['incremental_mrr']:,.2f}")
        with col_ad2:
            ad_spend_color = "green" if metrics['incremental_mrr'] > add_spend else "red"
            # Increased font-size for the 'add_spend' value
            st.markdown(f"**Additional Ad Spend (for comparison)**: <span style='color:{ad_spend_color}; font-weight:bold; font-size: 2em;'>${add_spend:,.2f}</span>", unsafe_allow_html=True)
            if metrics['incremental_mrr'] > add_spend:
                st.success(f"SEO's incremental MRR is ${metrics['incremental_mrr'] - add_spend:,.2f} higher than the additional ad spend!")
            else:
                st.warning(f"Additional ad spend is ${add_spend - metrics['incremental_mrr']:,.2f} higher than SEO's incremental MRR.")