import pandas as pd
import streamlit as st
import io

# ✅ Raw sample file URL
SAMPLE_FILE_URL = "https://huggingface.co/spaces/Em4e/seo-b2b-saas-forecasting-tool/raw/main/sample_gsc_data.csv"

st.set_page_config(page_title="SEO ROI Forecasting Tool for B2B SaaS", layout="wide")
st.title("📈 SEO ROI Forecasting Tool for B2B SaaS")

st.markdown("""
This app helps you estimate the **financial upside** of ranking improvements for your SEO keywords.  
<br>

👉 **Please make sure to check the sample file before uploading your own data**:  
📎 [Download sample CSV](https://huggingface.co/spaces/Em4e/seo-b2b-saas-forecasting-tool/blob/main/sample_gsc_data.csv)  
<br>

Developed by: [Emilija Gjorgjevska](https://www.linkedin.com/in/emilijagjorgjevska/)
""", unsafe_allow_html=True)


# === Inputs ===
with st.sidebar:
    st.header("🔧 Assumptions")
    uploaded_file = st.file_uploader("Upload Google Search Console CSV", type="csv")
    target_position = st.slider("Target SERP Position", min_value=1.0, max_value=10.0, step=0.5, value=4.0)
    conversion_rate = st.slider("Conversion Rate (Visitor → Signup %)", min_value=0.1, max_value=10.0, step=0.1, value=2.0)
    close_rate = st.slider("Close Rate (Signup → Customer %)", min_value=1.0, max_value=100.0, step=1.0, value=20.0)
    mrr_per_customer = st.slider("MRR per Customer ($)", min_value=10, max_value=1000, step=10, value=200)
    seo_cost = st.slider("Total SEO Investment ($)", min_value=1000, max_value=100000, step=1000, value=10000)

# === Load CSV ===
def load_csv():
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_csv(SAMPLE_FILE_URL)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# === Main ROI Logic ===
def calculate_roi(df):
    empty_df = pd.DataFrame()
    try:
        conversion = conversion_rate / 100
        close = close_rate / 100

        df_columns_lower = {col.lower(): col for col in df.columns}
        expected_cols = {
            'query': ['query', 'queries', 'keyword', 'keywords'],
            'impressions': ['impressions'],
            'position': ['position', 'avg. position', 'average position']
        }

        found_cols = {}
        for k, v_list in expected_cols.items():
            for v in v_list:
                if v in df_columns_lower:
                    found_cols[k] = df_columns_lower[v]
                    break
            if k not in found_cols:
                st.error(f"Missing required column for {k.upper()}")
                return None, empty_df

        df.rename(columns={found_cols[k]: k for k in found_cols}, inplace=True)

        ctr_benchmarks = {i: v for i, v in zip(range(1, 11), [0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01])}
        ctr_benchmarks.update({i: 0.005 for i in range(11, 21)})
        get_ctr = lambda pos: ctr_benchmarks.get(int(round(pos)), 0.005)

        df = df[(df['position'] >= 5) & (df['position'] <= 20)].copy()
        if df.empty:
            st.warning("No keywords between position 5–20.")
            return None, empty_df

        df['Current_CTR'] = df['position'].apply(get_ctr)
        df['Target_CTR'] = get_ctr(target_position)
        df['Projected_Clicks'] = df['impressions'] * df['Target_CTR']
        df['Current_Clicks'] = df['impressions'] * df['Current_CTR']
        df['Incremental_Clicks'] = df['Projected_Clicks'] - df['Current_Clicks']
        df = df[df['Incremental_Clicks'] > 0]

        if df.empty:
            st.warning("No incremental clicks projected. Adjust assumptions.")
            return None, empty_df

        df['Signups'] = df['Incremental_Clicks'] * conversion
        df['Customers'] = df['Signups'] * close
        df['MRR'] = df['Customers'] * mrr_per_customer

        total_clicks = df['Incremental_Clicks'].sum()
        total_signups = df['Signups'].sum()
        total_customers = df['Customers'].sum()
        total_mrr = df['MRR'].sum()
        roi = float('inf') if seo_cost == 0 else ((total_mrr - seo_cost) / seo_cost) * 100

        output_df = df[['query', 'MRR']].copy()
        output_df.rename(columns={'query': 'Keyword', 'MRR': 'Projected Incremental MRR ($)'}, inplace=True)

        def label(m):
            if m >= 2000:
                return "High ROI"
            elif m >= 500:
                return "Moderate ROI"
            return "Low Priority"

        output_df['Impact'] = output_df['Projected Incremental MRR ($)'].apply(label)
        output_df.sort_values(by=['Impact', 'Projected Incremental MRR ($)'], ascending=[True, False], inplace=True)

        return {
            "clicks": f"{total_clicks:,.0f}",
            "signups": f"{total_signups:,.1f}",
            "customers": f"{total_customers:,.1f}",
            "mrr": f"${total_mrr:,.2f}",
            "roi": f"{roi:,.2f}%"
        }, output_df

    except Exception as e:
        st.error(f"Error during ROI calculation: {e}")
        return None, empty_df


if st.button("Run Forecast"):
    df = load_csv()
    if df is not None:
        summary, table = calculate_roi(df)
        if summary:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Incremental Clicks", summary['clicks'])
            col2.metric("Projected Signups", summary['signups'])
            col3.metric("New Customers", summary['customers'])
            col4.metric("Incremental MRR", summary['mrr'])
            col5.metric("SEO ROI", summary['roi'])

            st.subheader("📊 Opportunity Keywords")
            st.dataframe(table, use_container_width=True)
