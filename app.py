import pandas as pd
import numpy as np
import streamlit as st
import requests

# ▶️ URL for your sample GSC CSV
# This URL points to a sample Google Search Console (GSC) data CSV file hosted on Hugging Face.
SAMPLE_FILE_URL = (
    "https://huggingface.co/spaces/Em4e/seo-b2b-saas-forecasting-tool/"
    "resolve/main/sample_gsc_data.csv"
)

# === Helper Functions ===
# === Load & normalize CSV ===
# This function handles loading the GSC data, either from an uploaded file or the sample URL.
# It also standardizes column names and simulates CPC values if missing.
@st.cache_data # Caches the output of this function to improve performance.
def load_csv(uploaded_file_obj):
    """
    Loads the GSC data from an uploaded CSV or a sample URL,
    normalizes column names, and ensures a 'cpc' column exists.
    Args:
        uploaded_file_obj (streamlit.uploaded_file_manager.UploadedFile): The file object
                                                                         uploaded by the user, or None.
    Returns:
        pd.DataFrame: The loaded and processed DataFrame, or None if an error occurs.
    """
    try:
        if uploaded_file_obj:
            df = pd.read_csv(uploaded_file_obj)
        else:
            df = pd.read_csv(SAMPLE_FILE_URL)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None
    # Convert all column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]
    # Check for 'cpc' column; if missing, simulate values
    if "cpc" not in df.columns:
        st.warning("No `cpc` column found—simulating CPC values between 0.50–3.00 USD (for testing purposes only!)")
        df["cpc"] = np.round(np.random.uniform(0.5, 3.0, size=len(df)), 2)
    return df

# === Core calculation ===
# This function performs the main calculations for SEO performance, including
# click-through rates, incremental clicks, avoided paid spend, and ROI.
@st.cache_data # Caches the output of this function to improve performance.
def calculate(
    df,
    target_position,
    conversion_rate,
    close_rate,
    mrr_per_customer,
    seo_cost,
    add_spend,
):
    """
    Performs core calculations for SEO forecasting based on GSC data and user inputs.
    Args:
        df (pd.DataFrame): The input DataFrame containing GSC data.
        target_position (float): The desired average search engine result page position.
        conversion_rate (float): Percentage of clicks that convert to signups.
        close_rate (float): Percentage of signups that become paying customers.
        mrr_per_customer (int): Monthly Recurring Revenue per customer.
        seo_cost (int): Total investment in SEO efforts.
        add_spend (int): Hypothetical additional ad spend for comparison.
    Returns:
        tuple: A dictionary of calculated metrics and a DataFrame with detailed results.
               Returns (None, pd.DataFrame()) if required columns are missing.
    """
    # Define required column mappings for flexibility in input CSVs
    required_columns = {
        "query": ["query", "keyword", "queries"],
        "impressions": ["impressions"],
        "position": ["position", "avg. position", "average position"],
        "cpc": ["cpc"],
    }
    found_columns = {}
    for key, options in required_columns.items():
        for opt in options:
            if opt in df.columns:
                found_columns[key] = opt
                break
        if key not in found_columns:
            st.error(f"Missing required column: {key}. Please ensure your CSV has one of {options}.")
            return None, pd.DataFrame()
    # Rename columns to a standardized format for easier processing
    df = df.rename(columns={found_columns[k]: k for k in found_columns})
    # Define Click-Through Rate (CTR) benchmarks by position
    # These are illustrative CTRs for positions 1-20
    ctr_benchmarks = {i: v for i, v in zip(range(1, 11), [0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01])}
    ctr_benchmarks.update({i: 0.005 for i in range(11, 21)})
    # Helper function to get CTR based on position, defaulting to 0.005 for positions > 20
    get_ctr = lambda p: ctr_benchmarks.get(int(round(p)), 0.005)
    
    df["current_ctr"] = df["position"].apply(get_ctr)
    
    # Optimized: Calculate the target CTR value once and assign it to the whole column
    target_ctr_value = ctr_benchmarks.get(int(round(target_position)), 0.005)
    df["target_ctr"] = target_ctr_value # Optimized line
    
    df["current_clicks"] = df["impressions"] * df["current_ctr"]
    df["projected_clicks"] = df["impressions"] * df["target_ctr"]
    df["incremental_clicks"] = df["projected_clicks"] - df["current_clicks"]
    df["avoided_paid_spend"] = df["incremental_clicks"] * df["cpc"]
    # --- Financial calculations ---
    total_avoided_paid_spend = df["avoided_paid_spend"].sum()
    net_savings_vs_paid = total_avoided_paid_spend - seo_cost
    total_incremental_conversions = df["incremental_clicks"].sum() * (
        conversion_rate / 100
    )
    total_incremental_customers = total_incremental_conversions * (close_rate / 100)
    incremental_mrr = total_incremental_customers * mrr_per_customer
    # SEO ROI calculation, handling division by zero for seo_cost
    if seo_cost > 0:
        seo_roi = (incremental_mrr - seo_cost) / seo_cost
    else:
        seo_roi = np.inf  # Undefined or very high if no SEO cost
    # Categorize impact for each query based on its current position relative to the target
    def categorize_impact(row):
        if row["position"] > target_position:
            return "🚀 Improvement"  # Position is worse than target, room for improvement
        elif (
            row["position"] <= target_position and row["incremental_clicks"] > 0
        ):
            return "✅ Maintain & Grow"  # Position is at or better than target, still gaining clicks
        else:
            return "🎯 Reached Target"  # Position is at or better than target, no further incremental clicks expected
    df["impact_category"] = df.apply(categorize_impact, axis=1)
    # Return calculated metrics and the detailed DataFrame
    return {
        "total_avoided_paid_spend": total_avoided_paid_spend,
        "net_savings_vs_paid": net_savings_vs_paid,
        "total_incremental_conversions": total_incremental_conversions,
        "total_incremental_customers": total_incremental_customers,
        "incremental_mrr": incremental_mrr,
        "seo_roi": seo_roi,
    }, df

# Set Streamlit page configuration for a wider layout and a descriptive title.
st.set_page_config(page_title="SEO ROI & Savings Forecasting", layout="wide")
st.title("📈 B2B SaaS SEO ROI & Savings Simulator")
st.markdown("App created by [Emilija Gjorgjevska](https://www.linkedin.com/in/emilijagjorgjevska/)")

# ℹ️ How the app works
# This section provides an expandable information box explaining the app's methodology.
with st.expander("ℹ️ How the app works", expanded=True):
    st.markdown(
        """
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
      <p>1. <b>Load your GSC data</b> (we lowercase all column names on load). If no file is uploaded, we use the default sample data. If no <code>cpc</code> column is present, we simulate values between 0.50 and 3.00 USD.</p>
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
        <li><b>Compare SEO's revenue generation directly against a specific paid ad budget.</b> For instance, if you're considering spending an extra X dollars on Google Ads, you can see whether your SEO's projected incremental MRR is higher or lower than that same amount.</li>
        <li><b>Visualize the efficiency of your SEO investment.</b> If your SEO investment generates significantly more incremental MRR than a comparable additional ad spend, it highlights SEO as a potentially more effective use of marketing funds.</li>
      </ul>
      <p>The "Ad Spend" metric will be <span style="color: green; font-weight: bold;">green</span> if your projected Incremental MRR from SEO is <b>greater than</b> this additional ad spend, and <span style="color: red; font-weight: bold;">red</span> if it is not.</p>
      
      <p>5. <b>Interpreting Results & Assumptions</b></p>
      <ul>
        <li><b>Target SERP Position:</b> The 'Target SERP Position' is an <u>aspirational average</u> you aim for among your <u>most important and achievable keywords</u>, rather than a literal expectation for every single query. In reality, not all keywords will reach the same position due to varying competition and relevance.</li>
        <li><b>High-Impact Queries:</b> While the model calculates for all queries, focus your analysis on the 'Detailed Keyword Performance' table. Look for queries with a '🚀 Improvement' impact category and high 'impressions' and 'incremental_clicks'. These are often your most promising opportunities for SEO effort.</li>
      </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

# — Sidebar inputs
# This section defines the input controls in the Streamlit sidebar, allowing users to
# adjust various parameters for the SEO forecasting.
with st.sidebar:
    st.header("🔧 Assumptions & Inputs")
    uploaded_file = st.file_uploader("Upload GSC CSV", type="csv")
    target_position = st.slider(
        "Target SERP Position",
        1.0,
        10.0,
        4.0,
        0.5,
        help="This is the **average search engine ranking you assume all your queries will achieve.** A lower number (e.g., position 1) indicates a higher, more visible ranking. This target position is used to project the future click-through rate for every query."
    )  # Desired average search engine result page position
    conversion_rate = st.slider(
        "Conversion Rate (% → signup)", 0.1, 10.0, 2.0, 0.1
    )  # Percentage of clicks that convert to signups
    close_rate = st.slider(
        "Close Rate (% → customer)", 1.0, 100.0, 20.0, 1.0
    )  # Percentage of signups that become paying customers
    mrr_per_customer = st.slider(
        "MRR per Customer ($)", 10, 1000, 200, 10
    )  # Monthly Recurring Revenue per customer
    seo_cost = st.slider(
        "Total SEO Investment ($)", 1_000, 100_000, 10_000, 1_000
    )  # Total investment in SEO efforts
    add_spend = st.slider(
        "Additional Ad Spend ($)", 0, 50_000, 0, 1_000
    )  # Hypothetical additional ad spend for comparison

# — Download sample CSV button
# Provides a button for users to download the sample GSC data CSV.
sample_bytes = requests.get(SAMPLE_FILE_URL).content
st.download_button(
    label="📥 Download sample CSV",
    data=sample_bytes,
    file_name="sample_gsc_data.csv",
    mime="text/csv",
)

# --- Main app logic ---
# This block orchestrates the flow of the Streamlit application.
df = load_csv(uploaded_file)  # Load the data first, passing the uploaded file object
if df is not None:  # Proceed only if data loading was successful
    # Perform the core calculations
    metrics, df_results = calculate(
        df.copy(),
        target_position,
        conversion_rate,
        close_rate,
        mrr_per_customer,
        seo_cost,
        add_spend,
    )
    if metrics is not None:  # Proceed only if calculations were successful
        st.write("---")
        st.header("📊 SEO Performance Summary")
        # Display key performance metrics in a 3-column layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Total Avoided Paid Spend 💰",
                value=f"${metrics['total_avoided_paid_spend']:,.2f}",
            )
        with col2:
            st.metric(
                label="Net Savings vs Paid 📈",
                value=f"${metrics['net_savings_vs_paid']:,.2f}",
            )
        with col3:
            st.metric(
                label="Incremental MRR (Monthly Recurring Revenue) 🚀",
                value=f"${metrics['incremental_mrr']:,.2f}",
            )
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric(
                label="Total Incremental Conversions 🎯",
                value=f"{metrics['total_incremental_conversions']:,.0f}",
            )
        with col5:
            st.metric(
                label="Total Incremental Customers 🤝",
                value=f"{metrics['total_incremental_customers']:,.0f}",
            )
        with col6:
            st.metric(
                label="SEO ROI (Return on Investment) 💰",
                value=f"{metrics['seo_roi']:.2%}",
            )
        st.write("---")
        st.header("Hypothetical Comparison: SEO vs. Additional Ad Spend")
        # Compare SEO's incremental MRR with a hypothetical additional ad spend
        col_ad1, col_ad2, col_advice = st.columns([1, 1, 1])
        with col_ad1:
            st.metric(
                label="Incremental MRR from SEO",
                value=f"${metrics['incremental_mrr']:,.2f}",
            )
        with col_ad2:
            st.metric(
                label="Additional Ad Spend", value=f"${add_spend:,.2f}"
            )
        with col_advice:
            if metrics["incremental_mrr"] > add_spend:
                advice_message = "SEO is a better investment!"
                advice_color = "green"
            else:
                advice_message = "Ad Spend may yield higher returns."
                advice_color = "red"
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <p style="font-size: 1.2em; margin-bottom: 0;">Advice</p>
                    <p style="color:{advice_color}; font-weight:bold; font-size: 1.5em; margin-top: 0;">{advice_message}</p>
                </div>
            """,
                unsafe_allow_html=True,
            )
        st.write("---")
        st.header("Detailed Keyword Performance")
        st.info("💡 **How to use this table:** Focus on queries with the '🚀 Improvement' impact category and high 'impressions'. These represent opportunities where improving your current position towards the 'Target SERP Position' can yield significant incremental clicks and avoided paid spend.")
        # Display a detailed table of keyword performance, sorted by incremental clicks
        st.dataframe(
            df_results[
                [
                    "query",
                    "impressions",
                    "position",
                    "current_ctr",
                    "target_ctr",
                    "current_clicks",
                    "projected_clicks",
                    "incremental_clicks",
                    "cpc",
                    "avoided_paid_spend",
                    "impact_category",
                ]
            ].sort_values(by="incremental_clicks", ascending=False),
            use_container_width=True,
        )