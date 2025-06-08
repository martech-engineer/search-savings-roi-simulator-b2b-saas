import pandas as pd
import numpy as np
import streamlit as st
import requests

# Constants
SAMPLE_FILE_URL = (
    "https://huggingface.co/spaces/Em4e/seo-b2b-saas-forecasting-tool/"
    "resolve/main/sample_gsc_data.csv"
)

# --- 1. Data Loading and Preprocessing (Single Responsibility Principle) ---
class DataLoader:
    """
    Handles loading GSC data from various sources and performs initial standardization.
    """
    def __init__(self, sample_file_url: str = SAMPLE_FILE_URL):
        self.sample_file_url = sample_file_url

    @st.cache_data
    # _self is correct for the instance itself
    def load_csv(_self, uploaded_file_obj: st.runtime.uploaded_file_manager.UploadedFile | None) -> pd.DataFrame | None:
        """
        Loads the GSC data from an uploaded CSV or a sample URL,
        normalizes column names, and ensures a 'cpc' column exists.

        Args:
            _self: The instance of the DataLoader class (ignored by Streamlit caching).
            uploaded_file_obj (streamlit.runtime.uploaded_file_manager.UploadedFile): The file object
                                                                             uploaded by the user, or None.
        Returns:
            pd.DataFrame: The loaded and processed DataFrame, or None if an error occurs.
        """
        try:
            if uploaded_file_obj:
                df = pd.read_csv(uploaded_file_obj)
            else:
                # Use _self.sample_file_url since _self is the instance
                df = pd.read_csv(_self.sample_file_url)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

        df.columns = [col.lower() for col in df.columns]

        if "cpc" not in df.columns:
            st.warning("No `cpc` column found—simulating CPC values between 0.50–3.00 USD (for testing purposes only!)")
            df["cpc"] = np.round(np.random.uniform(0.5, 3.0, size=len(df)), 2)
        return df

# --- 2. Core Calculation Logic (Single Responsibility Principle) ---
class SeoCalculator:
    """
    Performs core calculations for SEO forecasting.
    """
    def __init__(self):
        # Define Click-Through Rate (CTR) benchmarks by position
        self.ctr_benchmarks = {i: v for i, v in zip(range(1, 11), [0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01])}
        self.ctr_benchmarks.update({i: 0.005 for i in range(11, 21)})

        self.required_columns_map = {
            "query": ["query", "keyword", "queries"],
            "impressions": ["impressions"],
            "position": ["position", "avg. position", "average position"],
            "cpc": ["cpc"],
        }

    def _get_ctr(self, position: float) -> float:
        """Helper to get CTR based on position, defaulting to 0.005 for positions > 20."""
        return self.ctr_benchmarks.get(int(round(position)), 0.005)

    def _validate_and_rename_columns(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Validates required columns and renames them to a standardized format."""
        found_columns = {}
        for key, options in self.required_columns_map.items():
            for opt in options:
                if opt in df.columns:
                    found_columns[key] = opt
                    break
            if key not in found_columns:
                st.error(f"Missing required column: {key}. Please ensure your CSV has one of {options}.")
                return None
        return df.rename(columns={found_columns[k]: k for k in found_columns})

    @st.cache_data
    # _self is correct for the instance itself
    def calculate_metrics(
        _self, # Changed to _self
        df: pd.DataFrame,
        target_position: float,
        conversion_rate: float,
        close_rate: float,
        mrr_per_customer: int,
        seo_cost: int,
        add_spend: int,
    ) -> tuple[dict, pd.DataFrame] | tuple[None, pd.DataFrame]:
        """
        Performs core calculations for SEO forecasting based on GSC data and user inputs.

        Returns:
            tuple: A dictionary of calculated metrics and a DataFrame with detailed results.
                   Returns (None, pd.DataFrame()) if required columns are missing.
        """
        # Use _self.ctr_benchmarks and _self.required_columns_map, etc.
        df_processed = _self._validate_and_rename_columns(df.copy())
        if df_processed is None:
            return None, pd.DataFrame()

        df_processed["current_ctr"] = df_processed["position"].apply(_self._get_ctr)
        target_ctr_value = _self._get_ctr(target_position)
        df_processed["target_ctr"] = target_ctr_value

        df_processed["current_clicks"] = df_processed["impressions"] * df_processed["current_ctr"]
        df_processed["projected_clicks"] = df_processed["impressions"] * df_processed["target_ctr"]
        df_processed["incremental_clicks"] = df_processed["projected_clicks"] - df_processed["current_clicks"]
        df_processed["avoided_paid_spend"] = df_processed["incremental_clicks"] * df_processed["cpc"]

        # --- Financial calculations ---
        total_avoided_paid_spend = df_processed["avoided_paid_spend"].sum()
        net_savings_vs_paid = total_avoided_paid_spend - seo_cost
        total_incremental_conversions = df_processed["incremental_clicks"].sum() * (
            conversion_rate / 100
        )
        total_incremental_customers = total_incremental_conversions * (close_rate / 100)
        incremental_mrr = total_incremental_customers * mrr_per_customer

        if seo_cost > 0:
            seo_roi = (incremental_mrr - seo_cost) / seo_cost
        else:
            seo_roi = np.inf

        # Categorize impact for each query
        def categorize_impact(row):
            if row["position"] > target_position:
                return "🚀 Improvement"
            elif row["position"] <= target_position and row["incremental_clicks"] > 0:
                return "✅ Maintain & Grow"
            else:
                return "🎯 Reached Target"
        df_processed["impact_category"] = df_processed.apply(categorize_impact, axis=1)

        metrics = {
            "total_avoided_paid_spend": total_avoided_paid_spend,
            "net_savings_vs_paid": net_savings_vs_paid,
            "total_incremental_conversions": total_incremental_conversions,
            "total_incremental_customers": total_incremental_customers,
            "incremental_mrr": incremental_mrr,
            "seo_roi": seo_roi,
        }
        return metrics, df_processed

# --- 3. Streamlit User Interface (Single Responsibility Principle) ---
class SeoAppUI:
    """
    Manages the Streamlit user interface and presentation.
    """
    def __init__(self, data_loader: DataLoader, seo_calculator: SeoCalculator):
        self.data_loader = data_loader
        self.seo_calculator = seo_calculator
        self._set_page_config()

    def _set_page_config(self):
        st.set_page_config(page_title="SEO ROI & Savings Forecasting", layout="wide")
        st.title("B2B SaaS SEO ROI & Savings Simulator")
        st.markdown("App created by [Emilija Gjorgjevska](https://www.linkedin.com/in/emilijagjorgjevska/)")

    def _display_info_expander(self):
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

    def _get_sidebar_inputs(self) -> tuple:
        with st.sidebar:
            st.header("🔧 Assumptions & Inputs")
            uploaded_file = st.file_uploader("Upload queries data CSV", type="csv")
            target_position = st.slider(
                "Target SERP Position",
                1.0,
                10.0,
                4.0,
                0.5,
                help="This is the **average search engine ranking you assume all your queries will achieve.** A lower number (e.g., position 1) indicates a higher, more visible ranking. This target position is used to project the future click-through rate for every query."
            )
            conversion_rate = st.slider("Conversion Rate (% → signup)", 0.1, 10.0, 2.0, 0.1)
            close_rate = st.slider("Close Rate (% → customer)", 1.0, 100.0, 20.0, 1.0)
            mrr_per_customer = st.slider("MRR per Customer ($)", 10, 1000, 200, 10)
            seo_cost = st.slider("Total SEO Investment ($)", 1_000, 100_000, 10_000, 1_000)
            add_spend = st.slider("Additional Ad Spend ($)", 0, 50_000, 0, 1_000)

            sample_bytes = requests.get(SAMPLE_FILE_URL).content
            st.download_button(
                label="📥 Download sample CSV",
                data=sample_bytes,
                file_name="sample_gsc_data.csv",
                mime="text/csv",
            )
        return uploaded_file, target_position, conversion_rate, close_rate, mrr_per_customer, seo_cost, add_spend

    def _display_summary_metrics(self, metrics: dict):
        st.write("---")
        st.header("📊 SEO Performance Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Avoided Paid Spend 💰", f"${metrics['total_avoided_paid_spend']:,.2f}")
        with col2:
            st.metric("Net Savings vs Paid 📈", f"${metrics['net_savings_vs_paid']:,.2f}")
        with col3:
            st.metric("Incremental MRR (Monthly Recurring Revenue) 🚀", f"${metrics['incremental_mrr']:,.2f}")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Total Incremental Conversions 🎯", f"{metrics['total_incremental_conversions']:,.0f}")
        with col5:
            st.metric("Total Incremental Customers 🤝", f"{metrics['total_incremental_customers']:,.0f}")
        with col6:
            st.metric("SEO ROI (Return on Investment) 💰", f"{metrics['seo_roi']:.2%}")

    def _display_ad_spend_comparison(self, metrics: dict, add_spend: int):
        st.write("---")
        st.header("Hypothetical Comparison: SEO vs. Additional Ad Spend")
        col_ad1, col_ad2, col_advice = st.columns([1, 1, 1])
        with col_ad1:
            st.metric("Incremental MRR from SEO", f"${metrics['incremental_mrr']:,.2f}")
        with col_ad2:
            st.metric("Additional Ad Spend", value=f"${add_spend:,.2f}"
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

    def _display_detailed_performance_table(self, df_results: pd.DataFrame):
        st.write("---")
        st.header("Detailed Keyword Performance")
        st.info("💡 **How to use this table:** Focus on queries with the '🚀 Improvement' impact category and high 'impressions'. These represent opportunities where improving your current position towards the 'Target SERP Position' can yield significant incremental clicks and avoided paid spend.")
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

    def run(self):
        self._display_info_expander()
        uploaded_file, target_position, conversion_rate, close_rate, mrr_per_customer, seo_cost, add_spend = self._get_sidebar_inputs()

        # FIX: Call load_csv normally, Python handles the _self
        df = self.data_loader.load_csv(uploaded_file)

        if df is not None:
            # FIX: Call calculate_metrics normally, Python handles the _self
            metrics, df_results = self.seo_calculator.calculate_metrics(
                df,
                target_position,
                conversion_rate,
                close_rate,
                mrr_per_customer,
                seo_cost,
                add_spend,
            )

            if metrics is not None:
                self._display_summary_metrics(metrics)
                self._display_ad_spend_comparison(metrics, add_spend)
                self._display_detailed_performance_table(df_results)

# --- Main Application Entry Point ---
if __name__ == "__main__":
    data_loader = DataLoader()
    seo_calculator = SeoCalculator()
    app_ui = SeoAppUI(data_loader, seo_calculator)
    app_ui.run()