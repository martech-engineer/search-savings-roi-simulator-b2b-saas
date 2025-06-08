import pandas as pd
import numpy as np
import streamlit as st
import requests

# Constants
SAMPLE_FILE_URL = (
    "https://huggingface.co/spaces/Em4e/seo-b2b-saas-forecasting-tool/"
    "resolve/main/sample_keyword_data_cpc.csv"
)

# --- CACHED CSV LOADER ---
@st.cache_data
def load_csv(uploaded_file_obj: st.runtime.uploaded_file_manager.UploadedFile | None, sample_file_url: str) -> pd.DataFrame | None:
    try:
        if uploaded_file_obj:
            df = pd.read_csv(uploaded_file_obj)
        else:
            df = pd.read_csv(sample_file_url)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None
    df.columns = [col.strip().lower() for col in df.columns]
    if "cpc" not in df.columns:
        st.warning("No `cpc` column found—simulating CPC values between 0.50–3.00 USD.")
        df["cpc"] = np.round(np.random.uniform(0.5, 3.0, size=len(df)), 2)
    return df

# --- SEO Forecasting Core ---
class SeoCalculator:
    def __init__(self):
        self.ctr_benchmarks = {i: v for i, v in zip(range(1, 11), [0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01])}
        self.ctr_benchmarks.update({i: 0.005 for i in range(11, 21)})
        self.required_columns_map = {
            "query": ["query", "keyword", "queries"],
            "impressions": ["impressions"],
            "position": ["position", "avg. position", "average position"],
            "cpc": ["cpc"],
        }

    def _get_ctr(self, position: float) -> float:
        return self.ctr_benchmarks.get(int(round(position)), 0.005)

    def _validate_and_rename_columns(self, df: pd.DataFrame) -> pd.DataFrame | None:
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

    def calculate_metrics(
        self,
        df: pd.DataFrame,
        target_position: float,
        conversion_rate: float,
        close_rate: float,
        mrr_per_customer: int,
        seo_cost: int,
        add_spend: int,
    ) -> tuple[dict, pd.DataFrame] | tuple[None, pd.DataFrame]:
        df_processed = self._validate_and_rename_columns(df.copy())
        if df_processed is None:
            return None, pd.DataFrame()

        df_processed["current_ctr"] = df_processed["position"].apply(self._get_ctr)
        target_ctr_value = self._get_ctr(target_position)
        df_processed["target_ctr"] = target_ctr_value
        df_processed["current_clicks"] = df_processed["impressions"] * df_processed["current_ctr"]
        df_processed["projected_clicks"] = df_processed["impressions"] * df_processed["target_ctr"]
        df_processed["incremental_clicks"] = df_processed["projected_clicks"] - df_processed["current_clicks"]
        df_processed["avoided_paid_spend"] = df_processed["incremental_clicks"] * df_processed["cpc"]

        total_avoided_paid_spend = df_processed["avoided_paid_spend"].sum()
        net_savings_vs_paid = total_avoided_paid_spend - seo_cost
        total_incremental_conversions = df_processed["incremental_clicks"].sum() * (conversion_rate / 100)
        total_incremental_customers = total_incremental_conversions * (close_rate / 100)
        incremental_mrr = total_incremental_customers * mrr_per_customer
        seo_roi = (incremental_mrr - seo_cost) / seo_cost if seo_cost > 0 else np.inf

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

# --- Streamlit UI ---
class SeoAppUI:
    def __init__(self, seo_calculator: SeoCalculator):
        self.seo_calculator = seo_calculator
        self._set_page_config()

    def _set_page_config(self):
        st.set_page_config(page_title="SEO ROI & Savings Forecasting", layout="wide")
        st.title("Search & Savings ROI Simulator for B2B SaaS")
        st.markdown("App created by [Emilija Gjorgjevska](https://www.linkedin.com/in/emilijagjorgjevska/)")

    def _display_info_expander(self):
        with st.expander("ℹ️ How the app works [CLICK TO EXPAND]", expanded=False):
            st.markdown(
                """
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                <p>1. <b>Load your GSC data</b>. If no file is uploaded, a sample file is used. If no <code>cpc</code> is present, it will be simulated.</p>
                <p>2. <b>CTR benchmarks</b> are applied for positions 1–20 to calculate expected clicks.</p>
                <p>3. <b>Incremental Clicks = Projected - Current</b>. Financial impact is calculated from these changes.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _get_sidebar_inputs(self):
        with st.sidebar:
            st.header("🔧 Assumptions & Inputs")
            uploaded_file = st.file_uploader("Upload queries CSV data", type="csv")
            target_position = st.slider("Target SERP Position", 1.0, 10.0, 4.0, 0.5)
            conversion_rate = st.slider("Conversion Rate (% → signup)", 0.1, 10.0, 2.0, 0.1)
            close_rate = st.slider("Close Rate (% → customer)", 1.0, 100.0, 20.0, 1.0)
            mrr_per_customer = st.slider("MRR per Customer ($)", 10, 1000, 200, 10)
            seo_cost = st.slider("Total SEO Investment ($)", 1_000, 100_000, 10_000, 1_000)
            add_spend = st.slider("Additional Ad Spend ($)", 0, 50_000, 0, 1_000)
        return uploaded_file, target_position, conversion_rate, close_rate, mrr_per_customer, seo_cost, add_spend

    def _display_summary_metrics(self, metrics: dict):
        st.write("---")
        st.header("📊 SEO Performance Summary")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Avoided Paid Spend 💰", f"${metrics['total_avoided_paid_spend']:,.2f}")
        with col2: st.metric("Net Savings 📈", f"${metrics['net_savings_vs_paid']:,.2f}")
        with col3: st.metric("Incremental MRR 🚀", f"${metrics['incremental_mrr']:,.2f}")
        col4, col5, col6 = st.columns(3)
        with col4: st.metric("Conversions 🎯", f"{metrics['total_incremental_conversions']:,.0f}")
        with col5: st.metric("Customers 🤝", f"{metrics['total_incremental_customers']:,.0f}")
        with col6: st.metric("SEO ROI 💰", f"{metrics['seo_roi']:.2%}")

    def _display_ad_spend_comparison(self, metrics: dict, add_spend: int):
        st.write("---")
        st.header("Hypothetical Comparison: SEO vs. Ad Spend")
        col_ad1, col_ad2, col_advice = st.columns(3)
        with col_ad1: st.metric("Incremental MRR", f"${metrics['incremental_mrr']:,.2f}")
        with col_ad2: st.metric("Ad Spend", value=f"${add_spend:,.2f}")
        with col_advice:
            if metrics["incremental_mrr"] > add_spend:
                st.markdown("""
                <div style="background-color: #f0fff0; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 0 8px rgba(0,0,0,0.05);">
                    <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #333;">Advice</p>
                    <p style="margin: 10px 0 0 0; color: green; font-size: 2em; font-weight: bold;">SEO is a better investment!</p>
                </div>
                """, unsafe_allow_html=True)
            else:         
                st.markdown("""
                <div style="background-color: #fcd3d4; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 0 8px rgba(252, 211, 212, 1);">
                    <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #333;">Advice</p>
                    <p style="margin: 10px 0 0 0; color: #f76f72; font-size: 2em; font-weight: bold;">Consider Ad Spend.</p>
                </div>
                """, unsafe_allow_html=True)

    def _display_detailed_performance_table(self, df_results: pd.DataFrame):
        st.write("---")
        st.header("Detailed Keyword Performance")
        st.dataframe(df_results.sort_values(by="incremental_clicks", ascending=False), use_container_width=True)

    def run(self):
        self._display_info_expander()
        sample_bytes = requests.get(SAMPLE_FILE_URL).content
        st.download_button("📥 Download sample CSV", sample_bytes, file_name="sample_keyword_data_cpc.csv", mime="text/csv")
        uploaded_file, target_position, conversion_rate, close_rate, mrr_per_customer, seo_cost, add_spend = self._get_sidebar_inputs()
        df = load_csv(uploaded_file, SAMPLE_FILE_URL)
        if df is not None:
            metrics, df_results = self.seo_calculator.calculate_metrics(
                df, target_position, conversion_rate, close_rate, mrr_per_customer, seo_cost, add_spend
            )
            if metrics is not None:
                self._display_summary_metrics(metrics)
                self._display_ad_spend_comparison(metrics, add_spend)
                self._display_detailed_performance_table(df_results)

# --- App Runner ---
if __name__ == "__main__":
    calculator = SeoCalculator()
    app_ui = SeoAppUI(calculator)
    app_ui.run()