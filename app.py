import pandas as pd
import gradio as gr

# ✅ Direct raw URL from Hugging Face blob to raw
SAMPLE_FILE_URL = "https://huggingface.co/spaces/Em4e/seo-b2b-saas-forecasting-tool/raw/main/sample_gsc_data.csv"

def calculate_seo_roi(
    gsc_file,
    target_position: float,
    conversion_rate_percent: float,
    close_rate_percent: float,
    mrr_per_customer: float,
    seo_cost: float
):
    empty_df = pd.DataFrame()

    try:
        if gsc_file is None or isinstance(gsc_file, bool):
            df = pd.read_csv(SAMPLE_FILE_URL)
        else:
            df = pd.read_csv(gsc_file.name)

        conversion_rate = conversion_rate_percent / 100
        close_rate = close_rate_percent / 100

        df_columns_lower = {col.lower(): col for col in df.columns}
        expected_cols_variations = {
            'query': ['query', 'queries', 'keyword', 'keywords'],
            'page': ['page', 'pages', 'landing page', 'landing_page', 'url'],
            'impressions': ['impressions'],
            'position': ['position', 'avg. position', 'average position']
        }

        found_cols_map = {}
        for internal_name, variations in expected_cols_variations.items():
            for var in variations:
                if var in df_columns_lower:
                    found_cols_map[internal_name] = df_columns_lower[var]
                    break
            else:
                if internal_name in ['query', 'impressions', 'position']:
                    return (f"Missing critical column for {internal_name}. Found: {df.columns.tolist()}", "", "", "", "", empty_df)

        df_processed = df.copy()
        rename_dict = {found_cols_map[k]: k for k in found_cols_map}
        df_processed.rename(columns=rename_dict, inplace=True)

        ctr_benchmarks = {
            i: v for i, v in zip(range(1, 11), [0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01])
        }
        ctr_benchmarks.update({i: 0.005 for i in range(11, 21)})

        def get_ctr(pos):
            return ctr_benchmarks.get(int(round(pos)), 0.005)

        df_filtered = df_processed[(df_processed['position'] >= 5) & (df_processed['position'] <= 20)].copy()
        if df_filtered.empty:
            return ("No keywords in positions 5–20.", "", "", "", "", empty_df)

        df_filtered['Current_CTR'] = df_filtered['position'].apply(get_ctr)
        df_filtered['Target_CTR'] = get_ctr(target_position)
        df_filtered['Projected_Clicks'] = df_filtered['impressions'] * df_filtered['Target_CTR']
        df_filtered['Current_Clicks'] = df_filtered['impressions'] * df_filtered['Current_CTR']
        df_filtered['Incremental_Clicks'] = df_filtered['Projected_Clicks'] - df_filtered['Current_Clicks']
        df_filtered = df_filtered[df_filtered['Incremental_Clicks'] > 0]

        if df_filtered.empty:
            return ("No incremental clicks projected. Try different assumptions.", "", "", "", "", empty_df)

        df_filtered['Signups'] = df_filtered['Incremental_Clicks'] * conversion_rate
        df_filtered['New_Customers'] = df_filtered['Signups'] * close_rate
        df_filtered['Incremental_MRR'] = df_filtered['New_Customers'] * mrr_per_customer

        total_clicks = df_filtered['Incremental_Clicks'].sum()
        total_signups = df_filtered['Signups'].sum()
        total_customers = df_filtered['New_Customers'].sum()
        total_mrr = df_filtered['Incremental_MRR'].sum()
        roi = float('inf') if seo_cost == 0 else ((total_mrr - seo_cost) / seo_cost) * 100

        output_df = df_filtered[['query', 'Incremental_MRR']].copy()
        output_df.rename(columns={'query': 'Keyword', 'Incremental_MRR': 'Projected Incremental MRR ($)'}, inplace=True)

        def label_impact(mrr):
            if mrr >= 2000:
                return "High ROI"
            elif mrr >= 500:
                return "Moderate ROI"
            else:
                return "Low Priority"

        output_df['Business Impact'] = output_df['Projected Incremental MRR ($)'].apply(label_impact)

        sort_priority = {"High ROI": 0, "Moderate ROI": 1, "Low Priority": 2}
        output_df['__sort__'] = output_df['Business Impact'].map(sort_priority)
        output_df.sort_values(by=['__sort__', 'Projected Incremental MRR ($)'], ascending=[True, False], inplace=True)
        output_df.drop(columns='__sort__', inplace=True)

        return (
            f"{total_clicks:,.0f}",
            f"{total_signups:,.1f}",
            f"{total_customers:,.1f}",
            f"${total_mrr:,.2f}",
            f"{roi:,.2f}%",
            output_df
        )

    except Exception as e:
        return (f"Error: {e}", "", "", "", "", empty_df)


# Gradio UI components
input_gsc_file = gr.File(label="Upload Google Search Console CSV Export", file_types=[".csv"])
input_target_position = gr.Slider(minimum=1, maximum=10, step=0.5, value=4, label="Target SERP Position")
input_conversion_rate = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, value=2.0, label="Conversion Rate (Visitor to Signup %)")
input_close_rate = gr.Slider(minimum=1.0, maximum=100.0, step=1.0, value=20.0, label="Close Rate (Signup to Customer %)")
input_mrr = gr.Slider(minimum=10, maximum=1000, step=10, value=200, label="MRR per Customer ($)")
input_cost = gr.Slider(minimum=1000, maximum=100000, step=1000, value=10000, label="Total SEO Investment ($)")

output_clicks = gr.Textbox(label="Incremental Clicks")
output_signups = gr.Textbox(label="Projected Signups")
output_customers = gr.Textbox(label="Projected New Customers")
output_mrr = gr.Textbox(label="Projected Incremental MRR")
output_roi = gr.Textbox(label="SEO ROI")
output_df = gr.Dataframe(label="Opportunity Keywords & Business Impact", row_count=(5, "dynamic"), interactive=False)

app = gr.Interface(
    fn=calculate_seo_roi,
    inputs=[
        input_gsc_file, input_target_position, input_conversion_rate,
        input_close_rate, input_mrr, input_cost
    ],
    outputs=[
        output_clicks, output_signups, output_customers,
        output_mrr, output_roi, output_df
    ],
    title="SEO ROI Forecasting Tool for B2B SaaS",
    description="""
    <h3>📊 How This Tool Works:</h3>
    <p>This tool helps B2B SaaS teams translate SEO performance into financial impact. It acts as a 'what-if' planner to estimate how better keyword rankings can drive leads, MRR, and overall return on investment.</p>

    <ul>
    <li><b><span style='color: blue;'>1. Upload Your Data (or use default):</span></b> If you don’t upload a CSV, the app uses a sample file automatically: <a href='https://huggingface.co/spaces/Em4e/seo-b2b-saas-forecasting-tool/blob/main/sample_gsc_data.csv' target='_blank'>sample_gsc_data.csv</a></li>
    <li><b><span style='color: blue;'>2. Set Your Assumptions:</span></b> Define funnel conversion rates, MRR, and SEO budget.</li>
    <li><b><span style='color: blue;'>3. Identify Opportunities:</span></b> The tool filters keywords ranked between positions 5–20 and forecasts click + revenue uplift if improved.</li>
    <li><b><span style='color: blue;'>4. Prioritize by ROI:</span></b> Keywords are labeled with business impact and sorted by MRR gain.</li>
    </ul>

    <h4>Assumptions:</h4>
    <ul>
    <li><b>CTR Benchmarks:</b> Position 1: 25%, Position 2: 15%, ..., Position 10: 1%, beyond 10: 0.5%</li>
    <li><b>Conversion Rates:</b> Enter as percentages (e.g., 2 for 2%)</li>
    </ul>
    """
)

if __name__ == "__main__":
    app.launch(share=True)
