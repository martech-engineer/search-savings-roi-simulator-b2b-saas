---
title: SEO ROI Forecaster
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---


# 📊 SEO ROI Forecasting Tool for B2B SaaS

This tool helps growth and marketing leaders at B2B SaaS companies translate **organic keyword improvements** into **forecasted revenue** and **ROI**.

## 🔍 How it Works

1. **Upload your Google Search Console (GSC) export**  
   The file should contain columns like `query`, `impressions`, and `position`.

2. **Set your business assumptions**
   - Target ranking position (e.g., position 4)
   - Conversion rate: visitor → lead
   - Close rate: lead → customer
   - Monthly Recurring Revenue (MRR) per customer
   - Your total SEO investment

3. **Opportunity Identification**  
   The app focuses on keywords ranking in **positions 5–20**, simulating gains if they were improved to the target position.

4. **Business Impact Forecasting**  
   Using CTR benchmarks, the tool estimates:
   - Additional traffic
   - New leads and customers
   - Incremental MRR
   - ROI based on your SEO cost

5. **Prioritized Output**  
   Keywords are labeled as **High ROI**, **Moderate ROI**, or **Low Priority**, sorted by forecasted MRR.

---

## 📁 Example Input Format

```csv
query,impressions,position
b2b saas seo strategy,1234,11.4
keyword intent model,845,8.9
