---
name: csv-data-summarizer
description: Analyzes CSV files, generates summary stats, and plots quick visualizations using Python and pandas. 中文：分析 CSV 文件，自动生成统计摘要和可视化图表。
metadata:
  version: 2.1.1
  author: marketplace
  adapted_by: gegeda
  dependencies: python>=3.8, pandas>=2.0.0, matplotlib>=3.7.0, seaborn>=0.12.0
---

# CSV Data Summarizer

Analyzes CSV files and provides comprehensive summaries with statistical insights and visualizations.

## When to Use

Claude should use this Skill whenever the user:
- Uploads or references a CSV file
- Asks to summarize, analyze, or visualize tabular data ("分析这个CSV", "总结数据")
- Requests insights from tabular data

## Behavior

**Auto-execute immediately — do NOT ask the user what they want.** Load the CSV, detect the data type, run the full analysis, generate all relevant charts, and present everything at once.

### Analysis Steps:

1. **Load** → pandas DataFrame
2. **Inspect** → column types, date columns, numeric columns, categories
3. **Adapt** to data type:
   - Sales/E-commerce → time-series, revenue, product performance
   - Customer → distribution, segmentation, demographics
   - Financial → trends, stats, correlations
   - Operational → metrics, distributions, time-series
   - Survey → frequency, cross-tabs, distribution
   - Generic → adapt based on column types found
4. **Visualize** → only what makes sense for this data:
   - Time-series → only if date columns exist
   - Correlation heatmap → only if multiple numeric columns
   - Category distribution → only if categorical columns
   - Histograms → for numeric distributions
5. **Output** → data overview, key stats, missing analysis, charts, actionable insights
6. **Present** → complete analysis in one response, no follow-up questions

### Do / Don't

✅ "I'll analyze this data comprehensively right now."
✅ Immediately run analysis, generate all charts, provide complete insights

❌ "What would you like to do with this data?"
❌ Offering options, asking permission, partial output requiring follow-up

## Files

- `analyze.py` — Core analysis logic (`summarize_csv(file_path)`)
- `requirements.txt` — Python dependencies
- `examples/` — Example CSV for testing

## Notes

- Auto-detects date columns (columns containing 'date' in name)
- Handles missing data gracefully
- Adapts chart types to data structure
