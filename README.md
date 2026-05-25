# Thailand-Cambodia Escalation Risk Dashboard
 
An 8-signal geopolitical risk model and interactive dashboard tracking political instability in Thailand and Cambodia, built for the Moonshot-Labs Analyst Jam for the Intelligence Community (MAJIC), an NGA-affiliated hackathon focused on AI-driven geopolitical risk prediction.
 
## Overview
 
This tool ingests monthly ACLED conflict event data across three categories, demonstrations, civilian targeting, and political violence, and computes a composite escalation risk score using eight precursor signals derived from cross-correlation analysis against historical escalation episodes.
 
The output is a fully self-contained interactive HTML dashboard requiring no server or build step.
 
---
## Dashboard Preview
![14 Month Composite Risk Scores](assets/14%20Month%20Composite%20Risk%20Scores.png)

![Conflict Timeline Graphs](assets/Conflict%20Timeline%20Graphs.png)

![December 2025 Heatmap](assets/December%202025%20Heatmap.png)

![Displacement Zoom Cambodia](assets/Displacement%20Zoom%20in%20December%202025%20(Cambodia).png)

![Displacement Zoom Thailand](assets/Displacement%20Zoom%20in%20December%202025%20(Thailand).png)

![Precursor Signals Graphs](assets/Precursor%20Signals%20Graphs.png)
---
## Dashboard
 
| Tab | What It Shows |
|---|---|
| Overview | Monthly events, risk scores, fatalities, 3-month momentum, AI analyst briefing |
| Geographic Map | Leaflet map with displacement event markers, animated month-by-month |
| Risk Heatmap | 14-month risk score grid per country with hover tooltips |
| Event Breakdown | Stacked bar charts by conflict category |
| Raw Data | Filterable table of all monthly metrics |
| Precursor Signals | All 8 individual signal charts + composite score |
| Methodology | Full explanation of the model, signal weights, and limitations |
 
---
 
## The 8-Signal Model
 
All signals are min-max normalized to [0, 1] and combined via weighted sigmoid scoring:
 
```
risk_score = sigmoid(5 × (weighted_sum − 0.35))
```
 
| Signal | Weight | Method | Key Finding |
|---|---|---|---|
| Civilian Targeting Shift | 15% | Civilian targeting / (demonstrations + 1), 3mo smoothed | r=0.71 lag-1 correlation with future violence |
| Violence Intensity | 15% | Fatalities per event, 3mo smoothed | Rising lethality independent of event count signals intensification |
| Fatality Acceleration | 15% | Δ in 3mo rolling fatality totals | Captures whether deaths are trending up or down |
| Event Acceleration | 15% | 1st derivative of 3mo event momentum | Positive = worsening; negative = calming |
| Anomaly Detection | 12% | IQR-based outlier vs 12mo rolling window | Robust to heavy-tailed conflict distributions |
| Volatility Regime | 10% | 6mo vs 24mo standard deviation ratio | Ratio > 1.0 = unstable period; spikes precede escalation |
| Violence-Protest Divergence | 10% | Violence z-score minus protest z-score | Positive = violence outpacing protests (suppression signal) |
| Jerk | 8% | 2nd derivative of event momentum | Earliest detectable signal of a new escalation pattern |
 
**Precursor alert** fires when three leading indicators converge simultaneously: civilian targeting above 6-month mean, volatility ratio > 1.2, and positive event acceleration.
 
---
 
## Data Sources
 
- **[ACLED](https://acleddata.com/)** — Armed Conflict Location & Event Data (monthly xlsx exports, 3 categories × 2 countries)
- **[IDMC](https://www.internal-displacement.org/)** — Internal Displacement Monitoring Centre (event-level CSV with lat/lng)
- **[Google Gemini API](https://ai.google.dev/)** — Optional AI analyst narrative (requires API key)
---
 
## Setup
 
```bash
# Clone
git clone https://github.com/yourusername/risk-dashboard.git
cd risk-dashboard
 
# Install dependencies
pip install -r requirements.txt
 
# Place your data files in the project root
# (ACLED xlsx exports + IDMC CSVs — filenames configured in DATASETS at top of script)
 
# Optional: enable AI analyst summary
export GEMINI_API_KEY=your_key_here
 
# Run
python3 risk_dashboard.py
```
 
Open `risk_dashboard_outputs/risk_dashboard.html` in your browser. No server required.
 
---
 
## Project Structure
 
```
risk-dashboard/
├── risk_dashboard.py          # Main script — data loading, modeling, dashboard generation
├── requirements.txt           # Python dependencies
├── README.md
├── .gitignore                 # Excludes API keys, data files, and outputs
└── risk_dashboard_outputs/    # Generated dashboard (git-ignored)
    └── risk_dashboard.html
```
 
---
 
## Security
 
API keys are loaded exclusively from environment variables, never hardcoded.
 
```bash
export GEMINI_API_KEY=your_key_here
```
 
If the key is not set, the dashboard runs normally with an AI summary placeholder.
 
---
 
## Limitations
 
This is a quantitative momentum model, not a full geopolitical intelligence assessment. It does not account for diplomatic context, election cycles, or media coverage patterns. Spikes in ACLED reporting backlogs can produce false signals. The model is best used as a screening tool, not as a standalone prediction engine.
 
---
