# Thailand-Cambodia-Risk-Dashboard
An 8-signal geopolitical risk model and interactive dashboard tracking political instability in Thailand and Cambodia using ACLED conflict data and IDMC displacement data.
Built for the Moonshot-Labs Analyst Jam for the Intelligence Community (MAJIC) (an NGA-affiliated hackathon focused on AI-driven geopolitical risk prediction)

What It Does
This tool ingests monthly ACLED event data across three categories (demonstrations, civilian targeting, political violence) and computes a composite escalation risk score using eight precursor signals. The output is an interactive HTML dashboard with charts, heatmaps, a Leaflet geographic map, and an optional AI analyst briefing powered by the Gemini API.

The 8-Signal Model
SignalWeightDescriptionCivilian Targeting Shift15%Civilian targeting / demonstrations ratio (r=0.71 lag-1 predictor)Violence Intensity15%Fatalities per event, 3-month smoothedFatality Acceleration15%Change in 3-month rolling fatality totalsEvent Acceleration15%1st derivative of 3-month event momentumAnomaly (IQR)12%Robust outlier detection vs 12-month rolling windowVolatility Regime10%6-month vs 24-month standard deviation ratioViolence-Protest Divergence10%Suppression indicator (violence z-score minus protest z-score)Jerk8%2nd derivative of event momentum
All signals are min-max normalized to [0, 1] and combined via weighted sigmoid scoring.

Data Sources

ACLED — Armed Conflict Location & Event Data (monthly xlsx exports)
IDMC — Internal Displacement Monitoring Centre (event-level CSV)


Setup
bash# Clone the repo
git clone https://github.com/yourusername/risk-dashboard.git
cd risk-dashboard

# Install dependencies
pip install -r requirements.txt

# Add your data files to the project directory
# (ACLED xlsx files + IDMC CSVs — see DATASETS config in risk_dashboard.py)

# Optional: enable AI analyst summary
export GEMINI_API_KEY=your_key_here

# Run
python3 risk_dashboard.py
Open risk_dashboard_outputs/risk_dashboard.html in your browser.

Dashboard Tabs

Overview — Monthly events, risk scores, fatalities, 3-month momentum
Geographic Map — Leaflet map with displacement event markers, animated by month
Risk Heatmap — 14-month risk score grid with hover tooltips
Event Breakdown — Stacked bar charts by conflict category
Raw Data — Filterable table of all monthly metrics
Precursor Signals — All 8 individual signal charts + composite score
Methodology — Full explanation of the model and its limitations


Security Note
Never hardcode API keys. This project uses environment variables exclusively:
bashexport GEMINI_API_KEY=your_key_here

Limitations
This is a quantitative momentum model, not a full geopolitical assessment. It does not account for diplomatic context, election cycles, or media coverage patterns. Spikes in ACLED reporting backlogs can create false signals. Best used as a screening tool, not as a prediction engine.
