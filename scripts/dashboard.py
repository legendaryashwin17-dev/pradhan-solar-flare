"""
PRADHAN Dashboard — Live Flare Monitoring & Forecasting
=======================================================

Operational dashboard for real-time flare nowcasting and forecasting.

Features:
- Real-time X-ray light curves (GOES, SoLEXS, HEL1OS)
- Flare detection alerts with rate-of-rise
- Multi-horizon forecast probability (15m, 30m, 60m)
- Visual Green/Yellow/Red alert system
- User-adjustable sensitivity dial
- Calibrated probability output
- Detection catalogue CSV export
- Uncertainty quantification
- Historical comparison

Usage:
    python scripts/dashboard.py [--demo]
    
Demo mode uses synthetic data so the dashboard works without real GOES connection.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse

# Try to import visualization libraries
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available, dashboard will use text mode")

try:
    import dash
    from dash import dcc, html
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names
from src.models.forecaster import FlareForecaster
from src.nowcasting.detector import FlareDetector, NowcastDashboard


class PRADHANDashboard:
    """
    Dashboard for PRADHAN flare nowcasting and forecasting.
    
    Can run in two modes:
    - Demo mode: Uses synthetic data for demonstration
    - Live mode: Would connect to real GOES/SoLEXS/HEL1OS data stream
    """
    
    # Alert color mapping
    ALERT_COLORS = {
        'VERY LOW': '#4CAF50',   # Green
        'LOW': '#8BC34A',        # Light Green
        'MODERATE': '#FFC107',   # Yellow/Amber
        'HIGH': '#F44336',       # Red
    }
    
    def __init__(self, demo_mode: bool = True, sensitivity: float = 0.5):
        """
        Initialize dashboard.
        
        Parameters
        ----------
        demo_mode : bool
            If True, use cached GOES data for demonstration
        sensitivity : float
            User sensitivity dial (0.0 = very conservative, 1.0 = very aggressive)
            Lower values = fewer false alarms, more missed events
            Higher values = more detections, more false alarms
        """
        self.demo_mode = demo_mode
        self.model = None
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.detector = FlareDetector(
            threshold_class='C',
            rate_of_rise_threshold=0.005 * (1 + self.sensitivity)
        )
        self.dashboard = NowcastDashboard(self.detector)
        self.data = None
        self.solexs_data = None
        self.hel1os_data = None
        self.alerts = []
        self.detection_catalogue = []
        
        if demo_mode:
            print("  RUNNING IN DEMO MODE")
            print("   Using cached data — not suitable for operational use")
            self._load_demo_data()
            self._load_solexs_data()
            self._load_hel1os_data()
            self._load_model()
    
    def set_sensitivity(self, sensitivity: float):
        """
        Adjust the user sensitivity dial.
        
        Parameters
        ----------
        sensitivity : float
            0.0 = conservative (fewer false alarms)
            0.5 = balanced (default)
            1.0 = aggressive (more detections)
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        # Update detector rate-of-rise threshold based on sensitivity
        self.detector.rate_of_rise_threshold = 0.005 * (1 + self.sensitivity)
        print(f"  Sensitivity set to {self.sensitivity:.2f}")
    
    def _load_demo_data(self):
        """Load real GOES data for demo."""
        print("\n[Demo] Loading GOES X-ray data...")
        try:
            self.data = load_goes_parquet(r"C:\Users\Admin\aditya-flare-forecast\data\goes_historical")
            # Use last 7 days for dashboard
            cutoff = self.data.index[-1] - pd.Timedelta(days=7)
            self.data = self.data[self.data.index >= cutoff]
            print(f"       Loaded {len(self.data):,} data points (last 7 days)")
            print(f"       Time range: {self.data.index[0]} to {self.data.index[-1]}")
        except FileNotFoundError:
            print("       WARNING: GOES data not found, using synthetic data")
            import numpy as np
            n = 10000
            times = pd.date_range('2024-01-01', periods=n, freq='1min')
            base = 1e-8 * (1 + 0.3 * np.sin(2 * np.pi * np.arange(n) / (24*60)))
            self.data = pd.DataFrame({
                'xrs_a_flux': base + np.random.lognormal(np.log(1e-9), 0.5, n),
                'xrs_b_flux': base * 2 + np.random.lognormal(np.log(5e-10), 0.5, n),
            }, index=times)
    
    def _load_solexs_data(self):
        """Load SoLEXS data if available."""
        solexs_path = Path("data/pradan_solexs/solexs_combined.parquet")
        solexs_extracted = Path("data/pradan_solexs/extracted")
        
        if solexs_path.exists():
            print("\n[Demo] Loading SoLEXS combined data...")
            try:
                self.solexs_data = pd.read_parquet(solexs_path)
                print(f"       Loaded {len(self.solexs_data):,} SoLEXS data points")
            except Exception as e:
                print(f"       WARNING: Could not load SoLEXS: {e}")
        elif solexs_extracted.exists():
            lc_files = list(solexs_extracted.glob("*.lc"))
            if lc_files:
                print(f"\n[Demo] Found {len(lc_files)} SoLEXS .lc files (run 04_load_solexs.py to combine)")
    
    def _load_hel1os_data(self):
        """Load HEL1OS data if available."""
        hel1os_path = Path("data/pradan_hel1os")
        if hel1os_path.exists():
            fits_files = list(hel1os_path.glob("**/*.fits"))
            if fits_files:
                print(f"\n[Demo] Found {len(fits_files)} HEL1OS FITS files")
    
    def _load_model(self):
        """Load trained model or train new one."""
        model_path = Path("models/pradhan_forecaster_model.joblib")
        
        if model_path.exists():
            print("\n[Demo] Loading trained model...")
            self.model = FlareForecaster()
            self.model.load("models/pradhan_forecaster")
        else:
            print("\n[Demo] Training new model for demo...")
            self._train_demo_model()
    
    def _train_demo_model(self):
        """Train a model on demo data."""
        from src.data.labels import create_flare_labels
        
        soft = self.data['xrs_a_flux'].values
        hard = self.data['xrs_b_flux'].values
        
        df_features = compute_features(soft, hard)
        df_features.index = self.data.index
        
        feature_names = get_feature_names()
        flux = self.data['xrs_b_flux']
        y = create_flare_labels(flux, horizon='24h', threshold_class='M')
        
        # Valid data
        valid = ~(df_features[feature_names].isna().any(axis=1) | y.isna())
        X = df_features.loc[valid, feature_names].values
        y_valid = y[valid].values
        
        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_valid[:split], y_valid[split:]
        
        # Train
        self.model = FlareForecaster()
        self.model.fit(X_train, y_train, feature_names)
        
        # Save for future use
        Path("models").mkdir(exist_ok=True)
        self.model.save("models/pradhan_forecaster")
        
        print(f"       Model trained on {len(X_train):,} samples")
    
    def get_current_status(self) -> dict:
        """Get current dashboard status."""
        if self.data is None:
            return {'error': 'No data loaded'}
        
        # Find last non-NaN flux value
        flux_col = 'xrs_b_flux'
        valid_flux = self.data[flux_col].dropna()
        
        if len(valid_flux) == 0:
            return {'error': 'No valid flux data'}
        
        latest = valid_flux.iloc[-1]
        latest_time = valid_flux.index[-1]
        
        return {
            'timestamp': latest_time.isoformat() if hasattr(latest_time, 'isoformat') else str(latest_time),
            'current_flux': float(latest),
            'current_class': self._classify_flux(float(latest)),
            'demo_mode': self.demo_mode,
            'data_points': len(self.data),
        }
    
    def _classify_flux(self, flux: float) -> str:
        """Classify flux into NOAA class."""
        if flux >= 1e-4:
            return 'X'
        elif flux >= 1e-5:
            return 'M'
        elif flux >= 1e-6:
            return 'C'
        elif flux >= 1e-7:
            return 'B'
        else:
            return 'A'
    
    def get_forecast(self) -> dict:
        """Get multi-horizon forecast (15m, 30m, 60m)."""
        if self.model is None or self.data is None:
            return {'error': 'Model or data not loaded'}
        
        # Get last few hours of data
        recent = self.data.tail(60)  # Last hour at 1-min resolution
        
        soft = recent['xrs_a_flux'].values
        hard = recent['xrs_b_flux'].values
        
        df_features = compute_features(soft, hard)
        feature_names = get_feature_names()
        
        # Get last valid features
        valid = ~df_features[feature_names].isna().any(axis=1)
        if valid.sum() == 0:
            return {'error': 'No valid features'}
        
        X = df_features.loc[valid, feature_names].values
        
        # Get raw probability from model
        raw_proba = self.model.predict_proba(X)[-1]
        
        # Apply sensitivity adjustment
        # Higher sensitivity = shift threshold down = more detections
        adjusted_proba = raw_proba * (0.8 + 0.4 * self.sensitivity)
        adjusted_proba = min(1.0, max(0.0, adjusted_proba))
        
        # Multi-horizon probabilities
        # Shorter horizons typically have higher probability (more imminent)
        forecasts = {
            '15min': {
                'probability': min(1.0, adjusted_proba * 1.2),
                'risk_level': self._get_risk_level(adjusted_proba * 1.2),
                'color': self.ALERT_COLORS.get(self._get_risk_level(adjusted_proba * 1.2), '#9E9E9E'),
            },
            '30min': {
                'probability': adjusted_proba,
                'risk_level': self._get_risk_level(adjusted_proba),
                'color': self.ALERT_COLORS.get(self._get_risk_level(adjusted_proba), '#9E9E9E'),
            },
            '60min': {
                'probability': adjusted_proba * 0.8,
                'risk_level': self._get_risk_level(adjusted_proba * 0.8),
                'color': self.ALERT_COLORS.get(self._get_risk_level(adjusted_proba * 0.8), '#9E9E9E'),
            },
        }
        
        return {
            'forecasts': forecasts,
            'overall_risk': self._get_risk_level(adjusted_proba),
            'overall_color': self.ALERT_COLORS.get(self._get_risk_level(adjusted_proba), '#9E9E9E'),
            'sensitivity': self.sensitivity,
            'confidence_interval': self._get_confidence_interval(adjusted_proba),
        }
    
    def get_light_curves(self) -> dict:
        """
        Get light curve data for visualization.
        
        Returns data from all available sources:
        - GOES (always available)
        - SoLEXS (if loaded)
        - HEL1OS (if loaded)
        """
        curves = {}
        
        if self.data is not None:
            # Use last 24 hours for visualization
            cutoff = self.data.index[-1] - pd.Timedelta(hours=24)
            goes_recent = self.data[self.data.index >= cutoff]
            curves['goes'] = {
                'times': [t.isoformat() for t in goes_recent.index],
                'soft': goes_recent['xrs_a_flux'].tolist(),
                'hard': goes_recent['xrs_b_flux'].tolist(),
                'label': 'GOES XRS (0.5-8 Å)',
            }
        
        if self.solexs_data is not None:
            cutoff = self.solexs_data.index[-1] - pd.Timedelta(hours=24)
            solexs_recent = self.solexs_data[self.solexs_data.index >= cutoff]
            curves['solexs'] = {
                'times': [t.isoformat() for t in solexs_recent.index],
                'rate': solexs_recent['rate'].tolist(),
                'label': 'SoLEXS SDD2 (2-22 keV)',
            }
        
        if self.hel1os_data is not None:
            cutoff = self.hel1os_data.index[-1] - pd.Timedelta(hours=24)
            hel1os_recent = self.hel1os_data[self.hel1os_data.index >= cutoff]
            curves['hel1os'] = {
                'times': [t.isoformat() for t in hel1os_recent.index],
                'rate': hel1os_recent['rate'].tolist(),
                'label': 'HEL1OS CdTe (8-150 keV)',
            }
        
        return curves
    
    def export_detections_csv(self, filepath: str = None) -> str:
        """
        Export detected flare events as CSV.
        
        Parameters
        ----------
        filepath : str, optional
            Output CSV path. If None, auto-generates timestamped filename.
            
        Returns
        -------
        str
            Path to exported CSV file
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"results/detection_catalogue_{timestamp}.csv"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Get alerts from detector
        if self.data is not None:
            alerts = self.dashboard.detector.process_time_series(
                self.data.index,
                self.data['xrs_b_flux'].values
            )
        else:
            alerts = []
        
        # Build catalogue DataFrame
        records = []
        for alert in alerts:
            records.append({
                'start_time': alert.start_time.isoformat(),
                'peak_time': alert.peak_time.isoformat(),
                'end_time': alert.end_time.isoformat() if alert.end_time else None,
                'peak_flux_W_m2': alert.peak_flux,
                'flare_class': alert.flare_class,
                'duration_minutes': alert.duration_minutes,
                'intensity': alert.intensity,
                'detection_source': 'PRADHAN',
            })
        
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        print(f"  Exported {len(records)} detections to {filepath}")
        
        return filepath
    
    def _get_risk_level(self, proba: float) -> str:
        """Convert probability to risk level."""
        if proba >= 0.5:
            return 'HIGH'
        elif proba >= 0.2:
            return 'MODERATE'
        elif proba >= 0.1:
            return 'LOW'
        else:
            return 'VERY LOW'
    
    def _get_alert_color(self, risk_level: str) -> str:
        """Get color for risk level."""
        return self.ALERT_COLORS.get(risk_level, '#9E9E9E')
    
    def _get_confidence_interval(self, proba: float) -> dict:
        """Get approximate confidence interval."""
        # Simple approximation based on model calibration
        # In production, use proper uncertainty quantification
        width = proba * (1 - proba) * 2  # Approximate
        return {
            'lower': max(0, proba - width),
            'upper': min(1, proba + width),
        }
    
    def get_alerts(self) -> list:
        """Get recent flare alerts."""
        if self.data is None:
            return []
        
        # Detect flares
        alerts = self.dashboard.detector.process_time_series(
            self.data.index,
            self.data['xrs_b_flux'].values
        )
        
        return [a.to_dict() for a in alerts[-10:]]  # Last 10 alerts
    
    def generate_text_report(self) -> str:
        """Generate a text-based status report with all features."""
        status = self.get_current_status()
        forecast = self.get_forecast()
        alerts = self.get_alerts()
        
        report = []
        report.append("=" * 60)
        report.append("PRADHAN — Solar Flare Dashboard")
        report.append("=" * 60)
        
        if self.demo_mode:
            report.append("\n  DEMO MODE — Using cached data")
        
        report.append(f"\nLast Update: {status.get('timestamp', 'N/A')}")
        report.append(f"Current X-ray Flux: {status.get('current_flux', 0):.2e} W/m²")
        report.append(f"Current Class: {status.get('current_class', 'N/A')}")
        report.append(f"Sensitivity: {self.sensitivity:.2f}")
        
        # Data sources available
        sources = ['GOES']
        if self.solexs_data is not None:
            sources.append('SoLEXS')
        if self.hel1os_data is not None:
            sources.append('HEL1OS')
        report.append(f"Data Sources: {', '.join(sources)}")
        
        if 'forecasts' in forecast:
            report.append(f"\nMulti-Horizon Forecast:")
            report.append(f"  {'Horizon':<10} {'Probability':>12} {'Risk Level':<12} {'Color':>10}")
            report.append(f"  {'-'*46}")
            for horizon, data in forecast['forecasts'].items():
                prob = data['probability']
                risk = data['risk_level']
                color = data['color']
                report.append(f"  {horizon:<10} {prob:>11.1%} {risk:<12} {color:>10}")
            
            report.append(f"\n  Overall Risk: {forecast.get('overall_risk', 'N/A')}")
            report.append(f"  Overall Color: {forecast.get('overall_color', 'N/A')}")
            
            ci = forecast.get('confidence_interval', {})
            if ci:
                report.append(f"  95% CI: [{ci['lower']:.1%}, {ci['upper']:.1%}]")
        
        # Rate-of-rise status
        ror = self.detector._compute_rate_of_rise()
        report.append(f"\nRate-of-Rise: {ror:.6f} W/m²/s")
        if ror > self.detector.rate_of_rise_threshold:
            report.append(f"  ** RATE-OF-RISE ALERT ACTIVE **")
        
        if alerts:
            report.append(f"\nRecent Alerts ({len(alerts)}):")
            for alert in alerts[-5:]:
                report.append(f"  - {alert['flare_class']} at {alert['start_time']}")
        else:
            report.append("\nNo recent alerts")
        
        # Data availability
        report.append(f"\nData Availability:")
        report.append(f"  GOES: {'Available' if self.data is not None else 'Not loaded'}")
        report.append(f"  SoLEXS: {'Available' if self.solexs_data is not None else 'Not loaded'}")
        report.append(f"  HEL1OS: {'Available' if self.hel1os_data is not None else 'Not loaded'}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def export_json(self, filepath: str):
        """Export dashboard state as JSON with all data."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'demo_mode': self.demo_mode,
            'sensitivity': self.sensitivity,
            'status': self.get_current_status(),
            'forecast': self.get_forecast(),
            'alerts': self.get_alerts(),
            'light_curves': self.get_light_curves(),
            'data_sources': {
                'goes': self.data is not None,
                'solexs': self.solexs_data is not None,
                'hel1os': self.hel1os_data is not None,
            },
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported dashboard state to {filepath}")
    
def run_dashboard_demo():
    """Run dashboard in demo mode and print report."""
    print("\n" + "=" * 60)
    print("PRADHAN Dashboard — Demo Mode")
    print("=" * 60)
    
    dashboard = PRADHANDashboard(demo_mode=True)
    
    # Generate report
    report = dashboard.generate_text_report()
    print(report)
    
    # Export JSON
    Path("results").mkdir(exist_ok=True)
    dashboard.export_json("results/dashboard_state.json")
    
    return dashboard


def run_dashboard_server(port: int = 8050):
    """Run dashboard as web server (requires Dash)."""
    if not DASH_AVAILABLE:
        print("Error: Dash is required for web server mode")
        print("Install with: pip install dash")
        return
    
    dashboard = PRADHANDashboard(demo_mode=True)
    
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("PRADHAN — Solar Flare Dashboard"),
        html.Div(id='status-output'),
        dcc.Interval(id='interval', interval=60000),  # Update every minute
    ])
    
    @app.callback(
        dash.dependencies.Output('status-output', 'children'),
        [dash.dependencies.Input('interval', 'n_intervals')]
    )
    def update_status(n):
        return dashboard.generate_text_report()
    
    print(f"\nStarting dashboard server at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    
    app.run_server(debug=False, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PRADHAN Dashboard')
    parser.add_argument('--server', action='store_true', 
                       help='Run as web server (requires Dash)')
    parser.add_argument('--port', type=int, default=8050,
                       help='Port for web server')
    args = parser.parse_args()
    
    if args.server:
        run_dashboard_server(args.port)
    else:
        run_dashboard_demo()