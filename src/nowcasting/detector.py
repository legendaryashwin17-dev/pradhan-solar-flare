"""
PRADHAN Nowcasting — Threshold-Based Detection
===============================================

Simple threshold-based flare detection for operational nowcasting.

This is NOT a machine learning model — it's a rule-based system
that alerts when X-ray flux exceeds defined thresholds.

For operational use, this complements the ML forecast by providing
real-time alerts for ongoing events.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class FlareAlert:
    """Represents a detected flare event."""
    start_time: datetime
    peak_time: datetime
    end_time: Optional[datetime]
    peak_flux: float
    flare_class: str
    duration_minutes: Optional[float]
    intensity: str  # 'gradual', 'impulsive'
    
    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time.isoformat(),
            'peak_time': self.peak_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'peak_flux': self.peak_flux,
            'flare_class': self.flare_class,
            'duration_minutes': self.duration_minutes,
            'intensity': self.intensity,
        }


class FlareDetector:
    """
    Threshold-based flare detection for real-time nowcasting.
    
    Uses fixed flux thresholds to detect ongoing flare events.
    This is a simple but robust approach used in operational systems.
    
    NOTE: This detects events, it doesn't predict them.
    Use the ML model for prediction.
    """
    
    # NOAA flare class thresholds (W/m²)
    THRESHOLDS = {
        'A': 1e-8,
        'B': 1e-7,
        'C': 1e-6,
        'M': 1e-5,
        'X': 1e-4,
    }
    
    def __init__(
        self,
        threshold_class: str = 'C',
        min_duration_minutes: int = 1,
        cooldown_minutes: int = 10,
        rate_of_rise_threshold: float = 0.01,
        rate_of_rise_window: int = 5
    ):
        """
        Initialize detector.
        
        Parameters
        ----------
        threshold_class : str
            Minimum class to detect ('C', 'M', or 'X')
        min_duration_minutes : int
            Minimum duration to count as an event
        cooldown_minutes : int
            Minimum time between alerts for same class
        rate_of_rise_threshold : float
            Minimum rate of rise (W/m^2 per second) to trigger detection.
            A value of 0.01 means flux must increase by 1e-2 W/m^2/s.
            Set to 0 to disable rate-of-rise detection.
        rate_of_rise_window : int
            Number of samples to average for rate-of-rise calculation
        """
        self.threshold_flux = self.THRESHOLDS.get(threshold_class, 1e-6)
        self.threshold_class = threshold_class
        self.min_duration = timedelta(minutes=min_duration_minutes)
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.rate_of_rise_threshold = rate_of_rise_threshold
        self.rate_of_rise_window = rate_of_rise_window
        
        self.last_alert_time: Optional[datetime] = None
        self.last_alert_class: Optional[str] = None
        self.current_event: Optional[FlareAlert] = None
        self._flux_history: List[float] = []
        self._time_history: List[datetime] = []
        
    def classify_flare(self, flux: float) -> str:
        """Classify flux into NOAA class."""
        if flux >= self.THRESHOLDS['X']:
            return 'X'
        elif flux >= self.THRESHOLDS['M']:
            return 'M'
        elif flux >= self.THRESHOLDS['C']:
            return 'C'
        elif flux >= self.THRESHOLDS['B']:
            return 'B'
        else:
            return 'A'
    
    def _compute_rate_of_rise(self) -> float:
        """
        Compute the rate of rise of X-ray flux.
        
        Uses the last `rate_of_rise_window` samples to estimate
        the rate of change (W/m^2 per second).
        
        Returns
        -------
        float
            Rate of rise in W/m^2 per second. Positive = rising.
        """
        if len(self._flux_history) < 2 or len(self._time_history) < 2:
            return 0.0
        
        n = min(self.rate_of_rise_window, len(self._flux_history))
        recent_flux = self._flux_history[-n:]
        recent_time = self._time_history[-n:]
        
        if n < 2:
            return 0.0
        
        # Compute time differences in seconds
        dt_seconds = (recent_time[-1] - recent_time[0]).total_seconds()
        if dt_seconds <= 0:
            return 0.0
        
        # Linear regression slope for rate of rise
        df = recent_flux[-1] - recent_flux[0]
        return df / dt_seconds
    
    def _is_rate_of_rise_detected(self) -> bool:
        """
        Check if rate-of-rise detection criteria are met.
        
        Rate-of-rise detection triggers when:
        1. Flux is above a fraction of the threshold (pre-threshold)
        2. The rate of rise exceeds the configured threshold
        
        This provides EARLIER detection than pure threshold crossing.
        """
        if self.rate_of_rise_threshold <= 0:
            return False  # Rate-of-rise detection disabled
        
        if len(self._flux_history) < self.rate_of_rise_window:
            return False
        
        current_flux = self._flux_history[-1]
        ror = self._compute_rate_of_rise()
        
        # Pre-threshold detection: trigger at 70% of threshold if rising fast
        pre_threshold = self.threshold_flux * 0.7
        
        return (current_flux >= pre_threshold) and (ror >= self.rate_of_rise_threshold)
    
    def get_flare_subclass(self, flux: float, base_class: str) -> str:
        """Get numeric subclass (e.g., 'M2.5')."""
        if base_class == 'A':
            return 'A'
        
        threshold = self.THRESHOLDS[base_class]
        value = flux / threshold
        
        if value >= 10:
            # Cap at X9.9
            return f"{base_class}9.9"
        
        return f"{base_class}{value:.1f}"
    
    def detect_event(
        self,
        time: datetime,
        flux: float,
        flux_history: Optional[List[float]] = None
    ) -> Optional[FlareAlert]:
        """
        Detect if a flare event is occurring.
        
        Uses two complementary detection methods:
        1. Threshold crossing: flux exceeds NOAA class threshold
        2. Rate-of-rise: flux rising rapidly toward threshold
        
        Parameters
        ----------
        time : datetime
            Current time
        flux : float
            Current X-ray flux
        flux_history : list, optional
            Previous flux values for intensity classification
            
        Returns
        -------
        FlareAlert or None
            Alert if event detected, None otherwise
        """
        # Track history for rate-of-rise calculation
        self._flux_history.append(flux)
        self._time_history.append(time)
        if len(self._flux_history) > 100:
            self._flux_history = self._flux_history[-100:]
            self._time_history = self._time_history[-100:]
        
        above_threshold = flux >= self.threshold_flux
        rate_of_rise_detected = self._is_rate_of_rise_detected()
        
        # Start new event: either threshold crossed OR rate-of-rise detected
        if (above_threshold or rate_of_rise_detected) and self.current_event is None:
            # Start of new event
            detection_method = 'threshold' if above_threshold else 'rate_of_rise'
            self.current_event = FlareAlert(
                start_time=time,
                peak_time=time,
                end_time=None,
                peak_flux=flux,
                flare_class=self.get_flare_subclass(flux, self.classify_flare(flux)),
                duration_minutes=None,
                intensity='unknown'
            )
            return self.current_event
        
        elif (above_threshold or rate_of_rise_detected) and self.current_event is not None:
            # Continuing event
            if flux > self.current_event.peak_flux:
                self.current_event.peak_time = time
                self.current_event.peak_flux = flux
                self.current_event.flare_class = self.get_flare_subclass(
                    flux, self.classify_flare(flux)
                )
            return None  # Event already started, don't re-alert
        
        elif not above_threshold and self.current_event is not None:
            # End of event
            self.current_event.end_time = time
            duration = (time - self.current_event.start_time).total_seconds() / 60
            self.current_event.duration_minutes = duration
            
            # Classify intensity based on rise time and flux history
            if flux_history is not None and len(flux_history) > 0:
                rise_time = (self.current_event.peak_time - 
                           self.current_event.start_time).total_seconds() / 60
                if rise_time < 5:
                    self.current_event.intensity = 'impulsive'
                elif rise_time < 30:
                    self.current_event.intensity = 'gradual'
                else:
                    self.current_event.intensity = 'slow'
            
            # Check cooldown
            alert = self.current_event
            self.current_event = None
            
            if (self.last_alert_time is None or 
                time - self.last_alert_time > self.cooldown):
                self.last_alert_time = time
                self.last_alert_class = alert.flare_class
                return alert
            else:
                return None
        
        return None
    
    def process_time_series(
        self,
        times: pd.DatetimeIndex,
        fluxes: np.ndarray
    ) -> List[FlareAlert]:
        """
        Process a time series of flux measurements.
        
        Parameters
        ----------
        times : pd.DatetimeIndex
            Time stamps
        fluxes : np.ndarray
            Corresponding flux values
            
        Returns
        -------
        List[FlareAlert]
            Detected events
        """
        alerts = []
        flux_history = []
        
        for i, (t, f) in enumerate(zip(times, fluxes)):
            # Add previous flux to history (for intensity classification)
            if i > 0:
                flux_history.append(fluxes[i-1])
            
            alert = self.detect_event(t, f, flux_history)
            if alert is not None:
                alerts.append(alert)
        
        return alerts


class NowcastDashboard:
    """
    Real-time nowcasting dashboard data generator.
    
    Generates current status and recent alerts for display.
    """
    
    def __init__(self, detector: FlareDetector):
        self.detector = detector
        self.recent_alerts: List[FlareAlert] = []
        self.current_status = {
            'current_flux': 0,
            'current_class': 'A',
            'peak_flux_24h': 0,
            'peak_class_24h': 'A',
            'n_events_today': 0,
            'last_update': None,
        }
    
    def update(
        self,
        time: datetime,
        flux: float
    ) -> Optional[FlareAlert]:
        """Update dashboard with new data point."""
        # Update status
        self.current_status['current_flux'] = flux
        self.current_status['current_class'] = self.detector.classify_flare(flux)
        self.current_status['last_update'] = time
        
        if flux > self.current_status['peak_flux_24h']:
            self.current_status['peak_flux_24h'] = flux
            self.current_status['peak_class_24h'] = self.detector.classify_flare(flux)
        
        # Detect events
        alert = self.detector.detect_event(time, flux)
        
        if alert is not None:
            self.recent_alerts.append(alert)
            self.current_status['n_events_today'] += 1
            
            # Keep only last 24h of alerts
            cutoff = time - timedelta(hours=24)
            self.recent_alerts = [
                a for a in self.recent_alerts
                if a.start_time > cutoff
            ]
        
        return alert
    
    def get_status_dict(self) -> Dict:
        """Get current status as dictionary for JSON export."""
        return {
            'timestamp': self.current_status['last_update'].isoformat() 
                        if self.current_status['last_update'] else None,
            'current_flux': self.current_status['current_flux'],
            'current_class': self.current_status['current_class'],
            'peak_24h': {
                'flux': self.current_status['peak_flux_24h'],
                'class': self.current_status['peak_class_24h'],
            },
            'events_today': self.current_status['n_events_today'],
            'recent_alerts': [a.to_dict() for a in self.recent_alerts[-5:]],
        }


if __name__ == "__main__":
    # Test flare detection
    print("Testing FlareDetector...")
    
    from src.data.reader import get_sample_data
    
    # Generate sample data with known flares
    df = get_sample_data(n_points=2000)
    
    # Create detector
    detector = FlareDetector(threshold_class='C')
    dashboard = NowcastDashboard(detector)
    
    # Process data
    alerts = dashboard.detector.process_time_series(
        df.index, 
        df['xrs_b_flux'].values
    )
    
    print(f"\nDetected {len(alerts)} events:")
    for alert in alerts:
        print(f"  {alert.flare_class} at {alert.start_time} "
              f"(peak: {alert.peak_flux:.2e})")
    
    # Test dashboard
    print("\nDashboard status:")
    print(dashboard.get_status_dict())