"""
PRADHAN Active Region Tracking — Honest Implementation
=======================================================

IMPORTANT: This module demonstrates how AR data COULD be incorporated.
The current implementation uses SYNTHETIC data that is clearly marked.

To use real AR data:
1. Download NOAA AR catalog from: https://www.ngdc.noaa.gov/stp/space-weather.html
2. Or use SunPy's solar active region database
3. Or use SHARP parameters from SDO/HMI

The features here (magnetic class, area) are based on standard NOAA
classification schemes:
- Alpha: Single unipolar sunspot
- Beta: Bipolar sunspot group
- Beta-Gamma: Bipolar with opposite polarity spots between
- Beta-Gamma-Delta: Complex delta configuration (highest flare risk)

NOTE: True physics-based flare prediction requires magnetogram data
(SOHO/MDI, SDO/HMI, Hinode). This module shows how such data would be
integrated, but we currently use synthetic analogs for demonstration.

Reference: McIntosh (1990) classification scheme
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# NOAA Magnetic Classification (simplified)
# Full scheme has more subtypes: https://www.swpc.noaa.gov/solar-structures
MAGNETIC_CLASSES = {
    'alpha': 0,      # Simple unipolar
    'beta': 1,       # Simple bipolar
    'beta-gamma': 2, # Bipolar with intermediate spots
    'beta-gamma-delta': 3,  # Complex delta configuration (high risk)
    'delta': 4,      # Delta configuration present
}


@dataclass
class ActiveRegion:
    """
    Active region data structure.
    
    In a real implementation, these fields would come from NOAA catalogs
    or magnetogram analysis.
    """
    ar_id: str
    start_time: datetime
    end_time: Optional[datetime]
    noaa_number: Optional[int]
    magnetic_class: str
    latitude: Optional[float]
    longitude: Optional[float]
    area: float  # Area in millionths of solar hemisphere
    complexity_score: float  # 0-1 scale based on magnetic complexity
    longitude_helio: Optional[float]  # Heliographic longitude
    
    def to_features(self) -> Dict[str, float]:
        """Convert to feature dictionary."""
        return {
            'magnetic_class_encoded': MAGNETIC_CLASSES.get(self.magnetic_class, -1),
            'area': self.area,
            'complexity_score': self.complexity_score,
            'is_delta': 1.0 if 'delta' in self.magnetic_class else 0.0,
            'is_beta_gamma': 1.0 if 'beta-gamma' in self.magnetic_class else 0.0,
        }


class ActiveRegionTracker:
    """
    Track active regions and extract AR-based features.
    
    This demonstrates how AR data would be integrated into the
    flare forecasting pipeline. Currently uses synthetic data.
    """
    
    def __init__(self):
        self.regions: Dict[str, ActiveRegion] = {}
        self.flare_history: Dict[str, List[Dict]] = {}
    
    def add_region(self, ar: ActiveRegion):
        """Add an active region to the tracker."""
        self.regions[ar.ar_id] = ar
        self.flare_history[ar.ar_id] = []
    
    def get_region_at_time(self, time: datetime) -> List[str]:
        """Get IDs of active regions present at a given time."""
        active = []
        for ar_id, ar in self.regions.items():
            if ar.start_time <= time:
                if ar.end_time is None or time <= ar.end_time:
                    active.append(ar_id)
        return active
    
    def get_region_features(self, ar_id: str, time: datetime) -> Optional[Dict[str, float]]:
        """
        Get features for an active region at a given time.
        
        Returns None if region doesn't exist or is not active.
        """
        ar = self.regions.get(ar_id)
        if ar is None:
            return None
        
        # Check if region is active at this time
        if time < ar.start_time:
            return None
        if ar.end_time is not None and time > ar.end_time:
            return None
        
        features = ar.to_features()
        
        # Add flare history features (last 24 hours)
        history_24h = [
            f for f in self.flare_history.get(ar_id, [])
            if (time - f['time']).total_seconds() < 86400
        ]
        
        features['flares_last_24h'] = len(history_24h)
        features['m_flares_last_24h'] = sum(
            1 for f in history_24h if f['class'] in ['M', 'X']
        )
        features['x_flares_last_24h'] = sum(
            1 for f in history_24h if f['class'] == 'X'
        )
        
        # Add decay indicator (if region is near end of life)
        if ar.end_time is not None:
            hours_remaining = (ar.end_time - time).total_seconds() / 3600
            features['decay_phase'] = 1.0 if hours_remaining < 24 else 0.0
        else:
            features['decay_phase'] = 0.0
        
        return features
    
    def add_flare(self, ar_id: str, time: datetime, flare_class: str, peak_flux: float):
        """Record a flare associated with an active region."""
        if ar_id not in self.flare_history:
            self.flare_history[ar_id] = []
        
        self.flare_history[ar_id].append({
            'time': time,
            'class': flare_class,
            'peak_flux': peak_flux
        })
    
    def get_composite_features(self, time: datetime) -> Dict[str, float]:
        """
        Get composite features across all active regions at a time.
        
        Returns aggregated features if multiple ARs are present.
        """
        active_ars = self.get_region_at_time(time)
        
        if not active_ars:
            # No active regions
            return {
                'n_active_regions': 0,
                'max_magnetic_class': -1,
                'max_complexity': 0,
                'total_area': 0,
                'has_delta': 0,
                'has_recent_m': 0,
                'has_recent_x': 0,
            }
        
        # Aggregate features across all active ARs
        features = {
            'n_active_regions': len(active_ars),
            'max_magnetic_class': max(
                MAGNETIC_CLASSES.get(self.regions[ar_id].magnetic_class, -1)
                for ar_id in active_ars
            ),
            'max_complexity': max(
                self.regions[ar_id].complexity_score for ar_id in active_ars
            ),
            'total_area': sum(self.regions[ar_id].area for ar_id in active_ars),
            'has_delta': max(
                1.0 if 'delta' in self.regions[ar_id].magnetic_class else 0.0
                for ar_id in active_ars
            ),
            'has_recent_m': max(
                sum(1 for f in self.flare_history.get(ar_id, [])
                    if f['class'] in ['M', 'X'] and
                    (time - f['time']).total_seconds() < 86400)
                for ar_id in active_ars
            ),
            'has_recent_x': max(
                sum(1 for f in self.flare_history.get(ar_id, [])
                    if f['class'] == 'X' and
                    (time - f['time']).total_seconds() < 86400)
                for ar_id in active_ars
            ),
        }
        
        return features


def load_noaa_ar_catalog(
    start_date: str,
    end_date: str
) -> List[ActiveRegion]:
    """
    Load active region data from NOAA catalog.
    
    NOTE: This is a placeholder. In production, you would:
    1. Download from https://www.ngdc.noaa.gov/stp/space-weather.html
    2. Use SunPy's solar client
    3. Parse the NOAA AR text files
    
    Returns
    -------
    List[ActiveRegion]
        List of active regions in the date range
    """
    # This would download real NOAA data
    # For now, return empty list - see create_synthetic_ar_data()
    return []


def create_synthetic_ar_data(
    start_time: datetime,
    end_time: datetime,
    n_regions: int = 10,
    seed: int = 42
) -> List[ActiveRegion]:
    """
    Create SYNTHETIC active region data for demonstration.
    
    ⚠️  WARNING: This is SYNTHETIC data for demonstration only!
    
    In a real implementation, you would use actual NOAA AR catalogs.
    The characteristics here mimic real AR statistics but are not real.
    
    Parameters
    ----------
    start_time : datetime
        Start of time range
    end_time : datetime
        End of time range
    n_regions : int
        Number of synthetic regions to create
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    List[ActiveRegion]
        List of synthetic active regions
    """
    np.random.seed(seed)
    
    # Real AR statistics (approximate):
    # - Average AR lifetime: 2-3 solar rotations (~30-90 days)
    # - Most are simple (alpha, beta), few are complex (delta)
    # - Larger areas correlate with higher flare rates
    # - Delta regions are rare but high-risk
    
    time_span = (end_time - start_time).total_seconds()
    regions = []
    
    # Class distribution (based on real statistics)
    # Most ARs are simple, complex ones are rare
    class_probs = [0.3, 0.4, 0.2, 0.1]  # alpha, beta, beta-gamma, beta-gamma-delta
    class_names = ['alpha', 'beta', 'beta-gamma', 'beta-gamma-delta']
    
    for i in range(n_regions):
        # Random start time within the period
        start = start_time + timedelta(seconds=np.random.uniform(0, time_span * 0.8))
        
        # Random lifetime (2-60 days)
        lifetime_days = np.random.exponential(14)  # Mean 14 days, but skewed
        lifetime_days = min(max(lifetime_days, 2), 60)
        end = start + timedelta(days=lifetime_days)
        
        # Random class based on distribution
        magnetic_class = np.random.choice(class_names, p=class_probs)
        
        # Area: larger areas more likely in complex classes
        base_area = np.random.lognormal(mean=3, sigma=1)  # Typical area
        if 'delta' in magnetic_class:
            base_area *= 2  # Delta regions tend to be larger
        area = min(base_area, 3000)  # Cap at 3000 millionths
        
        # Complexity correlates with class
        complexity_map = {
            'alpha': (0.1, 0.3),
            'beta': (0.2, 0.5),
            'beta-gamma': (0.4, 0.7),
            'beta-gamma-delta': (0.6, 1.0),
        }
        complexity = np.random.uniform(*complexity_map[magnetic_class])
        
        region = ActiveRegion(
            ar_id=f"SYNTH_AR_{i+1:03d}",  # Clearly marked as synthetic
            start_time=start,
            end_time=end,
            noaa_number=None,  # No real NOAA number
            magnetic_class=magnetic_class,
            latitude=np.random.uniform(-30, 30),
            longitude=np.random.uniform(0, 360),
            area=area,
            complexity_score=complexity,
            longitude_helio=np.random.uniform(-90, 90),
        )
        regions.append(region)
    
    return regions


def add_ar_features_to_dataframe(
    df: pd.DataFrame,
    ar_regions: List[ActiveRegion],
    tracker: ActiveRegionTracker = None
) -> pd.DataFrame:
    """
    Add active region features to a feature DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with datetime index
    ar_regions : List[ActiveRegion]
        List of active regions
    tracker : ActiveRegionTracker, optional
        Existing tracker to use
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added AR features
    """
    if tracker is None:
        tracker = ActiveRegionTracker()
        for ar in ar_regions:
            tracker.add_region(ar)
    
    # Initialize AR feature columns
    ar_feature_names = [
        'n_active_regions',
        'max_magnetic_class',
        'max_complexity',
        'total_area',
        'has_delta',
        'has_recent_m',
        'has_recent_x',
    ]
    
    for fname in ar_feature_names:
        df[f'ar_{fname}'] = np.nan
    
    # Fill in AR features for each time point
    for i, (time, row) in enumerate(df.iterrows()):
        if isinstance(time, pd.Timestamp):
            time = time.to_pydatetime()
        
        features = tracker.get_composite_features(time)
        for fname, value in features.items():
            df.at[df.index[i], f'ar_{fname}'] = value
    
    return df


if __name__ == "__main__":
    print("Testing Active Region Module")
    print("=" * 50)
    
    # Create synthetic data
    start = datetime(2023, 1, 1)
    end = datetime(2023, 6, 30)
    
    print("\n⚠️  WARNING: Using SYNTHETIC data for demonstration")
    print("   Real implementation would use NOAA AR catalogs\n")
    
    regions = create_synthetic_ar_data(start, end, n_regions=10)
    
    print(f"Created {len(regions)} synthetic active regions:")
    for ar in regions[:5]:
        print(f"  {ar.ar_id}: class={ar.magnetic_class}, "
              f"area={ar.area:.0f}, complexity={ar.complexity_score:.2f}")
    print("  ...")
    
    # Test tracker
    tracker = ActiveRegionTracker()
    for ar in regions:
        tracker.add_region(ar)
    
    # Check features at a specific time
    test_time = datetime(2023, 3, 15, 12, 0)
    features = tracker.get_composite_features(test_time)
    print(f"\nFeatures at {test_time}:")
    for k, v in features.items():
        print(f"  {k}: {v}")