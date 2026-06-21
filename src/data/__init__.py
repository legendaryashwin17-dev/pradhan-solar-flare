# PRADHAN — Solar Flare Nowcasting with Statistical Features
# =============================================================================
#
# Scientific Note:
# This system uses STATISTICAL features derived from X-ray light curves,
# NOT physics-based parameters. True physics-based forecasting requires
# vector magnetogram data (e.g., SDO/HMI) for magnetic free energy calculations.
#
# The 19 features here are proxies that correlate with flaring activity,
# validated through empirical research in the literature.
# =============================================================================

from .reader import load_goes_parquet, load_solexs_lc, load_solexs_directory, validate_data
from .features import compute_features, get_feature_names
from .labels import (
    create_flare_labels,
    FLUX_THRESHOLDS,
    FORECAST_HORIZONS,
    NOAA_SCALE_LABELS,
)

__all__ = [
    'load_goes_parquet',
    'load_solexs_lc',
    'load_solexs_directory',
    'validate_data',
    'compute_features',
    'get_feature_names',
    'create_flare_labels',
    'FLUX_THRESHOLDS',
    'FORECAST_HORIZONS',
    'NOAA_SCALE_LABELS',
]
