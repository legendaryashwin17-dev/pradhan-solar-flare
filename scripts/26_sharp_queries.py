"""
SHARP Query Builder for PRADHAN

Generates ready-to-use JSOC query strings for the dates we need:
  1. Solar Cycle 25 peak: 2024 X-class flares (May-Oct 2024)
  2. 2026 current period: April-June 2026 (matches our GOES-18 + HEL1OS data)

JSOC Manual Export Page: http://jsoc.stanford.edu/ajax/lookdata.html
"""

# ─────────────────────────────────────────────────────────────────────
# Query 1: 2024 X-class flares (Solar Cycle 25 peak)
# ─────────────────────────────────────────────────────────────────────
# These are the M+ flares from the catalog we want SHARP data for.
# Covers: M5.0 (Jan), M5.2 (Feb), X8.7 (May), X1.1 (May), M2.0 (May),
# M2.4 (Oct), M1.0 (Oct), etc.

QUERY_2024_XFLARES = {
    "name": "Solar Cycle 25 X/M flares (2024)",
    "series": "hmi.sharp_cea_720s",
    "ds": "hmi.sharp_cea_720s[2024-01-01 00:00:00_TAI-2024-12-31 23:59:59_TAI]",
    "keywords": "USFLUX,TOTUSJH,TOTUSJZ,TOTPOT,R_VALUE,SAVNCPP,MEANPOT",
    "format": "csv",
    "op": "export",
    "description": "All SHARP data for the year 2024 (full year of cycle 25 ramp-up)"
}

# ─────────────────────────────────────────────────────────────────────
# Query 2: 2026 current period (matches our GOES-18 + HEL1OS coverage)
# ─────────────────────────────────────────────────────────────────────
# This is the period we have GOES-18 NetCDF files for.
# 63 files covering 2026-04-19 through 2026-06-20.

QUERY_2026_CURRENT = {
    "name": "2026 Apr-Jun (matches HEL1OS/GOES data)",
    "series": "hmi.sharp_cea_720s",
    "ds": "hmi.sharp_cea_720s[2026-04-15 00:00:00_TAI-2026-06-20 23:59:59_TAI]",
    "keywords": "USFLUX,TOTUSJH,TOTUSJZ,TOTPOT,R_VALUE,SAVNCPP,MEANPOT",
    "format": "csv",
    "op": "export",
    "description": "SHARP data for our active observation window"
}

# ─────────────────────────────────────────────────────────────────────
# Query 3: Tight around X8.7 flare (May 14, 2024) — for the headline result
# ─────────────────────────────────────────────────────────────────────
# Smaller, faster query for the iconic X8.7 event from AR 13664.

QUERY_X87_FLARES = {
    "name": "X8.7 flare window (AR 13664)",
    "series": "hmi.sharp_cea_720s",
    "ds": "hmi.sharp_cea_720s[2024-05-13 00:00:00_TAI-2024-05-15 23:59:59_TAI]",
    "keywords": "USFLUX,TOTUSJH,TOTUSJZ,TOTPOT,R_VALUE,SAVNCPP,MEANPOT",
    "format": "csv",
    "op": "export",
    "description": "SHARP data 24h before and after the X8.7 flare"
}


def build_url(query):
    """Build a direct JSOC export URL that can be pasted into a browser."""
    base = "http://jsoc.stanford.edu/cgi-bin/jsoc_fetch"
    params = f"op={query['op']}&ds={query['ds']}&key={query['keywords']}&format={query['format']}"
    return f"{base}?{params}"


def build_web_form_query(query):
    """Build the manual form query text for the JSOC web interface."""
    return f"""DS (dataseries):    {query['ds']}
Keywords:           {query['keywords']}
Time format:        YYYY-MM-DD HH:MM:SS_TAI
Export format:      {query['format']}
"""


def main():
    for q in [QUERY_2024_XFLARES, QUERY_2026_CURRENT, QUERY_X87_FLARES]:
        print('=' * 70)
        print(f"Query: {q['name']}")
        print(f"Description: {q['description']}")
        print('=' * 70)
        print()
        print("WEB FORM QUERY (paste in lookdata.html):")
        print(build_web_form_query(q))
        print()
        print("DIRECT URL (paste in browser address bar):")
        print(build_url(q))
        print()
        print()


if __name__ == '__main__':
    main()