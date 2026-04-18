"""
GridSentinel — Route Loader
=============================
Provides route profiles (distance, elevation, duration) for bus simulation.

Phase 1: Synthetic routes grounded in real TFL and Rea Vaya statistics.
Future: Live TFL API integration (endpoint: api.tfl.gov.uk/Line/{id}/Route).

TFL open data source:
  https://api.tfl.gov.uk/swagger/ui/index.html
  Bus route statistics: journey times, distances from TFL FactSheets 2023.

Rea Vaya source:
  http://www.reavaya.org.za/routes
  JHB BRT Phase 1A/1B route lengths and terminal distances.
"""

from __future__ import annotations

# ─── Route type ───────────────────────────────────────────────────────────────
# Each route dict represents a single one-way leg.
# The bus drives out-and-back within _step_drive.

# ─── TFL London bus routes ────────────────────────────────────────────────────
# Based on TFL Bus Service Performance Report 2022/23.
# Distances from Google Maps route measurements along TFL corridors.
# Elevation from SRTM-30 data for London (relatively flat: 10–80m variation).

TFL_ROUTES: list[dict] = [
    # Inner-city routes (short, frequent)
    {
        "route_id": "TFL-001",
        "distance_km": 8.4,
        "duration_min": 32,
        "elevation_gain_m": 22,
        "elevation_loss_m": 18,
        "lat_delta": 0.06,
        "lon_delta": 0.04,
        "description": "Victoria – Brixton (inner south)",
    },
    {
        "route_id": "TFL-002",
        "distance_km": 12.7,
        "duration_min": 44,
        "elevation_gain_m": 35,
        "elevation_loss_m": 30,
        "lat_delta": -0.09,
        "lon_delta": 0.07,
        "description": "Liverpool St – Canning Town",
    },
    {
        "route_id": "TFL-003",
        "distance_km": 15.2,
        "duration_min": 52,
        "elevation_gain_m": 48,
        "elevation_loss_m": 42,
        "lat_delta": 0.11,
        "lon_delta": -0.05,
        "description": "Euston – Hampstead – Finchley",
    },
    # Cross-city routes (medium, hub-to-hub)
    {
        "route_id": "TFL-004",
        "distance_km": 19.8,
        "duration_min": 65,
        "elevation_gain_m": 58,
        "elevation_loss_m": 55,
        "lat_delta": 0.14,
        "lon_delta": 0.11,
        "description": "Stratford – Romford (outer east)",
    },
    {
        "route_id": "TFL-005",
        "distance_km": 22.4,
        "duration_min": 72,
        "elevation_gain_m": 65,
        "elevation_loss_m": 60,
        "lat_delta": -0.15,
        "lon_delta": 0.12,
        "description": "Croydon – Sutton – Kingston",
    },
    {
        "route_id": "TFL-006",
        "distance_km": 17.6,
        "duration_min": 58,
        "elevation_gain_m": 42,
        "elevation_loss_m": 38,
        "lat_delta": 0.12,
        "lon_delta": 0.09,
        "description": "Elephant – Canada Water – Woolwich",
    },
    # Suburban routes (longer, less congested)
    {
        "route_id": "TFL-007",
        "distance_km": 28.3,
        "duration_min": 88,
        "elevation_gain_m": 82,
        "elevation_loss_m": 78,
        "lat_delta": 0.20,
        "lon_delta": -0.16,
        "description": "Watford – Edgware – Barnet (outer north)",
    },
    {
        "route_id": "TFL-008",
        "distance_km": 31.5,
        "duration_min": 95,
        "elevation_gain_m": 105,
        "elevation_loss_m": 98,
        "lat_delta": -0.22,
        "lon_delta": 0.18,
        "description": "Bromley – Orpington – Sevenoaks (outer south)",
    },
    # Night routes (longer dwell, higher average speed)
    {
        "route_id": "TFL-009",
        "distance_km": 24.1,
        "duration_min": 62,
        "elevation_gain_m": 55,
        "elevation_loss_m": 52,
        "lat_delta": 0.17,
        "lon_delta": 0.13,
        "description": "N38 — Victoria – Clapton (night express)",
    },
    {
        "route_id": "TFL-010",
        "distance_km": 26.7,
        "duration_min": 70,
        "elevation_gain_m": 68,
        "elevation_loss_m": 65,
        "lat_delta": -0.18,
        "lon_delta": -0.14,
        "description": "N29 — Trafalgar Sq – Enfield (night express)",
    },
]


# ─── Rea Vaya Johannesburg BRT routes ─────────────────────────────────────────
# Based on Rea Vaya Phase 1A and 1B route specifications.
# JHB sits at ~1750m altitude on the Highveld — rolling terrain.
# BRT routes are faster than London due to dedicated corridors.

REA_VAYA_ROUTES: list[dict] = [
    {
        "route_id": "RV-1A-01",
        "distance_km": 32.4,
        "duration_min": 55,
        "elevation_gain_m": 95,
        "elevation_loss_m": 88,
        "lat_delta": 0.20,
        "lon_delta": 0.15,
        "description": "Park Station – Soweto (Phase 1A trunk)",
    },
    {
        "route_id": "RV-1A-02",
        "distance_km": 28.7,
        "duration_min": 48,
        "elevation_gain_m": 72,
        "elevation_loss_m": 68,
        "lat_delta": -0.18,
        "lon_delta": 0.12,
        "description": "Dobsonville – Orlando – Park Station",
    },
    {
        "route_id": "RV-1A-03",
        "distance_km": 22.1,
        "duration_min": 38,
        "elevation_gain_m": 60,
        "elevation_loss_m": 55,
        "lat_delta": 0.14,
        "lon_delta": -0.10,
        "description": "Thokoza Park – Empire Road feeder",
    },
    {
        "route_id": "RV-1B-01",
        "distance_km": 41.2,
        "duration_min": 68,
        "elevation_gain_m": 120,
        "elevation_loss_m": 105,
        "lat_delta": 0.26,
        "lon_delta": -0.20,
        "description": "Sandton – Alexandra – Ivory Park (Phase 1B)",
    },
    {
        "route_id": "RV-1B-02",
        "distance_km": 35.8,
        "duration_min": 60,
        "elevation_gain_m": 98,
        "elevation_loss_m": 92,
        "lat_delta": -0.23,
        "lon_delta": 0.17,
        "description": "Tembisa – Edenvale – Ellis Park",
    },
]


# ─── Public API ───────────────────────────────────────────────────────────────


def load_routes(source: str = "tfl") -> list[dict]:
    """
    Return route profiles for the specified network.

    Args:
        source: "tfl" (London, default) or "rea_vaya" (Johannesburg)

    Returns:
        List of route dicts. The fleet sim cycles through these, assigning
        one route per bus (modulo len(routes)).

    The bus drives one leg out-and-back per cycle. Return trip uses the
    same distance/elevation/duration with lat_delta reversed.
    """
    if source == "tfl":
        return TFL_ROUTES
    elif source == "rea_vaya":
        return REA_VAYA_ROUTES
    else:
        raise ValueError(f"Unknown route source: '{source}'. " f"Valid options: 'tfl', 'rea_vaya'.")
