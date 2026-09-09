"""API v1 routes."""
from django.urls import path

from apps.core import views as core_views
from apps.datasets import views as dataset_views
from apps.experiments import views as experiment_views
from apps.forecasting import views as forecast_views
from apps.insights import views as insight_views
from apps.pol4 import views as pol4_views

urlpatterns = [
    # -- system ------------------------------------------------------------
    path("health/", core_views.health, name="health"),
    path("system/", core_views.system_info, name="system-info"),

    # -- dashboard ---------------------------------------------------------
    path("dashboard/summary/", forecast_views.dashboard_summary, name="dashboard-summary"),

    # -- forecasts ---------------------------------------------------------
    path("forecasts/", forecast_views.forecast_list, name="forecast-list"),
    path("forecasts/timeseries/", forecast_views.forecast_timeseries, name="forecast-timeseries"),
    path("forecasts/drivers/", forecast_views.forecast_drivers, name="forecast-drivers"),
    path("forecasts/peaks/", forecast_views.forecast_peaks, name="forecast-peaks"),
    path("forecasts/overview/", forecast_views.forecast_overview, name="forecast-overview"),
    path("forecasts/heatmap/", forecast_views.forecast_heatmap, name="forecast-heatmap"),
    path("forecasts/narrative/", forecast_views.forecast_narrative, name="forecast-narrative"),

    # -- anomalies ---------------------------------------------------------
    path("anomalies/", forecast_views.anomaly_list, name="anomaly-list"),

    # -- models ------------------------------------------------------------
    path("models/", forecast_views.model_list, name="model-list"),
    path("models/leaderboard/", forecast_views.model_leaderboard, name="model-leaderboard"),

    # -- backtests ---------------------------------------------------------
    path("backtests/", forecast_views.backtest_list, name="backtest-list"),
    path("backtests/metrics/", forecast_views.backtest_metrics, name="backtest-metrics"),

    # -- insights ----------------------------------------------------------
    path("insights/", insight_views.insight_list, name="insight-list"),
    path("insights/opportunities/", insight_views.opportunity_list, name="insight-opportunities"),
    path("insights/data-quality/", insight_views.data_quality, name="insight-data-quality"),

    # -- Pol 4 demand intelligence -----------------------------------------
    # Read-only, served from the artefacts `python -m ml.pol4.pipeline` writes.
    # Nothing here opens search_data.csv or fits a model.
    path("pol4/overview/", pol4_views.overview, name="pol4-overview"),
    path("pol4/forecast/", pol4_views.forecast, name="pol4-forecast"),
    path("pol4/heatmap/", pol4_views.heatmap, name="pol4-heatmap"),
    path("pol4/cities/", pol4_views.city_list, name="pol4-cities"),
    path("pol4/cities/<str:city_code>/", pol4_views.city_detail, name="pol4-city-detail"),
    path("pol4/cities/<str:city_code>/pickup/", pol4_views.city_pickup, name="pol4-city-pickup"),
    path("pol4/stability/", pol4_views.stability, name="pol4-stability"),
    path("pol4/model-performance/", pol4_views.model_performance, name="pol4-model-performance"),
    path("pol4/reports/", pol4_views.report_list, name="pol4-reports"),
    path("pol4/reports/<str:kind>/", pol4_views.report_preview, name="pol4-report-preview"),
    path("pol4/reports/<str:kind>.csv", pol4_views.report_download, name="pol4-report-download"),

    # -- data lab ----------------------------------------------------------
    path("datasets/", dataset_views.dataset_list, name="dataset-list"),
    path("datasets/upload/", dataset_views.dataset_upload, name="dataset-upload"),
    path("datasets/profile/", dataset_views.dataset_profile, name="dataset-profile"),
    path("datasets/map/", dataset_views.dataset_map, name="dataset-map"),
    path("datasets/validate/", dataset_views.dataset_validate, name="dataset-validate"),
    path("datasets/contract/", dataset_views.contract_detail, name="dataset-contract"),

    # -- training ----------------------------------------------------------
    path("training/", experiment_views.training_list, name="training-list"),
    path("training/run/", experiment_views.training_run, name="training-run"),
    path("training/<str:run_id>/", experiment_views.training_detail, name="training-detail"),
    path("experiments/", experiment_views.experiment_list, name="experiment-list"),
]
