from . import datasets, models, runs

feature_routers = [
    runs.router,
    datasets.router,
    models.router,
]
