from . import admin_audit, admin_invitations, admin_users, auth, datasets, runs, sessions

all_routers = [
    auth.router,
    admin_users.router,
    admin_invitations.router,
    admin_audit.router,
    runs.router,
    datasets.router,
    sessions.router,
]
