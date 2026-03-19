from . import admin_audit, admin_invitations, admin_users, auth, sessions
from ..routers import feature_routers

all_routers = [
    auth.router,
    admin_users.router,
    admin_invitations.router,
    admin_audit.router,
    *feature_routers,
    sessions.router,
]
