import type { PropsWithChildren } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../features/auth/auth-context";

export function RequireAuth({ children }: PropsWithChildren) {
  const { me, loading } = useAuth();
  const location = useLocation();

  if (loading) return <div className="auth-page"><div className="panel auth-card">Loading session…</div></div>;
  if (!me) return <Navigate to={`/login?next=${encodeURIComponent(location.pathname + location.search + location.hash)}`} replace />;
  if (me.must_change_password && location.pathname !== "/profile/change-password") {
    return <Navigate to={`/profile/change-password?next=${encodeURIComponent(location.pathname + location.search + location.hash)}`} replace />;
  }
  return <>{children}</>;
}

export function RequireAdmin({ children }: PropsWithChildren) {
  const { me } = useAuth();
  if (me?.role !== "admin") {
    return <Navigate to="/runs" replace />;
  }
  return <>{children}</>;
}

export function RedirectIfAuthenticated({ children }: PropsWithChildren) {
  const { me, loading } = useAuth();
  if (loading) return <div className="auth-page"><div className="panel auth-card">Loading session…</div></div>;
  if (me) {
    if (me.must_change_password) return <Navigate to="/profile/change-password" replace />;
    return <Navigate to="/runs" replace />;
  }
  return <>{children}</>;
}
