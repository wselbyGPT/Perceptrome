import type { PropsWithChildren } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../features/auth/auth-context";
import { LoadingState } from "../components/ui/states";

function LoadingSession() {
  return <div className="auth-page"><div className="auth-card panel"><LoadingState title="Loading session" message="Checking your authenticated session and account policy." /></div></div>;
}

function nextPathname(location: ReturnType<typeof useLocation>) {
  return encodeURIComponent(location.pathname + location.search + location.hash);
}

export function RequireAuth({ children }: PropsWithChildren) {
  const { authState } = useAuth();
  const location = useLocation();

  if (authState === "authenticating") return <LoadingSession />;
  if (authState === "anonymous" || authState === "email_unverified") {
    return <Navigate to={`/login?next=${nextPathname(location)}`} replace />;
  }
  if (authState === "forbidden") {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
}

export function RequireAdmin({ children }: PropsWithChildren) {
  const { authState, me } = useAuth();
  if (authState === "authenticating") return <LoadingSession />;
  if (authState !== "authenticated" && authState !== "must_change_password") {
    return <Navigate to="/login" replace />;
  }
  if (me?.role !== "admin") return <Navigate to="/dashboard" replace />;
  return <>{children}</>;
}

export function RedirectIfAuthenticated({ children }: PropsWithChildren) {
  const { authState } = useAuth();
  if (authState === "authenticating") return <LoadingSession />;
  if (authState === "must_change_password") return <Navigate to="/security" replace />;
  if (authState === "authenticated") return <Navigate to="/dashboard" replace />;
  if (authState === "forbidden") return <Navigate to="/dashboard" replace />;
  return <>{children}</>;
}
