import type { PropsWithChildren, ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../features/auth/auth-context";

export function AppShell({ title, subtitle, actions, children }: PropsWithChildren<{ title: string; subtitle: string; actions?: ReactNode }>) {
  const { me, logoutAndClear } = useAuth();
  const navigate = useNavigate();

  return (
    <div className="app-shell">
      <header className="topbar panel">
        <div className="brand">
          <h1>{title}</h1>
          <p className="subtitle">{subtitle}</p>
        </div>

        <div className="auth-controls toolbar">
          <span id="whoami" aria-live="polite">{me ? `${me.email} (${me.role})` : ""}</span>
          <button id="change-password-btn" type="button" className="btn btn--secondary" onClick={() => navigate('/profile/change-password')}>
            Change password
          </button>
          <Link to="/datasets" className="btn btn--secondary">Datasets</Link>
          <button
            id="logout-btn"
            type="button"
            className="btn btn--secondary"
            onClick={async () => {
              await logoutAndClear();
              navigate('/login');
            }}
          >
            Logout
          </button>
          {actions}
        </div>
      </header>
      {children}
    </div>
  );
}
