import type { PropsWithChildren } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../features/auth/auth-context";
import { StatusBadge } from "../components/ui/states";

const primaryNav = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/runs', label: 'Runs' },
  { to: '/models', label: 'Models' },
  { to: '/datasets', label: 'Datasets' },
  { to: '/profile', label: 'Profile' },
  { to: '/security', label: 'Security' },
];

const adminNav = [
  { to: '/admin/users', label: 'Users' },
  { to: '/admin/invitations', label: 'Invitations' },
  { to: '/admin/audit-log', label: 'Audit Log' },
  { to: '/admin/system', label: 'System' },
];

function ShellNavLink({ to, label, disabled = false }: { to: string; label: string; disabled?: boolean }) {
  if (disabled) {
    return <span className="shell-nav__link shell-nav__link--disabled" aria-disabled="true">{label}</span>;
  }

  return (
    <NavLink to={to} className={({ isActive }) => `shell-nav__link${isActive ? ' shell-nav__link--active' : ''}`}>
      {label}
    </NavLink>
  );
}

function titleFromPath(pathname: string) {
  const segments = pathname.split('/').filter(Boolean);
  if (!segments.length) return { title: 'Dashboard', subtitle: 'Overview' };
  const title = segments.map((segment) => segment.replace(/-/g, ' ')).map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1)).join(' / ');
  const subtitle = pathname.startsWith('/admin') ? 'Administrative controls and system operations.' : 'Authenticated workspace';
  return { title, subtitle };
}

export function AppShell({ children }: PropsWithChildren) {
  const { me, logoutAndClear } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const { title, subtitle } = titleFromPath(location.pathname);
  const isAdmin = me?.role === 'admin';

  return (
    <div className="shell-frame">
      <aside className="shell-sidebar panel">
        <div className="shell-sidebar__brand">
          <p className="eyebrow">Perceptrome</p>
          <h1>Control Center</h1>
          <p className="muted">Shared workspace shell for runs, datasets, profiles, and admin operations.</p>
        </div>

        <nav className="shell-nav stack" aria-label="Primary">
          <div className="stack-sm">
            <span className="shell-nav__section">Workspace</span>
            {primaryNav.map((item) => <ShellNavLink key={item.to} to={item.to} label={item.label} />)}
          </div>
          <div className="stack-sm">
            <div className="cluster shell-nav__section-row">
              <span className="shell-nav__section">Admin</span>
              {isAdmin ? <StatusBadge label="admin" tone="admin" /> : <StatusBadge label="restricted" tone="muted" />}
            </div>
            {adminNav.map((item) => <ShellNavLink key={item.to} to={item.to} label={item.label} disabled={!isAdmin} />)}
          </div>
        </nav>
      </aside>

      <div className="shell-main">
        <header className="shell-topbar panel">
          <div>
            <p className="eyebrow">{title}</p>
            <h2>{title}</h2>
            <p className="muted">{subtitle}</p>
          </div>
          <div className="shell-topbar__actions">
            <div className="shell-usercard">
              <strong>{me?.username ?? me?.email ?? 'Anonymous'}</strong>
              <span>{me?.email ?? ''}</span>
              <div className="cluster">
                <StatusBadge label={me?.role ?? 'unknown'} tone={me?.role} />
                <StatusBadge label={me?.is_active ? 'active' : 'inactive'} />
              </div>
            </div>
            <div className="cluster">
              <button type="button" className="btn btn--secondary" onClick={() => navigate('/security')}>Security</button>
              <button
                type="button"
                className="btn btn--secondary"
                onClick={async () => {
                  await logoutAndClear();
                  navigate('/login');
                }}
              >
                Logout
              </button>
            </div>
          </div>
        </header>

        <main className="shell-content">{children}</main>
      </div>
    </div>
  );
}
