import { FeedbackNotice, StatusBadge } from "../../components/ui/states";
import { useAuth } from "../auth/auth-context";

export function ProfilePage() {
  const { me } = useAuth();

  return (
    <div className="page-section stack-lg">
      <section className="panel stack">
        <div>
          <p className="eyebrow">Profile</p>
          <h1>Account details</h1>
          <p className="muted">A read-only snapshot of the current authenticated user.</p>
        </div>
        <div className="definition-grid">
          <div><span className="label">Email</span><strong>{me?.email ?? 'n/a'}</strong></div>
          <div><span className="label">Username</span><strong>{me?.username ?? 'Not set'}</strong></div>
          <div><span className="label">Role</span><StatusBadge label={me?.role ?? 'unknown'} tone={me?.role} /></div>
          <div><span className="label">Status</span><StatusBadge label={me?.is_active ? 'active' : 'inactive'} /></div>
          <div><span className="label">Email verification</span><StatusBadge label={me?.email_verified_at ? 'verified' : 'pending'} tone={me?.email_verified_at ? 'success' : 'warning'} /></div>
          <div><span className="label">Password policy</span><StatusBadge label={me?.must_change_password ? 'must change' : 'current'} tone={me?.must_change_password ? 'warning' : 'success'} /></div>
        </div>
      </section>
      <FeedbackNotice title="Need to update your password?" message="Use the Security page for password rotation and session-sensitive account actions." tone="info" />
    </div>
  );
}
