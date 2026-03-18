import { FeedbackNotice, StatusBadge } from "../../components/ui/states";

const sampleInvitations = [
  { email: 'new.admin@perceptrome.local', role: 'admin', status: 'pending', sentAt: '2026-03-18T09:00:00Z' },
  { email: 'researcher@perceptrome.local', role: 'user', status: 'accepted', sentAt: '2026-03-17T14:15:00Z' },
];

export function AdminInvitationsPage() {
  return (
    <div className="page-section stack-lg">
      <section className="panel stack">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>Invitations</h1>
          <p className="muted">A lightweight operator view for invitation state until the full workflow is wired to backend APIs.</p>
        </div>
        <FeedbackNotice title="Invitation management" message="This page is routed through the shared shell now, so the eventual invitation endpoints can slot in without changing navigation." tone="info" />
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Email</th><th>Role</th><th>Status</th><th>Sent</th></tr></thead>
            <tbody>
              {sampleInvitations.map((invite) => (
                <tr key={invite.email}>
                  <td>{invite.email}</td>
                  <td><StatusBadge label={invite.role} tone={invite.role} /></td>
                  <td><StatusBadge label={invite.status} tone={invite.status === 'accepted' ? 'success' : 'warning'} /></td>
                  <td className="mono">{invite.sentAt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
