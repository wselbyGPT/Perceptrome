import type { ReactNode } from "react";

export function LoadingState({ title = "Loading", message = "Please wait while we fetch the latest data.", compact = false }: { title?: string; message?: string; compact?: boolean }) {
  return (
    <div className={`state-card${compact ? ' state-card--compact' : ''}`} role="status" aria-live="polite">
      <div className="state-card__icon state-card__icon--loading" aria-hidden="true" />
      <div>
        <h3>{title}</h3>
        <p>{message}</p>
      </div>
    </div>
  );
}

export function EmptyState({ title, message, action }: { title: string; message: string; action?: ReactNode }) {
  return (
    <div className="state-card state-card--empty">
      <div className="state-card__icon" aria-hidden="true">∅</div>
      <div>
        <h3>{title}</h3>
        <p>{message}</p>
      </div>
      {action ? <div className="state-card__action">{action}</div> : null}
    </div>
  );
}

export function ErrorState({ title = "Something went wrong", message, action }: { title?: string; message: string; action?: ReactNode }) {
  return (
    <div className="state-card state-card--error" role="alert">
      <div className="state-card__icon" aria-hidden="true">!</div>
      <div>
        <h3>{title}</h3>
        <p>{message}</p>
      </div>
      {action ? <div className="state-card__action">{action}</div> : null}
    </div>
  );
}

export function FeedbackNotice({ title, message, tone = "info", actions }: { title: string; message: string; tone?: "info" | "success" | "warning" | "error"; actions?: ReactNode }) {
  return (
    <div className={`feedback-notice feedback-notice--${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <div>
        <strong>{title}</strong>
        <p>{message}</p>
      </div>
      {actions ? <div className="feedback-notice__actions">{actions}</div> : null}
    </div>
  );
}

const statusToneMap: Record<string, string> = {
  queued: 'warning',
  running: 'info',
  completed: 'success',
  failed: 'danger',
  canceled: 'muted',
  active: 'success',
  inactive: 'muted',
  admin: 'admin',
  user: 'user',
  connected: 'success',
  disconnected: 'danger',
  pending: 'warning',
};

export function StatusBadge({ label, tone }: { label: string; tone?: string }) {
  const normalized = (tone ?? statusToneMap[label.toLowerCase()] ?? 'neutral').toLowerCase();
  return <span className={`badge badge--${normalized}`}>{label}</span>;
}
