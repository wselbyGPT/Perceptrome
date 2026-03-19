import type { ReactNode } from "react";

export function WorkspacePage({
  eyebrow,
  title,
  description,
  actions,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="page-section stack-lg">
      <section className="hero-card panel">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
          <p className="hero-copy">{description}</p>
        </div>
        {actions ? <div className="hero-actions">{actions}</div> : null}
      </section>
      {children}
    </div>
  );
}
