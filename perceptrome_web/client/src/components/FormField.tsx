import type { ReactNode } from "react";

type Props = {
  label: string;
  htmlFor: string;
  error?: string;
  children: ReactNode;
  help?: ReactNode;
};

export function FormField({ label, htmlFor, error, children, help }: Props) {
  return (
    <label className="input-group" htmlFor={htmlFor}>
      <span className="label">{label}</span>
      {children}
      {help ? <span className="field-help">{help}</span> : null}
      <span className="field-error">{error ?? ""}</span>
    </label>
  );
}
