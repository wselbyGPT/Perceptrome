type Props = {
  message?: string | null;
  tone?: "plain" | "ok" | "error";
};

export function MessageBanner({ message, tone = "plain" }: Props) {
  return <div className={`msg${tone === "plain" ? "" : ` ${tone}`}`}>{message ?? ""}</div>;
}
