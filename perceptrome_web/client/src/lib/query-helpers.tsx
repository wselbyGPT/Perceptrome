import type { ReactNode } from "react";
import { ErrorState, FeedbackNotice, LoadingState } from "../components/ui/states";

export function QueryRetryButton({ onRetry, label = "Retry" }: { onRetry: () => void; label?: string }) {
  return <button className="btn btn--secondary" type="button" onClick={onRetry}>{label}</button>;
}

export function QueryBoundary({
  isLoading,
  error,
  loadingTitle,
  loadingMessage,
  errorTitle,
  errorMessage,
  onRetry,
  children,
}: {
  isLoading: boolean;
  error: unknown;
  loadingTitle: string;
  loadingMessage: string;
  errorTitle?: string;
  errorMessage: string;
  onRetry?: () => void;
  children: ReactNode;
}) {
  if (isLoading) {
    return <LoadingState title={loadingTitle} message={loadingMessage} />;
  }

  if (error) {
    return (
      <ErrorState
        title={errorTitle}
        message={error instanceof Error ? error.message : errorMessage}
        action={onRetry ? <QueryRetryButton onRetry={onRetry} /> : undefined}
      />
    );
  }

  return <>{children}</>;
}

export function ActionFeedback({
  title,
  message,
  tone,
}: {
  title: string;
  message: string;
  tone: "info" | "success" | "warning" | "error";
}) {
  return <FeedbackNotice title={title} message={message} tone={tone} />;
}
