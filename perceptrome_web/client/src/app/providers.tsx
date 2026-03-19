import type { PropsWithChildren } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { AuthProvider } from "../features/auth/auth-context";
import { RunsLiveUpdatesProvider } from "../features/runs/live-updates";
import { queryClient } from "./queryClient";

export function AppProviders({ children }: PropsWithChildren) {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <RunsLiveUpdatesProvider>{children}</RunsLiveUpdatesProvider>
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
