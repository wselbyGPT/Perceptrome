import { Outlet } from "react-router-dom";
import { AppShell } from "../../app/app-shell";

export function AuthenticatedLayout() {
  return (
    <AppShell>
      <Outlet />
    </AppShell>
  );
}
