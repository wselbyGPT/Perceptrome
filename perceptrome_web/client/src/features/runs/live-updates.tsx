import { createContext, useContext, useEffect, useMemo, useState, type PropsWithChildren } from "react";
import { useAuth } from "../auth/auth-context";
import { buildNextUrl, makeWebSocketUrl } from "../../lib/utils";

export type RunsLiveUpdatesContextValue = {
  socket: WebSocket | null;
  connectionState: "idle" | "connecting" | "connected" | "disconnected";
};

const RunsLiveUpdatesContext = createContext<RunsLiveUpdatesContextValue>({
  socket: null,
  connectionState: "idle",
});

export function RunsLiveUpdatesProvider({ children }: PropsWithChildren) {
  const { me } = useAuth();
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connectionState, setConnectionState] = useState<RunsLiveUpdatesContextValue["connectionState"]>("idle");

  useEffect(() => {
    if (!me || me.must_change_password) {
      setSocket((existing) => {
        existing?.close();
        return null;
      });
      setConnectionState("idle");
      return;
    }

    const ws = new WebSocket(makeWebSocketUrl());
    setConnectionState("connecting");
    ws.addEventListener("open", () => {
      console.log("[ws] connected");
      setConnectionState("connected");
    });
    ws.addEventListener("error", (event) => {
      console.warn("[ws] error", event);
    });
    ws.addEventListener("close", (event) => {
      console.warn(`[ws] closed code=${event.code} reason=${event.reason || "(none)"}`);
      setConnectionState("disconnected");
      if (event.code === 4401) {
        window.location.replace(`/login?next=${buildNextUrl()}`);
        return;
      }
      if (event.code === 4403) {
        window.location.replace(`/security?next=${buildNextUrl()}`);
      }
    });
    setSocket(ws);

    return () => {
      ws.close();
      setSocket((current) => (current === ws ? null : current));
    };
  }, [me]);

  const value = useMemo(() => ({ socket, connectionState }), [socket, connectionState]);
  return <RunsLiveUpdatesContext.Provider value={value}>{children}</RunsLiveUpdatesContext.Provider>;
}

export function useRunsLiveUpdates() {
  return useContext(RunsLiveUpdatesContext);
}
