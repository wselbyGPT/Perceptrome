import { createContext, useContext, useEffect, useMemo, useState, type PropsWithChildren } from "react";
import { useAuth } from "../auth/auth-context";
import { buildNextUrl } from "../../lib/utils";
import { makeWebSocketUrl } from "../../lib/utils";

type WsContextValue = {
  socket: WebSocket | null;
};

const WsContext = createContext<WsContextValue>({ socket: null });

export function RunsWebSocketProvider({ children }: PropsWithChildren) {
  const { me } = useAuth();
  const [socket, setSocket] = useState<WebSocket | null>(null);

  useEffect(() => {
    if (!me || me.must_change_password) {
      setSocket((existing) => {
        existing?.close();
        return null;
      });
      return;
    }

    const ws = new WebSocket(makeWebSocketUrl());
    ws.addEventListener("open", () => {
      console.log("[ws] connected");
    });
    ws.addEventListener("error", (event) => {
      console.warn("[ws] error", event);
    });
    ws.addEventListener("close", (event) => {
      console.warn(`[ws] closed code=${event.code} reason=${event.reason || "(none)"}`);
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

  const value = useMemo(() => ({ socket }), [socket]);
  return <WsContext.Provider value={value}>{children}</WsContext.Provider>;
}

export function useRunsWebSocket() {
  return useContext(WsContext);
}
