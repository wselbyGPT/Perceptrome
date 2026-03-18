import { createContext, useContext, useEffect, useMemo, useState, type PropsWithChildren } from "react";
import { getMe, logout, type Me } from "../../auth_api";

type AuthContextValue = {
  me: Me | null;
  loading: boolean;
  refresh: () => Promise<void>;
  setMe: (me: Me | null) => void;
  logoutAndClear: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: PropsWithChildren) {
  const [me, setMe] = useState<Me | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    try {
      const nextMe = await getMe();
      setMe(nextMe);
    } catch {
      setMe(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const value = useMemo<AuthContextValue>(() => ({
    me,
    loading,
    refresh,
    setMe,
    logoutAndClear: async () => {
      try {
        await logout();
      } finally {
        setMe(null);
      }
    },
  }), [me, loading]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}
