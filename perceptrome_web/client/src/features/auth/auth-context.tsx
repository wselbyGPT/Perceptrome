import { createContext, useContext, useMemo, type PropsWithChildren } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  changePassword,
  getMe,
  login,
  logout,
  type LoginInput,
  type Me,
} from "../../auth_api";
import { isFrontendApiError } from "../../lib/api-client";

export type AuthState =
  | "anonymous"
  | "authenticating"
  | "authenticated"
  | "must_change_password"
  | "email_unverified"
  | "forbidden";

type AuthContextValue = {
  me: Me | null;
  authState: AuthState;
  loading: boolean;
  refresh: () => Promise<Me | null>;
  setMe: (me: Me | null) => void;
  loginWithPassword: (input: LoginInput) => Promise<Me>;
  logoutAndClear: () => Promise<void>;
  changePasswordAndRefresh: (currentPassword: string, newPassword: string) => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);
const authQueryKey = ["auth", "me"] as const;

function deriveAuthState(me: Me | null, isPending: boolean, error: unknown, isAuthenticating: boolean): AuthState {
  if (isPending || isAuthenticating) return "authenticating";
  if (me?.must_change_password) return "must_change_password";
  if (me) return "authenticated";
  if (isFrontendApiError(error)) {
    if (error.status === 403) return "forbidden";
    if (error.status === 401 && error.message === "Email verification required") return "email_unverified";
  }
  return "anonymous";
}

export function AuthProvider({ children }: PropsWithChildren) {
  const queryClient = useQueryClient();
  const meQuery = useQuery({
    queryKey: authQueryKey,
    queryFn: getMe,
    retry: false,
  });

  const loginMutation = useMutation({
    mutationFn: (input: LoginInput) => login(input.email, input.password),
    onSuccess: (me) => {
      queryClient.setQueryData(authQueryKey, me);
    },
  });

  const logoutMutation = useMutation({
    mutationFn: logout,
    onSettled: () => {
      queryClient.setQueryData(authQueryKey, null);
    },
  });

  const changePasswordMutation = useMutation({
    mutationFn: ({ currentPassword, newPassword }: { currentPassword: string; newPassword: string }) =>
      changePassword(currentPassword, newPassword),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: authQueryKey });
    },
  });

  const value = useMemo<AuthContextValue>(() => {
    const me = meQuery.data ?? null;
    return {
      me,
      authState: deriveAuthState(me, meQuery.isPending, meQuery.error, loginMutation.isPending),
      loading: meQuery.isPending || loginMutation.isPending,
      refresh: async () => {
        try {
          const result = await queryClient.fetchQuery({ queryKey: authQueryKey, queryFn: getMe });
          return result;
        } catch {
          queryClient.setQueryData(authQueryKey, null);
          return null;
        }
      },
      setMe: (nextMe) => {
        queryClient.setQueryData(authQueryKey, nextMe);
      },
      loginWithPassword: async (input) => loginMutation.mutateAsync(input),
      logoutAndClear: async () => {
        await logoutMutation.mutateAsync();
      },
      changePasswordAndRefresh: async (currentPassword, newPassword) => {
        await changePasswordMutation.mutateAsync({ currentPassword, newPassword });
      },
    };
  }, [changePasswordMutation, loginMutation, logoutMutation, meQuery.data, meQuery.error, meQuery.isPending, queryClient]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
