import { apiRequest } from "./lib/api-client";

export type Me = {
  id: string;
  email: string;
  username?: string | null;
  role: "admin" | "user" | string;
  is_active: boolean;
  must_change_password: boolean;
  email_verified_at?: string | null;
};

export type AdminUser = {
  id: string;
  email: string;
  username?: string | null;
  role: "admin" | "user" | string;
  is_active: boolean;
  must_change_password: boolean;
  email_verified_at?: string | null;
  email_verification_sent_at?: string | null;
  created_at: string;
  last_login_at?: string | null;
  locked_until?: string | null;
  failed_login_count: number;
  account_state: "active" | "suspended" | string;
  is_locked: boolean;
};

export type AdminUserListResponse = {
  users: AdminUser[];
  total: number;
};

export type AdminCreateUserInput = {
  email: string;
  password: string;
  username?: string | null;
  role?: "admin" | "user";
  is_active?: boolean;
  must_change_password?: boolean;
};

export type AdminUserUpdateInput = {
  username?: string | null;
  role?: "admin" | "user";
  is_active?: boolean;
  must_change_password?: boolean;
};

export type AdminUserFilters = {
  search?: string;
  role?: "admin" | "user" | "all";
  state?: "active" | "suspended" | "all";
  verification?: "verified" | "pending" | "all";
  must_change_password?: boolean | null;
};

export type AdminUserActionResponse = {
  message: string;
  user: AdminUser;
  revoked_session_count: number;
};

export type LoginInput = {
  email: string;
  password: string;
};

export async function getMe(): Promise<Me> {
  return apiRequest<Me>("/api/auth/me", { method: "GET" });
}

export async function login(email: string, password: string): Promise<Me> {
  return apiRequest<Me>("/api/auth/login", {
    method: "POST",
    body: { email, password },
  });
}

export async function verifyEmail(token: string): Promise<string> {
  const out = await apiRequest<{ message: string }>("/api/auth/verify-email", {
    method: "POST",
    body: { token },
  });
  return out.message;
}

export async function resendVerification(email: string): Promise<string> {
  const out = await apiRequest<{ message: string }>("/api/auth/resend-verification", {
    method: "POST",
    body: { email },
  });
  return out.message;
}

export async function forgotPassword(email: string): Promise<string> {
  const out = await apiRequest<{ message: string }>("/api/auth/forgot-password", {
    method: "POST",
    body: { email },
  });
  return out.message;
}

export async function resetPassword(token: string, newPassword: string): Promise<string> {
  const out = await apiRequest<{ message: string }>("/api/auth/reset-password", {
    method: "POST",
    body: { token, new_password: newPassword },
  });
  return out.message;
}

export async function logout(): Promise<void> {
  await apiRequest<{ message: string }>("/api/auth/logout", {
    method: "POST",
  });
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  await apiRequest<{ message: string }>("/api/auth/change-password", {
    method: "POST",
    body: {
      current_password: currentPassword,
      new_password: newPassword,
    },
  });
}

export async function adminListUsers(filters: AdminUserFilters = {}): Promise<AdminUserListResponse> {
  const params = new URLSearchParams();
  if (filters.search?.trim()) params.set("search", filters.search.trim());
  if (filters.role && filters.role !== "all") params.set("role", filters.role);
  if (filters.state && filters.state !== "all") params.set("state", filters.state);
  if (filters.verification && filters.verification !== "all") params.set("verification", filters.verification);
  if (typeof filters.must_change_password === "boolean") params.set("must_change_password", String(filters.must_change_password));
  const query = params.toString();
  return apiRequest<AdminUserListResponse>(`/api/admin/users${query ? `?${query}` : ""}`, { method: "GET" });
}

export async function adminCreateUser(input: AdminCreateUserInput): Promise<AdminUser> {
  return apiRequest<AdminUser>("/api/admin/users", {
    method: "POST",
    body: {
      email: input.email,
      password: input.password,
      username: input.username || null,
      role: input.role ?? "user",
      is_active: input.is_active ?? true,
      must_change_password: input.must_change_password ?? true,
    },
  });
}

export async function adminUpdateUser(userId: string, input: AdminUserUpdateInput): Promise<AdminUser> {
  return apiRequest<AdminUser>(`/api/admin/users/${userId}`, {
    method: "PATCH",
    body: input,
  });
}

async function adminUserAction(userId: string, action: string): Promise<AdminUserActionResponse> {
  return apiRequest<AdminUserActionResponse>(`/api/admin/users/${userId}/${action}`, { method: "POST" });
}

export const adminSuspendUser = (userId: string) => adminUserAction(userId, "suspend");
export const adminActivateUser = (userId: string) => adminUserAction(userId, "activate");
export const adminForceResetUser = (userId: string) => adminUserAction(userId, "force-reset");
export const adminResendVerificationUser = (userId: string) => adminUserAction(userId, "resend-verification");
export const adminRevokeUserSessions = (userId: string) => adminUserAction(userId, "revoke-sessions");
