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
};

export type AdminCreateUserInput = {
  email: string;
  password: string;
  username?: string | null;
  role?: "admin" | "user";
  is_active?: boolean;
  must_change_password?: boolean;
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

export async function adminListUsers(): Promise<AdminUser[]> {
  return apiRequest<AdminUser[]>("/api/admin/users", { method: "GET" });
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
