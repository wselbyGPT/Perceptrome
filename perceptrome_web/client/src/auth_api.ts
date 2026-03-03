// client/src/auth_api.ts
export type Me = {
  id: string;
  email: string;
  username?: string | null;
  role: "admin" | "user" | string;
  is_active: boolean;
  must_change_password: boolean;
};

export type AdminUser = {
  id: string;
  email: string;
  username?: string | null;
  role: "admin" | "user" | string;
  is_active: boolean;
  must_change_password: boolean;
};

export type AdminCreateUserInput = {
  email: string;
  password: string;
  username?: string | null;
  role?: "admin" | "user";
  is_active?: boolean;
  must_change_password?: boolean;
};

type ApiErrorShape = {
  detail?: string;
};

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...init,
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });

  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const data = (await res.json()) as ApiErrorShape;
      if (data?.detail) message = data.detail;
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(message);
  }

  const text = await res.text();
  return (text ? JSON.parse(text) : {}) as T;
}

export async function getMe(): Promise<Me> {
  return apiFetch<Me>("/api/auth/me", { method: "GET" });
}

export async function login(email: string, password: string): Promise<Me> {
  return apiFetch<Me>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function logout(): Promise<void> {
  await apiFetch<{ message: string }>("/api/auth/logout", {
    method: "POST",
  });
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  await apiFetch<{ message: string }>("/api/auth/change-password", {
    method: "POST",
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
}

export async function adminListUsers(): Promise<AdminUser[]> {
  return apiFetch<AdminUser[]>("/api/admin/users", { method: "GET" });
}

export async function adminCreateUser(input: AdminCreateUserInput): Promise<AdminUser> {
  return apiFetch<AdminUser>("/api/admin/users", {
    method: "POST",
    body: JSON.stringify({
      email: input.email,
      password: input.password,
      username: input.username || null,
      role: input.role ?? "user",
      is_active: input.is_active ?? true,
      must_change_password: input.must_change_password ?? true,
    }),
  });
}
