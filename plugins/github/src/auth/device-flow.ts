/**
 * GitHub device-flow helpers. Uses the public OAuth client_id only —
 * NEVER bundle a GitHub App private key.
 */

export interface DeviceFlowStartResult {
  deviceCode: string;
  userCode: string;
  verificationUri: string;
  expiresIn: number;
  interval: number;
}

export interface DeviceFlowTokenResult {
  accessToken: string;
  tokenType: string;
  scope?: string;
}

export interface DeviceFlowClientOptions {
  /** Public OAuth App client id (safe to ship). */
  clientId: string;
  fetchImpl?: typeof fetch;
}

export async function startGithubDeviceFlow(
  options: DeviceFlowClientOptions,
): Promise<DeviceFlowStartResult> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl("https://github.com/login/device/code", {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "User-Agent": "rayvan-plugin-github",
    },
    body: JSON.stringify({
      client_id: options.clientId,
      scope: "repo workflow",
    }),
  });
  if (!response.ok) {
    throw new Error(`GitHub device flow start failed (${response.status})`);
  }
  const body = (await response.json()) as {
    device_code: string;
    user_code: string;
    verification_uri: string;
    expires_in: number;
    interval: number;
  };
  return {
    deviceCode: body.device_code,
    userCode: body.user_code,
    verificationUri: body.verification_uri,
    expiresIn: body.expires_in,
    interval: body.interval,
  };
}

export async function pollGithubDeviceFlow(
  options: DeviceFlowClientOptions & { deviceCode: string },
): Promise<
  | { status: "pending" | "slow_down" }
  | { status: "complete"; token: DeviceFlowTokenResult }
  | { status: "denied" | "expired" }
> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(
    "https://github.com/login/oauth/access_token",
    {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        "User-Agent": "rayvan-plugin-github",
      },
      body: JSON.stringify({
        client_id: options.clientId,
        device_code: options.deviceCode,
        grant_type: "urn:ietf:params:oauth:grant-type:device_code",
      }),
    },
  );
  if (!response.ok) {
    throw new Error(`GitHub device flow poll failed (${response.status})`);
  }
  const body = (await response.json()) as {
    error?: string;
    access_token?: string;
    token_type?: string;
    scope?: string;
  };
  if (body.access_token) {
    return {
      status: "complete",
      token: {
        accessToken: body.access_token,
        tokenType: body.token_type ?? "bearer",
        scope: body.scope,
      },
    };
  }
  if (body.error === "authorization_pending") {
    return { status: "pending" };
  }
  if (body.error === "slow_down") {
    return { status: "slow_down" };
  }
  if (body.error === "access_denied") {
    return { status: "denied" };
  }
  if (body.error === "expired_token") {
    return { status: "expired" };
  }
  return { status: "pending" };
}
