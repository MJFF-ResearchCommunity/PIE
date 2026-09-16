/** Where the development server proxies /api: `python -m pie.imaging.viewer serve --port N`. */
export const DEFAULT_API_PORT = 8765;

export function apiPort(value: string | undefined): number {
  if (value === undefined || value === "") return DEFAULT_API_PORT;
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65535)
    throw new Error(
      `PIE_VIEWER_API_PORT must be a TCP port (1-65535), got "${value}"`,
    );
  return port;
}

/** In development the page and API are separate servers; in production the API serves the page. */
export function serviceUnavailableMessage(dev: boolean, port: number) {
  return dev
    ? `The imaging service is unavailable. Start it with "python -m pie.imaging.viewer serve --port ${port}" (the dev proxy target, set by PIE_VIEWER_API_PORT), then reload.`
    : "The imaging service stopped responding. Restart python -m pie.imaging.viewer serve, then reload.";
}
