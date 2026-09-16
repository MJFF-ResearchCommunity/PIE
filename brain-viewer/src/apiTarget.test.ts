import { describe, expect, it } from "vitest";
import {
  DEFAULT_API_PORT,
  apiPort,
  serviceUnavailableMessage,
} from "./apiTarget";

describe("dev API target", () => {
  it("defaults to the server's default port and accepts an override", () => {
    expect(apiPort(undefined)).toBe(DEFAULT_API_PORT);
    expect(apiPort("")).toBe(8765);
    expect(apiPort("9001")).toBe(9001);
  });

  it("rejects values that are not a TCP port", () => {
    for (const bad of ["0", "65536", "80a", "1.5", "-1"])
      expect(() => apiPort(bad)).toThrow(/PIE_VIEWER_API_PORT/);
  });

  it("names the proxied port in development and no fixed port in production", () => {
    expect(serviceUnavailableMessage(true, 9001)).toContain("--port 9001");
    expect(serviceUnavailableMessage(false, 9001)).not.toMatch(/\d{4}/);
  });
});
