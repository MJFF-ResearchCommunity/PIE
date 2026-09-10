import { describe, it, expect, vi } from "vitest";
import { INITIAL_SURFACE_LIGHTING, probeLightingTarget } from "./renderSafety";

function context(status = 10, lost = false) {
  return {
    FRAMEBUFFER_BINDING: 1,
    TEXTURE_BINDING_3D: 2,
    FRAMEBUFFER: 3,
    TEXTURE_3D: 4,
    RGBA8: 5,
    COLOR_ATTACHMENT0: 6,
    FRAMEBUFFER_COMPLETE: 10,
    NO_ERROR: 0,
    isContextLost: vi.fn(() => lost),
    getParameter: vi.fn((key: number) =>
      key === 1 ? "previous framebuffer" : "previous texture",
    ),
    createTexture: vi.fn(() => "probe texture"),
    createFramebuffer: vi.fn(() => "probe framebuffer"),
    bindTexture: vi.fn(),
    bindFramebuffer: vi.fn(),
    texStorage3D: vi.fn(),
    framebufferTextureLayer: vi.fn(),
    checkFramebufferStatus: vi.fn(() => status),
    deleteTexture: vi.fn(),
    deleteFramebuffer: vi.fn(),
    getError: vi.fn(() => 0),
  };
}

describe("graphics safety", () => {
  it("starts without the optional gradient lighting pass", () => {
    expect(INITIAL_SURFACE_LIGHTING).toBe(false);
  });
  it("checks both ends of the actual 3D target and restores GL state", () => {
    const gl = context();
    expect(
      probeLightingTarget(
        gl as unknown as WebGL2RenderingContext,
        [256, 256, 128],
      ),
    ).toBe(true);
    expect(gl.framebufferTextureLayer.mock.calls.map((c) => c[4])).toEqual([
      0, 127,
    ]);
    expect(gl.bindFramebuffer).toHaveBeenLastCalledWith(
      3,
      "previous framebuffer",
    );
    expect(gl.bindTexture).toHaveBeenLastCalledWith(4, "previous texture");
    expect(gl.deleteTexture).toHaveBeenCalledWith("probe texture");
    expect(gl.deleteFramebuffer).toHaveBeenCalledWith("probe framebuffer");
  });
  it("rejects unsupported framebuffers and frees the probe resources", () => {
    const gl = context(36061);
    expect(
      probeLightingTarget(
        gl as unknown as WebGL2RenderingContext,
        [256, 256, 256],
      ),
    ).toBe(false);
    expect(gl.deleteTexture).toHaveBeenCalledOnce();
    expect(gl.deleteFramebuffer).toHaveBeenCalledOnce();
  });
  it("does not allocate into a lost context", () => {
    const gl = context(10, true);
    expect(
      probeLightingTarget(
        gl as unknown as WebGL2RenderingContext,
        [256, 256, 256],
      ),
    ).toBe(false);
    expect(gl.createTexture).not.toHaveBeenCalled();
  });
});
