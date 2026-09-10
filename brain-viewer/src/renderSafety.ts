/** Check the actual render target, rather than assuming browser/GPU support. */
export function probeLightingTarget(
  gl: WebGL2RenderingContext,
  dims: number[],
): boolean {
  if (gl.isContextLost()) return false;
  const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  const previousTexture = gl.getParameter(gl.TEXTURE_BINDING_3D);
  const texture = gl.createTexture();
  const framebuffer = gl.createFramebuffer();
  if (!texture || !framebuffer) {
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(framebuffer);
    return false;
  }
  try {
    gl.bindTexture(gl.TEXTURE_3D, texture);
    gl.texStorage3D(gl.TEXTURE_3D, 1, gl.RGBA8, dims[0], dims[1], dims[2]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    for (const layer of [0, dims[2] - 1]) {
      gl.framebufferTextureLayer(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        texture,
        0,
        layer,
      );
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        return false;
    }
    return !gl.isContextLost() && gl.getError() === gl.NO_ERROR;
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
    gl.bindTexture(gl.TEXTURE_3D, previousTexture);
    gl.deleteFramebuffer(framebuffer);
    gl.deleteTexture(texture);
  }
}

/** The optional lighting path must never run automatically on initial load. */
export const INITIAL_SURFACE_LIGHTING = false;
