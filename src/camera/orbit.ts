import { inverse, multiply } from '../math.ts';

export class OrbitCamera {
  theta = 0.8;
  phi = 0.6;
  radius = 400;
  targetX = 0;
  targetY = 0;
  targetZ = 0;
  fov = 75 * Math.PI / 180;
  near = 1;
  far = 10000;

  private _viewMatrix: Float32Array = new Float32Array(16);
  private _projMatrix: Float32Array = new Float32Array(16);
  private _viewProjMatrix: Float32Array = new Float32Array(16);

  private _isDragging = false;
  private _prevMouseX = 0;
  private _prevMouseY = 0;

  get eyeX() { return this.radius * Math.sin(this.phi) * Math.sin(this.theta); }
  get eyeY() { return this.radius * Math.cos(this.phi); }
  get eyeZ() { return this.radius * Math.sin(this.phi) * Math.cos(this.theta); }

  get viewMatrix(): Float32Array { return this._viewMatrix; }
  get projMatrix(): Float32Array { return this._projMatrix; }
  get viewProjMatrix(): Float32Array {
    multiply(this._projMatrix, this._viewMatrix, this._viewProjMatrix);
    return this._viewProjMatrix;
  }

  invViewProjMatrix(out: Float32Array) {
    const vp = this.viewProjMatrix;
    inverse(vp, out);
  }

  update(aspect: number) {
    this.updateAspect(aspect);
    this.updateView();
  }

  updateAspect(aspect: number) {
    this._projMatrix = this._perspective(this.fov, aspect, this.near, this.far) as unknown as Float32Array;
  }

  private _perspective(fovY: number, aspect: number, near: number, far: number) {
    const f = 1 / Math.tan(fovY / 2);
    const nf = 1 / (near - far);
    const m = new Float32Array(16);
    m[0] = f / aspect; m[1] = 0; m[2] = 0; m[3] = 0;
    m[4] = 0; m[5] = f; m[6] = 0; m[7] = 0;
    m[8] = 0; m[9] = 0; m[10] = (far + near) * nf; m[11] = -1;
    m[12] = 0; m[13] = 0; m[14] = 2 * far * near * nf; m[15] = 0;
    return m;
  }

  updateView() {
    this._viewMatrix = this._lookAt(
      this.eyeX, this.eyeY, this.eyeZ,
      this.targetX, this.targetY, this.targetZ,
      0, 1, 0
    ) as unknown as Float32Array;
  }

  private _lookAt(ex: number, ey: number, ez: number, cx: number, cy: number, cz: number, ux: number, uy: number, uz: number) {
    let zx = ex - cx, zy = ey - cy, zz = ez - cz;
    const zLen = Math.sqrt(zx*zx + zy*zy + zz*zz);
    zx /= zLen; zy /= zLen; zz /= zLen;

    let xx = uy*zz - uz*zy, xy = uz*zx - ux*zz, xz = ux*zy - uy*zx;
    const xLen = Math.sqrt(xx*xx + xy*xy + xz*xz);
    xx /= xLen; xy /= xLen; xz /= xLen;

    const yx = zy*xz - zz*xy, yy = zz*xx - zx*xz, yz = zx*xy - zy*xx;

    const m = new Float32Array(16);
    m[0] = xx; m[1] = yx; m[2] = zx; m[3] = 0;
    m[4] = xy; m[5] = yy; m[6] = zy; m[7] = 0;
    m[8] = xz; m[9] = yz; m[10] = zz; m[11] = 0;
    m[12] = -(xx*ex + xy*ey + xz*ez);
    m[13] = -(yx*ex + yy*ey + yz*ez);
    m[14] = -(zx*ex + zy*ey + zz*ez);
    m[15] = 1;
    return m;
  }

  attach(canvas: HTMLCanvasElement) {
    canvas.addEventListener('pointerdown', (e) => {
      this._isDragging = true;
      this._prevMouseX = e.clientX;
      this._prevMouseY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!this._isDragging) return;
      this.theta -= (e.clientX - this._prevMouseX) * 0.005;
      this.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.phi + (e.clientY - this._prevMouseY) * 0.005));
      this._prevMouseX = e.clientX;
      this._prevMouseY = e.clientY;
    });

    canvas.addEventListener('pointerup', () => { this._isDragging = false; });
    canvas.addEventListener('pointercancel', () => { this._isDragging = false; });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.radius *= 1 + e.deltaY * 0.001;
      this.radius = Math.max(50, Math.min(2000, this.radius));
    }, { passive: false });
  }
}
