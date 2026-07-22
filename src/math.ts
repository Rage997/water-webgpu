export function perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[1] = 0; m[2] = 0; m[3] = 0;
  m[4] = 0;
  m[5] = f;
  m[6] = 0; m[7] = 0;
  m[8] = 0; m[9] = 0;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[12] = 0; m[13] = 0;
  m[14] = 2 * far * near * nf;
  m[15] = 0;
  return m;
}

export function lookAt(eyeX: number, eyeY: number, eyeZ: number, centerX: number, centerY: number, centerZ: number, upX: number, upY: number, upZ: number): Float32Array {
  let zx = eyeX - centerX, zy = eyeY - centerY, zz = eyeZ - centerZ;
  const zLen = Math.sqrt(zx*zx + zy*zy + zz*zz);
  zx /= zLen; zy /= zLen; zz /= zLen;

  let xx = upY * zz - upZ * zy;
  let xy = upZ * zx - upX * zz;
  let xz = upX * zy - upY * zx;
  const xLen = Math.sqrt(xx*xx + xy*xy + xz*xz);
  xx /= xLen; xy /= xLen; xz /= xLen;

  let yx = zy * xz - zz * xy;
  let yy = zz * xx - zx * xz;
  let yz = zx * xy - zy * xx;

  const m = new Float32Array(16);
  m[0] = xx; m[1] = yx; m[2] = zx; m[3] = 0;
  m[4] = xy; m[5] = yy; m[6] = zy; m[7] = 0;
  m[8] = xz; m[9] = yz; m[10] = zz; m[11] = 0;
  m[12] = -(xx*eyeX + xy*eyeY + xz*eyeZ);
  m[13] = -(yx*eyeX + yy*eyeY + yz*eyeZ);
  m[14] = -(zx*eyeX + zy*eyeY + zz*eyeZ);
  m[15] = 1;
  return m;
}

export function multiply(a: Float32Array, b: Float32Array, out: Float32Array): Float32Array {
  for (let i = 0; i < 4; i++) {
    const ai0 = a[i], ai1 = a[i+4], ai2 = a[i+8], ai3 = a[i+12];
    out[i]     = ai0*b[0]  + ai1*b[1]  + ai2*b[2]  + ai3*b[3];
    out[i+4]   = ai0*b[4]  + ai1*b[5]  + ai2*b[6]  + ai3*b[7];
    out[i+8]   = ai0*b[8]  + ai1*b[9]  + ai2*b[10] + ai3*b[11];
    out[i+12]  = ai0*b[12] + ai1*b[13] + ai2*b[14] + ai3*b[15];
  }
  return out;
}

export function inverse(m: Float32Array, out: Float32Array): Float32Array {
  const a00=m[0],a01=m[1],a02=m[2],a03=m[3];
  const a10=m[4],a11=m[5],a12=m[6],a13=m[7];
  const a20=m[8],a21=m[9],a22=m[10],a23=m[11];
  const a30=m[12],a31=m[13],a32=m[14],a33=m[15];
  const b00=a00*a11-a01*a10,b01=a00*a12-a02*a10,b02=a00*a13-a03*a10;
  const b03=a01*a12-a02*a11,b04=a01*a13-a03*a11,b05=a02*a13-a03*a12;
  const b06=a20*a31-a21*a30,b07=a20*a32-a22*a30,b08=a20*a33-a23*a30;
  const b09=a21*a32-a22*a31,b10=a21*a33-a23*a31,b11=a22*a33-a23*a32;
  const det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
  if (!det) return out;
  const invDet = 1/det;
  out[0]=(a11*b11-a12*b10+a13*b09)*invDet;
  out[1]=(-a01*b11+a02*b10-a03*b09)*invDet;
  out[2]=(a31*b05-a32*b04+a33*b03)*invDet;
  out[3]=(-a21*b05+a22*b04-a23*b03)*invDet;
  out[4]=(-a10*b11+a12*b08-a13*b07)*invDet;
  out[5]=(a00*b11-a02*b08+a03*b07)*invDet;
  out[6]=(-a30*b05+a32*b02-a33*b01)*invDet;
  out[7]=(a20*b05-a22*b02+a23*b01)*invDet;
  out[8]=(a10*b10-a11*b08+a13*b06)*invDet;
  out[9]=(-a00*b10+a01*b08-a03*b06)*invDet;
  out[10]=(a30*b04-a31*b02+a33*b00)*invDet;
  out[11]=(-a20*b04+a21*b02-a23*b00)*invDet;
  out[12]=(-a10*b09+a11*b07-a12*b06)*invDet;
  out[13]=(a00*b09-a01*b07+a02*b06)*invDet;
  out[14]=(-a30*b03+a31*b01-a32*b00)*invDet;
  out[15]=(a20*b03-a21*b01+a22*b00)*invDet;
  return out;
}
