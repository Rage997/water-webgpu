// @ts-ignore - three.js examples ship without bundled type declarations
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
// @ts-ignore - three.js core ships without bundled type declarations here
import { HalfFloatType } from 'three';

interface LoadRequest {
  id: number;
  url: string;
}

interface LoadResponse {
  id: number;
  success: boolean;
  data?: Uint16Array;
  width?: number;
  height?: number;
  error?: string;
}

self.addEventListener('message', (event: MessageEvent<LoadRequest>) => {
  const { id, url } = event.data;

  const loader = new EXRLoader();
  loader.setDataType(HalfFloatType);

  loader.load(
    url,
    (texture: unknown) => {
      if (
        texture &&
        typeof texture === 'object' &&
        'image' in texture &&
        texture.image &&
        typeof texture.image === 'object' &&
        'data' in texture.image &&
        'width' in texture.image &&
        'height' in texture.image
      ) {
        const raw = texture.image.data;
        if (!(raw instanceof Uint16Array)) {
          const response: LoadResponse = {
            id,
            success: false,
            error: `EXR data is not Uint16Array (got ${raw?.constructor?.name})`,
          };
          self.postMessage(response);
          return;
        }

        const response: LoadResponse = {
          id,
          success: true,
          data: raw,
          width: texture.image.width as number,
          height: texture.image.height as number,
        };
        // Transfer the ArrayBuffer to avoid copying
        self.postMessage(response, { transfer: [raw.buffer] });
      } else {
        const response: LoadResponse = {
          id,
          success: false,
          error: 'Invalid EXR texture format',
        };
        self.postMessage(response);
      }
    },
    undefined,
    (error: unknown) => {
      const response: LoadResponse = {
        id,
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
      self.postMessage(response);
    }
  );
});
