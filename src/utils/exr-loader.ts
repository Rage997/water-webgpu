export interface EquirectData {
  // Float16-encoded RGBA, ready to upload straight into an rgba16float texture.
  data: Uint16Array;
  width: number;
  height: number;
}

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

let worker: Worker | null = null;
let nextRequestId = 0;
const pendingRequests = new Map<number, {
  resolve: (data: EquirectData) => void;
  reject: (error: Error) => void;
}>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./exr-loader.worker.ts', import.meta.url), { type: 'module' });
    worker.addEventListener('message', (event: MessageEvent<LoadResponse>) => {
      const { id, success, data, width, height, error } = event.data;
      const pending = pendingRequests.get(id);
      if (!pending) return;

      pendingRequests.delete(id);

      if (success && data && width && height) {
        pending.resolve({ data, width, height });
      } else {
        pending.reject(new Error(error || 'Unknown error loading EXR'));
      }
    });
  }
  return worker;
}

export async function loadEXR(url: string): Promise<EquirectData> {
  return new Promise((resolve, reject) => {
    const id = nextRequestId++;
    pendingRequests.set(id, { resolve, reject });

    const request: LoadRequest = { id, url };
    getWorker().postMessage(request);
  });
}
