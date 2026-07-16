/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_FORCE_DEV_GATEWAYS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
