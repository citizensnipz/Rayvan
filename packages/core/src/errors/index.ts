export type RayvanErrorCode =
  | "not_found"
  | "validation_failed"
  | "unauthorized"
  | "conflict"
  | "internal";

export class RayvanError extends Error {
  readonly code: RayvanErrorCode;

  constructor(code: RayvanErrorCode, message: string) {
    super(message);
    this.name = "RayvanError";
    this.code = code;
  }
}
