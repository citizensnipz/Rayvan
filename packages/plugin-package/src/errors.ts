export class PluginPackageError extends Error {
  constructor(
    message: string,
    readonly code:
      | "INVALID_ARCHIVE"
      | "CHECKSUM_MISMATCH"
      | "MANIFEST_INVALID"
      | "SIGNATURE_REJECTED"
      | "LAYOUT_INVALID"
      | "HOST_MISMATCH",
  ) {
    super(message);
    this.name = "PluginPackageError";
  }
}
