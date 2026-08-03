/** Hard caps for `.rayvan-plugin` archives (zip-bomb / DoS mitigation). */
export const PLUGIN_PACKAGE_LIMITS = {
  /** Maximum compressed archive size on disk. */
  maxArchiveBytes: 32 * 1024 * 1024,
  /** Maximum number of non-directory entries. */
  maxEntries: 512,
  /** Maximum size of a single uncompressed entry. */
  maxEntryBytes: 32 * 1024 * 1024,
  /** Maximum total uncompressed payload size. */
  maxTotalUncompressedBytes: 64 * 1024 * 1024,
} as const;
