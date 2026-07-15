export type {
  ConfigurationDerivedFinding,
  ConfigurationMatrixCellStatus,
  ConfigurationMatrixCellViewModel,
  ConfigurationMatrixColumnViewModel,
  ConfigurationMatrixRowViewModel,
  ConfigurationMatrixViewModel,
} from "./types.js";
export {
  buildConfigurationDerivedFindings,
  buildConfigurationMatrix,
} from "./build-matrix.js";
export {
  filterConfigurationMatrix,
  type ConfigurationMatrixFilters,
} from "./filter-matrix.js";
