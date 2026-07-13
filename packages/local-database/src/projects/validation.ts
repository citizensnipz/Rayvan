import { InvalidProjectNameError } from "./errors.js";

export function validateProjectName(name: string): string {
  const trimmed = name.trim();
  if (trimmed.length === 0) {
    throw new InvalidProjectNameError();
  }
  return trimmed;
}
