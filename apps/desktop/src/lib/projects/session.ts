import { invoke } from "@tauri-apps/api/core";

export async function loadCurrentProjectId(): Promise<string | null> {
  return invoke<string | null>("get_current_project_id");
}

export async function saveCurrentProjectId(
  projectId: string | null,
): Promise<void> {
  await invoke("set_current_project_id", { projectId });
}
