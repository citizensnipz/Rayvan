import { ProjectService } from "@rayvan/local-database";

import { tauriProjectRepository } from "./repository.js";

export const projectService = new ProjectService(tauriProjectRepository);
