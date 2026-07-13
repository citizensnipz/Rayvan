import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type PropsWithChildren,
} from "react";
import type { Project } from "@rayvan/core";

import { projectService } from "../../lib/projects/service.js";

interface ProjectsContextValue {
  projects: Project[];
  loading: boolean;
  error: string | null;
  includeArchived: boolean;
  setIncludeArchived: (value: boolean) => void;
  refresh: () => Promise<void>;
  createProject: (input: {
    name: string;
    description?: string;
  }) => Promise<Project>;
  updateProject: (
    id: string,
    input: { name?: string; description?: string },
  ) => Promise<Project>;
  archiveProject: (id: string) => Promise<Project>;
  restoreProject: (id: string) => Promise<Project>;
  deleteProject: (id: string) => Promise<void>;
  getProject: (id: string) => Promise<Project | null>;
}

const ProjectsContext = createContext<ProjectsContextValue | null>(null);

export function ProjectsProvider({ children }: PropsWithChildren) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [includeArchived, setIncludeArchived] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const nextProjects = await projectService.list({ includeArchived });
      setProjects(nextProjects);
    } catch (refreshError) {
      setError(
        refreshError instanceof Error
          ? refreshError.message
          : "Failed to load projects",
      );
    } finally {
      setLoading(false);
    }
  }, [includeArchived]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const createProject = useCallback(
    async (input: { name: string; description?: string }) => {
      const project = await projectService.create(input);
      await refresh();
      return project;
    },
    [refresh],
  );

  const updateProject = useCallback(
    async (id: string, input: { name?: string; description?: string }) => {
      const project = await projectService.update(id, input);
      await refresh();
      return project;
    },
    [refresh],
  );

  const archiveProject = useCallback(
    async (id: string) => {
      const project = await projectService.archive(id);
      await refresh();
      return project;
    },
    [refresh],
  );

  const restoreProject = useCallback(
    async (id: string) => {
      const project = await projectService.restore(id);
      await refresh();
      return project;
    },
    [refresh],
  );

  const deleteProject = useCallback(
    async (id: string) => {
      await projectService.delete(id);
      await refresh();
    },
    [refresh],
  );

  const getProject = useCallback((id: string) => projectService.getById(id), []);

  const value = useMemo(
    () => ({
      projects,
      loading,
      error,
      includeArchived,
      setIncludeArchived,
      refresh,
      createProject,
      updateProject,
      archiveProject,
      restoreProject,
      deleteProject,
      getProject,
    }),
    [
      projects,
      loading,
      error,
      includeArchived,
      refresh,
      createProject,
      updateProject,
      archiveProject,
      restoreProject,
      deleteProject,
      getProject,
    ],
  );

  return (
    <ProjectsContext.Provider value={value}>{children}</ProjectsContext.Provider>
  );
}

export function useProjects(): ProjectsContextValue {
  const context = useContext(ProjectsContext);
  if (!context) {
    throw new Error("useProjects must be used within ProjectsProvider");
  }
  return context;
}
