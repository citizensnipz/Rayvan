import type { ProjectId } from "@rayvan/core";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PropsWithChildren,
} from "react";

import { useProjects } from "../features/projects/ProjectsContext.js";
import {
  loadCurrentProjectId,
  saveCurrentProjectId,
} from "../lib/projects/session.js";

interface CurrentProjectContextValue {
  currentProjectId: ProjectId | null;
  setCurrentProjectId: (id: ProjectId | null) => void;
  clearCurrentProject: () => void;
  sessionReady: boolean;
}

const CurrentProjectContext = createContext<CurrentProjectContextValue | null>(
  null,
);

export function CurrentProjectProvider({ children }: PropsWithChildren) {
  const { projects, loading } = useProjects();
  const [currentProjectId, setCurrentProjectIdState] =
    useState<ProjectId | null>(null);
  const [sessionReady, setSessionReady] = useState(false);
  const restoredRef = useRef(false);

  const persistSelection = useCallback((id: ProjectId | null) => {
    void saveCurrentProjectId(id).catch((error: unknown) => {
      console.error("Failed to persist current project selection", error);
    });
  }, []);

  const setCurrentProjectId = useCallback(
    (id: ProjectId | null) => {
      setCurrentProjectIdState(id);
      persistSelection(id);
    },
    [persistSelection],
  );

  const clearCurrentProject = useCallback(() => {
    setCurrentProjectIdState(null);
    persistSelection(null);
  }, [persistSelection]);

  useEffect(() => {
    if (loading || restoredRef.current) {
      return;
    }

    let cancelled = false;

    async function restoreSession() {
      try {
        const savedId = await loadCurrentProjectId();
        if (cancelled) {
          return;
        }

        const activeProjects = projects.filter(
          (project) => project.status === "active",
        );
        const savedStillActive = activeProjects.find(
          (project) => project.id === savedId,
        );

        if (savedStillActive) {
          setCurrentProjectIdState(savedStillActive.id);
        } else if (activeProjects[0]) {
          setCurrentProjectIdState(activeProjects[0].id);
          void saveCurrentProjectId(activeProjects[0].id).catch(() => {
            /* preference restore fallback is best-effort */
          });
        } else {
          setCurrentProjectIdState(null);
        }
      } catch (error) {
        console.error("Failed to restore current project selection", error);
        const activeProjects = projects.filter(
          (project) => project.status === "active",
        );
        if (activeProjects[0]) {
          setCurrentProjectIdState(activeProjects[0].id);
        }
      } finally {
        if (!cancelled) {
          restoredRef.current = true;
          setSessionReady(true);
        }
      }
    }

    void restoreSession();

    return () => {
      cancelled = true;
    };
  }, [loading, projects]);

  useEffect(() => {
    if (!sessionReady || !currentProjectId) {
      return;
    }

    const stillPresent = projects.some(
      (project) =>
        project.id === currentProjectId && project.status === "active",
    );

    if (!stillPresent) {
      const fallback = projects.find((project) => project.status === "active");
      setCurrentProjectIdState(fallback?.id ?? null);
      void saveCurrentProjectId(fallback?.id ?? null).catch(() => {
        /* preference sync is best-effort */
      });
    }
  }, [projects, currentProjectId, sessionReady]);

  const value = useMemo(
    () => ({
      currentProjectId,
      setCurrentProjectId,
      clearCurrentProject,
      sessionReady,
    }),
    [currentProjectId, setCurrentProjectId, clearCurrentProject, sessionReady],
  );

  return (
    <CurrentProjectContext.Provider value={value}>
      {children}
    </CurrentProjectContext.Provider>
  );
}

export function useCurrentProject(): CurrentProjectContextValue {
  const context = useContext(CurrentProjectContext);
  if (!context) {
    throw new Error(
      "useCurrentProject must be used within CurrentProjectProvider",
    );
  }
  return context;
}

export function useResolvedCurrentProject() {
  const { projects, loading } = useProjects();
  const {
    currentProjectId,
    setCurrentProjectId,
    clearCurrentProject,
    sessionReady,
  } = useCurrentProject();

  const currentProject = useMemo(
    () =>
      projects.find((project) => project.id === currentProjectId) ?? null,
    [projects, currentProjectId],
  );

  return {
    currentProjectId,
    currentProject,
    setCurrentProjectId,
    clearCurrentProject,
    hasProject: currentProject !== null,
    sessionReady: sessionReady && !loading,
  };
}
