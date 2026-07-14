import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type PropsWithChildren,
} from "react";

import { useCurrentProject } from "../CurrentProjectContext.js";
import {
  APP_SECTION_DEFINITIONS,
  type AppSection,
} from "./sections.js";

interface AppNavigationContextValue {
  activeSection: AppSection;
  setActiveSection: (section: AppSection) => void;
  isCreatingProject: boolean;
  openCreateProject: () => void;
  closeCreateProject: () => void;
}

const AppNavigationContext = createContext<AppNavigationContextValue | null>(
  null,
);

function sectionRequiresProject(section: AppSection): boolean {
  return (
    APP_SECTION_DEFINITIONS.find((definition) => definition.id === section)
      ?.requiresProject ?? true
  );
}

export function AppNavigationProvider({ children }: PropsWithChildren) {
  const { currentProjectId } = useCurrentProject();
  const [activeSection, setActiveSectionState] =
    useState<AppSection>("overview");
  const [isCreatingProject, setIsCreatingProject] = useState(false);

  const setActiveSection = useCallback((section: AppSection) => {
    setActiveSectionState(section);
    setIsCreatingProject(false);
  }, []);

  const openCreateProject = useCallback(() => {
    setIsCreatingProject(true);
  }, []);

  const closeCreateProject = useCallback(() => {
    setIsCreatingProject(false);
  }, []);

  useEffect(() => {
    if (!currentProjectId && sectionRequiresProject(activeSection)) {
      setActiveSectionState("overview");
    }
  }, [currentProjectId, activeSection]);

  const value = useMemo(
    () => ({
      activeSection,
      setActiveSection,
      isCreatingProject,
      openCreateProject,
      closeCreateProject,
    }),
    [
      activeSection,
      setActiveSection,
      isCreatingProject,
      openCreateProject,
      closeCreateProject,
    ],
  );

  return (
    <AppNavigationContext.Provider value={value}>
      {children}
    </AppNavigationContext.Provider>
  );
}

export function useAppNavigation(): AppNavigationContextValue {
  const context = useContext(AppNavigationContext);
  if (!context) {
    throw new Error(
      "useAppNavigation must be used within AppNavigationProvider",
    );
  }
  return context;
}
