import { AppNavigationProvider } from "./navigation/AppNavigationContext.js";
import { CurrentProjectProvider } from "./CurrentProjectContext.js";
import { ProjectsProvider } from "../features/projects/ProjectsContext.js";
import { DesktopShell } from "./shell/DesktopShell.js";
import { ThemeProvider } from "./theme/ThemeContext.js";

export function App() {
  return (
    <ThemeProvider>
      <ProjectsProvider>
        <CurrentProjectProvider>
          <AppNavigationProvider>
            <DesktopShell />
          </AppNavigationProvider>
        </CurrentProjectProvider>
      </ProjectsProvider>
    </ThemeProvider>
  );
}
