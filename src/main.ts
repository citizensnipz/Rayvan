import { invoke } from "@tauri-apps/api/core";
import "./styles.css";

type NetworkStatus = "notConnected" | "connecting" | "connected";

interface ApplicationStatus {
  networkStatus: NetworkStatus;
}

const networkStatusLabels: Record<NetworkStatus, string> = {
  notConnected: "Not connected",
  connecting: "Connecting",
  connected: "Connected",
};

async function renderApplicationStatus(): Promise<void> {
  const statusElement = document.querySelector<HTMLParagraphElement>("#network-status");

  if (!statusElement) {
    throw new Error("The network status element is missing.");
  }

  const applicationStatus = await invoke<ApplicationStatus>("get_application_status");
  statusElement.textContent = `Network status: ${networkStatusLabels[applicationStatus.networkStatus]}`;
}

void renderApplicationStatus().catch((error: unknown) => {
  console.error("Unable to read the application status from the Rayvan core.", error);
});
