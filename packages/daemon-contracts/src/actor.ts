export type RayvanActor =
  | {
      type: "user";
      id: string;
      displayName?: string;
    }
  | {
      type: "mcp_client";
      id: string;
      displayName: string;
    }
  | {
      type: "plugin";
      pluginId: string;
    }
  | {
      type: "rayvan";
      id: string;
    }
  | {
      type: "system";
      id: string;
    }
  | {
      type: "desktop";
      id: string;
      displayName?: string;
    };
