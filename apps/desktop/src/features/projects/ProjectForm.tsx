import { useState, type FormEvent } from "react";
import { Button, Input, TextArea } from "@rayvan/ui";
import type { Project } from "@rayvan/core";

interface ProjectFormProps {
  initialValues?: Pick<Project, "name" | "description">;
  submitLabel: string;
  onSubmit: (input: { name: string; description?: string }) => Promise<void>;
  onCancel: () => void;
}

export function ProjectForm({
  initialValues,
  submitLabel,
  onSubmit,
  onCancel,
}: ProjectFormProps) {
  const [name, setName] = useState(initialValues?.name ?? "");
  const [description, setDescription] = useState(
    initialValues?.description ?? "",
  );
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      await onSubmit({
        name,
        description: description.trim() || undefined,
      });
    } catch (submitError) {
      setError(
        submitError instanceof Error
          ? submitError.message
          : "Failed to save project",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={(event) => void handleSubmit(event)} style={{ maxWidth: "32rem" }}>
      <div style={{ display: "grid", gap: "1rem" }}>
        <label style={{ display: "grid", gap: "0.35rem" }}>
          <span>Name</span>
          <Input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Payments API"
            required
          />
        </label>

        <label style={{ display: "grid", gap: "0.35rem" }}>
          <span>Description</span>
          <TextArea
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            placeholder="Optional summary of what this project represents"
          />
        </label>

        {error ? (
          <p style={{ margin: 0, color: "var(--color-danger)" }} role="alert">
            {error}
          </p>
        ) : null}

        <div style={{ display: "flex", gap: "0.75rem" }}>
          <Button type="submit" disabled={submitting}>
            {submitting ? "Saving..." : submitLabel}
          </Button>
          <Button type="button" onClick={onCancel} disabled={submitting}>
            Cancel
          </Button>
        </div>
      </div>
    </form>
  );
}
