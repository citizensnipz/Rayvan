import { EmptyState } from "@rayvan/ui";

interface FeaturePlaceholderProps {
  title: string;
  description: string;
}

export function FeaturePlaceholder({
  title,
  description,
}: FeaturePlaceholderProps) {
  return (
    <section>
      <h2 style={{ marginTop: 0 }}>{title}</h2>
      <EmptyState title={`${title} coming soon`} description={description} />
    </section>
  );
}
