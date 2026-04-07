import { Globe, ShieldCheck, Sparkles, Github } from "lucide-react";

const features = [
  {
    icon: Globe,
    title: "Works Everywhere",
    description: "ChatGPT, Claude, Gemini, Perplexity — one extension for all platforms.",
  },
  {
    icon: ShieldCheck,
    title: "Privacy First",
    description: "100% local processing. No cloud, no data collection, no tracking.",
  },
  {
    icon: Sparkles,
    title: "Zero Setup",
    description: "Install once, works forever. No API keys, no configuration needed.",
  },
  {
    icon: Github,
    title: "Open Source",
    description: "Fully auditable code. Community-driven development you can trust.",
  },
];

export function FeaturesGrid() {
  return (
    <section className="relative border-y border-border py-24">
      <div className="container mx-auto px-6">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="mb-4 text-3xl font-bold text-foreground md:text-5xl">
            Built for <span className="gradient-text">everyone</span>
          </h2>
          <p className="mb-16 text-muted-foreground">Simple, private, and powerful by design.</p>
        </div>

        <div className="mx-auto grid max-w-4xl gap-6 md:grid-cols-2">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="group rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm transition-all hover:border-primary/30"
            >
              <feature.icon className="mb-4 h-8 w-8 text-primary" />
              <h3 className="mb-2 text-lg font-semibold text-foreground">{feature.title}</h3>
              <p className="text-sm text-muted-foreground">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
