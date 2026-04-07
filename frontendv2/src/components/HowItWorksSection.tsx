import { Keyboard, Wand2, Zap, ShieldCheck } from "lucide-react";

const steps = [
  {
    icon: Keyboard,
    title: "You type normally",
    description: "Write your prompts as you always do. No changes to your workflow.",
  },
  {
    icon: Wand2,
    title: "Distill compresses invisibly",
    description: "In <200ms, locally on your device. No data ever leaves your browser.",
  },
  {
    icon: Zap,
    title: "AI responds faster",
    description: "Fewer tokens in = faster responses + more messages in your quota.",
  },
];

export function HowItWorksSection() {
  return (
    <section id="how-it-works" className="relative py-24 border-y border-border">
      <div className="container mx-auto px-6">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="mb-4 text-3xl font-bold text-foreground md:text-5xl">
            How it <span className="gradient-text">works</span>
          </h2>
          <p className="mb-16 text-muted-foreground">Three steps. Zero friction.</p>
        </div>

        <div className="mx-auto grid max-w-5xl gap-8 md:grid-cols-3">
          {steps.map((step, i) => (
            <div key={step.title} className="relative text-center">
              {/* Connector line */}
              {i < steps.length - 1 && (
                <div className="absolute right-0 top-10 hidden h-px w-full translate-x-1/2 bg-gradient-to-r from-primary/40 to-transparent md:block" />
              )}
              <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-2xl border border-border bg-secondary/50">
                <step.icon className="h-8 w-8 text-primary" />
              </div>
              <div className="mb-2 font-mono text-xs text-accent">Step {i + 1}</div>
              <h3 className="mb-2 text-lg font-semibold text-foreground">{step.title}</h3>
              <p className="text-sm text-muted-foreground">{step.description}</p>
            </div>
          ))}
        </div>

        <div className="mt-12 flex items-center justify-center gap-2 text-sm text-muted-foreground">
          <ShieldCheck className="h-4 w-4 text-accent" />
          100% local processing. Your conversations never leave your device.
        </div>
      </div>
    </section>
  );
}
