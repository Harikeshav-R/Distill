import { Button } from "@/components/ui/button";
import { Chrome, Droplets } from "lucide-react";

export function CTASection() {
  return (
    <section className="relative py-32">
      <div className="absolute inset-0 bg-radial-glow" />
      <div className="container relative z-10 mx-auto px-6 text-center">
        <Droplets className="mx-auto mb-6 h-12 w-12 text-primary animate-pulse-glow" />
        <h2 className="mb-4 text-4xl font-bold text-foreground md:text-6xl">
          Your AI. <span className="gradient-text">Twice as far.</span>
        </h2>
        <p className="mx-auto mb-10 max-w-md text-muted-foreground">
          Free forever. Premium at $4.99/mo for power users.
        </p>
        <Button size="lg" className="glow-primary gap-2 px-10 text-base" asChild>
          <a href="#">
            <Chrome className="h-5 w-5" />
            Add Distill to Chrome
          </a>
        </Button>
      </div>
    </section>
  );
}
