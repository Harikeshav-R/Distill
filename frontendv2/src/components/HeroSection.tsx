import { Button } from "@/components/ui/button";
import { Chrome, ArrowDown } from "lucide-react";
import { PromptCompression } from "@/components/PromptCompression";

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center pt-16 overflow-hidden">
      {/* Soft radial glow behind headline */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] rounded-full bg-primary/8 blur-[160px]" />
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] rounded-full bg-accent/5 blur-[120px]" />

      <div className="container relative z-10 mx-auto px-6 py-20">
        <div className="mx-auto max-w-4xl text-center">
          {/* Headline */}
          <h1
            className="mb-6 text-5xl font-extrabold leading-tight tracking-tight text-foreground md:text-7xl opacity-0 animate-fade-in"
          >
            <span className="gradient-text">Twice</span> the messages.{"\n"}
            <br className="hidden sm:block" />
            Same free tier.
          </h1>

          {/* Subheadline */}
          <p
            className="mx-auto mb-10 max-w-2xl text-lg text-muted-foreground md:text-xl opacity-0 animate-fade-in"
            style={{ animationDelay: "0.15s" }}
          >
            Distill removes the words AI doesn't need from your prompts — so you
            get more out of your free ChatGPT and Claude limits. No setup, no
            cloud, runs locally in your browser.
          </p>

          {/* CTAs */}
          <div
            className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center opacity-0 animate-fade-in"
            style={{ animationDelay: "0.25s" }}
          >
            <Button
              size="lg"
              variant="secondary"
              className="gap-2 px-8 text-base bg-foreground text-background hover:bg-foreground/90 font-semibold"
              asChild
            >
              <a href="#">
                <Chrome className="h-5 w-5" />
                Add to Chrome — It's Free
              </a>
            </Button>
            <a
              href="#how-it-works"
              className="inline-flex items-center gap-1.5 text-sm text-muted-foreground/80 transition-colors hover:text-foreground font-medium border border-border/60 rounded-full px-4 py-2 hover:border-border"
            >
              See how it works
              <ArrowDown className="h-4 w-4" />
            </a>
          </div>

          {/* Before/After Compression Demo */}
          <div
            className="mt-16 opacity-0 animate-fade-in-up"
            style={{ animationDelay: "0.45s" }}
          >
            <PromptCompression />
          </div>
        </div>
      </div>
    </section>
  );
}
