import { useEffect, useState } from "react";

const originalWords = [
  { text: "Can", keep: true },
  { text: "you", keep: true },
  { text: "please", keep: false },
  { text: "help", keep: true },
  { text: "me", keep: true },
  { text: "write", keep: true },
  { text: "a", keep: true },
  { text: "really", keep: false },
  { text: "good", keep: false },
  { text: "and", keep: false },
  { text: "detailed", keep: false },
  { text: "Python", keep: true },
  { text: "function", keep: true },
  { text: "that", keep: true },
  { text: "is", keep: false },
  { text: "able", keep: false },
  { text: "to", keep: false },
  { text: "sort", keep: true },
  { text: "a", keep: true },
  { text: "list", keep: true },
  { text: "of", keep: true },
  { text: "numbers", keep: true },
  { text: "in", keep: true },
  { text: "ascending", keep: true },
  { text: "order", keep: true },
  { text: "efficiently?", keep: true },
];

export function PromptCompression() {
  const [showStrike, setShowStrike] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setShowStrike(true), 1200);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="mx-auto max-w-3xl overflow-hidden rounded-xl border border-border bg-card/60 backdrop-blur-sm shadow-lg">
      {/* Browser chrome mockup */}
      <div className="flex items-center gap-2 border-b border-border bg-secondary/40 px-4 py-3">
        <div className="flex items-center gap-1.5">
          <div className="h-3 w-3 rounded-full bg-destructive/50" />
          <div className="h-3 w-3 rounded-full bg-accent/30" />
          <div className="h-3 w-3 rounded-full bg-primary/30" />
        </div>
        <div className="ml-2 flex-1 rounded-md bg-secondary/60 px-3 py-1 text-xs text-muted-foreground font-mono">
          chatgpt.com
        </div>
        <span className="font-mono text-xs text-muted-foreground">
          {showStrike ? "52% compressed" : "analyzing…"}
        </span>
      </div>

      {/* Prompt content */}
      <div className="p-6">
        <p className="mb-3 font-mono text-xs text-muted-foreground uppercase tracking-wider">
          Your prompt
        </p>
        <div className="text-left text-base leading-relaxed md:text-lg">
          {originalWords.map((word, i) => (
            <span key={i} className="inline">
              <span
                className={`relative inline transition-all duration-500 ${
                  showStrike && !word.keep
                    ? "text-muted-foreground/30 line-through decoration-destructive/60"
                    : "text-foreground"
                }`}
                style={{
                  transitionDelay: showStrike ? `${i * 40}ms` : "0ms",
                }}
              >
                {word.text}
              </span>{" "}
            </span>
          ))}
        </div>

        {showStrike && (
          <div className="mt-5 border-t border-border pt-5 text-left">
            <p className="mb-2 font-mono text-xs text-accent uppercase tracking-wider">
              What AI actually receives
            </p>
            <p className="text-base text-foreground md:text-lg">
              {originalWords
                .filter((w) => w.keep)
                .map((w) => w.text)
                .join(" ")}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
