import { Frown, Clock, Users } from "lucide-react";

const problems = [
  {
    icon: Frown,
    title: "ChatGPT Free",
    stat: "15-20 messages / 3 hours",
    description: "Hit your limit mid-conversation",
  },
  {
    icon: Clock,
    title: "Claude Free",
    stat: "50 messages / day",
    description: "Run out before lunch",
  },
  {
    icon: Users,
    title: "65% of users",
    stat: "Hit limits weekly",
    description: "A universal frustration",
  },
];

export function ProblemSection() {
  return (
    <section className="relative py-24">
      <div className="absolute inset-0 bg-radial-glow opacity-50" />
      <div className="container relative z-10 mx-auto px-6">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="mb-4 text-3xl font-bold text-foreground md:text-5xl">
            AI has a <span className="gradient-text">limits</span> problem
          </h2>
          <p className="mb-16 text-muted-foreground">
            You're paying for tokens you don't need. Filler words, redundancy, and fluff
            eat your message quota. <span className="text-foreground font-medium">Distill fixes that.</span>
          </p>
        </div>

        <div className="mx-auto grid max-w-4xl gap-6 md:grid-cols-3">
          {problems.map((problem) => (
            <div
              key={problem.title}
              className="group rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm transition-all hover:border-primary/30 hover:glow-subtle"
            >
              <problem.icon className="mb-4 h-8 w-8 text-destructive/70" />
              <h3 className="mb-1 text-lg font-semibold text-foreground">{problem.title}</h3>
              <p className="mb-3 font-mono text-sm text-accent">{problem.stat}</p>
              <p className="text-sm text-muted-foreground">{problem.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
