import { Gauge, Target, Timer, Zap } from "lucide-react";

const stats = [
  { icon: Gauge, value: "52%", label: "Average compression", color: "text-primary" },
  { icon: Target, value: "95.7%", label: "Accuracy preserved", color: "text-accent" },
  { icon: Timer, value: "<200ms", label: "Processing time", color: "text-primary" },
  { icon: Zap, value: "82.5%", label: "Faster AI responses", color: "text-accent" },
];

export function StatsBar() {
  return (
    <section className="relative border-y border-border bg-secondary/30 py-12">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
          {stats.map((stat) => (
            <div key={stat.label} className="flex flex-col items-center text-center">
              <stat.icon className={`mb-3 h-6 w-6 ${stat.color}`} />
              <span className={`font-mono text-3xl font-bold ${stat.color} md:text-4xl`}>
                {stat.value}
              </span>
              <span className="mt-1 text-sm text-muted-foreground">{stat.label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
