import { Check, X } from "lucide-react";

const plans = [
  {
    name: "ChatGPT Plus",
    price: "$20/mo",
    features: [
      { text: "Unlimited messages", included: true },
      { text: "Single platform", included: true },
      { text: "Requires subscription", included: false },
    ],
    highlight: false,
  },
  {
    name: "Distill",
    price: "Free",
    features: [
      { text: "2x your existing messages", included: true },
      { text: "Works on all platforms", included: true },
      { text: "No subscription needed", included: true },
    ],
    highlight: true,
  },
  {
    name: "Claude Pro",
    price: "$20/mo",
    features: [
      { text: "Unlimited messages", included: true },
      { text: "Single platform", included: true },
      { text: "Requires subscription", included: false },
    ],
    highlight: false,
  },
];

export function ComparisonSection() {
  return (
    <section className="relative py-24">
      <div className="container mx-auto px-6">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="mb-4 text-3xl font-bold text-foreground md:text-5xl">
            <span className="font-mono text-accent">1/10th</span> the cost.{" "}
            <span className="gradient-text">Same benefits.</span>
          </h2>
          <p className="mb-16 text-muted-foreground">Why pay $20/mo when you can double your free tier?</p>
        </div>

        <div className="mx-auto grid max-w-4xl gap-6 md:grid-cols-3">
          {plans.map((plan) => (
            <div
              key={plan.name}
              className={`rounded-xl border p-6 transition-all ${
                plan.highlight
                  ? "border-primary/50 bg-primary/5 glow-subtle"
                  : "border-border bg-card/50"
              }`}
            >
              <h3 className="mb-1 text-lg font-semibold text-foreground">{plan.name}</h3>
              <p className={`mb-6 font-mono text-2xl font-bold ${plan.highlight ? "text-primary" : "text-muted-foreground"}`}>
                {plan.price}
              </p>
              <ul className="space-y-3">
                {plan.features.map((feature) => (
                  <li key={feature.text} className="flex items-center gap-2 text-sm">
                    {feature.included ? (
                      <Check className="h-4 w-4 text-accent" />
                    ) : (
                      <X className="h-4 w-4 text-destructive/60" />
                    )}
                    <span className="text-muted-foreground">{feature.text}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
