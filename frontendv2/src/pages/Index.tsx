import { Navbar } from "@/components/Navbar";
import { HeroSection } from "@/components/HeroSection";
import { StatsBar } from "@/components/StatsBar";
import { ProblemSection } from "@/components/ProblemSection";
import { HowItWorksSection } from "@/components/HowItWorksSection";
import { ComparisonSection } from "@/components/ComparisonSection";
import { FeaturesGrid } from "@/components/FeaturesGrid";
import { CTASection } from "@/components/CTASection";
import { Droplets } from "lucide-react";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main>
        <HeroSection />
        <StatsBar />
        <ProblemSection />
        <HowItWorksSection />
        <ComparisonSection />
        <FeaturesGrid />
        <CTASection />
      </main>
      <footer className="border-t border-border py-8">
        <div className="container mx-auto flex items-center justify-between px-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <Droplets className="h-4 w-4 text-primary" />
            <span>Distill</span>
          </div>
          <p>© 2025 Distill. Open source & free forever.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
