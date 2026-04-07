import { Droplets } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Navbar() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
      <div className="container mx-auto flex h-16 items-center justify-between px-6">
        <div className="flex items-center gap-2">
          <Droplets className="h-6 w-6 text-primary" />
          <span className="text-lg font-bold text-foreground">Distill</span>
        </div>
        <Button size="sm" asChild>
          <a href="#">Add to Chrome</a>
        </Button>
      </div>
    </header>
  );
}
