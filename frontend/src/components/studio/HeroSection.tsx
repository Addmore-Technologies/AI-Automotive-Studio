import { Button } from "@/components/ui/button";

export function HeroSection() {
  return (
    <section id="top" className="hero-section">
      <div className="hero-inner">

        <h1 className="display-xl hero-heading">
          THE NEXT
          <br />
          GENERATION OF
          <br />
          AUTOMOTIVE STUDIO.
        </h1>

        <div className="hero-image-frame">
          <img
            src="/images/hero_supercar.png"
            alt="Black hypercar inside a premium LED automotive photography studio"
            fetchPriority="high"
            width={1200}
            height={618}
            className="hero-photograph"
          />
        </div>

        <div className="hero-bottom">
          <p className="hero-description">
            A considered setting for every vehicle.
            <br />
            Photography and film, brought into the studio.
          </p>

          <Button asChild variant="studio" size="studio">
            <a href="#workspace">
              Enter the studio <span aria-hidden>→</span>
            </a>
          </Button>
        </div>

      </div>
    </section>
  );
}
