import { ImageOff } from "lucide-react";
import { useState } from "react";

type PlotImageProps = {
  src: string;
  alt: string;
  className?: string;
  loading?: "lazy" | "eager";
};

export function PlotImage({ src, alt, className = "", loading = "lazy" }: PlotImageProps) {
  const [failed, setFailed] = useState(false);
  if (!src || failed) {
    return (
      <div className={`plot-image-fallback ${className}`} role="img" aria-label={`${alt}; image unavailable`}>
        <ImageOff size={28} />
        <span>Preview unavailable</span>
      </div>
    );
  }
  return <img className={className} src={src} alt={alt} loading={loading} onError={() => setFailed(true)} />;
}
