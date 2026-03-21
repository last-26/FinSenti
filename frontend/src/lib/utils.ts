import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

export function sentimentColor(sentiment: string): string {
  switch (sentiment) {
    case "positive":
      return "text-positive";
    case "negative":
      return "text-negative";
    case "neutral":
      return "text-neutral";
    default:
      return "text-gray-500";
  }
}

export function sentimentBg(sentiment: string): string {
  switch (sentiment) {
    case "positive":
      return "bg-positive-light";
    case "negative":
      return "bg-negative-light";
    case "neutral":
      return "bg-neutral-light";
    default:
      return "bg-gray-100";
  }
}

export function marketSignalLabel(signal: string): string {
  switch (signal) {
    case "bullish":
      return "Bullish";
    case "bearish":
      return "Bearish";
    case "hold":
      return "Hold";
    default:
      return signal;
  }
}
