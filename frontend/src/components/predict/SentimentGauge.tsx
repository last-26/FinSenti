"use client";

import { cn } from "@/lib/utils";

interface SentimentGaugeProps {
  probabilities: Record<string, number>;
}

export function SentimentGauge({ probabilities }: SentimentGaugeProps) {
  const items = [
    { label: "Negative", value: probabilities.negative ?? 0, color: "bg-red-500" },
    { label: "Neutral", value: probabilities.neutral ?? 0, color: "bg-amber-500" },
    { label: "Positive", value: probabilities.positive ?? 0, color: "bg-green-500" },
  ];

  return (
    <div className="space-y-3">
      {items.map((item) => (
        <div key={item.label}>
          <div className="flex justify-between text-sm mb-1">
            <span className="font-medium text-gray-700">{item.label}</span>
            <span className="text-gray-500">{(item.value * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className={cn("h-3 rounded-full transition-all duration-500", item.color)}
              style={{ width: `${Math.max(item.value * 100, 1)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
