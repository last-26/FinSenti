"use client";

import type { SentimentResult as SentimentResultType } from "@/lib/api";
import { cn, formatConfidence, marketSignalLabel, sentimentBg, sentimentColor } from "@/lib/utils";
import { SentimentGauge } from "./SentimentGauge";

interface SentimentResultProps {
  result: SentimentResultType;
}

export function SentimentResult({ result }: SentimentResultProps) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className={cn("px-6 py-4", sentimentBg(result.sentiment))}>
        <div className="flex items-center justify-between">
          <div>
            <span className={cn("text-2xl font-bold capitalize", sentimentColor(result.sentiment))}>
              {result.sentiment}
            </span>
            <span className="ml-3 text-lg text-gray-600">
              {formatConfidence(result.confidence)}
            </span>
          </div>
          <div className="text-right">
            <span className="text-sm text-gray-500">Signal: </span>
            <span className="font-semibold text-gray-700">
              {marketSignalLabel(result.market_signal)}
            </span>
          </div>
        </div>
      </div>

      <div className="px-6 py-4 space-y-4">
        {/* Probability gauge */}
        <SentimentGauge probabilities={result.probabilities} />

        {/* Entities */}
        {result.entities.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Entities Detected</h4>
            <div className="flex flex-wrap gap-2">
              {result.entities.map((entity) => (
                <span
                  key={entity}
                  className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs font-medium"
                >
                  {entity}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Meta */}
        <div className="flex items-center justify-between text-xs text-gray-400 pt-2 border-t">
          <span>Model: {result.model_used}</span>
          <span>{result.inference_time_ms.toFixed(1)} ms</span>
        </div>
      </div>
    </div>
  );
}
