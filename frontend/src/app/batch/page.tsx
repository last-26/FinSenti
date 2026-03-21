"use client";

import { useState } from "react";
import { predictBatch, type BatchResponse } from "@/lib/api";
import { cn, sentimentColor } from "@/lib/utils";

export default function BatchPage() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const texts = input
      .split("\n")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    if (texts.length === 0) return;

    setIsLoading(true);
    setError("");
    try {
      const res = await predictBatch(texts);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Batch Analysis</h1>
        <p className="text-gray-500 mt-1">Analyze multiple texts at once (one per line)</p>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <form onSubmit={handleSubmit} className="space-y-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={"Enter one text per line...\n\nExample:\nTesla reported record earnings\nFed raised interest rates\nApple stock remained flat"}
            className="w-full h-48 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-sm font-mono"
          />
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-400">
              {input.split("\n").filter((t) => t.trim()).length} texts
            </span>
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? "Analyzing..." : "Analyze All"}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          {/* Summary */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-green-700">{result.summary.positive}</div>
              <div className="text-xs text-green-600">Positive</div>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-amber-700">{result.summary.neutral}</div>
              <div className="text-xs text-amber-600">Neutral</div>
            </div>
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-red-700">{result.summary.negative}</div>
              <div className="text-xs text-red-600">Negative</div>
            </div>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">
                {(result.summary.avg_confidence * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-blue-600">Avg Confidence</div>
            </div>
          </div>

          {/* Results table */}
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 bg-gray-50">
                  <th className="text-left py-3 px-4 font-medium text-gray-600">Text</th>
                  <th className="text-center py-3 px-4 font-medium text-gray-600">Sentiment</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-600">Confidence</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-600">Time</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((r, i) => (
                  <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4 text-gray-700 max-w-md truncate">{r.text}</td>
                    <td className="py-3 px-4 text-center">
                      <span className={cn("font-medium capitalize", sentimentColor(r.sentiment))}>
                        {r.sentiment}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-mono">
                      {(r.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4 text-right text-gray-400">
                      {r.inference_time_ms.toFixed(1)} ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="px-4 py-2 bg-gray-50 text-xs text-gray-500 text-right">
              Total: {result.total_inference_time_ms.toFixed(0)} ms
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
