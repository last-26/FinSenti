"use client";

import { useState } from "react";
import { predict, type SentimentResult as SentimentResultType } from "@/lib/api";
import { SentimentInput } from "@/components/predict/SentimentInput";
import { SentimentResult } from "@/components/predict/SentimentResult";

export default function PredictPage() {
  const [result, setResult] = useState<SentimentResultType | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>("");

  const handleSubmit = async (text: string) => {
    setIsLoading(true);
    setError("");
    try {
      const res = await predict(text);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Sentiment Analysis</h1>
        <p className="text-gray-500 mt-1">Enter financial text to classify sentiment</p>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <SentimentInput onSubmit={handleSubmit} isLoading={isLoading} />
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {result && <SentimentResult result={result} />}
    </div>
  );
}
