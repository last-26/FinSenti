"use client";

import { useState } from "react";

interface SentimentInputProps {
  onSubmit: (text: string) => void;
  isLoading: boolean;
}

const EXAMPLES = [
  "Tesla reported record Q4 deliveries beating analyst expectations by 12%",
  "Fed raised interest rates by 25 basis points",
  "Apple announced a new stock buyback program worth $90B",
  "Oil prices remained steady amid OPEC uncertainty",
  "Shares dropped 15% after disappointing guidance",
  "The board will meet on Tuesday to discuss Q2 results",
];

export function SentimentInput({ onSubmit, isLoading }: SentimentInputProps) {
  const [text, setText] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim()) {
      onSubmit(text.trim());
    }
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="space-y-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter financial text to analyze sentiment..."
          className="w-full h-32 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-sm"
          maxLength={512}
        />
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">{text.length}/512</span>
          <button
            type="submit"
            disabled={!text.trim() || isLoading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? "Analyzing..." : "Analyze Sentiment"}
          </button>
        </div>
      </form>

      <div>
        <p className="text-xs text-gray-500 mb-2">Try an example:</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              onClick={() => setText(example)}
              className="px-3 py-1 text-xs bg-gray-100 text-gray-600 rounded-full hover:bg-gray-200 transition-colors text-left"
            >
              {example.length > 50 ? example.slice(0, 50) + "..." : example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
