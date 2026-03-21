"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getHealth, getHistory, type HealthStatus, type HistoryEntry } from "@/lib/api";
import { cn, sentimentBg, sentimentColor } from "@/lib/utils";

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [recent, setRecent] = useState<HistoryEntry[]>([]);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setError(e.message));
    getHistory(1, 5).then((r) => setRecent(r.entries)).catch(() => {});
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">Financial Sentiment Analysis Pipeline</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
          API unavailable: {error}
        </div>
      )}

      {/* Status cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <h3 className="text-sm font-medium text-gray-500">API Status</h3>
          <p className={cn("text-lg font-semibold mt-1", health ? "text-green-600" : "text-gray-400")}>
            {health ? "Online" : "Checking..."}
          </p>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <h3 className="text-sm font-medium text-gray-500">Active Model</h3>
          <p className="text-lg font-semibold mt-1 text-gray-900">
            {health?.model_name ?? "None"}
          </p>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <h3 className="text-sm font-medium text-gray-500">Device</h3>
          <p className="text-lg font-semibold mt-1 text-gray-900">
            {health?.device?.toUpperCase() ?? "-"}
          </p>
        </div>
      </div>

      {/* Quick actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Link
          href="/predict"
          className="bg-blue-600 text-white rounded-lg p-6 hover:bg-blue-700 transition-colors"
        >
          <h3 className="text-lg font-semibold">Analyze Text</h3>
          <p className="text-blue-200 text-sm mt-1">
            Enter financial text and get sentiment prediction
          </p>
        </Link>
        <Link
          href="/batch"
          className="bg-white border border-gray-200 rounded-lg p-6 hover:bg-gray-50 transition-colors"
        >
          <h3 className="text-lg font-semibold text-gray-900">Batch Analysis</h3>
          <p className="text-gray-500 text-sm mt-1">
            Analyze multiple texts at once
          </p>
        </Link>
      </div>

      {/* Recent predictions */}
      {recent.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold text-gray-900">Recent Predictions</h2>
            <Link href="/history" className="text-sm text-blue-600 hover:underline">
              View all
            </Link>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 divide-y">
            {recent.map((entry) => (
              <div key={entry.id} className="px-4 py-3 flex items-center justify-between">
                <p className="text-sm text-gray-700 truncate max-w-md">{entry.text}</p>
                <div className="flex items-center gap-3">
                  <span className={cn("text-sm font-medium capitalize", sentimentColor(entry.sentiment))}>
                    {entry.sentiment}
                  </span>
                  <span className="text-xs text-gray-400">
                    {(entry.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
