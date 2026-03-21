"use client";

import { useEffect, useState } from "react";
import { getHistory, type HistoryEntry } from "@/lib/api";
import { cn, sentimentColor } from "@/lib/utils";

export default function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [error, setError] = useState("");
  const pageSize = 20;

  useEffect(() => {
    getHistory(page, pageSize)
      .then((res) => {
        setEntries(res.entries);
        setTotal(res.total);
      })
      .catch((e) => setError(e.message));
  }, [page]);

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Prediction History</h1>
        <p className="text-gray-500 mt-1">{total} total predictions</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {entries.length === 0 && !error ? (
        <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
          <p className="text-gray-500">No predictions yet. Try the predict page.</p>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50">
                <th className="text-left py-3 px-4 font-medium text-gray-600">Text</th>
                <th className="text-center py-3 px-4 font-medium text-gray-600">Sentiment</th>
                <th className="text-right py-3 px-4 font-medium text-gray-600">Confidence</th>
                <th className="text-right py-3 px-4 font-medium text-gray-600">Model</th>
                <th className="text-right py-3 px-4 font-medium text-gray-600">Date</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.id} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-4 text-gray-700 max-w-sm truncate">{entry.text}</td>
                  <td className="py-3 px-4 text-center">
                    <span className={cn("font-medium capitalize", sentimentColor(entry.sentiment))}>
                      {entry.sentiment}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right font-mono">
                    {(entry.confidence * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right text-gray-500">{entry.model_used}</td>
                  <td className="py-3 px-4 text-right text-gray-400 text-xs">
                    {new Date(entry.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between px-4 py-3 bg-gray-50">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <span className="text-sm text-gray-500">
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
