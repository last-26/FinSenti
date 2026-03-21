"use client";

import type { RunSummary } from "@/lib/api";

interface MetricsTableProps {
  runs: RunSummary[];
}

export function MetricsTable({ runs }: MetricsTableProps) {
  if (runs.length === 0) {
    return <p className="text-gray-500 text-sm">No runs found.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="text-left py-3 px-4 font-medium text-gray-600">Run</th>
            <th className="text-left py-3 px-4 font-medium text-gray-600">Base Model</th>
            <th className="text-left py-3 px-4 font-medium text-gray-600">Status</th>
            <th className="text-right py-3 px-4 font-medium text-gray-600">F1 Macro</th>
            <th className="text-right py-3 px-4 font-medium text-gray-600">Accuracy</th>
            <th className="text-right py-3 px-4 font-medium text-gray-600">Latency (p50)</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id} className="border-b border-gray-100 hover:bg-gray-50">
              <td className="py-3 px-4">
                <div className="font-medium text-gray-900">{run.run_name}</div>
                <div className="text-xs text-gray-400">{run.run_id.slice(0, 8)}...</div>
              </td>
              <td className="py-3 px-4 text-gray-600">{run.base_model ?? "-"}</td>
              <td className="py-3 px-4">
                <span
                  className={`px-2 py-0.5 rounded text-xs font-medium ${
                    run.status === "FINISHED"
                      ? "bg-green-50 text-green-700"
                      : "bg-yellow-50 text-yellow-700"
                  }`}
                >
                  {run.status}
                </span>
              </td>
              <td className="py-3 px-4 text-right font-mono">
                {run.f1_macro != null ? run.f1_macro.toFixed(4) : "-"}
              </td>
              <td className="py-3 px-4 text-right font-mono">
                {run.accuracy != null ? run.accuracy.toFixed(4) : "-"}
              </td>
              <td className="py-3 px-4 text-right font-mono">
                {run.latency_p50_ms != null ? `${run.latency_p50_ms.toFixed(1)} ms` : "-"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
