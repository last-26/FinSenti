"use client";

import { useEffect, useState } from "react";
import {
  getExperiments,
  getExperimentRuns,
  type ExperimentSummary,
  type RunSummary,
} from "@/lib/api";
import { MetricsTable } from "@/components/experiments/MetricsTable";

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExp, setSelectedExp] = useState<string>("");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getExperiments()
      .then((exps) => {
        setExperiments(exps);
        if (exps.length > 0) {
          setSelectedExp(exps[0].experiment_id);
        }
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (selectedExp) {
      getExperimentRuns(selectedExp)
        .then(setRuns)
        .catch((e) => setError(e.message));
    }
  }, [selectedExp]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Experiments</h1>
        <p className="text-gray-500 mt-1">MLflow experiment tracking results</p>
      </div>

      {error && (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded-lg text-sm">
          MLflow unavailable: {error}
        </div>
      )}

      {/* Experiment selector */}
      {experiments.length > 0 && (
        <div className="flex gap-2">
          {experiments.map((exp) => (
            <button
              key={exp.experiment_id}
              onClick={() => setSelectedExp(exp.experiment_id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedExp === exp.experiment_id
                  ? "bg-blue-600 text-white"
                  : "bg-white border border-gray-200 text-gray-700 hover:bg-gray-50"
              }`}
            >
              {exp.experiment_name}
              <span className="ml-2 opacity-60">({exp.run_count})</span>
            </button>
          ))}
        </div>
      )}

      {/* Runs table */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <MetricsTable runs={runs} />
      </div>
    </div>
  );
}
