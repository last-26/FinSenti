"use client";

import { useEffect, useState } from "react";
import { getModels, switchModel, type ModelInfo } from "@/lib/api";
import { ModelCard } from "@/components/models/ModelCard";

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState("");

  const fetchModels = () => {
    getModels().then(setModels).catch((e) => setError(e.message));
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleSwitch = async (name: string) => {
    try {
      await switchModel(name);
      fetchModels();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to switch model");
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Models</h1>
        <p className="text-gray-500 mt-1">Manage and switch between trained models</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {models.length === 0 && !error && (
        <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
          <p className="text-gray-500">No models found. Train a model first.</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {models.map((model) => (
          <ModelCard key={model.name} model={model} onSwitch={handleSwitch} />
        ))}
      </div>
    </div>
  );
}
