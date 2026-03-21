"use client";

import type { ModelInfo } from "@/lib/api";
import { cn } from "@/lib/utils";

interface ModelCardProps {
  model: ModelInfo;
  onSwitch?: (name: string) => void;
}

export function ModelCard({ model, onSwitch }: ModelCardProps) {
  return (
    <div
      className={cn(
        "bg-white rounded-lg border p-5",
        model.is_active ? "border-blue-300 ring-2 ring-blue-100" : "border-gray-200"
      )}
    >
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-semibold text-gray-900">{model.name}</h3>
          <p className="text-sm text-gray-500 mt-1">{model.base_model}</p>
        </div>
        <span
          className={cn(
            "px-2 py-0.5 rounded text-xs font-medium",
            model.is_active
              ? "bg-blue-50 text-blue-700"
              : "bg-gray-100 text-gray-600"
          )}
        >
          {model.status}
        </span>
      </div>

      {!model.is_active && onSwitch && (
        <button
          onClick={() => onSwitch(model.name)}
          className="mt-4 w-full px-4 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200 transition-colors"
        >
          Switch to this model
        </button>
      )}

      {model.is_active && (
        <div className="mt-4 text-center text-xs text-blue-600 font-medium">
          Currently Active
        </div>
      )}
    </div>
  );
}
