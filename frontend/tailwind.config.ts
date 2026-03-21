import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        positive: {
          DEFAULT: "#22c55e",
          light: "#dcfce7",
        },
        negative: {
          DEFAULT: "#ef4444",
          light: "#fee2e2",
        },
        neutral: {
          DEFAULT: "#f59e0b",
          light: "#fef3c7",
        },
      },
    },
  },
  plugins: [],
};

export default config;
