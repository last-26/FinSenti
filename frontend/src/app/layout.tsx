import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "FinSenti - Financial Sentiment Analysis",
  description: "ML-powered financial sentiment analysis pipeline",
};

const navItems = [
  { href: "/", label: "Dashboard" },
  { href: "/predict", label: "Predict" },
  { href: "/batch", label: "Batch" },
  { href: "/experiments", label: "Experiments" },
  { href: "/models", label: "Models" },
  { href: "/history", label: "History" },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen flex flex-col">
          <header className="bg-white border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between h-16">
                <Link href="/" className="text-xl font-bold text-gray-900">
                  FinSenti
                </Link>
                <nav className="flex space-x-1">
                  {navItems.map((item) => (
                    <Link
                      key={item.href}
                      href={item.href}
                      className="px-3 py-2 rounded-md text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
                    >
                      {item.label}
                    </Link>
                  ))}
                </nav>
              </div>
            </div>
          </header>
          <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
            {children}
          </main>
          <footer className="bg-white border-t border-gray-200 py-4">
            <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
              FinSenti - Financial Sentiment Analysis MLOps Pipeline
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
