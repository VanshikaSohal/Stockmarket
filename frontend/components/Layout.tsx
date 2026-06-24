import { ReactNode, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import {
  Activity,
  BarChart3,
  CandlestickChart,
  Cpu,
  FlaskConical,
  GitCompareArrows,
  Home,
  LineChart,
  Menu,
  X,
  ChevronRight,
  BookOpen,
} from 'lucide-react';

interface NavItem {
  label: string;
  href: string;
  icon: ReactNode;
}

const navItems: NavItem[] = [
  { label: 'Overview', href: '/', icon: <Home size={18} /> },
  { label: 'Risk Metrics', href: '/risk-metrics', icon: <Activity size={18} /> },
  { label: 'Portfolio Optimization', href: '/portfolio-optimization', icon: <BarChart3 size={18} /> },
  { label: 'Volatility Modeling', href: '/volatility', icon: <CandlestickChart size={18} /> },
  { label: 'Machine Learning', href: '/machine-learning', icon: <Cpu size={18} /> },
  { label: 'Bayesian Inference', href: '/bayesian', icon: <FlaskConical size={18} /> },
  { label: 'Backtesting', href: '/backtesting', icon: <GitCompareArrows size={18} /> },
  { label: 'API Reference', href: '/api', icon: <BookOpen size={18} /> },
];

export default function Layout({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen flex">
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 bg-white border-r border-gray-200 transform transition-transform duration-200 ease-in-out lg:translate-x-0 lg:static lg:inset-auto ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="h-full flex flex-col">
          <div className="flex items-center justify-between h-16 px-6 border-b border-gray-100">
            <Link href="/" className="flex items-center gap-2">
              <LineChart className="text-primary-600" size={24} />
              <span className="font-bold text-gray-900 text-sm">Portfolio Risk Analyzer</span>
            </Link>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden text-gray-400 hover:text-gray-600"
            >
              <X size={20} />
            </button>
          </div>
          <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
            {navItems.map((item) => {
              const isActive = router.pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={isActive ? 'nav-link-active' : 'nav-link-inactive'}
                  onClick={() => setSidebarOpen(false)}
                >
                  <span className={isActive ? 'text-primary-600' : 'text-gray-400'}>{item.icon}</span>
                  <span className="flex-1">{item.label}</span>
                  {isActive && <ChevronRight size={14} className="text-primary-400" />}
                </Link>
              );
            })}
          </nav>
          <div className="px-4 py-4 border-t border-gray-100">
            <p className="text-xs text-gray-400">
              v1.0.0 &middot; FastAPI + Next.js
            </p>
          </div>
        </div>
      </aside>

      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div className="flex-1 flex flex-col min-w-0">
        <header className="sticky top-0 z-30 bg-white/80 backdrop-blur-sm border-b border-gray-200 lg:hidden">
          <div className="flex items-center h-14 px-4">
            <button
              onClick={() => setSidebarOpen(true)}
              className="text-gray-500 hover:text-gray-700 mr-3"
            >
              <Menu size={20} />
            </button>
            <LineChart className="text-primary-600" size={20} />
            <span className="ml-2 font-semibold text-sm text-gray-900">Portfolio Risk Analyzer</span>
          </div>
        </header>
        <main className="flex-1">
          {children}
        </main>
      </div>
    </div>
  );
}
