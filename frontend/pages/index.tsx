import Link from 'next/link';
import {
  Activity, BarChart3, CandlestickChart, Cpu, FlaskConical, GitCompareArrows,
  TrendingUp, Shield, DollarSign, Brain, BookOpen, ChevronRight, Star,
} from 'lucide-react';

const features = [
  {
    title: 'Risk Metrics',
    description: 'VaR, CVaR, Sharpe, Sortino, Calmar ratios with rolling and parametric methods for comprehensive risk assessment.',
    icon: Activity,
    href: '/risk-metrics',
    color: 'text-blue-600 bg-blue-50',
  },
  {
    title: 'Portfolio Optimization',
    description: 'Minimum variance, maximum Sharpe, Black-Litterman, Hierarchical Risk Parity, and CVaR optimization.',
    icon: BarChart3,
    href: '/portfolio-optimization',
    color: 'text-emerald-600 bg-emerald-50',
  },
  {
    title: 'Volatility Modeling',
    description: 'GARCH/EGARCH/GJR-GARCH, DCC-GARCH, HAR-RV realized volatility with forecasting.',
    icon: CandlestickChart,
    href: '/volatility',
    color: 'text-purple-600 bg-purple-50',
  },
  {
    title: 'Machine Learning',
    description: 'Random Forest, XGBoost, LightGBM, CatBoost, stacking ensembles, LSTM/GRU, TFT, N-BEATS.',
    icon: Cpu,
    href: '/machine-learning',
    color: 'text-amber-600 bg-amber-50',
  },
  {
    title: 'Bayesian Inference',
    description: 'Hierarchical regression, horseshoe priors, PyMC/NumPyro backends, LOO-CV model comparison.',
    icon: FlaskConical,
    href: '/bayesian',
    color: 'text-rose-600 bg-rose-50',
  },
  {
    title: 'Backtesting',
    description: 'Rebalancing-based strategy backtester with transaction cost models and performance attribution.',
    icon: GitCompareArrows,
    href: '/backtesting',
    color: 'text-cyan-600 bg-cyan-50',
  },
];

const portfolioUniverse = [
  { ticker: 'AAPL', name: 'Apple', sector: 'Technology', annReturn: '30.2%', sharpe: '0.89' },
  { ticker: 'MSFT', name: 'Microsoft', sector: 'Technology', annReturn: '25.4%', sharpe: '0.77' },
  { ticker: 'GOOGL', name: 'Alphabet', sector: 'Communication', annReturn: '26.2%', sharpe: '0.74' },
  { ticker: 'AMZN', name: 'Amazon', sector: 'Consumer Disc.', annReturn: '23.7%', sharpe: '0.60' },
  { ticker: 'WMT', name: 'Walmart', sector: 'Consumer Staples', annReturn: '20.9%', sharpe: '0.84' },
  { ticker: 'JPM', name: 'JPMorgan', sector: 'Financials', annReturn: '18.9%', sharpe: '0.52' },
  { ticker: 'XOM', name: 'Exxon Mobil', sector: 'Energy', annReturn: '19.0%', sharpe: '0.49' },
  { ticker: 'UNH', name: 'UnitedHealth', sector: 'Health Care', annReturn: '17.1%', sharpe: '0.51' },
  { ticker: 'V', name: 'Visa', sector: 'Financials', annReturn: '14.9%', sharpe: '0.46' },
  { ticker: 'JNJ', name: 'J&J', sector: 'Health Care', annReturn: '4.6%', sharpe: '0.13' },
  { ticker: 'PFE', name: 'Pfizer', sector: 'Health Care', annReturn: '1.4%', sharpe: '-0.02' },
];

const keyResults = [
  { portfolio: 'Equal Weight', annReturn: '18.4%', sharpe: '0.82', maxDD: '-29.0%', cumulative: '+126.1%' },
  { portfolio: 'Minimum Variance', annReturn: '18.4%', sharpe: '0.82', maxDD: '-29.0%', cumulative: '+126.1%' },
  { portfolio: 'Maximum Sharpe', annReturn: '24.1%', sharpe: '1.08', maxDD: '-20.9%', cumulative: '+199.0%' },
];

const techStack = [
  { category: 'Language', items: 'Python 3.10+' },
  { category: 'API Framework', items: 'FastAPI, uvicorn, structlog' },
  { category: 'Finance', items: 'arch, scipy, statsmodels, yfinance' },
  { category: 'ML/DL', items: 'scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch' },
  { category: 'Bayesian', items: 'PyMC, NumPyro, ArviZ' },
  { category: 'Infrastructure', items: 'Docker, GitHub Actions, Prometheus' },
  { category: 'Frontend', items: 'Next.js, TypeScript, Tailwind CSS, Recharts' },
];

export default function Home() {
  return (
    <div className="bg-white">
      <section className="relative overflow-hidden border-b border-gray-100">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-50/40 via-white to-accent-50/40" />
        <div className="relative section-container py-20 lg:py-28">
          <div className="max-w-4xl">
            <div className="flex items-center gap-2 mb-4">
              <span className="badge-blue">v1.0.0</span>
              <span className="badge-green">Production Ready</span>
              <span className="badge-purple">MIT Licensed</span>
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-gray-900 leading-tight mb-6 text-balance">
              Portfolio Risk Analyzer
            </h1>
            <p className="text-xl text-gray-500 mb-8 max-w-3xl leading-relaxed">
              An institutional-grade quantitative finance pipeline combining classical portfolio theory,
              machine learning forecasting, and Bayesian inference across an 11-stock universe.
            </p>
            <div className="flex flex-wrap gap-3">
              <Link
                href="/risk-metrics"
                className="inline-flex items-center gap-2 px-6 py-3 bg-primary-600 text-white rounded-xl font-semibold hover:bg-primary-700 transition-colors shadow-sm"
              >
                Explore Metrics <ChevronRight size={18} />
              </Link>
              <Link
                href="/api"
                className="inline-flex items-center gap-2 px-6 py-3 bg-white border border-gray-200 text-gray-700 rounded-xl font-semibold hover:bg-gray-50 transition-colors"
              >
                <BookOpen size={18} /> API Reference
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="section-container">
        <h2 className="section-title">Technical Capabilities</h2>
        <p className="section-subtitle">
          Six integrated modules covering the full quantitative finance pipeline from data collection to production deployment.
        </p>
        <div className="feature-grid">
          {features.map((feature) => (
            <Link key={feature.title} href={feature.href} className="card p-6 hover:border-primary-200 group">
              <div className={`inline-flex p-3 rounded-xl ${feature.color} mb-4`}>
                <feature.icon size={22} />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-primary-600 transition-colors">
                {feature.title}
              </h3>
              <p className="text-sm text-gray-500 leading-relaxed">{feature.description}</p>
            </Link>
          ))}
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-b border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Key Results</h2>
          <p className="section-subtitle">
            Performance comparison across portfolio strategies on 5-year historical data (2020-2024).
          </p>
          <div className="overflow-x-auto rounded-xl border border-gray-200">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="table-header">Portfolio Strategy</th>
                  <th className="table-header">Annualized Return</th>
                  <th className="table-header">Sharpe Ratio</th>
                  <th className="table-header">Max Drawdown</th>
                  <th className="table-header">5yr Cumulative</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {keyResults.map((r) => (
                  <tr key={r.portfolio} className="hover:bg-gray-50/50">
                    <td className="table-cell font-medium text-gray-900">{r.portfolio}</td>
                    <td className="table-cell">{r.annReturn}</td>
                    <td className="table-cell">
                      <span className="font-mono font-semibold text-emerald-600">{r.sharpe}</span>
                    </td>
                    <td className="table-cell text-red-500">{r.maxDD}</td>
                    <td className="table-cell font-semibold text-primary-600">{r.cumulative}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="section-container">
        <h2 className="section-title">Portfolio Universe</h2>
        <p className="section-subtitle">
          11 large-cap US equities across 6 GICS sectors, 2020-01-03 to 2024-12-27.
        </p>
        <div className="overflow-x-auto rounded-xl border border-gray-200">
          <table className="min-w-full divide-y divide-gray-200">
            <thead>
              <tr>
                <th className="table-header">Ticker</th>
                <th className="table-header">Company</th>
                <th className="table-header">Sector</th>
                <th className="table-header">Annualized Return</th>
                <th className="table-header">Sharpe Ratio</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-100">
              {portfolioUniverse.map((asset) => (
                <tr key={asset.ticker} className="hover:bg-gray-50/50">
                  <td className="table-cell font-mono font-semibold text-gray-900">{asset.ticker}</td>
                  <td className="table-cell text-gray-700">{asset.name}</td>
                  <td className="table-cell">
                    <span className="badge-blue">{asset.sector}</span>
                  </td>
                  <td className="table-cell font-mono">{asset.annReturn}</td>
                  <td className="table-cell font-mono">{asset.sharpe}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Technology Stack</h2>
          <p className="section-subtitle">
            Built with modern tools across the full stack.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {techStack.map((t) => (
              <div key={t.category} className="card p-4">
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">{t.category}</p>
                <p className="text-sm font-medium text-gray-800">{t.items}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="section-container py-12 text-center">
        <div className="flex items-center justify-center gap-1 text-sm text-gray-400">
          <Star size={14} className="text-amber-400" />
          Built with Python, TypeScript, and quantitative intuition
        </div>
      </section>
    </div>
  );
}
