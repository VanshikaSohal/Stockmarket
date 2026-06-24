import CodeBlock from '../components/CodeBlock';
import { BookOpen, Globe, Shield, Lock, Activity, BarChart3, Cpu, FlaskConical } from 'lucide-react';

const endpoints = [
  {
    group: 'System',
    icon: <Globe size={16} />,
    items: [
      { method: 'GET', path: '/api/v1/health', desc: 'Liveness/readiness probe', auth: false },
      { method: 'GET', path: '/api/v1/config', desc: 'Active configuration (sanitized)', auth: false },
      { method: 'GET', path: '/metrics', desc: 'Prometheus metrics', auth: false },
      { method: 'GET', path: '/docs', desc: 'Swagger UI (OpenAPI)', auth: false },
      { method: 'GET', path: '/redoc', desc: 'ReDoc documentation', auth: false },
    ],
  },
  {
    group: 'Risk Metrics',
    icon: <Activity size={16} />,
    items: [
      { method: 'POST', path: '/api/v1/risk/metrics', desc: 'Full risk summary (VaR, CVaR, Sharpe, Sortino, Calmar)', auth: false },
      { method: 'POST', path: '/api/v1/risk/var', desc: 'Value at Risk (historical or parametric)', auth: false },
      { method: 'POST', path: '/api/v1/risk/cvar', desc: 'Conditional Value at Risk', auth: false },
      { method: 'POST', path: '/api/v1/risk/sharpe', desc: 'Annualized Sharpe ratio', auth: false },
      { method: 'POST', path: '/api/v1/risk/sortino', desc: 'Annualized Sortino ratio', auth: false },
    ],
  },
  {
    group: 'Portfolio Optimization',
    icon: <BarChart3 size={16} />,
    items: [
      { method: 'POST', path: '/api/v1/optimize/minvar', desc: 'Minimum variance weights (SLSQP)', auth: false },
      { method: 'POST', path: '/api/v1/optimize/sharpe', desc: 'Maximum Sharpe ratio weights', auth: false },
      { method: 'POST', path: '/api/v1/optimize/black-litterman', desc: 'Black-Litterman posterior returns + weights', auth: false },
      { method: 'POST', path: '/api/v1/optimize/hrp', desc: 'Hierarchical Risk Parity weights', auth: false },
      { method: 'POST', path: '/api/v1/optimize/cvar', desc: 'CVaR-minimizing portfolio weights', auth: false },
    ],
  },
  {
    group: 'Volatility',
    icon: <Activity size={16} />,
    items: [
      { method: 'POST', path: '/api/v1/volatility/garch', desc: 'GARCH/EGARCH/GJR-GARCH fit + forecast', auth: false },
      { method: 'POST', path: '/api/v1/volatility/garch/dcc', desc: 'DCC-GARCH dynamic correlation', auth: false },
      { method: 'POST', path: '/api/v1/volatility/har', desc: 'HAR-RV realized volatility model', auth: false },
    ],
  },
  {
    group: 'Backtesting',
    icon: <Activity size={16} />,
    items: [
      { method: 'POST', path: '/api/v1/backtest', desc: 'Rebalancing-based portfolio backtest', auth: false },
    ],
  },
  {
    group: 'ML / Bayesian',
    icon: <Cpu size={16} />,
    items: [
      { method: 'POST', path: '/api/v1/predict/bayesian/hierarchical', desc: 'Hierarchical Bayesian regression', auth: false },
      { method: 'POST', path: '/api/v1/predict/bayesian/horseshoe', desc: 'Horseshoe prior sparse regression', auth: false },
    ],
  },
  {
    group: 'Authentication',
    icon: <Lock size={16} />,
    items: [
      { method: 'POST', path: '/api/v1/auth/token', desc: 'JWT access token (username + password)', auth: false },
    ],
  },
];

const requestExample = `# Compute risk metrics
curl -X POST http://localhost:8000/api/v1/risk/metrics \\
  -H "Content-Type: application/json" \\
  -d '{"returns": [0.01, -0.005, 0.002, -0.01, 0.008], "confidence": 0.95}'

# Optimize portfolio (max Sharpe)
curl -X POST http://localhost:8000/api/v1/optimize/sharpe \\
  -H "Content-Type: application/json" \\
  -d '{"returns_matrix": [[0.01, 0.02], [-0.01, 0.01], [0.005, -0.005]], "weights": [0.5, 0.5]}'

# Fit GARCH model
curl -X POST http://localhost:8000/api/v1/volatility/garch \\
  -H "Content-Type: application/json" \\
  -d '{"returns": [0.01, -0.005, 0.002], "p": 1, "q": 1, "vol": "GARCH"}'`;

const pythonExample = `import requests

API = "http://localhost:8000/api/v1"

# Risk metrics
res = requests.post(f"{API}/risk/metrics", json={
    "returns": [0.01, -0.005, 0.002],
    "confidence": 0.95,
})
print(res.json())

# Portfolio optimization
res = requests.post(f"{API}/optimize/hrp", json={
    "returns_matrix": [[0.01, 0.02], [-0.01, 0.01]],
    "weights": [0.5, 0.5],
})
print(res.json()["weights"])`;

export default function ApiPage() {
  return (
    <div className="bg-white">
      <div className="section-container">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-gray-900 text-white">
            <BookOpen size={20} />
          </div>
          <div>
            <h1 className="section-title mb-0">API Reference</h1>
            <p className="section-subtitle mb-0">
              Complete REST API documentation for the Portfolio Risk Analyzer. All endpoints return JSON responses.
            </p>
          </div>
        </div>
      </div>

      <section className="section-container pt-0">
        <div className="card p-6 mb-8">
          <h2 className="font-semibold text-gray-900 mb-4">Quick Start</h2>
          <p className="text-sm text-gray-500 mb-4">
            Start the API server locally and make requests:
          </p>
          <div className="space-y-6">
            <CodeBlock
              title="Terminal"
              code={`# Install with all extras
uv sync --group dev --all-extras

# Start development server
uvicorn src.api.main:app --reload --port 8000

# Server running at http://localhost:8000
# API docs at http://localhost:8000/docs`}
            />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <CodeBlock title="curl" code={requestExample} />
              <CodeBlock title="Python" code={pythonExample} />
            </div>
          </div>
        </div>

        <h2 className="text-2xl font-bold text-gray-900 mb-6">Endpoints</h2>
        <div className="space-y-8">
          {endpoints.map((group) => (
            <div key={group.group} className="card overflow-hidden">
              <div className="card-header flex items-center gap-2">
                <span className="text-gray-400">{group.icon}</span>
                <h3 className="font-semibold text-gray-900">{group.group}</h3>
                <span className="ml-auto text-xs text-gray-400">{group.items.length} endpoints</span>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full">
                  <thead>
                    <tr>
                      <th className="table-header w-20">Method</th>
                      <th className="table-header">Path</th>
                      <th className="table-header">Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {group.items.map((ep) => (
                      <tr key={ep.path} className="hover:bg-gray-50/50">
                        <td className="table-cell">
                          <span className={`inline-flex px-2 py-0.5 rounded text-xs font-bold ${
                            ep.method === 'GET' ? 'bg-green-50 text-green-700' : 'bg-blue-50 text-blue-700'
                          }`}>
                            {ep.method}
                          </span>
                        </td>
                        <td className="table-cell font-mono text-xs text-gray-800">{ep.path}</td>
                        <td className="table-cell text-sm text-gray-600">{ep.desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Request / Response Schemas</h2>
          <p className="section-subtitle">
            All endpoints use Pydantic-validated JSON schemas. Full OpenAPI spec available at <code className="text-primary-600">/openapi.json</code>.
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <CodeBlock
              title="ReturnsPayload"
              code={`{
  "returns": [float],        // Daily return series
  "confidence": 0.95,        // VaR/CVaR confidence
  "risk_free_rate": 0.02,    // Annual risk-free
  "periods": 252             // Trading days/year
}`}
            />
            <CodeBlock
              title="PortfolioReturnsPayload"
              code={`{
  "returns_matrix": [[float]], // rows=dates, cols=assets
  "weights": [float],          // initial weights
  "risk_free_rate": 0.02
}`}
            />
            <CodeBlock
              title="GARCHPayload"
              code={`{
  "returns": [float],
  "p": 1, "q": 1,
  "vol": "GARCH | EGARCH | GJRGARCH",
  "dist": "normal | studentst | ...",
  "horizon": 5
}`}
            />
            <CodeBlock
              title="BlackLittermanPayload"
              code={`{
  "returns_matrix": [[float]],
  "market_caps": [float],
  "view_asset_pairs": [["A","B"]],
  "view_returns": [float],
  "view_confidences": [float],
  "risk_aversion": 2.5
}`}
            />
          </div>
        </div>
      </section>

      <section className="section-container">
        <h2 className="section-title">Authentication</h2>
        <p className="section-subtitle">
          JWT-based authentication is optional and disabled by default. Enable via <code className="text-primary-600">PRA_API_AUTH_ENABLED=true</code>.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">Obtain Token</h3>
            <CodeBlock
              code={`curl -X POST http://localhost:8000/api/v1/auth/token \\
  -H "Content-Type: application/json" \\
  -d '{"username": "admin", "password": "your-secret"}'`}
            />
          </div>
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">Use Token</h3>
            <CodeBlock
              code={`curl http://localhost:8000/api/v1/risk/metrics \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer <your-token>" \\
  -d '{"returns": [0.01, -0.005]}'`}
            />
          </div>
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-gray-100 py-12 text-center">
        <p className="text-sm text-gray-400">
          Full OpenAPI specification available at <code className="text-primary-600 font-mono">/openapi.json</code>
        </p>
      </section>
    </div>
  );
}
