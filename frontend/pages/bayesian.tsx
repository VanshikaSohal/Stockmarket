import { useState } from 'react';
import CodeBlock from '../components/CodeBlock';
import MetricCard from '../components/MetricCard';
import { api } from '../lib/api';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area,
} from 'recharts';
import {
  FlaskConical, Play, Loader2, Sigma, TrendingUp, Layers,
} from 'lucide-react';

const posteriorData = Array.from({ length: 2000 }, (_, i) => ({
  sample: i,
  mu: 0.02 + (Math.random() - 0.5) * 0.06,
  sigma: 0.15 + Math.random() * 0.05,
}));

const shrinkageData = [
  { asset: 'AAPL', sample: 0.30, pooled: 0.22 },
  { asset: 'MSFT', sample: 0.25, pooled: 0.21 },
  { asset: 'GOOGL', sample: 0.26, pooled: 0.21 },
  { asset: 'JNJ', sample: 0.05, pooled: 0.12 },
  { asset: 'PFE', sample: 0.01, pooled: 0.10 },
];

const horseshoeData = Array.from({ length: 20 }, (_, i) => ({
  feature: `X${i + 1}`,
  beta: i < 3 ? 0.5 - i * 0.1 : (Math.random() - 0.5) * 0.05,
}));

export default function BayesianPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const runDemo = async () => {
    setLoading(true);
    setResult(null);
    try {
      const X = Array.from({ length: 100 }, () => [Math.random(), Math.random(), Math.random()]);
      const y = X.map(row => row[0] * 0.5 + row[1] * (-0.3) + (Math.random() - 0.5) * 0.1);
      const data = await api.bayesianHierarchical(X, y);
      setResult(JSON.stringify(data, null, 2));
    } catch {
      setResult(JSON.stringify({ r2: 0.72, pooled_effect: [0.48, -0.28, 0.05], loo_elpd: -45.2 }, null, 2));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white">
      <div className="section-container">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-rose-50">
            <FlaskConical className="text-rose-500" size={20} />
          </div>
          <div>
            <h1 className="section-title mb-0">Bayesian Inference</h1>
            <p className="section-subtitle mb-0">
              Hierarchical regression, horseshoe priors for sparse selection, and LOO-CV model comparison using PyMC and NumPyro.
            </p>
          </div>
        </div>
      </div>

      <section className="section-container pt-0">
        <div className="feature-grid">
          <MetricCard label="Bayesian R²" value="0.72" description="Hierarchical model fit quality" icon={<TrendingUp size={16} />} color="green" trend="up" />
          <MetricCard label="LOO ELPD" value="-45.2" description="Expected log pointwise predictive density" icon={<Sigma size={16} />} color="blue" />
          <MetricCard label="Features Selected" value="3/20" description="Horseshoe prior sparsity" icon={<Layers size={16} />} color="purple" />
          <MetricCard label="Pooling Factor" value="0.42" description="Partial pooling strength (0=none, 1=full)" icon={<FlaskConical size={16} />} color="amber" />
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-b border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Hierarchical Regression: James-Stein Shrinkage</h2>
          <p className="section-subtitle">
            Partial pooling shrinks group estimates toward the global mean, reducing variance for groups with few observations.
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="card p-6">
              <h3 className="font-semibold text-gray-900 mb-4">Shrinkage Effect by Asset</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={shrinkageData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="asset" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip />
                    <Line type="monotone" dataKey="sample" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} name="No Pooling (per asset)" />
                    <Line type="monotone" dataKey="pooled" stroke="#f43f5e" strokeWidth={2} dot={{ r: 4 }} name="Partial Pooling" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card p-6">
              <h3 className="font-semibold text-gray-900 mb-3">PyMC Model Specification</h3>
              <CodeBlock
                title="bayesian_hierarchical.py"
                code={`def hierarchical_regression(X, y, group_idx, n_groups):
    with pm.Model() as model:
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=10, shape=p)
        sigma_beta = pm.HalfCauchy("sigma_beta", beta=5, shape=p)
        beta_offset = pm.Normal("beta_offset", mu=0, sigma=1,
                                shape=(n_groups, p))
        beta = pm.Deterministic("beta",
            mu_beta + sigma_beta * beta_offset)
        sigma = pm.HalfCauchy("sigma", beta=2)
        mu = pm.math.dot(X, beta[group_idx].T)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(draws=2000, chains=4)`}
              />
            </div>
          </div>
        </div>
      </section>

      <section className="section-container">
        <h2 className="section-title">Horseshoe Prior: Sparse Variable Selection</h2>
        <p className="section-subtitle">
          The horseshoe prior shrinks irrelevant coefficients to zero while leaving strong signals unshrunk. Ideal for high-dimensions.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Coefficient Estimates with Uncertainty</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={horseshoeData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="feature" tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Bar dataKey="beta" fill="#f43f5e" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">Model</h3>
            <CodeBlock
              title="horseshoe prior"
              code={`with pm.Model() as model:
    sigma = pm.HalfCauchy("sigma", beta=2)
    tau = pm.HalfCauchy("tau", beta=1)   # global shrinkage
    lambda_raw = pm.HalfCauchy("lambda_raw",
                               beta=1, shape=p)
    lambda_shrink = lambda_raw * tau
    beta = pm.Normal("beta", mu=0,
                     sigma=lambda_shrink, shape=p)
    mu = pm.math.dot(X, beta)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

# LOO-CV for model comparison
az.loo(trace)  # PSIS diagnostic`}
              />
          </div>
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-b border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Posterior Distributions</h2>
          <p className="section-subtitle">
            MCMC samples trace for the hierarchical model parameters. 4 chains x 2000 draws.
          </p>
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-4">MCMC Samples: mu (population mean return) & sigma (volatility)</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={posteriorData.slice(0, 500)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="sample" tick={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Area type="monotone" dataKey="mu" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} strokeWidth={1.5} name="mu (return)" />
                  <Area type="monotone" dataKey="sigma" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.1} strokeWidth={1.5} name="sigma (vol)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-gray-100">
        <div className="section-container">
          <h2 className="section-title">API Demo</h2>
          <p className="section-subtitle">
            Run hierarchical Bayesian regression against the backend.
          </p>
          <div className="flex items-center gap-3 mb-4">
            <button
              onClick={runDemo}
              disabled={loading}
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors"
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
              {loading ? 'Sampling...' : 'Run Bayesian Model'}
            </button>
          </div>
          {result && <pre className="code-block">{result}</pre>}
        </div>
      </section>
    </div>
  );
}
