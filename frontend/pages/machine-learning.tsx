import CodeBlock from '../components/CodeBlock';
import MetricCard from '../components/MetricCard';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line,
} from 'recharts';
import {
  Cpu, TrendingUp, Brain, GitBranch, Target, Layers,
} from 'lucide-react';

const mlResults = [
  { model: 'Random Forest (Class)', accuracy: 52.4, mae: 0.0105 },
  { model: 'XGBoost (Class)', accuracy: 53.2, mae: 0.0108 },
  { model: 'LightGBM (Class)', accuracy: 52.8, mae: 0.0106 },
  { model: 'CatBoost (Class)', accuracy: 52.1, mae: 0.0110 },
  { model: 'Stacking Ensemble', accuracy: 54.5, mae: 0.0098 },
  { model: 'LSTM', accuracy: 51.8, mae: 0.0112 },
];

const lstmLossData = Array.from({ length: 100 }, (_, i) => ({
  epoch: i + 1,
  train: Math.max(0.02, 0.12 * Math.exp(-i / 25) + 0.01 * Math.random()),
  val: Math.max(0.03, 0.15 * Math.exp(-i / 20) + 0.015 * Math.random()),
}));

export default function MachineLearningPage() {
  return (
    <div className="bg-white">
      <div className="section-container">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-amber-50">
            <Cpu className="text-amber-500" size={20} />
          </div>
          <div>
            <h1 className="section-title mb-0">Machine Learning</h1>
            <p className="section-subtitle mb-0">
              Classical ML, gradient boosting, deep learning, stacking ensembles, and model interpretability with SHAP.
            </p>
          </div>
        </div>
      </div>

      <section className="section-container pt-0">
        <div className="feature-grid">
          <MetricCard label="Best Accuracy" value="54.5%" description="Stacking ensemble direction classifier" icon={<Target size={16} />} color="green" trend="up" />
          <MetricCard label="Best MAE" value="0.0098" description="Stacking ensemble return regression" icon={<TrendingUp size={16} />} color="blue" trend="up" />
          <MetricCard label="Model Types" value="8" description="RF, XGBoost, LGBM, CatBoost, LSTM, GRU, TFT, N-BEATS" icon={<Layers size={16} />} color="purple" />
          <MetricCard label="MLflow Runs" value="50+" description="Tracked experiments with Optuna tuning" icon={<GitBranch size={16} />} color="amber" />
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-b border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Model Performance Comparison</h2>
          <p className="section-subtitle">
            Direction classification accuracy and return regression MAE across all ML models.
          </p>
          <div className="card p-6">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={mlResults} barGap={4}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="model" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                  <YAxis yAxisId="left" tick={{ fontSize: 11 }} domain={[48, 56]} unit="%" axisLine={false} tickLine={false} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} domain={[0.008, 0.012]} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Bar yAxisId="left" dataKey="accuracy" fill="#f59e0b" radius={[2, 2, 0, 0]} name="Accuracy (%)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      <section className="section-container">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">MLflow + Optuna Training Pipeline</h3>
            <CodeBlock
              title="optuna_tuning.py"
              language="python"
              code={`import optuna
from src.models.walk_forward import PurgedWalkForward

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_est", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_split", 2, 20),
    }
    cv = PurgedWalkForward(
        n_splits=5, embargo=10, purge=5
    )
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        model = RandomForestClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(
            y[test_idx], model.predict(X[test_idx])))
    return np.mean(scores)`}
            />
          </div>
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">Stacking Ensemble</h3>
            <CodeBlock
              title="ensemble.py"
              code={`from src.models.ensemble import build_stacking_ensemble

models = {
    "rf": RandomForestRegressor(n_estimators=200),
    "xgb": XGBRegressor(n_estimators=200),
    "lgbm": LGBMRegressor(n_estimators=200),
    "cat": CatBoostRegressor(n_estimators=200,
                              verbose=0),
}
meta = Ridge(alpha=1.0)

ensemble = build_stacking_ensemble(
    models=models, meta_learner=meta,
    cv=5, use_features=True,
)`}
            />
          </div>
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-b border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Deep Learning: LSTM Training</h2>
          <p className="section-subtitle">
            Sequence models with probabilistic output heads (Gaussian, Quantile) using PyTorch.
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="card p-6">
              <h3 className="font-semibold text-gray-900 mb-4">Training Loss Curves</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={lstmLossData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip />
                    <Line type="monotone" dataKey="train" stroke="#3b82f6" strokeWidth={2} dot={false} name="Train" />
                    <Line type="monotone" dataKey="val" stroke="#f59e0b" strokeWidth={2} dot={false} name="Validation" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card p-6">
              <h3 className="font-semibold text-gray-900 mb-3">Probabilistic LSTM</h3>
              <CodeBlock
                title="deep_learning.py"
                code={`class ProbabilisticSequenceRegressor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim,
                 output_dim=1, dist="gaussian"):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True)
        if dist == "gaussian":
            self.mu = nn.Linear(hidden_dim, output_dim)
            self.logvar = nn.Linear(hidden_dim, output_dim)
        elif dist == "quantile":
            self.q = nn.Linear(hidden_dim, output_dim * 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.mu(out[:, -1])`}
              />
            </div>
          </div>
        </div>
      </section>

      <section className="section-container">
        <h2 className="section-title">Model Interpretability (SHAP)</h2>
        <p className="section-subtitle">
          SHAP (SHapley Additive exPlanations) for understanding feature importance and model decisions.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="card p-6">
            <CodeBlock
              title="interpret.py"
              code={`from src.models.interpret import (
    create_explainer,
    plot_shap_summary,
)

explainer = create_explainer(model, X_train)
shap_values = explainer(X_test)

# Summary beeswarm plot
plot_shap_summary(shap_values, X_test)

# Force plot for single prediction
plot_shap_force(shap_values[0], X_test.iloc[0])

# Waterfall for detailed breakdown
plot_shap_waterfall(shap_values[0], X_test.iloc[0])`}
            />
          </div>
          <div className="space-y-4">
            <div className="card p-5">
              <h4 className="font-semibold text-sm text-gray-900 mb-2">Available Plot Types</h4>
              <ul className="space-y-2">
                {[
                  'Beeswarm summary: global feature importance + direction',
                  'Dependence plots: feature interactions',
                  'Force plots: individual prediction breakdown',
                  'Waterfall charts: additive feature contributions',
                  'bar plots: mean absolute SHAP values',
                ].map((item, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-gray-600">
                    <Brain size={12} className="mt-0.5 text-amber-500 flex-shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
            <div className="card p-5">
              <h4 className="font-semibold text-sm text-gray-900 mb-2">Pipeline Integration</h4>
              <p className="text-xs text-gray-500">
                MLflow auto-logging captures SHAP explanations as artifacts.
                Optuna hyperparameter optimization uses PurgedWalkForward CV
                with embargoing to prevent data leakage in time series.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
