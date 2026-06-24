import { ReactNode } from 'react';

interface MetricCardProps {
  label: string;
  value: string;
  description?: string;
  icon?: ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  color?: 'blue' | 'green' | 'purple' | 'amber' | 'red';
}

const colorClasses = {
  blue: 'border-l-blue-500 bg-blue-50/50',
  green: 'border-l-emerald-500 bg-emerald-50/50',
  purple: 'border-l-purple-500 bg-purple-50/50',
  amber: 'border-l-amber-500 bg-amber-50/50',
  red: 'border-l-red-500 bg-red-50/50',
};

const trendIcons: Record<string, string> = {
  up: '\u2191',
  down: '\u2193',
  neutral: '\u2192',
};

export default function MetricCard({ label, value, description, icon, trend, color = 'blue' }: MetricCardProps) {
  return (
    <div className={`card border-l-4 ${colorClasses[color]}`}>
      <div className="card-body">
        <div className="flex items-center justify-between mb-1">
          <span className="metric-label">{label}</span>
          {icon && <span className="text-gray-400">{icon}</span>}
        </div>
        <div className="flex items-baseline gap-2">
          <span className="metric-value">{value}</span>
          {trend && (
            <span className={`text-sm ${
              trend === 'up' ? 'text-emerald-600' : trend === 'down' ? 'text-red-500' : 'text-gray-400'
            }`}>
              {trendIcons[trend]}
            </span>
          )}
        </div>
        {description && <p className="mt-1 text-xs text-gray-400">{description}</p>}
      </div>
    </div>
  );
}
