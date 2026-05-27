import { useQuery } from '@tanstack/react-query';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  fetchCoverage,
  fetchFeatureImportance,
  fetchHealth,
  fetchPerformance,
} from '../lib/api';
import type {
  FeatureImportanceEntry,
  RecommendationsRecord,
  SyntheticStatReport,
} from '../types/api';
import HelpIcon from '../components/HelpIcon';

const STAT_ORDER = ['points', 'rebounds', 'assists', 'threes_made'];
const STAT_LABEL: Record<string, string> = {
  points: 'Points',
  rebounds: 'Rebounds',
  assists: 'Assists',
  threes_made: '3PM',
};

function bucketMidpoint(label: string): number {
  const [lo, hi] = label.split('-').map(Number);
  return (lo + hi) / 2;
}

function calibrationChartData(report: SyntheticStatReport) {
  return Object.entries(report.calibration)
    .map(([bucket, b]) => {
      const predicted = bucketMidpoint(bucket);
      return {
        bucket,
        predicted,
        actual: b.hit_rate * 100,
        ideal: predicted,
        n: b.n,
      };
    })
    .sort((a, b) => a.predicted - b.predicted);
}

function StatCard({ stat, report }: { stat: string; report: SyntheticStatReport }) {
  const lift = report.vs_baseline_lift;
  const liftPct =
    report.baseline_l10_mae && report.baseline_l10_mae > 0
      ? ((lift ?? 0) / report.baseline_l10_mae) * 100
      : null;
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="font-semibold">{STAT_LABEL[stat] ?? stat}</h3>
        <span className="text-xs text-slate-500 font-mono">n={report.n.toLocaleString()}</span>
      </div>
      <dl className="grid grid-cols-2 gap-y-1 text-sm mb-4">
        <dt className="text-slate-500">
          MAE
          <HelpIcon topic="MAE" />
        </dt>
        <dd className="font-mono text-right">{report.mae.toFixed(3)}</dd>
        <dt className="text-slate-500">
          RMSE
          <HelpIcon topic="RMSE" />
        </dt>
        <dd className="font-mono text-right">{report.rmse.toFixed(3)}</dd>
        <dt className="text-slate-500">
          Log loss
          <HelpIcon topic="Log loss" />
        </dt>
        <dd className="font-mono text-right">{report.log_loss.toFixed(4)}</dd>
        <dt className="text-slate-500">
          Baseline MAE
          <HelpIcon topic="Baseline MAE" />
        </dt>
        <dd className="font-mono text-right text-slate-400">
          {report.baseline_l10_mae?.toFixed(3) ?? '—'}
        </dd>
        <dt className="text-slate-500">
          vs baseline
          <HelpIcon topic="vs baseline" />
        </dt>
        <dd
          className={`font-mono text-right ${
            (lift ?? 0) > 0 ? 'text-emerald-600' : 'text-rose-600'
          }`}
        >
          {lift != null ? `${lift > 0 ? '+' : ''}${lift.toFixed(3)}` : '—'}
          {liftPct != null && (
            <span className="text-xs text-slate-400 ml-1">
              ({liftPct > 0 ? '+' : ''}
              {liftPct.toFixed(1)}%)
            </span>
          )}
        </dd>
      </dl>

      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={calibrationChartData(report)}
            margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="predicted"
              type="number"
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              tick={{ fontSize: 10 }}
              label={{
                value: 'predicted P(over) %',
                position: 'insideBottom',
                dy: 8,
                style: { fontSize: 10 },
              }}
            />
            <YAxis
              type="number"
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              tick={{ fontSize: 10 }}
              label={{
                value: 'actual hit %',
                angle: -90,
                position: 'insideLeft',
                style: { fontSize: 10 },
              }}
            />
            <Tooltip
              formatter={(value: number, name: string) =>
                name === 'actual' || name === 'predicted'
                  ? `${value.toFixed(1)}%`
                  : value.toLocaleString()
              }
            />
            {/* Diagonal = perfect calibration */}
            <Line
              type="linear"
              dataKey="ideal"
              stroke="#cbd5e1"
              strokeDasharray="4 4"
              dot={false}
              isAnimationActive={false}
              name="ideal"
            />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#2563eb"
              strokeWidth={2}
              dot={{ r: 3 }}
              isAnimationActive={false}
              name="actual"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="h-24 mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={calibrationChartData(report)}
            margin={{ top: 0, right: 10, bottom: 0, left: 0 }}
          >
            <XAxis
              dataKey="bucket"
              tick={{ fontSize: 9 }}
              interval={0}
              angle={-30}
              textAnchor="end"
              height={40}
            />
            <YAxis hide />
            <Tooltip formatter={(v: number) => v.toLocaleString()} />
            <Bar dataKey="n" name="samples">
              {calibrationChartData(report).map((d) => (
                <Cell
                  key={d.bucket}
                  fill={d.n < 10 ? '#fda4af' : d.n < 50 ? '#fdba74' : '#94a3b8'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-slate-400 mt-1">
        Bucket sample sizes — orange/red = small N, low confidence in that bucket's hit rate.
      </p>
    </div>
  );
}

function RecRow({
  label,
  rec,
}: {
  label: string;
  rec: RecommendationsRecord | undefined;
}) {
  if (!rec || rec.n === 0) {
    return (
      <tr className="border-t border-slate-100">
        <td className="px-3 py-2 font-mono text-slate-600">{label}</td>
        <td className="px-3 py-2 text-right text-slate-400" colSpan={5}>
          No recommendations at +5% EV / -110.
        </td>
      </tr>
    );
  }
  const settled = rec.n - rec.pushes;
  const losses = settled - rec.wins;
  // -110 break-even is 52.38%. Anything noticeably above is +ROI.
  const breakeven = 100 / 210;
  let cls = 'text-slate-600';
  if (rec.win_rate != null) {
    if (rec.win_rate > breakeven + 0.02) cls = 'text-emerald-600 font-semibold';
    else if (rec.win_rate >= breakeven) cls = 'text-amber-600';
    else cls = 'text-rose-600';
  }
  return (
    <tr className="border-t border-slate-100">
      <td className="px-3 py-2 font-mono text-slate-600">{label}</td>
      <td className="px-3 py-2 text-right font-mono">{rec.n}</td>
      <td className="px-3 py-2 text-right font-mono text-emerald-700">{rec.wins}</td>
      <td className="px-3 py-2 text-right font-mono text-rose-700">{losses}</td>
      <td className="px-3 py-2 text-right font-mono text-slate-400">
        {rec.pushes > 0 ? rec.pushes : '—'}
      </td>
      <td className={`px-3 py-2 text-right font-mono ${cls}`}>
        {rec.win_rate != null ? `${(rec.win_rate * 100).toFixed(1)}%` : '—'}
      </td>
    </tr>
  );
}

function RecommendationRecordCard({
  result,
}: {
  result: Record<string, SyntheticStatReport>;
}) {
  // Aggregate the per-stat OVER and UNDER subrecords into one overall record.
  let totalN = 0;
  let totalWins = 0;
  let totalPushes = 0;
  let totalOverN = 0;
  let totalOverWins = 0;
  let totalUnderN = 0;
  let totalUnderWins = 0;
  for (const stat of STAT_ORDER) {
    const r = result[stat]?.recommendations;
    if (!r) continue;
    totalN += r.n;
    totalWins += r.wins;
    totalPushes += r.pushes;
    totalOverN += r.over.n;
    totalOverWins += r.over.wins;
    totalUnderN += r.under.n;
    totalUnderWins += r.under.wins;
  }
  if (totalN === 0) {
    return null;
  }
  const totalSettled = totalN - totalPushes;
  const totalWinRate = totalSettled > 0 ? totalWins / totalSettled : null;
  const breakeven = 100 / 210; // 52.38% at -110

  return (
    <section className="mb-6">
      <h2 className="text-lg font-semibold mb-3">Recommendation record</h2>
      <p className="text-sm text-slate-500 mb-4">
        Per stat, how often the model's recommended side hit. A recommendation
        fires when EV at synthetic <span className="font-mono">-110</span> odds
        clears <span className="font-mono">+5%</span> on either side. PASS rows
        excluded. Break-even at <span className="font-mono">-110</span> is{' '}
        <span className="font-mono">{(breakeven * 100).toFixed(2)}%</span>.
      </p>
      <div className="rounded-lg border border-slate-200 bg-white overflow-hidden">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-50 text-xs uppercase text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">Stat / Side</th>
              <th className="px-3 py-2 text-right">Recs</th>
              <th className="px-3 py-2 text-right">W</th>
              <th className="px-3 py-2 text-right">L</th>
              <th className="px-3 py-2 text-right">Push</th>
              <th className="px-3 py-2 text-right">Win rate</th>
            </tr>
          </thead>
          <tbody>
            {STAT_ORDER.map((stat) => {
              const rec = result[stat]?.recommendations;
              return (
                <RecRow
                  key={stat}
                  label={STAT_LABEL[stat] ?? stat}
                  rec={rec}
                />
              );
            })}
            <tr className="border-t-2 border-slate-300 bg-slate-50">
              <td className="px-3 py-2 font-semibold">Overall</td>
              <td className="px-3 py-2 text-right font-mono font-semibold">
                {totalN}
              </td>
              <td className="px-3 py-2 text-right font-mono text-emerald-700 font-semibold">
                {totalWins}
              </td>
              <td className="px-3 py-2 text-right font-mono text-rose-700 font-semibold">
                {totalSettled - totalWins}
              </td>
              <td className="px-3 py-2 text-right font-mono text-slate-400">
                {totalPushes > 0 ? totalPushes : '—'}
              </td>
              <td
                className={`px-3 py-2 text-right font-mono font-semibold ${
                  totalWinRate == null
                    ? 'text-slate-400'
                    : totalWinRate > breakeven + 0.02
                      ? 'text-emerald-600'
                      : totalWinRate >= breakeven
                        ? 'text-amber-600'
                        : 'text-rose-600'
                }`}
              >
                {totalWinRate != null
                  ? `${(totalWinRate * 100).toFixed(1)}%`
                  : '—'}
              </td>
            </tr>
          </tbody>
        </table>
        <div className="px-3 py-2 text-xs text-slate-400 border-t border-slate-100 grid grid-cols-2 gap-2">
          <span>
            OVER: {totalOverWins} / {totalOverN}{' '}
            {totalOverN > 0
              ? `(${((totalOverWins / totalOverN) * 100).toFixed(1)}%)`
              : ''}
          </span>
          <span className="text-right">
            UNDER: {totalUnderWins} / {totalUnderN}{' '}
            {totalUnderN > 0
              ? `(${((totalUnderWins / totalUnderN) * 100).toFixed(1)}%)`
              : ''}
          </span>
        </div>
      </div>
    </section>
  );
}

function StatusPill({ stat, lift }: { stat: string; lift: number | null | undefined }) {
  const label = STAT_LABEL[stat] ?? stat;
  if (lift == null || Number.isNaN(lift)) {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-mono bg-slate-100 text-slate-500">
        {label} —
      </span>
    );
  }
  let cls = 'bg-rose-100 text-rose-700';
  if (lift > 0.1) cls = 'bg-emerald-100 text-emerald-700';
  else if (lift > 0) cls = 'bg-amber-100 text-amber-800';
  const sign = lift > 0 ? '+' : '';
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-mono ${cls}`}>
      {label} {sign}
      {lift.toFixed(2)}
    </span>
  );
}

function FeatureImportanceCard({
  stat,
  entries,
  coverage,
}: {
  stat: string;
  entries: FeatureImportanceEntry[];
  coverage: Record<string, number> | undefined;
}) {
  // Top 15 by importance, render in DESCENDING order.
  const top = entries.slice(0, 15);
  // Recharts vertical BarChart renders bottom-up, so reverse for top-of-chart = highest.
  const chartData = top.slice().reverse().map((e) => ({
    feature: e.feature,
    importance: e.importance * 100,
  }));

  // Coverage table: sort ascending so dead features bubble up.
  const covRows = coverage
    ? Object.entries(coverage)
        .map(([f, p]) => ({ feature: f, pct: p * 100 }))
        .sort((a, b) => a.pct - b.pct)
    : [];

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h3 className="font-semibold mb-3">{STAT_LABEL[stat] ?? stat}</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h4 className="text-xs font-semibold text-slate-500 uppercase mb-2">
            Top 15 features by importance
          </h4>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ top: 4, right: 24, bottom: 4, left: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(v) => `${v.toFixed(0)}%`}
                  domain={[0, 'dataMax']}
                />
                <YAxis
                  dataKey="feature"
                  type="category"
                  tick={{ fontSize: 10 }}
                  width={140}
                  interval={0}
                />
                <Tooltip formatter={(v: number) => `${v.toFixed(2)}%`} />
                <Bar dataKey="importance" fill="#2563eb" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <h4 className="text-xs font-semibold text-slate-500 uppercase mb-2">
            Feature coverage (% non-null in training set)
          </h4>
          <div className="overflow-y-auto h-72 border border-slate-100 rounded text-xs">
            <table className="min-w-full">
              <thead className="bg-slate-50 sticky top-0">
                <tr>
                  <th className="px-2 py-1 text-left font-medium text-slate-500">Feature</th>
                  <th className="px-2 py-1 text-right font-medium text-slate-500">Coverage</th>
                </tr>
              </thead>
              <tbody>
                {covRows.map((row) => {
                  const pctClass =
                    row.pct < 20
                      ? 'bg-rose-50 text-rose-700'
                      : row.pct < 80
                      ? 'bg-amber-50 text-amber-800'
                      : 'text-slate-600';
                  return (
                    <tr key={row.feature} className={`border-t border-slate-100 ${pctClass}`}>
                      <td className="px-2 py-1 font-mono">{row.feature}</td>
                      <td className="px-2 py-1 text-right font-mono">
                        {row.pct.toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-slate-400 mt-1">
            Red &lt;20%, amber &lt;80%. Dead features bubble to the top.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function PerformanceLogPage() {
  const healthQ = useQuery({ queryKey: ['health'], queryFn: fetchHealth });
  const perfQ = useQuery({ queryKey: ['performance'], queryFn: fetchPerformance });
  const importanceQ = useQuery({
    queryKey: ['feature-importance'],
    queryFn: () => fetchFeatureImportance(15),
  });
  const coverageQ = useQuery({ queryKey: ['coverage'], queryFn: fetchCoverage });

  const synth = perfQ.data?.synthetic;
  const real = perfQ.data?.real;
  const importance = importanceQ.data?.stats ?? {};
  const coverage = coverageQ.data?.coverage ?? {};

  return (
    <div className="p-6 mx-auto max-w-6xl">
      <h1 className="text-2xl font-bold mb-1">Performance</h1>
      <p className="text-sm text-slate-500 mb-4">
        Backtest results from <code className="font-mono">scripts/run_backtest.py</code>. Re-run
        with <code className="font-mono">--save</code> to refresh.
      </p>

      <details className="mb-6 rounded-lg border border-slate-200 bg-white p-4 text-sm">
        <summary className="cursor-pointer font-semibold text-slate-700">
          How to read this page
        </summary>
        <div className="mt-3 grid gap-3 md:grid-cols-2 text-slate-600">
          <div>
            <h3 className="font-semibold text-slate-800 mb-1">Per-stat metrics</h3>
            <ul className="list-disc list-inside space-y-1">
              <li>
                <span className="font-semibold">MAE</span> — average size of the error in stat
                units. Lower is better.
              </li>
              <li>
                <span className="font-semibold">RMSE</span> — same idea but penalises big misses.
                Gap between RMSE and MAE = how often the model has outliers.
              </li>
              <li>
                <span className="font-semibold">Log loss</span> — how confident-and-correct the
                over/under probabilities are. The naive "always 50%" model gets 0.693, so anything
                below means real signal.
              </li>
              <li>
                <span className="font-semibold">Baseline MAE</span> — MAE you'd get by predicting
                the player's last-10-games mean. The bar to clear.
              </li>
              <li>
                <span className="font-semibold">vs baseline</span> — units of MAE the model beats
                that bar by. Green = useful; red = adding noise.
              </li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-slate-800 mb-1">Calibration plot</h3>
            <p className="mb-2">
              Each point is a probability bucket. X-axis = what the model said the chance of
              hitting the over was; Y-axis = how often it actually did.
            </p>
            <ul className="list-disc list-inside space-y-1">
              <li>
                <span className="font-semibold">On the dashed diagonal</span> = well-calibrated
                (model said 70%, hit 70%).
              </li>
              <li>
                <span className="font-semibold">Below the diagonal</span> = over-confident (model
                claimed more than reality delivered).
              </li>
              <li>
                <span className="font-semibold">Above the diagonal</span> = under-confident — when
                the model says 60%, it actually hits more. That's exploitable signal.
              </li>
            </ul>
            <p className="mt-2 text-xs text-slate-500">
              Bars under each curve = sample size in that bucket. Red/orange = too few samples to
              trust the hit-rate.
            </p>
          </div>
          <div className="md:col-span-2 text-xs text-slate-500 border-t border-slate-100 pt-3">
            <span className="font-semibold text-slate-600">Synthetic vs real:</span> the synthetic
            backtest grades against a stand-in line (the player's L10 mean rounded to .5). It
            measures model quality. The real-line backtest grades against actual sportsbook lines
            and produces ROI numbers, but needs weeks of captured props to fill in.
          </div>
        </div>
      </details>

      {synth && (
        <section className="mb-6 flex flex-wrap gap-2 items-center">
          <span className="text-xs text-slate-500 mr-1">vs L10 baseline:</span>
          {STAT_ORDER.map((s) => (
            <StatusPill
              key={s}
              stat={s}
              lift={synth.result[s]?.vs_baseline_lift ?? null}
            />
          ))}
        </section>
      )}

      <section className="rounded-lg border border-slate-200 bg-white p-4 mb-6">
        <div className="flex items-baseline gap-4 flex-wrap">
          <h2 className="font-semibold">Service</h2>
          {healthQ.isLoading && <span className="text-slate-500 text-sm">Checking…</span>}
          {healthQ.error && <span className="text-rose-600 text-sm">API not reachable.</span>}
          {healthQ.data && (
            <span className="text-sm text-slate-600">
              status=<span className="font-mono">{healthQ.data.status ?? '—'}</span> · models=
              <span className="font-mono">{healthQ.data.models_loaded ?? '—'}</span> ·{' '}
              <span className="font-mono">
                {Array.isArray(healthQ.data.stats) && healthQ.data.stats.length > 0
                  ? healthQ.data.stats.join(', ')
                  : '—'}
              </span>
            </span>
          )}
        </div>
      </section>

      <section className="mb-6">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-lg font-semibold">Synthetic backtest</h2>
          {synth && (
            <span className="text-xs text-slate-500 font-mono">
              {synth.start} → {synth.end} · generated {synth.generated_at.slice(0, 10)}
            </span>
          )}
        </div>
        <p className="text-sm text-slate-500 mb-4">
          Walks completed games, predicts each as-of game date (no leakage), grades against the
          actual stat using the L10-rounded line as the synthetic over/under. Calibration plot:
          predicted P(over) bucket vs. actual hit rate. Closer to the dashed diagonal = better
          calibrated.
        </p>
        {perfQ.isLoading && <div className="text-slate-500">Loading report…</div>}
        {perfQ.isError && (
          <div className="rounded border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
            Couldn't load performance report. Try{' '}
            <code className="font-mono">curl http://localhost:8000/api/v1/performance/</code>.
          </div>
        )}
        {!synth && perfQ.isSuccess && (
          <div className="rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
            No synthetic report yet. Run:
            <pre className="mt-2 text-xs font-mono bg-white p-2 rounded border border-amber-200 overflow-x-auto">
              python scripts/run_backtest.py synthetic --start 2024-03-01 --end 2024-04-14 --save
            </pre>
          </div>
        )}
        {synth && (
          <div className="grid gap-4 md:grid-cols-2">
            {STAT_ORDER.filter((s) => synth.result[s]).map((stat) => (
              <StatCard key={stat} stat={stat} report={synth.result[stat]} />
            ))}
          </div>
        )}
      </section>

      {synth && <RecommendationRecordCard result={synth.result} />}

      <section className="mb-6">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-lg font-semibold">Model insights</h2>
          {coverageQ.data && (
            <span className="text-xs text-slate-500 font-mono">
              coverage from train: {coverageQ.data.train_end} · {coverageQ.data.feature_columns.length} features
            </span>
          )}
        </div>
        <p className="text-sm text-slate-500 mb-4">
          What the model is actually leaning on — and what it's ignoring. Importance is XGBoost gain
          averaged across all 6 sub-models per stat, normalised so the bars sum to ~100%. Coverage
          shows what fraction of training rows had a non-null value for each feature; near-zero
          coverage means the model effectively never saw that signal.
        </p>
        {(importanceQ.isLoading || coverageQ.isLoading) && (
          <div className="text-slate-500">Loading model insights…</div>
        )}
        {(importanceQ.isError || coverageQ.isError) && (
          <div className="rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
            Insights endpoints not ready yet. Make sure models are trained and a coverage report
            exists at <code className="font-mono">data/reports/feature_coverage.json</code>.
          </div>
        )}
        {!importanceQ.isLoading && Object.keys(importance).length > 0 && (
          <div className="grid gap-4">
            {STAT_ORDER.filter((s) => importance[s]).map((stat) => (
              <FeatureImportanceCard
                key={stat}
                stat={stat}
                entries={importance[stat]}
                coverage={coverage[stat]}
              />
            ))}
          </div>
        )}
      </section>

      <section>
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-lg font-semibold">Real-line backtest</h2>
          {real && (
            <span className="text-xs text-slate-500 font-mono">
              {real.start} → {real.end} · generated {real.generated_at.slice(0, 10)}
            </span>
          )}
        </div>
        {!real && (
          <div className="rounded border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
            <p className="mb-2">
              Real-line ROI numbers will appear once a few weeks of captured prop_lines have
              accumulated. Until then this section stays empty.
            </p>
            <p className="text-xs text-slate-500 font-mono">
              python scripts/run_backtest.py real --start … --end … --save
            </p>
          </div>
        )}
        {real && (
          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <h3 className="font-semibold mb-3">Overall</h3>
              <dl className="grid grid-cols-2 gap-y-1 text-sm">
                <dt className="text-slate-500">Bets</dt>
                <dd className="font-mono text-right">{real.result.n_bets.toLocaleString()}</dd>
                <dt className="text-slate-500">Win rate</dt>
                <dd className="font-mono text-right">
                  {(real.result.win_rate * 100).toFixed(1)}%
                </dd>
                <dt className="text-slate-500">Staked</dt>
                <dd className="font-mono text-right">
                  ${real.result.total_staked.toFixed(2)}
                </dd>
                <dt className="text-slate-500">Returned</dt>
                <dd className="font-mono text-right">
                  ${real.result.total_returned.toFixed(2)}
                </dd>
                <dt className="text-slate-500">ROI</dt>
                <dd
                  className={`font-mono text-right font-semibold ${
                    real.result.roi >= 0 ? 'text-emerald-600' : 'text-rose-600'
                  }`}
                >
                  {real.result.roi >= 0 ? '+' : ''}
                  {(real.result.roi * 100).toFixed(2)}%
                </dd>
              </dl>
            </div>
            {Object.entries(real.result.by_stat).map(([stat, r]) => (
              <div key={stat} className="rounded-lg border border-slate-200 bg-white p-4">
                <h3 className="font-semibold mb-3">{STAT_LABEL[stat] ?? stat}</h3>
                <dl className="grid grid-cols-2 gap-y-1 text-sm">
                  <dt className="text-slate-500">Bets</dt>
                  <dd className="font-mono text-right">{r.n_bets.toLocaleString()}</dd>
                  <dt className="text-slate-500">Win rate</dt>
                  <dd className="font-mono text-right">{(r.win_rate * 100).toFixed(1)}%</dd>
                  <dt className="text-slate-500">ROI</dt>
                  <dd
                    className={`font-mono text-right font-semibold ${
                      r.roi >= 0 ? 'text-emerald-600' : 'text-rose-600'
                    }`}
                  >
                    {r.roi >= 0 ? '+' : ''}
                    {(r.roi * 100).toFixed(2)}%
                  </dd>
                </dl>
              </div>
            ))}
          </div>
        )}
      </section>

    </div>
  );
}
