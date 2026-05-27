import { Fragment, useMemo } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  fetchPredictionsVsActual,
  fetchSlate,
  getPlayer,
  getPlayerRecent,
} from '../lib/api';
import {
  formatOdds,
  formatPercent,
  formatSignedPercent,
  formatStatLabel,
  impliedProbability,
} from '../lib/format';
import RecBadge from '../components/RecBadge';
import HelpIcon from '../components/HelpIcon';
import type { RecentPredictionComparison, SlateProp } from '../types/api';

function isAxios404(err: unknown): boolean {
  const e = err as { response?: { status?: number } } | null;
  return e?.response?.status === 404;
}

const COMPARISON_STAT_ORDER = ['points', 'rebounds', 'assists', 'threes_made'];
const COMPARISON_TOLERANCE: Record<string, [number, number]> = {
  // [within = green, within = amber, else red]
  points: [3, 7],
  rebounds: [1.5, 3],
  assists: [1.5, 3],
  threes_made: [0.75, 1.5],
};

function errorColorClass(stat: string, err: number | null): string {
  if (err == null) return 'text-slate-400';
  const [good, ok] = COMPARISON_TOLERANCE[stat] ?? [3, 7];
  const a = Math.abs(err);
  if (a <= good) return 'text-emerald-600';
  if (a <= ok) return 'text-amber-600';
  return 'text-rose-600';
}

export default function PlayerDetailPage() {
  const { playerId } = useParams<{ playerId: string }>();
  const id = Number(playerId);

  const playerQ = useQuery({
    queryKey: ['player', id],
    queryFn: () => getPlayer(id),
    enabled: !Number.isNaN(id),
  });
  const recentQ = useQuery({
    queryKey: ['player-recent', id],
    queryFn: () => getPlayerRecent(id, 20),
    enabled: !Number.isNaN(id),
  });
  const slateQ = useQuery({
    queryKey: ['slate', 0],
    queryFn: () => fetchSlate(undefined, 0),
    enabled: !Number.isNaN(id),
  });
  const compareQ = useQuery({
    queryKey: ['player-compare', id],
    queryFn: () => fetchPredictionsVsActual(id, 3),
    enabled: !Number.isNaN(id),
  });

  const todaysProps = useMemo<SlateProp[]>(() => {
    const all = slateQ.data?.props ?? [];
    return all
      .filter((p) => p.player_id === id)
      .sort((a, b) => {
        if (a.stat_type !== b.stat_type) return a.stat_type.localeCompare(b.stat_type);
        return a.book.localeCompare(b.book);
      });
  }, [slateQ.data, id]);

  if (playerQ.isLoading || recentQ.isLoading) {
    return <PlayerDetailSkeleton />;
  }
  if (playerQ.error) {
    const notFound = isAxios404(playerQ.error);
    return (
      <div className="p-8">
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-rose-700">
          <h2 className="font-semibold mb-1">
            {notFound ? 'Player not found' : "Couldn't load player"}
          </h2>
          <p className="text-sm">
            {notFound
              ? `No player with id ${id} in the database.`
              : 'API request failed. Is uvicorn running on :8000?'}
          </p>
          <Link
            to="/"
            className="inline-block mt-3 text-sm rounded border border-rose-300 px-3 py-1 hover:bg-rose-100"
          >
            Back to slate
          </Link>
        </div>
      </div>
    );
  }

  const player = playerQ.data;
  const recent = (recentQ.data ?? []).slice().reverse();
  const chartData = recent.map((r) => ({
    date: r.game_date.slice(5),
    pts: r.points ?? 0,
    reb: r.rebounds ?? 0,
    ast: r.assists ?? 0,
    threes: r.threes_made ?? 0,
  }));

  const matchupLabel = todaysProps[0]?.matchup;
  const teamLabel = todaysProps[0]?.team_abbr;

  return (
    <div className="p-6 mx-auto max-w-5xl">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">{player?.full_name}</h1>
        <p className="text-sm text-slate-500">
          {player?.position ?? '—'}
          {teamLabel ? ` · ${teamLabel}` : ''}
          {matchupLabel ? ` · ${matchupLabel}` : ''}
        </p>
      </header>

      {todaysProps.length > 0 && (
        <section className="rounded-lg border border-slate-200 bg-white p-4 mb-6">
          <h2 className="font-semibold mb-2">
            Today's props ({todaysProps.length})
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-3 py-2 text-left">Stat</th>
                  <th className="px-3 py-2 text-left">Book</th>
                  <th className="px-3 py-2 text-right">Line</th>
                  <th className="px-3 py-2 text-right">
                    Proj
                    <HelpIcon topic="Proj" />
                  </th>
                  <th className="px-3 py-2 text-right">
                    P(Over)
                    <HelpIcon topic="P(Over)" />
                  </th>
                  <th className="px-3 py-2 text-right">
                    EV Over
                    <HelpIcon topic="EV Over" />
                  </th>
                  <th className="px-3 py-2 text-right">
                    EV Under
                    <HelpIcon topic="EV Under" />
                  </th>
                  <th className="px-3 py-2 text-right">Odds</th>
                  <th className="px-3 py-2 text-center">
                    Rec
                    <HelpIcon topic="REC" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {todaysProps.map((p) => (
                  <Fragment key={`${p.stat_type}-${p.book}`}>
                    <tr className="border-t border-slate-100">
                      <td className="px-3 py-2 font-mono">{formatStatLabel(p.stat_type)}</td>
                      <td className="px-3 py-2 text-slate-500">{p.book}</td>
                      <td className="px-3 py-2 text-right font-mono">{p.line.toFixed(1)}</td>
                      <td className="px-3 py-2 text-right font-mono">
                        {p.predicted_mean.toFixed(1)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono">
                        {formatPercent(p.over_probability)}
                      </td>
                      <td
                        className={`px-3 py-2 text-right font-mono ${
                          p.expected_value_over > 0 ? 'text-emerald-700' : 'text-slate-500'
                        }`}
                      >
                        {formatSignedPercent(p.expected_value_over)}
                      </td>
                      <td
                        className={`px-3 py-2 text-right font-mono ${
                          p.expected_value_under > 0 ? 'text-emerald-700' : 'text-slate-500'
                        }`}
                      >
                        {formatSignedPercent(p.expected_value_under)}
                      </td>
                      <td className="px-3 py-2 text-right text-slate-500 font-mono">
                        {formatOdds(p.over_odds)} / {formatOdds(p.under_odds)}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className="inline-flex items-center gap-1">
                          <RecBadge recommendation={p.recommendation} />
                          {p.sharp_book_disagreement && (
                            <span
                              className="text-amber-600 text-sm leading-none cursor-help"
                              title={
                                `Sharp book disagreement: the book is shading ${p.book_favored_side} hard (` +
                                `${p.book_favored_side === 'OVER' ? formatOdds(p.over_odds) : formatOdds(p.under_odds)}` +
                                `) but our model only gives ${p.book_favored_side} a ` +
                                `${formatPercent(p.book_favored_side === 'OVER' ? p.over_probability : 1 - p.over_probability)} ` +
                                'chance. Books move lines this aggressively when they have information we lack ' +
                                '(lineup news, in-game injury, sharp action). The +EV on the opposite side is ' +
                                'most likely mirage. Check news before betting.'
                              }
                              aria-label="Sharp book disagreement warning"
                            >
                              ⚠
                            </span>
                          )}
                        </span>
                      </td>
                    </tr>
                    {p.sharp_book_disagreement && p.recommendation !== 'PASS' && (
                      <tr className="bg-amber-50 border-t border-amber-200">
                        <td
                          colSpan={9}
                          className="px-3 py-2 text-xs text-amber-800"
                        >
                          <span className="font-semibold">⚠ Sharp book disagreement.</span>{' '}
                          The book is pricing {p.book_favored_side} at{' '}
                          <span className="font-mono">
                            {p.book_favored_side === 'OVER'
                              ? formatOdds(p.over_odds)
                              : formatOdds(p.under_odds)}
                          </span>{' '}
                          (
                          {formatPercent(
                            impliedProbability(
                              p.book_favored_side === 'OVER' ? p.over_odds : p.under_odds,
                            ),
                            0,
                          )}{' '}
                          implied), meaning they're heavily favoring{' '}
                          {p.book_favored_side}. Our model only gives{' '}
                          {p.book_favored_side} a{' '}
                          {formatPercent(
                            p.book_favored_side === 'OVER'
                              ? p.over_probability
                              : 1 - p.over_probability,
                          )}{' '}
                          chance, which is why the {p.recommendation} looks +EV. When books
                          shade lines this hard, they usually have information we don't (lineup
                          news, in-game injury, sharp action) — the {p.recommendation} edge is
                          most likely mirage. Check news before betting.
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {compareQ.data && compareQ.data.length > 0 && (
        <section className="rounded-lg border border-slate-200 bg-white p-4 mb-6">
          <h2 className="font-semibold mb-3">
            Predictions vs actuals (last {compareQ.data.length} games)
            <HelpIcon topic="Predictions vs actuals" />
          </h2>
          <div className="grid gap-3 md:grid-cols-3">
            {compareQ.data.map((g) => (
              <ComparisonCard key={g.game_id} game={g} />
            ))}
          </div>
        </section>
      )}

      <section className="rounded-lg border border-slate-200 bg-white p-4 mb-6">
        <h2 className="font-semibold mb-2">Recent form</h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="pts"
                stroke="#2563eb"
                name="Points"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="reb"
                stroke="#16a34a"
                name="Rebounds"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="ast"
                stroke="#ea580c"
                name="Assists"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="threes"
                stroke="#9333ea"
                name="3PM"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-4 mb-6">
        <h2 className="font-semibold mb-2">Last {recent.length} games</h2>
        <table className="min-w-full text-sm">
          <thead className="bg-slate-50 text-xs uppercase text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">Date</th>
              <th className="px-3 py-2 text-right">MIN</th>
              <th className="px-3 py-2 text-right">PTS</th>
              <th className="px-3 py-2 text-right">REB</th>
              <th className="px-3 py-2 text-right">AST</th>
              <th className="px-3 py-2 text-right">3PM</th>
            </tr>
          </thead>
          <tbody>
            {recent.map((r) => (
              <tr key={r.game_date} className="border-t border-slate-100">
                <td className="px-3 py-2 font-mono">{r.game_date}</td>
                <td className="px-3 py-2 text-right font-mono">
                  {r.minutes?.toFixed(0) ?? '—'}
                </td>
                <td className="px-3 py-2 text-right font-mono">{r.points ?? '—'}</td>
                <td className="px-3 py-2 text-right font-mono">{r.rebounds ?? '—'}</td>
                <td className="px-3 py-2 text-right font-mono">{r.assists ?? '—'}</td>
                <td className="px-3 py-2 text-right font-mono">{r.threes_made ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function ComparisonCard({ game }: { game: RecentPredictionComparison }) {
  const matchupLabel = game.opponent_abbr
    ? `${game.is_home ? 'vs' : '@'} ${game.opponent_abbr}`
    : '';
  return (
    <div className="rounded border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-sm font-mono text-slate-700">{game.game_date}</div>
        <div className="text-xs text-slate-500">
          {matchupLabel}
          {game.minutes != null && (
            <span className="ml-2 text-slate-400">{game.minutes.toFixed(0)} min</span>
          )}
        </div>
      </div>
      <table className="w-full text-xs">
        <thead className="text-[10px] uppercase tracking-wide text-slate-400">
          <tr>
            <th className="text-left font-medium py-0.5">Stat</th>
            <th className="text-right font-medium py-0.5">Pred</th>
            <th className="text-right font-medium py-0.5">Actual</th>
            <th className="text-right font-medium py-0.5">Err</th>
          </tr>
        </thead>
        <tbody>
          {COMPARISON_STAT_ORDER.map((stat) => {
            const c = game.stats[stat];
            if (!c) return null;
            const errCls = errorColorClass(stat, c.error);
            const errStr =
              c.error == null
                ? '—'
                : `${c.error >= 0 ? '+' : ''}${c.error.toFixed(1)}`;
            return (
              <tr key={stat} className="border-t border-slate-200">
                <td className="py-1 font-mono text-slate-600">{formatStatLabel(stat)}</td>
                <td className="py-1 text-right font-mono text-slate-700">
                  {c.predicted.toFixed(1)}
                </td>
                <td className="py-1 text-right font-mono text-slate-700">
                  {c.actual != null ? c.actual.toFixed(0) : '—'}
                </td>
                <td className={`py-1 text-right font-mono font-semibold ${errCls}`}>
                  {errStr}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PlayerDetailSkeleton() {
  return (
    <div className="p-6 mx-auto max-w-5xl animate-pulse" aria-busy="true">
      <div className="mb-6">
        <div className="h-7 bg-slate-200 rounded w-64 mb-2" />
        <div className="h-4 bg-slate-100 rounded w-40" />
      </div>
      {[0, 1, 2].map((i) => (
        <section
          key={i}
          className="rounded-lg border border-slate-200 bg-white p-4 mb-6"
        >
          <div className="h-5 bg-slate-200 rounded w-48 mb-3" />
          <div className="h-32 bg-slate-100 rounded" />
        </section>
      ))}
    </div>
  );
}
