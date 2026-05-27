import { useEffect, useMemo, useState } from 'react';
import { useQueries, useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { fetchRecommendationRecord, fetchSlate, fetchSlateAnchor } from '../lib/api';
import { formatSignedPercent, formatStatLabel } from '../lib/format';
import RecBadge from '../components/RecBadge';
import HelpIcon from '../components/HelpIcon';
import type { RecRecordResponse, SlateProp, SlateResponse } from '../types/api';

// Rolling 7-day window centered on today: 3 days past + today + 3 days ahead.
const WINDOW_DAYS = 7;
const WINDOW_BACK = 3;

function isoDate(d: Date): string {
  // YYYY-MM-DD in local time. Slate API treats target_date as a local date.
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function buildWindow(startDate: Date, days: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < days; i++) {
    const d = new Date(startDate);
    d.setDate(d.getDate() + i);
    out.push(isoDate(d));
  }
  return out;
}

function shortDayLabel(iso: string, todayIso: string, tomorrowIso: string): string {
  if (iso === todayIso) return 'Today';
  if (iso === tomorrowIso) return 'Tomorrow';
  // Parse as local YYYY-MM-DD by constructing through the components.
  const [y, m, d] = iso.split('-').map(Number);
  const dt = new Date(y, m - 1, d);
  return dt.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
}

type PlayerSlate = {
  player_id: number;
  player_name: string;
  team_abbr: string;
  is_home: boolean;
  props: SlateProp[];
  best_prop: SlateProp;
  best_side: 'over' | 'under';
  best_ev: number;
};

type GameGroup = {
  game_id: number;
  game_date: string;
  home_abbr: string;
  away_abbr: string;
  home_players: PlayerSlate[];
  away_players: PlayerSlate[];
};

function aggregateByGame(props: SlateProp[]): GameGroup[] {
  const games = new Map<number, GameGroup>();

  const playerKey = (gameId: number, playerId: number) => `${gameId}:${playerId}`;
  const playerMap = new Map<string, PlayerSlate>();

  for (const p of props) {
    let game = games.get(p.game_id);
    if (!game) {
      game = {
        game_id: p.game_id,
        game_date: p.game_date,
        home_abbr: p.home_abbr,
        away_abbr: p.away_abbr,
        home_players: [],
        away_players: [],
      };
      games.set(p.game_id, game);
    }

    const key = playerKey(p.game_id, p.player_id);
    let player = playerMap.get(key);
    if (!player) {
      player = {
        player_id: p.player_id,
        player_name: p.player_name,
        team_abbr: p.team_abbr,
        is_home: p.is_home,
        props: [],
        best_prop: p,
        best_side: 'over',
        best_ev: -Infinity,
      };
      playerMap.set(key, player);
      (p.is_home ? game.home_players : game.away_players).push(player);
    }
    player.props.push(p);
    const ev = Math.max(p.expected_value_over, p.expected_value_under);
    if (ev > player.best_ev) {
      player.best_ev = ev;
      player.best_prop = p;
      player.best_side = p.expected_value_over >= p.expected_value_under ? 'over' : 'under';
    }
  }

  for (const game of games.values()) {
    game.home_players.sort((a, b) => b.best_ev - a.best_ev);
    game.away_players.sort((a, b) => b.best_ev - a.best_ev);
  }

  // Sort games chronologically by date, then by matchup label for stable order
  // when several games share a tip date (which is the typical case).
  return Array.from(games.values()).sort((a, b) => {
    if (a.game_date !== b.game_date) return a.game_date.localeCompare(b.game_date);
    return `${a.away_abbr}@${a.home_abbr}`.localeCompare(`${b.away_abbr}@${b.home_abbr}`);
  });
}

function formatRelativeTime(ms: number): string {
  const sec = Math.floor(ms / 1000);
  if (sec < 5) return 'just now';
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  return `${hr}h ago`;
}

function SlateSkeleton() {
  return (
    <div className="space-y-5" aria-busy="true" aria-label="Loading slate">
      {[0, 1, 2].map((i) => (
        <section
          key={i}
          className="rounded-lg border border-slate-200 bg-white overflow-hidden animate-pulse"
        >
          <div className="px-4 py-2.5 bg-slate-200 h-9" />
          <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-slate-200">
            {[0, 1].map((side) => (
              <div key={side}>
                <div className="px-4 py-2 border-b border-slate-100">
                  <div className="h-4 w-12 bg-slate-200 rounded" />
                </div>
                <ul className="divide-y divide-slate-100">
                  {[0, 1, 2, 3].map((j) => (
                    <li key={j} className="px-4 py-2.5">
                      <div className="flex justify-between gap-3">
                        <div className="h-4 bg-slate-200 rounded w-32" />
                        <div className="h-4 bg-slate-200 rounded w-20" />
                      </div>
                      <div className="mt-2 h-3 bg-slate-100 rounded w-40" />
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function RecRecordBadge({ data }: { data: RecRecordResponse | undefined }) {
  if (!data || data.n_recommendations === 0) return null;
  const settled = data.n_recommendations - data.pushes;
  const losses = settled - data.wins;
  const breakeven = 100 / 210; // 52.38% at -110
  const rate = data.win_rate;
  let cls = 'bg-slate-100 text-slate-600';
  if (rate != null) {
    if (rate > breakeven + 0.02) cls = 'bg-emerald-100 text-emerald-700';
    else if (rate >= breakeven) cls = 'bg-amber-100 text-amber-800';
    else cls = 'bg-rose-100 text-rose-700';
  }
  return (
    <div className="mt-2 flex items-baseline gap-2 flex-wrap">
      <span
        className={`inline-flex items-baseline gap-1 px-2 py-0.5 rounded-full text-xs font-mono ${cls}`}
        title={
          `Recommendations from ${data.start} to ${data.end} graded against actuals. ` +
          `Wins / losses count only OVER/UNDER recommendations (PASS excluded). ` +
          `Break-even at -110 is 52.4%. NOTE: ${data.note}`
        }
      >
        <span className="font-semibold">
          Recent: {data.wins}-{losses}
        </span>
        {rate != null && <span>({(rate * 100).toFixed(1)}%)</span>}
      </span>
      <span className="text-[11px] text-slate-400">
        {data.start} → {data.end} · {data.n_recommendations} recs
      </span>
    </div>
  );
}

function describeError(err: unknown): string {
  if (!err) return 'Unknown error.';
  // Axios errors expose response on the error object; we deliberately avoid
  // importing axios types here to keep the helper portable.
  const e = err as { response?: { status?: number }; message?: string; code?: string };
  if (e.code === 'ERR_NETWORK' || e.message === 'Network Error') {
    return 'API not reachable. Is uvicorn running on :8000?';
  }
  if (e.response?.status) {
    return `API returned ${e.response.status}. Check backend logs.`;
  }
  return e.message ?? String(err);
}

export default function SlatePage() {
  const [minEdge, setMinEdge] = useState(0);
  const [filterStat, setFilterStat] = useState<string>('all');
  const [activeDate, setActiveDate] = useState<string | null>(null);
  // Re-render every 10s so the "last refreshed" label stays accurate without
  // refetching. React Query handles the actual refetch on its own interval.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 10_000);
    return () => window.clearInterval(id);
  }, []);

  const anchorQ = useQuery({
    queryKey: ['slate-anchor'],
    queryFn: fetchSlateAnchor,
    staleTime: 5 * 60_000,
    refetchOnWindowFocus: false,
  });

  // Window strategy: 7 days centered on today (3 back / today / 3 forward).
  // If daily ingest is stale, push the window backward enough so that the
  // most recent prop date is still inside it — otherwise the user opens the
  // page and sees nothing.
  const todayDate = new Date();
  const todayIso = isoDate(todayDate);
  const tomorrowDate = new Date(todayDate);
  tomorrowDate.setDate(todayDate.getDate() + 1);
  const tomorrowIso = isoDate(tomorrowDate);

  const dates = useMemo(() => {
    let start = new Date(todayDate);
    start.setDate(todayDate.getDate() - WINDOW_BACK);
    const latestIso = anchorQ.data?.latest_prop_date;
    if (latestIso && latestIso < todayIso) {
      // Stale ingest. Anchor window's last day at latestIso, but never push
      // past today by more than WINDOW_BACK days into the future.
      const [y, m, d] = latestIso.split('-').map(Number);
      const latest = new Date(y, m - 1, d);
      const staleStart = new Date(latest);
      staleStart.setDate(latest.getDate() - (WINDOW_DAYS - 1));
      if (staleStart < start) start = staleStart;
    }
    return buildWindow(start, WINDOW_DAYS);
  }, [anchorQ.data?.latest_prop_date, todayIso]);

  const dayQueries = useQueries({
    queries: dates.map((d) => ({
      queryKey: ['slate', d, minEdge],
      queryFn: () => fetchSlate(d, minEdge),
      // Today refreshes every minute; future days don't need polling.
      refetchInterval: d === todayIso ? 60_000 : false,
      // Future days are stable; let them stay warm.
      staleTime: d === todayIso ? 0 : 5 * 60_000,
    })),
  });

  const recRecordQ = useQuery({
    queryKey: ['rec-record', 7],
    queryFn: () => fetchRecommendationRecord(7),
    staleTime: 5 * 60_000,
    refetchOnWindowFocus: false,
  });

  // Filter each day's props by the current stat filter and produce a tidy
  // per-day summary used by both the tab strip and the body.
  type DaySummary = {
    date: string;
    data: SlateResponse | undefined;
    isLoading: boolean;
    error: unknown;
    filteredProps: SlateProp[];
    games: ReturnType<typeof aggregateByGame>;
    statTypes: string[];
    updatedAt: number;
  };
  const daySummaries: DaySummary[] = useMemo(() => {
    return dayQueries.map((q, i) => {
      const data = q.data;
      const all = data?.props ?? [];
      const filtered = filterStat === 'all'
        ? all
        : all.filter((p) => p.stat_type === filterStat);
      return {
        date: dates[i],
        data,
        isLoading: q.isLoading,
        error: q.error,
        filteredProps: filtered,
        games: aggregateByGame(filtered),
        statTypes: Array.from(new Set(all.map((p) => p.stat_type))).sort(),
        updatedAt: q.dataUpdatedAt,
      };
    });
  }, [dayQueries, filterStat, dates]);

  // Only show tabs for days that have at least one prop (post-filter) — or
  // are still loading their initial response (so the user sees a tab populate).
  const visibleDays = daySummaries.filter(
    (d) => d.isLoading || d.filteredProps.length > 0,
  );

  // Decide the active tab. Sticky to user selection if that day is still
  // present; otherwise default to today if visible, else the first visible day.
  const activeDay =
    visibleDays.find((d) => d.date === activeDate)
      ?? visibleDays.find((d) => d.date === todayIso)
      ?? visibleDays[0];

  const allStats = useMemo(() => {
    const s = new Set<string>();
    for (const d of daySummaries) for (const t of d.statTypes) s.add(t);
    return Array.from(s).sort();
  }, [daySummaries]);

  // Surface fatal errors (network down etc) only if EVERY day failed —
  // otherwise the per-day card shows the per-day error.
  const everyDayErrored =
    dayQueries.length > 0 && dayQueries.every((q) => q.error != null);
  if (everyDayErrored) {
    return (
      <div className="p-8">
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-rose-700">
          <h2 className="font-semibold mb-1">Couldn't load slate</h2>
          <p className="text-sm">{describeError(dayQueries[0]!.error)}</p>
        </div>
      </div>
    );
  }

  const totalWeekProps = daySummaries.reduce(
    (acc, d) => acc + d.filteredProps.length,
    0,
  );
  const totalWeekGames = visibleDays.reduce((acc, d) => acc + d.games.length, 0);

  return (
    <div className="p-6 mx-auto max-w-6xl">
      <header className="mb-6 flex items-baseline justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold">This Week's Slate</h1>
          <p className="text-sm text-slate-500">
            {totalWeekGames} {totalWeekGames === 1 ? 'game' : 'games'} across{' '}
            {visibleDays.length}{' '}
            {visibleDays.length === 1 ? 'day' : 'days'} ·{' '}
            {totalWeekProps} props
            {activeDay?.updatedAt && activeDay.updatedAt > 0 && (
              <>
                {' '}· {shortDayLabel(activeDay.date, todayIso, tomorrowIso)} updated{' '}
                {formatRelativeTime(now - activeDay.updatedAt)}
              </>
            )}
          </p>
          <RecRecordBadge data={recRecordQ.data} />
          {anchorQ.data &&
            anchorQ.data.days_stale != null &&
            anchorQ.data.days_stale > 1 && (
              <p
                className="mt-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1 inline-block"
                title="Showing the most recent window we have prop data for. Run scripts/ingest_props.py to refresh."
              >
                Daily prop ingest is {anchorQ.data.days_stale} days stale (latest:{' '}
                {anchorQ.data.latest_prop_date}). Showing the most recent window
                we have data for.
              </p>
            )}
        </div>
        <div className="flex items-center gap-4 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-slate-600">
              Min EV
              <HelpIcon topic="EV" />
            </span>
            <select
              value={minEdge}
              onChange={(e) => setMinEdge(Number(e.target.value))}
              className="rounded border border-slate-300 bg-white px-2 py-1"
            >
              <option value={0}>0%</option>
              <option value={0.03}>+3%</option>
              <option value={0.05}>+5%</option>
              <option value={0.08}>+8%</option>
              <option value={0.1}>+10%</option>
            </select>
          </label>
          <label className="flex items-center gap-2">
            <span className="text-slate-600">Stat</span>
            <select
              value={filterStat}
              onChange={(e) => setFilterStat(e.target.value)}
              className="rounded border border-slate-300 bg-white px-2 py-1"
            >
              <option value="all">All</option>
              {allStats.map((s) => (
                <option key={s} value={s}>
                  {formatStatLabel(s)}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      {visibleDays.length === 0 ? (
        dayQueries.some((q) => q.isLoading) ? (
          <SlateSkeleton />
        ) : (
          <div className="rounded-lg border border-slate-200 bg-white p-8 text-center text-slate-500">
            <p className="mb-3">
              No games in the next {WINDOW_DAYS} days
              {minEdge > 0 || filterStat !== 'all'
                ? ` matching the current filter (Min EV ${(minEdge * 100).toFixed(0)}%${
                    filterStat !== 'all' ? `, ${formatStatLabel(filterStat)} only` : ''
                  })`
                : ''}.
            </p>
            {(minEdge > 0 || filterStat !== 'all') && (
              <button
                type="button"
                onClick={() => {
                  setMinEdge(0);
                  setFilterStat('all');
                }}
                className="text-sm rounded border border-slate-300 px-3 py-1 text-slate-700 hover:bg-slate-50"
              >
                Reset filters
              </button>
            )}
          </div>
        )
      ) : (
        <>
          <DayTabStrip
            days={visibleDays}
            activeDate={activeDay?.date ?? null}
            onSelect={setActiveDate}
            todayIso={todayIso}
            tomorrowIso={tomorrowIso}
          />
          {activeDay && (
            <DayPanel
              day={activeDay}
              minEdge={minEdge}
              filterStat={filterStat}
              onResetFilters={() => {
                setMinEdge(0);
                setFilterStat('all');
              }}
            />
          )}
        </>
      )}
    </div>
  );
}

function DayTabStrip({
  days,
  activeDate,
  onSelect,
  todayIso,
  tomorrowIso,
}: {
  days: { date: string; games: unknown[]; filteredProps: unknown[]; isLoading: boolean }[];
  activeDate: string | null;
  onSelect: (d: string) => void;
  todayIso: string;
  tomorrowIso: string;
}) {
  return (
    <div
      role="tablist"
      aria-label="Slate days"
      className="mb-4 flex flex-wrap gap-1 border-b border-slate-200"
    >
      {days.map((d) => {
        const isActive = d.date === activeDate;
        const label = shortDayLabel(d.date, todayIso, tomorrowIso);
        const count = d.games.length;
        return (
          <button
            key={d.date}
            role="tab"
            aria-selected={isActive}
            type="button"
            onClick={() => onSelect(d.date)}
            className={`px-3 py-2 -mb-px text-sm border-b-2 transition-colors ${
              isActive
                ? 'border-slate-900 text-slate-900 font-semibold'
                : 'border-transparent text-slate-500 hover:text-slate-700'
            }`}
          >
            <span>{label}</span>
            {d.isLoading ? (
              <span className="ml-1.5 inline-block w-4 h-3 align-middle bg-slate-200 rounded animate-pulse" />
            ) : (
              <span
                className={`ml-1.5 inline-flex items-center justify-center min-w-[1.25rem] px-1 rounded text-xs font-mono ${
                  isActive ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-500'
                }`}
              >
                {count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

function DayPanel({
  day,
  minEdge,
  filterStat,
  onResetFilters,
}: {
  day: {
    date: string;
    isLoading: boolean;
    error: unknown;
    filteredProps: SlateProp[];
    games: ReturnType<typeof aggregateByGame>;
  };
  minEdge: number;
  filterStat: string;
  onResetFilters: () => void;
}) {
  if (day.isLoading) return <SlateSkeleton />;
  if (day.error) {
    return (
      <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-rose-700">
        <p className="text-sm">{describeError(day.error)}</p>
      </div>
    );
  }
  if (day.games.length === 0) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-8 text-center text-slate-500">
        <p className="mb-3">
          No props for this day meet the current filter
          {minEdge > 0 || filterStat !== 'all'
            ? ` (Min EV ${(minEdge * 100).toFixed(0)}%${
                filterStat !== 'all' ? `, ${formatStatLabel(filterStat)} only` : ''
              })`
            : ''}.
        </p>
        {(minEdge > 0 || filterStat !== 'all') && (
          <button
            type="button"
            onClick={onResetFilters}
            className="text-sm rounded border border-slate-300 px-3 py-1 text-slate-700 hover:bg-slate-50"
          >
            Reset filters
          </button>
        )}
      </div>
    );
  }
  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
        <span>
          Each row shows the player's
          <span className="font-semibold mx-1 text-slate-600">
            Best EV
            <HelpIcon topic="Best EV" />
          </span>
          and a
          <span className="font-semibold mx-1 text-slate-600">
            Rec
            <HelpIcon topic="REC" />
          </span>
          tag when the model has a strong opinion.
        </span>
        <span className="text-slate-400">Click any player for full details.</span>
      </div>
      <div className="space-y-5">
        {day.games.map((g) => (
          <GameCard key={g.game_id} game={g} />
        ))}
      </div>
    </>
  );
}

function GameCard({ game }: { game: GameGroup }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white overflow-hidden">
      <header className="px-4 py-2.5 bg-slate-900 text-white text-sm font-semibold tracking-wide flex items-baseline justify-between">
        <span>
          {game.away_abbr} @ {game.home_abbr}
        </span>
        <span className="text-xs font-mono text-slate-400 font-normal">
          {game.game_date}
        </span>
      </header>
      <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-slate-200">
        <TeamColumn label={game.away_abbr} sublabel="Away" players={game.away_players} />
        <TeamColumn label={game.home_abbr} sublabel="Home" players={game.home_players} />
      </div>
    </section>
  );
}

function TeamColumn({
  label,
  sublabel,
  players,
}: {
  label: string;
  sublabel: string;
  players: PlayerSlate[];
}) {
  return (
    <div>
      <div className="px-4 py-2 border-b border-slate-100 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-slate-700">{label}</h3>
        <span className="text-xs uppercase tracking-wide text-slate-400">{sublabel}</span>
      </div>
      {players.length === 0 ? (
        <div className="px-4 py-6 text-sm text-slate-400">No props</div>
      ) : (
        <ul className="divide-y divide-slate-100">
          {players.map((p) => (
            <PlayerRow key={p.player_id} player={p} />
          ))}
        </ul>
      )}
    </div>
  );
}

function PlayerRow({ player }: { player: PlayerSlate }) {
  const statTypes = Array.from(new Set(player.props.map((pr) => pr.stat_type)));
  const best = player.best_prop;
  const evColor =
    player.best_ev >= 0.05
      ? 'text-emerald-600'
      : player.best_ev > 0
        ? 'text-emerald-500'
        : 'text-slate-400';
  return (
    <li>
      <Link
        to={`/players/${player.player_id}`}
        className="block px-4 py-2.5 hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center justify-between gap-3">
          <div className="font-medium text-slate-900 truncate">{player.player_name}</div>
          <div className="flex items-center gap-2 text-xs flex-shrink-0">
            <span className="font-mono text-slate-600">
              {formatStatLabel(best.stat_type)} {player.best_side === 'over' ? 'o' : 'u'}
              {best.line.toFixed(1)}
            </span>
            <span className={`font-mono font-semibold ${evColor}`}>
              {formatSignedPercent(player.best_ev)}
            </span>
            {best.recommendation !== 'PASS' && (
              <RecBadge recommendation={best.recommendation} />
            )}
            {best.sharp_book_disagreement && best.recommendation !== 'PASS' && (
              <span
                className="text-amber-600 text-sm leading-none cursor-help"
                title={
                  `Sharp book disagreement — the book is heavily favoring ${best.book_favored_side} ` +
                  `but our model gives that side a low probability. Click in to see details. ` +
                  `Books shade lines this hard when they have information we lack.`
                }
                aria-label="Sharp book disagreement warning"
              >
                ⚠
              </span>
            )}
          </div>
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-1 text-[11px] text-slate-400">
          {statTypes.map((s) => (
            <span
              key={s}
              className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-slate-500"
            >
              {formatStatLabel(s)}
            </span>
          ))}
          <span className="ml-0.5">
            · {player.props.length} prop{player.props.length === 1 ? '' : 's'}
          </span>
        </div>
      </Link>
    </li>
  );
}
