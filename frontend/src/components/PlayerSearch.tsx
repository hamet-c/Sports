import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { searchPlayers } from '../lib/api';

export default function PlayerSearch() {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [focusIdx, setFocusIdx] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  // Debounce: only fire the query if it's been ≥200ms since the last keystroke
  // AND the query has at least 2 chars (avoid fetching huge result sets on 'a').
  const [debounced, setDebounced] = useState('');
  useEffect(() => {
    if (query.trim().length < 2) {
      setDebounced('');
      return;
    }
    const id = window.setTimeout(() => setDebounced(query.trim()), 200);
    return () => window.clearTimeout(id);
  }, [query]);

  const { data: results = [], isFetching } = useQuery({
    queryKey: ['player-search', debounced],
    queryFn: () => searchPlayers(debounced),
    enabled: debounced.length >= 2,
    staleTime: 30_000,
  });

  // Close on click-outside.
  useEffect(() => {
    function onDown(e: MouseEvent) {
      if (!containerRef.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, []);

  // Reset focus index when results change so arrow-keys start from the top.
  useEffect(() => {
    setFocusIdx(0);
  }, [results.length]);

  function pickResult(index: number) {
    const r = results[index];
    if (!r) return;
    navigate(`/players/${r.id}`);
    setQuery('');
    setOpen(false);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open || results.length === 0) {
      if (e.key === 'Enter' && results.length === 1) pickResult(0);
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setFocusIdx((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setFocusIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      pickResult(focusIdx);
    } else if (e.key === 'Escape') {
      setOpen(false);
    }
  }

  return (
    <div className="relative" ref={containerRef}>
      <input
        type="text"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
        placeholder="Find a player…"
        aria-label="Search players"
        className="w-56 rounded border border-slate-300 bg-white px-3 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-slate-400"
      />
      {open && debounced.length >= 2 && (
        <div className="absolute z-20 mt-1 w-72 rounded border border-slate-200 bg-white shadow-lg max-h-80 overflow-y-auto">
          {isFetching && results.length === 0 && (
            <div className="px-3 py-2 text-sm text-slate-500">Searching…</div>
          )}
          {!isFetching && results.length === 0 && (
            <div className="px-3 py-2 text-sm text-slate-500">
              No players match "{debounced}".
            </div>
          )}
          {results.map((p, i) => (
            <button
              key={p.id}
              type="button"
              onMouseEnter={() => setFocusIdx(i)}
              onClick={() => pickResult(i)}
              className={`block w-full text-left px-3 py-1.5 text-sm ${
                i === focusIdx ? 'bg-slate-100' : 'hover:bg-slate-50'
              }`}
            >
              <span className="font-medium">{p.full_name}</span>
              {p.position && (
                <span className="ml-2 text-xs text-slate-400">{p.position}</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
