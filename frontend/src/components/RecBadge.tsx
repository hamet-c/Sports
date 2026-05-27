interface Props {
  recommendation: 'OVER' | 'UNDER' | 'PASS';
}

export default function RecBadge({ recommendation }: Props) {
  const cls =
    recommendation === 'OVER'
      ? 'bg-emerald-100 text-emerald-700 ring-emerald-300'
      : recommendation === 'UNDER'
        ? 'bg-rose-100 text-rose-700 ring-rose-300'
        : 'bg-slate-100 text-slate-500 ring-slate-200';
  return (
    <span
      className={`inline-block rounded px-2 py-0.5 text-xs font-semibold ring-1 ring-inset ${cls}`}
    >
      {recommendation}
    </span>
  );
}
