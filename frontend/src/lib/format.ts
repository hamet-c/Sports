export function formatPercent(p: number, digits = 1): string {
  return `${(p * 100).toFixed(digits)}%`;
}

export function formatSignedPercent(p: number, digits = 1): string {
  const v = (p * 100).toFixed(digits);
  return p >= 0 ? `+${v}%` : `${v}%`;
}

export function formatOdds(odds: number): string {
  return odds > 0 ? `+${odds}` : `${odds}`;
}

/**
 * American odds → implied probability (including the book's vig).
 * Mirrors backend app.services.edge.implied_probability.
 */
export function impliedProbability(odds: number): number {
  if (odds > 0) return 100 / (odds + 100);
  return Math.abs(odds) / (Math.abs(odds) + 100);
}

export function formatStatLabel(stat: string): string {
  switch (stat) {
    case 'points':
      return 'PTS';
    case 'rebounds':
      return 'REB';
    case 'assists':
      return 'AST';
    case 'threes_made':
      return '3PM';
    case 'pra':
      return 'PRA';
    default:
      return stat.toUpperCase();
  }
}
