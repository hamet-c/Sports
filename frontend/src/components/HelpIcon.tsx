const HELP_TEXTS: Record<string, string> = {
  // ───── Slate / prop metrics ─────
  EV:
    'Expected value of a bet, expressed as % of stake. ' +
    'EV = P(win) × payout − P(lose) × 1. Positive EV means a long-run profitable bet at these odds. ' +
    "Each player row shows the highest EV across all of that player's props; click into the player to see every side and book.",
  'Best EV':
    'Highest expected value across every (stat × book × side) combination we have for this player today. ' +
    "It's the model's strongest opinion on this player — and what would drive a recommendation if it clears the threshold.",
  'EV Over':
    'Expected value of betting OVER at the listed odds. Positive (green) = +EV; negative = -EV. ' +
    'Computed as P(over) × decimal_payout − (1 − P(over)).',
  'EV Under':
    'Expected value of betting UNDER at the listed odds. Positive (green) = +EV; negative = -EV. ' +
    'Computed as (1 − P(over)) × decimal_payout − P(over).',
  REC:
    'Recommendation. OVER or UNDER when the model has at least the configured edge threshold (default +5% EV) on that side; ' +
    'PASS otherwise. PASS means the model has an opinion, but not a strong enough one to bet at these odds.',
  Rec: 'See REC.',
  'P(Over)':
    'Modelled probability that the player goes OVER the line, after calibration. ' +
    'Combined with the offered odds, this is what produces the EV numbers.',
  Proj:
    'The mean (expected value) of the model\'s distribution for this stat. ' +
    "Useful for spotting big gaps between our number and the book's line, but the EV columns are what you should actually bet on (they account for odds).",
  'Predictions vs actuals':
    "The model's mean prediction for each stat in this player's last 3 completed games, " +
    'computed using only data available BEFORE the game (no leakage). Predictions reflect the ' +
    'CURRENT trained model — re-running them after a retrain will produce different numbers. ' +
    'Color shows error magnitude: green = close, amber = moderate, red = big miss.',

  // ───── Performance / backtest metrics ─────
  MAE:
    'Mean absolute error. Average distance, in stat units, between predicted mean and actual outcome. ' +
    'Lower is better. For points, an MAE of 4.5 means the prediction is on average 4.5 points off.',
  RMSE:
    'Root mean squared error. Like MAE but punishes big misses more heavily. ' +
    'If RMSE is much larger than MAE, predictions occasionally have large errors.',
  'Log loss':
    'Negative log likelihood of the actual over/under outcome under the predicted probability. ' +
    'Lower is better. A model that always predicts 50% gets ~0.693 — anything below that is information.',
  'Baseline MAE':
    "MAE if you just predicted the player's last-10-games mean for the stat. " +
    'The dumbest reasonable predictor — the model has to beat this to add value.',
  'vs baseline':
    'How many stat units of MAE the model improves on the L10 baseline. ' +
    "Positive (green) means the model is better than just averaging the player's recent games.",
  'Sharp book disagreement':
    "The book is shading one side hard (implied probability ≥70%, ~-233 odds or shorter) " +
    'AND our model disagrees on that same side (our probability ≤50%). When a book moves a ' +
    'line this aggressively, they usually have information we lack — lineup news, in-game ' +
    'injury reports, or sharp action. The +EV we show on the opposite side is most likely ' +
    'mirage. Treat these recommendations with extra skepticism and check news before betting.',
};

export default function HelpIcon({ topic }: { topic: string }) {
  const text = HELP_TEXTS[topic];
  if (!text) return null;
  return (
    <span
      className="inline-flex items-center justify-center w-3.5 h-3.5 ml-1 text-[9px] rounded-full border border-slate-300 text-slate-400 cursor-help align-middle hover:border-slate-400 hover:text-slate-600"
      title={text}
      aria-label={`Help: ${topic}`}
    >
      ?
    </span>
  );
}
