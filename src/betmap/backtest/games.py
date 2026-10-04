"""Walk-forward backtest of the game model against historical closing lines.

For each week, the model is fit only on games that kicked off before that week, then
scored against nflverse closing lines. Bets are simulated at the closing price, which
is the hardest number to beat; an edge here is meaningful, a loss here is the default.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean

from betmap.models.game_model import Game, GameModel, ModelConfig, side_prob
from betmap.odds.math import american_to_decimal, devig, expected_value


@dataclass(frozen=True)
class ClosingLines:
    """nflverse conventions: spread_line > 0 means the home team is favored by that much."""

    spread_line: float
    home_spread_odds: float
    away_spread_odds: float
    total_line: float
    over_odds: float
    under_odds: float
    home_moneyline: float | None
    away_moneyline: float | None


@dataclass
class MarketResult:
    bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    profit: float = 0.0  # flat 1-unit stakes
    claimed_ev: list[float] = field(default_factory=list)

    @property
    def roi(self) -> float | None:
        return self.profit / self.bets if self.bets else None

    def record(self, ev: float, outcome: int, decimal: float) -> None:
        """outcome: 1 win, 0 push, -1 loss."""
        self.bets += 1
        self.claimed_ev.append(ev)
        if outcome > 0:
            self.wins += 1
            self.profit += decimal - 1
        elif outcome < 0:
            self.losses += 1
            self.profit -= 1
        else:
            self.pushes += 1


@dataclass
class BacktestReport:
    games: int = 0
    margin_mae_model: float = 0.0
    margin_mae_market: float = 0.0
    total_mae_model: float = 0.0
    total_mae_market: float = 0.0
    brier_model: float | None = None  # home win probability
    brier_market: float | None = None
    markets: dict[str, MarketResult] = field(default_factory=lambda: defaultdict(MarketResult))
    by_season: dict[int, dict[str, MarketResult]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(MarketResult))
    )


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def backtest(
    history: list[tuple[Game, int, int, ClosingLines | None]],
    test_seasons: list[int],
    config: ModelConfig | None = None,
    min_ev: float = 0.02,
    train_days: float = 4 * 365,
    model_weight: float = 1.0,
) -> BacktestReport:
    """`history` is (game, season, week, closing lines) for every game, test seasons included.

    Bets use `model_weight` * model + (1 - model_weight) * devigged closing price.
    """
    config = config or ModelConfig()

    def bet_two_way(market, p_model, odds_a, odds_b, outcome_a):
        """Side A wins when outcome_a > 0; side B is the other side."""
        p_market = devig([odds_a, odds_b])[0]
        p = model_weight * p_model + (1 - model_weight) * p_market
        _consider(report, season, market, p, odds_a, outcome_a, min_ev)
        _consider(report, season, market, 1 - p, odds_b, -outcome_a, min_ev)

    report = BacktestReport()
    margin_err_model, margin_err_market, total_err_model, total_err_market = [], [], [], []
    brier_model, brier_market = [], []

    weeks: dict[tuple[int, int], list] = defaultdict(list)
    for item in history:
        game, season, week, lines = item
        if season in test_seasons and game.is_final and lines is not None:
            weeks[(season, week)].append(item)
    games = [g for g, *_ in history]

    for (season, week), slate in sorted(weeks.items()):
        as_of = min(g.kickoff for g, *_ in slate)
        window = [
            g
            for g in games
            if g.is_final and g.kickoff < as_of and (as_of - g.kickoff).days <= train_days
        ]
        if not window:
            continue  # nothing to learn from yet (start of the available history)
        model = GameModel(config).fit(window, as_of)
        for game, _, _, lines in slate:
            pred = model.predict(game.home, game.away, game.neutral)
            margin = game.home_score - game.away_score
            total = game.home_score + game.away_score
            report.games += 1
            margin_err_model.append(abs(margin - pred.margin))
            margin_err_market.append(abs(margin - lines.spread_line))
            total_err_model.append(abs(total - pred.total))
            total_err_market.append(abs(total - lines.total_line))

            if lines.home_moneyline and lines.away_moneyline and margin != 0:
                home_ml = american_to_decimal(lines.home_moneyline)
                away_ml = american_to_decimal(lines.away_moneyline)
                p_model = pred.home_win()
                p_market = devig([home_ml, away_ml])[0]
                won = 1.0 if margin > 0 else 0.0
                brier_model.append((p_model - won) ** 2)
                brier_market.append((p_market - won) ** 2)
                bet_two_way("h2h", p_model, home_ml, away_ml, _sign(margin))

            # Spread: home covers when margin > spread_line.
            bet_two_way(
                "spreads",
                side_prob(pred.margin, pred.margin_sd, lines.spread_line),
                american_to_decimal(lines.home_spread_odds),
                american_to_decimal(lines.away_spread_odds),
                _sign(margin - lines.spread_line),
            )
            bet_two_way(
                "totals",
                pred.over(lines.total_line),
                american_to_decimal(lines.over_odds),
                american_to_decimal(lines.under_odds),
                _sign(total - lines.total_line),
            )

    if report.games:
        report.margin_mae_model = fmean(margin_err_model)
        report.margin_mae_market = fmean(margin_err_market)
        report.total_mae_model = fmean(total_err_model)
        report.total_mae_market = fmean(total_err_market)
    if brier_model:
        report.brier_model = fmean(brier_model)
        report.brier_market = fmean(brier_market)
    return report


def _consider(
    report: BacktestReport,
    season: int,
    market: str,
    prob: float,
    decimal: float,
    outcome: int,
    min_ev: float,
) -> None:
    ev = expected_value(prob, decimal)
    if ev >= min_ev:
        report.markets[market].record(ev, outcome, decimal)
        report.by_season[season][market].record(ev, outcome, decimal)
