# MNQ H24 fixed-clock decomposition interpretation

## Evidence boundary

This note consumes the successful hosted Actions run `33378807316` for PR #17 at research head `f8eb2c6f6374a852e1aa3c6e9628f9e907f7dd50`.

Immutable artifacts:

- `mnq-h24-clock-h24_vol05`, artifact `9753242729`, digest `sha256:67d0562ea755ef58c2a1c526f144eb2652cc68db344458e44c04722669c194c8`
- `mnq-h24-clock-h24_vol10`, artifact `9753341695`, digest `sha256:357ac55ec5c6689200bf6c1d63c713c60286a620b05f1020af9db9658fc681ba`

Both matrix jobs completed `MNQ_H24_CLOCK_DECOMPOSITION=PASS` against the exact pinned deep MNQ source. This is explanatory research evidence only. It does not authorize model selection, StrategySpec/runtime changes, broker actions, or promotion.

## What survived the clock split

The earlier aggregate-positive H24 phases are not broad all-day effects. Most high-count fixed-clock buckets are flat-to-negative after 2 bp, especially in the stricter 1.0x target.

The repeatable positive concentration is around two neighboring fixed UTC events:

| Target | Family | 16:48Z mean net | Signals | t-stat | 18:00Z mean net | Signals | t-stat |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H24 0.5x | baseline20 | +0.2723% | 21 | 1.25 | +0.0227% | 42 | 0.22 |
| H24 0.5x | expanded+regime | +0.1615% | 54 | 1.74 | +0.1179% | 113 | 2.19 |
| H24 1.0x | baseline20 | +0.2027% | 15 | 0.85 | +0.1088% | 17 | 0.60 |
| H24 1.0x | expanded+regime | +0.3211% | 38 | 2.39 | +0.1638% | 56 | 1.70 |

The strongest single cell is H24 1.0x expanded+regime at 16:48Z: 38 signals, +0.3211% mean net after 2 bp, 63.2% positive rate, t=2.39, with 8 of 13 observed quarters positive. The broader neighboring 18:00Z cell is also positive for expanded+regime in both targets, with substantially more samples.

## Falsifiers and cautions

This is not yet a clock-routing rule.

1. Sample counts at 16:48Z are much lower than the daytime buckets, so its attractive magnitude is vulnerable to event concentration.
2. The 18:00Z result is more populated but less uniformly strong across baseline20 and expanded+regime, especially at H24 0.5x baseline20.
3. Several high-count daytime buckets are negative, which argues against treating the original phase result as a general phase-wide edge.
4. Multiple fixed clocks were inspected. The next test must therefore be predeclared rather than selecting another clock after seeing these results.

## Next causal test

The smallest useful continuation is a predeclared **16:48Z / 18:00Z session-neighborhood falsification**, not another broad clock scan.

Hold the two clock events fixed and test whether the effect survives across independent calendar partitions and causal market-state strata without optimizing new thresholds. At minimum report:

- pre-2025 versus 2025+ separately;
- quarterly sign consistency and signal counts;
- high/normal/low realized-volatility strata defined from information available before the decision bar;
- expanded+regime versus baseline20 on identical rows;
- 2 bp and a stricter cost sensitivity;
- no promotion unless the sign survives outside the period carrying the largest contribution.

If the neighboring 16:48Z and 18:00Z effect fails outside its dominant calendar/state partition, reject clock specialization and return to broader causal state features. If it survives with adequate support, the next candidate should be a session/state specialist whose clock inputs are fixed by this evidence rather than searched again.
