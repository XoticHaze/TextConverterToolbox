# MNQ Specialist Model Program

Research-only program contract. This does not authorize runtime, broker, short execution, StrategySpec mutation, or model promotion.

## Objective

Develop independently falsifiable specialist tasks in parallel, then compare multiple challenger model families per task and replay only out-of-sample specialist outputs through an explainable orchestration policy.

Current H24 directional work is one research family, not a prerequisite for starting the other tasks.

## Parallel task packets

### A. Long opportunity
Question: when flat, is initiating/increasing long exposure economically favorable after costs?

Keep the corrected H24/event/session family as one challenger path, not a universal incumbent. Evaluate net expectancy, coverage, quarter/week stability, adverse tails, and time in market. Classification metrics are diagnostic only.

### B. Long HOLD / EXIT
Question: conditional on an already-open hypothetical long, does sufficient remaining edge exist to continue holding?

Create position-path observations from causal historical entry events. Challenger targets should include remaining forward return, additional MFE before MAE, time-to-edge-decay, and HOLD/EXIT classification or expected remaining value. Compare learned policies with fixed horizon, trailing, volatility/ATR, and existing deterministic exit baselines. Hard risk exits remain outside model authority.

### C. Short opportunity
Question: when flat, is initiating short exposure economically favorable after costs?

Train downside opportunity independently rather than assuming symmetry with long labels. Preserve downside-specific volatility, speed, regime, session, and adverse-excursion behavior. Research and historical replay only. Short execution remains disabled unless separately authorized later.

### D. Regime / cross-market context
Question: in what causal market state should each directional specialist participate or abstain?

Use only information available at prediction time. Candidate context includes trend/volatility state, session, cross-market relationships, rates/commodity/equity context when provenance is available, and causal state transitions. Do not select a lucky clock/session from OOS results and promote it directly; learn participation prospectively with past-only fits.

### E. Short HOLD / EXIT
Start when C yields enough causal hypothetical short trajectories. Mirror B's methodology but validate independently because downside trajectories and volatility are asymmetric.

### F. Orchestration replay
Start once at least two specialist tasks have aligned OOS predictions. Replay LONG / HOLD / EXIT / FLAT / SHORT decisions without retraining on the replay period. Begin with deterministic policies and abstention/conflict rules before testing a learned controller.

## Multi-challenger contract

Each task should retain multiple serious challengers rather than one permanent model. At minimum keep a simple interpretable baseline plus materially different nonlinear challenger families where sample size permits. Existing H24 family work already demonstrates this pattern with logistic and tree challengers.

Do not choose a family on a single aggregate score. Preserve paired OOS receipts so AutoTuner/Research Intelligence can determine whether a challenger adds economic value, stability, coverage, or useful disagreement.

## Shared OOS prediction contract

Every specialist artifact should be alignable by:

- timestamp / event identity
- instrument and data/corpus provenance
- task and direction
- horizon / target identity
- model family and model identity
- training cutoff / validation fold
- prediction/action and confidence or expected value when available
- realized evaluation outcome stored separately from training features
- cost assumption
- coverage / abstention state
- regime/context identity when used

Prediction artifacts must remain research evidence. They do not become trading authority by existing.

## Acceptance shape

A specialist earns continued challenger status through economic OOS evidence, not accuracy alone. Relevant gates include positive expectancy after realistic costs, adequate sample/coverage, stability across time/regimes, tail/drawdown behavior, and improvement versus simple task-specific baselines. A model may remain useful as a veto/context specialist even if it is not independently tradable, but that value must be demonstrated with paired OOS replay.

## Authority split

Research execution: create/falsify specialist targets and model challengers in parallel.

AutoTuner / Research Intelligence: consume immutable OOS evidence, compare challengers, retire weak candidates, and evaluate orchestration combinations.

MM-IBKR integration: consume only separately accepted outputs under existing strategy, risk, execution, and operator authority.

Deployment remains serialized even while research is parallel.
