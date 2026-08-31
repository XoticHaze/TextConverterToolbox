# H24 time-fixed evidence — research only

This note records evidence after correcting the deep MNQ source timestamp contract. It is not promotion authority.

## Source contract

`mbytes21/MNQ_DATA` `Last.csv` timestamps are empirically verified as America/New_York local one-minute bar-close labels. Research normalization is DST-aware local -> UTC, then minus one minute to UTC bar start.

Focused regression CI verifies:
- winter `2026-01-21 00:01` -> `2026-01-21T05:00Z`;
- summer `2025-06-10 00:01` -> `2025-06-10T04:00Z`.

## Corrected deep vs AXB same-timestamp parity

Receipt: `bb3e0b21330b236bfdbb9a46577c3eb51672b94494488fb2feec823b8fa64531`

Common 2026 rows: 2,311.

- median baseline20 feature correlation: `0.9998544051`
- minimum finite feature correlation: `0.9890917714`
- Ridge score correlation: `0.9987313651`
- Ridge direction agreement: `0.9948074427`
- logistic direction agreement: `0.9935093033`
- future H24 point-move correlation: `0.9980682058`
- future move-sign agreement: `0.9974037213`

Decision: reject the prior source-disagreement interpretation. It was caused by the deep timestamp contract bug.

## Corrected frozen 2026 H24 expected-move receipt

Receipt: `37c84321accd0e3a961b5617a4b6ac8017824cbedc5b5ecf5c8489f7dca82caa`

At a 1 MNQ-point cost, six complete weeks have policy-level phase summaries:

- Ridge expected-move: 4/6 positive weeks; median `+9.3247` points; mean `+11.3666`; p10 `-28.2679`; worst `-47.0804`; median coverage `0.5409`.
- logistic comparator: 3/6 positive weeks; median `+2.4408` points; mean `+14.6390`; p10 `-20.5609`; worst `-21.0321`; median coverage `0.5330`.

Ridge is not universally superior: on the six paired weeks it beats logistic 3/6. Its stronger median/breadth coexists with a worse negative tail. Preserve both as distinct evidence.

## Independent AXB later extension

The corrected parity receipt contains eight AXB-only later weeks after the deep source ends. At a 1-point cost, frozen Ridge is positive in 7/8 weeks with median approximately `+33.47` points. On seven weeks where logistic also has a valid phase summary, Ridge's paired median advantage is approximately `+10.93` points and it wins 4/7.

This extension is a later-time independent-source challenge, not a promotion verdict.

## MAE risk

The corrected-timestamp MAE rerun produced 57 valid policy weeks versus the original reporting guard of 60. The model/policy contract is unchanged. A descriptive receipt is being emitted with the original 60-week gate preserved explicitly; a below-60 result cannot satisfy that original reporting/promotion evidence gate.
