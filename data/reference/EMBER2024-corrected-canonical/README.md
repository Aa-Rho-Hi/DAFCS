# EMBER2024 Corrected Canonical Eval Split

This directory is the canonical local corrected evaluation bundle derived from
the currently hosted EMBER2024 raw test and challenge shards.

It contains:

- `2024-09-22_2024-12-14_test.jsonl`
- `2023-09-24_2024-12-14_challenge_malicious.jsonl`
- `X_test.dat`
- `y_test.dat`
- `X_challenge.dat`
- `y_challenge.dat`

Important notes:

- The currently hosted official raw `*_test.zip` archives redownload cleanly to
  `1,212,000` raw rows, not the `606,000` rows stated in the README/paper.
- Global deduplication by `sha256` yields `605,929` unique corrected test rows.
- The challenge set redownloads cleanly at `6,315` rows with no duplicates.
- This bundle is therefore the canonical corrected local evaluation set used for
  all local official-model reruns in this workspace.

Corrected row counts:

- Test: `605,929`
- Challenge: `6,315`

Corrected test file-type counts:

- `Win32`: `359,994`
- `Win64`: `119,993`
- `Dot_Net`: `59,953`
- `APK`: `48,000`
- `PDF`: `12,000`
- `ELF`: `5,989`

Challenge file-type counts:

- `Win32`: `3,225`
- `Win64`: `814`
- `Dot_Net`: `829`
- `APK`: `256`
- `PDF`: `805`
- `ELF`: `386`
