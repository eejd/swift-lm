# Upstream and adopted work (`local` branch)

Primary upstream: `SharpAI/SwiftLM` (GitHub fork parent). Secondary upstreams tracked: none.

`local` is upstream's release tag `b861` (the version of the MacPorts port `swiftlm`) plus the topic branches below. `main` mirrors `SharpAI/SwiftLM` `main` (fast-forward only).

## Adopted from other repos

None.

## Topic branches

| Topic | Upstream PR | Status |
|---|---|---|
| `topic/tool-calls-declared-tools` | none yet | not-submitted (no upstream contact before the quantbench + agentbench soak, ADR-0008); tracked in eejd/swift-lm#7, #8, #9. `git apply --check` of the Sources diff passes against `SharpAI/SwiftLM` `main` (`ee816ed`); it was not built there |

`topic/tool-calls-declared-tools`: every generation call of the chat endpoint handed the tool-call parser no tools (`tools: nil`), and the raw-completion endpoint the same. For the parser `nil` means "no schema to check against": it accepts any function name and reads every argument as text. The request's `tools` are now passed on every path, `[]` when the request has none, and `/v1/completions` passes `[]`. This fixes JSON answers with a `name` key returned as `tool_calls` (#9) and integer/boolean/array/object arguments returned as strings (#7: Qwen3.5-4B). It needs two changes in `mlx-swift-lm` (`eejd/mlx-swift-lm`: `topic/declared-tools-authoritative`, for #9, and `topic/minicpm5-tool-call-format`, for #8): the server change alone types arguments and stops extraction for formats whose parser is already tool-aware, not for the Llama 3 JSON parser and not for MiniCPM5. **Merge and port order:** this topic is only safe together with the library topics. Alone, with `mlx-swift-lm` still at `SharpAI@4d76a2e`, `tools: []` makes the old Llama 3 parser consume an unmarked `{"name": ...}` answer and reject it as undeclared (content lost instead of a wrong `tool_calls`). Merge `eejd/mlx-swift-lm` #1 and #2 first and bump the port's `mlx-swift-lm` distfile in the same port change as this one. The `mlx-swift-lm` gitlink of this branch can only be bumped to a commit of `eejd/mlx-swift-lm` that exists after those merges, so it is unchanged here. **Behavior change:** because the request's tools are now passed, a model answer that is a call to an undeclared function is rejected by the library's fail-closed rule and only logged: the client gets HTTP 200, empty content, no `tool_calls`, `finish_reason: stop` (measured: Hermes-3-Llama-3.2-3B `Jane Doe`, Llama-3.2-1B `2+2`; before, a wrong `tool_calls` entry). Surfacing rejections to the client: eejd/swift-lm#12. Not covered by this topic: the `--dflash` path (`Server.swift` DFlash branch, no tool parsing) and `MLXInferenceCore.InferenceEngine` (never passes tools or parses calls). The pinned `mlx-swift-lm` submodule (`SharpAI/mlx-swift-lm@4d76a2e`) is unchanged on this branch; the port takes `mlx-swift-lm` from a distfile.

## Origin

Developed 2026-10-22 from the T0 engine runs of eejd/py-quantbench (#217, #287) against SwiftLM b861.
