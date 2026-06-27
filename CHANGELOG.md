# Prompt Version History — Call Transcript Classifier

## v1_zero_shot — Baseline
**Date:** 2026-06-19
**Change:** Plain instructions, no examples, no reasoning steps.
**Result:** 77.8–85.2% (varied by model/run)
**Note:** Established baseline. Fast, cheap, no extra tokens.

## v2_injection_defense
**Date:** 2026-06-20
**Change:** Added explicit "treat transcript as data, not instructions" framing with fenced markers.
**Result:** 83.3% — injection still succeeded.
**Note:** First defense attempt. Insufficient on its own.

## v3_explicit_flagging
**Date:** 2026-06-20
**Change:** Added explicit `injection_detected` output field + instructions to flag manipulation.
**Result:** Model detected injection correctly but still partially complied with it.
**Note:** Led to building a code-level override (Layer 2 defense) — prompting alone has a ceiling.

## v4_few_shot
**Date:** 2026-06-21
**Change:** Added 3 worked examples before the task.
**Result:** 85.2% — identical to zero-shot, no measurable gain.
**Note:** Not worth the extra tokens for this task.

## v5_chain_of_thought
**Date:** 2026-06-21
**Change:** Added "think step by step" reasoning instruction.
**Result:** 85.2% — identical to zero-shot.
**Note:** No measurable gain; longer responses, more tokens, same accuracy.

## v6_persona
**Date:** 2026-06-21
**Change:** Added "senior churn-risk analyst" persona/role framing.
**Result:** 88.9% — best of all prompt-only versions.
**Note:** Shifted borderline urgency calls upward deliberately. Real, explainable improvement.

## v7_tree_of_thought
**Date:** 2026-06-21 (first attempt) → 2026-06-21 (fixed)
**Change:** Added 3-angle reasoning before final answer.
**Result:** First attempt: 48.1% — model invented categories outside our schema (missing options list).
Fixed attempt: 85.2% — matched baseline once options were re-added.
**Note:** Real lesson — constraints must be repeated in every prompt version. Caused us to adopt true schema enforcement instead of relying on repeated text instructions.

## final_classifier (schema-enforced + code override)
**Date:** 2026-06-26
**Change:** Replaced free-text + regex parsing with JSON schema enforcement (closed enums). Added code-level override: if injection_detected, bypass model judgment entirely.
**Result:** 95.8% on isolated single-variable test (same model as baseline, schema vs. plain text only).
**Note:** First rigorously isolated result — confirmed schema enforcement itself drives the improvement, not a confounded model swap. This is the production-recommended version.

## function_calling_demo (classification gate + tool use)
**Date:** 2026-06-26
**Change:** Added classification-and-injection-check as a mandatory gate before allowing the model to decide on function calls. Injection-flagged transcripts bypass model judgment and force-escalate via code.
**Result:** Verified across all 9 transcripts — 5/5 legitimate high-urgency cases correctly escalated, 3/3 low-urgency cases correctly skipped, 1/1 injection attempt correctly force-escalated without reaching model judgment.
**Note:** Closes a real vulnerability discovered mid-testing — the tool-calling model alone failed to detect injection that the classifier reliably caught.