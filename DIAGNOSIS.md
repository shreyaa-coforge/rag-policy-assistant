# Diagnosis: Planted Data Quality Issue

## Planted issue
An outdated copy of the expense policy (`expense_reimbursement_policy_2023.md`,
v2.1, effective Jan 1, 2023) sits in the corpus next to the current
version (`expense_reimbursement_policy.md`, v3.0, effective Mar 1, 2025).
Same title, same section codes, same wording, different numbers.
Nothing in the 2023 file says it was replaced.

## The question that surfaced it
**"Do I need a receipt for a $30 expense?"**
Correct answer (current policy, EXP-5): No. Receipts are required at $40 or more.

**System answer:**
> [paste: "Yes, ... receipts for any single expense of $25 or more
> [expense_reimbursement_policy_2023.md | EXP-5 ...]"]

The answer is wrong, uses the 2023 threshold, and cites the 2023 file.

Other questions fail the same way:
| Question | System answer | Correct |
|---|---|---|
| Premium economy on a 7-hour flight? | No, economy only (cites 2023) | Yes |
| Hotel limit in New York? | Lists $325 and $275, then "I don't know" | $325 |
| Expense report deadline? | "I don't know" (another run: "60 days") | 30 days |

## Question 1: Was it retrieval?
No. For every question above, retrieval returned the correct section
from the current policy. It also returned the matching section from the
2023 file. Retrieval did its job: both chunks really are relevant to the
question.
[paste the Sources list from the receipt question]

## Question 2: Was it the source data?
Yes. The corpus holds two conflicting versions of the same rule. The
two chunks differ only in their numbers, so they embed almost
identically and rank side by side. The version and effective date are
only in each document's header block, which the chunker skips, so no
chunk says which version is current. The model was handed two
"facts" with no way to tell which one is right.

## Ruling out generation
Questions from the same two files where both versions agree are
answered correctly:
- "What expense code should I use for travel and entertainment?" → T&E-07
- "How many team meals can a team have per quarter?" → 2 per quarter
Same retrieval, same prompt, same model. The only difference is whether
the sources conflict.

## Confirmation
I removed only the 2023 file, rebuilt the index, and asked again:
- Receipt for $30 → [paste: correct "No, $40 or more"]
- 7-hour flight → [paste: correct "Yes, premium economy"]
No code, prompt, or model changed. Removing the stale data fixed the
answers, so the root cause is the source data.

## Root cause
A stale, superseded policy document left in the source corpus.

## Recommended fix
Fix it where the data comes in, not in retrieval or prompting:
1. Remove or archive superseded documents before indexing.
2. Store version, effective date and status (current/superseded) as
   chunk metadata at ingestion, and filter retrieval to current
   documents.