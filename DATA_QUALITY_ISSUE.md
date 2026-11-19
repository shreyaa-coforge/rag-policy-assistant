# Planted Data Quality Issue

## Type

**Outdated duplicate document with conflicting details.**

## Where it is

| File | Version | Effective Date | Status |
|---|---|---|---|
| `expense_reimbursement_policy.md` | 3.0 | March 1, 2025 | Current |
| `expense_reimbursement_policy_2023.md` | 2.1 | January 1, 2023 | Outdated, never removed from the corpus |

Both files share the same title, the same 11 top-level sections (EXP-1 to EXP-11), the same tables, and the same wording. The only differences are the version, the effective date, and the numbers listed below. Nothing inside the 2023 file says it has been superseded. This mirrors a common real-world problem: the old policy stays on the shared drive after a new version is published.

The other two documents (`remote_work_policy.md` and `pto_policy.md`) are current and contain no planted issues.

## Conflicting details

| Section | Topic | Current (v3.0, 2025) | Outdated (v2.1, 2023) |
|---|---|---|---|
| EXP-3.1 | Flights of 6 hours or longer | Premium economy | Economy |
| EXP-3.2 | Lodging, standard cities | $250 per night | $200 per night |
| EXP-3.2 | Lodging, high-cost cities | $325 per night | $275 per night |
| EXP-3.2 | Lodging, international | $300 per night | $250 per night |
| EXP-3.3 | Mileage rate | $0.70 per mile | $0.655 per mile |
| EXP-4.1 | Client meals | $100 per person | $75 per person |
| EXP-4.2 | Daily meal allowance | $75 per day | $50 per day |
| EXP-4.2 | Breakfast / Lunch / Dinner | $15 / $20 / $40 | $10 / $15 / $25 |
| EXP-4.3 | Team meals | $60 per person | $45 per person |
| EXP-5 | Receipt threshold | $40 or more | $25 or more |
| EXP-5 | Missing receipt declarations | 3 per year | 5 per year |
| EXP-6.1 | Submission deadline | 30 days | 60 days |
| EXP-6.1 | Hard cutoff | 90 days | 120 days |
| EXP-6.1 | Payment timing | 10 business days | 15 business days |
| EXP-7 | Manager-only approval limit | Up to $2,500 | Up to $1,500 |
| EXP-7 | Department head tier | $2,500.01 to $10,000 | $1,500.01 to $7,500 |
| EXP-7 | VP of Finance tier | Over $10,000 | Over $7,500 |

## Questions that should surface the issue

| Question | Correct Answer (2025 policy) |
|---|---|
| What is the daily meal allowance when traveling for business? | $75 per day |
| How long do I have to submit an expense report? | 30 days |
| What is the hotel limit per night in New York? | $325 |
| Do I need a receipt for a $30 expense? | No, receipts are required at $40 or more |
| Can I fly premium economy on a 7-hour flight? | Yes |

Because the two versions embed almost identically, retrieval will often return chunks from both files. The LLM then receives two conflicting "facts" and may give the old number, mix the two, or hedge. Retrieval is doing its job, since the chunks really are relevant. The root cause is that the source corpus contains a stale document with no version or status information attached to its chunks.
