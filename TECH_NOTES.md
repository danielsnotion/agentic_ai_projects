# Part A — City Library Management System — Technical Notes

## Table of contents

- [Part A — Short technical notes](#part-a--short-technical-notes)
- [Design highlights](#design-highlights)
- [Assumptions & extensibility](#assumptions--extensibility)
- [Part B — Retail analysis (summary)](#part-b--retail-analysis-summary)

---

## Part A — Short technical notes

This prototype implements a minimal, easy-to-reason-about city library management system suitable for demos and as a foundation for later extension. The implementation keeps data in-memory using dictionaries keyed by unique IDs for books and members, which yields O(1) CRUD operations and simplifies state management. The borrow/return workflow performs presence and availability validation and updates the book record and the member's borrowed list in a single transactional step to avoid inconsistent state.

A lightweight `borrow_log` offers traceability without enforcing a strict schema; it can be extended to include due dates, fines, or payment events. Search functionality uses case-insensitive substring matching on title and author for predictable, fast lookups. Reporting includes "available by genre", "members with loans", and "most popular genre" (tracked by a Counter incremented on borrow).

The code is intentionally decoupled from storage: switching to file-based or database-backed persistence requires only replacing the storage layer while preserving the service API. The package includes a `__main__` demo that runs a realistic borrow/return flow and prints sample reports suitable for capture as screenshots.

## Design highlights

- Data model: in-memory dicts keyed by `book_id` and `member_id`.
- Workflow: validation → atomic update of book and member state.
- Traceability: append-only `borrow_log` for events and auditing.
- Search: case-insensitive substring match on title/author.
- Reporting: genre availability, active loans, and popularity counters.
- Extensible: storage swap and richer log schema are straightforward.

## Assumptions & extensibility

- Each physical copy uses a unique `book_id`.
- No reservations/waitlists or automated due-date/fine logic in the prototype.
- `availability` is a boolean flag on a copy; add availability count or status for multi-copy support.
- To add fines, due dates, or reservations: extend `borrow_log`, add scheduled tasks or background workers, and add persistence.

---

## Part B — Retail analysis (summary)

This part uses only pandas, numpy, and matplotlib. It performs data cleaning (key fields), parses dates, and derives `Year`, `Month`, `MonthName`, and `DayOfWeek`. Required outputs:

1. `01_transactions_by_city.png`
2. `02_payment_method_distribution.png`
3. `03_monthly_revenue_trend.png`
4. `04_total_revenue_by_season.png`
5. `05_avg_spend_per_season.png`

The script produces concise console insights and action items and saves a small aggregates workbook for downstream review.
