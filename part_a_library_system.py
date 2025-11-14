#!/usr/bin/env python3
"""
part_a_library_system.py
"""

from __future__ import annotations
import datetime
import logging
from typing import List, Dict, Optional, Tuple
import pathlib

import pandas as pd

# Configuration
DEFAULT_LOAN_DAYS = 14

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("LibrarySystem")


class LibrarySystem:
    """
    LibrarySystem manages books, members and borrow/return logs in-memory backed by CSV files.

    It provides methods to load and persist data, perform borrow/return operations,
    query inventory and generate simple reports. Instances keep in-memory pandas
    DataFrames and a derived borrowed-map for quick lookups.
    """

    def __init__(self,
                 books_csv: str = "book_records.csv",
                 members_csv: str = "member_records.csv",
                 borrow_log_csv: str = "borrow_log.csv",
                 loan_days: int = DEFAULT_LOAN_DAYS):
        """
        Initialize the LibrarySystem.

        Args:
            books_csv: path to books CSV file.
            members_csv: path to members CSV file.
            borrow_log_csv: path to borrow/return log CSV file.
            loan_days: default number of days for a loan when borrowing.
        """
        # Resolve the `requirements` folder relative to this module's location so
        # CSV lookups work even if the process CWD is different.
        base_dir = pathlib.Path(__file__).resolve().parent
        requirements_dir = base_dir / "requirements"
        # Ensure the directory exists so reads/writes won't fail when saving
        requirements_dir.mkdir(parents=True, exist_ok=True)

        self.books_csv = requirements_dir / books_csv
        self.members_csv = requirements_dir / members_csv
        self.borrow_log_csv = requirements_dir / borrow_log_csv

        self.loan_days = int(loan_days)

        # DataFrames will be used as primary in-memory data structures
        self.books_df = pd.DataFrame()  # columns expected: Book ID, Title, Author, Genre, Availability (Available/Issued)
        self.members_df = pd.DataFrame()  # columns expected: Member ID, Name, Age, Contact Info
        self.borrow_log_df = pd.DataFrame(columns=["timestamp", "member_id", "book_id", "action", "due_date"])

        self._load_books()
        self._load_members()
        self._load_borrow_log()
        # Recompute current borrowed state if needed
        self._recompute_current_borrowed_map()

    # ---------------- Loading ----------------
    def _load_books(self) -> None:
        """
        Load books from the configured CSV into `self.books_df`.

        If the CSV does not exist an empty DataFrame with expected columns is created.
        Also normalizes an "available" boolean column for runtime operations.
        """
        if not self.books_csv.exists():
            logger.warning("Books CSV not found: %s (starting empty)", self.books_csv)
            # create empty with expected columns
            self.books_df = pd.DataFrame(columns=["Book ID", "Title", "Author", "Genre", "Availability"])
            return
        self.books_df = pd.read_csv(self.books_csv, dtype=str).fillna("")
        # Normalize Availability into boolean column 'available' for easier operations
        self.books_df["Availability"] = self.books_df["Availability"].astype(str)
        self.books_df["available"] = self.books_df["Availability"].str.lower().isin(
            ["available", "true", "1", "yes", "y"])
        logger.info("Loaded %d books", len(self.books_df))

    def _load_members(self) -> None:
        """
        Load members from the configured CSV into `self.members_df`.

        If the CSV is missing an empty DataFrame with expected columns is created.
        Attempts to coerce the Age column to numeric when present.
        """
        if not self.members_csv.exists():
            logger.warning("Members CSV not found: %s (starting empty)", self.members_csv)
            self.members_df = pd.DataFrame(columns=["Member ID", "Name", "Age", "Contact Info"])
            return
        self.members_df = pd.read_csv(self.members_csv, dtype=str).fillna("")
        # Keep Age as numeric where possible
        if "Age" in self.members_df.columns:
            try:
                self.members_df["Age"] = pd.to_numeric(self.members_df["Age"], errors="coerce")
            except Exception:
                pass
        logger.info("Loaded %d members", len(self.members_df))

    def _load_borrow_log(self) -> None:
        """
        Load the borrow/return log from CSV into `self.borrow_log_df`.

        If the CSV does not exist an empty log with expected columns is initialized.
        Ensures the log has consistent columns for downstream processing.
        """
        if not self.borrow_log_csv.exists():
            # start with empty borrow_log_df
            self.borrow_log_df = pd.DataFrame(columns=["timestamp", "member_id", "book_id", "action", "due_date"])
            return
        self.borrow_log_df = pd.read_csv(self.borrow_log_csv, dtype=str).fillna("")
        # Ensure consistent columns
        for col in ["timestamp", "member_id", "book_id", "action", "due_date"]:
            if col not in self.borrow_log_df.columns:
                self.borrow_log_df[col] = ""
        logger.info("Loaded %d borrow-log entries", len(self.borrow_log_df))

    # ---------------- Persisting ----------------
    def save_state(self, save_books: bool = True, save_members: bool = True) -> None:
        """
        Persist current in-memory data to CSV files.

        Args:
            save_books: whether to write books CSV.
            save_members: whether to write members CSV.
        """
        if save_books:
            self._save_books_csv()
        if save_members:
            self._save_members_csv()
        self._save_borrow_log_csv()

    def _save_books_csv(self) -> None:
        """
        Write the books DataFrame to the configured CSV file.

        Maps the internal boolean availability to a human-friendly "Availability" column.
        """
        out_df = self.books_df.copy()
        out_df["Availability"] = out_df["available"].map({True: "Available", False: "Issued"})
        # Ensure columns order
        cols = ["Book ID", "Title", "Author", "Genre", "Availability"]
        for c in cols:
            if c not in out_df.columns:
                out_df[c] = ""
        out_df.to_csv(self.books_csv, index=False, columns=cols)
        logger.info("Saved %d books to %s", len(out_df), self.books_csv)

    def _save_members_csv(self) -> None:
        """
        Write the members DataFrame to the configured CSV file.

        Ensures expected columns are present before writing.
        """
        out_df = self.members_df.copy()
        cols = ["Member ID", "Name", "Age", "Contact Info"]
        for c in cols:
            if c not in out_df.columns:
                out_df[c] = ""
        out_df.to_csv(self.members_csv, index=False, columns=cols)
        logger.info("Saved %d members to %s", len(out_df), self.members_csv)

    def _save_borrow_log_csv(self) -> None:
        """
        Write the borrow/return log DataFrame to the configured CSV file.

        Ensures the log contains the expected columns before writing.
        """
        out_df = self.borrow_log_df.copy()
        cols = ["timestamp", "member_id", "book_id", "action", "due_date"]
        for c in cols:
            if c not in out_df.columns:
                out_df[c] = ""
        out_df.to_csv(self.borrow_log_csv, index=False, columns=cols)
        logger.info("Saved %d borrow-log records to %s", len(out_df), self.borrow_log_csv)

    # -------------- Internal helpers ----------------
    def _recompute_current_borrowed_map(self) -> None:
        """
        Recompute mapping of currently-borrowed books per member from borrow_log_df.

        Will be used to answer members_with_borrowed_books and to validate returns.
        """
        # Start from empty set for each member
        # We'll process log in chronological order; if timestamp missing we'll use order present
        df = self.borrow_log_df.copy()
        # Ensure ordering: if timestamp column parseable, sort; else keep input order
        try:
            df["__ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values(by="__ts").drop(columns="__ts")
        except Exception:
            pass

        # Build a dict member_id -> list of currently borrowed book_ids
        borrowed_map: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            member_id = str(row.get("member_id", "")).strip()
            book_id = str(row.get("book_id", "")).strip()
            action = str(row.get("action", "")).strip().lower()
            if not member_id or not book_id:
                continue
            lst = borrowed_map.setdefault(member_id, [])
            if action == "borrow":
                if book_id not in lst:
                    lst.append(book_id)
            elif action == "return":
                if book_id in lst:
                    lst.remove(book_id)
        # store as attribute for quick reference
        self._borrowed_map = borrowed_map
        # Reconcile per-book availability based on the computed borrowed_map.
        # Any book present in borrowed_map values is marked unavailable (False); do not
        # clobber existing availability flags loaded from CSV when a borrow-log entry
        # is not present. That preserves 'Issued' states that came from the CSV.
        try:
            # Ensure column exists; if missing, assume available True by default
            if "available" not in self.books_df.columns:
                self.books_df["available"] = True
            else:
                # fill any missing values with True (do not overwrite existing booleans)
                self.books_df["available"] = self.books_df["available"].fillna(True)

            # mark borrowed ones as False
            borrowed_ids = {bid for bids in borrowed_map.values() for bid in bids}
            if borrowed_ids:
                mask = self.books_df["Book ID"].isin(list(borrowed_ids))
                self.books_df.loc[mask, "available"] = False
        except Exception:
            # best-effort: do not raise if something unexpected in DataFrame
            pass

    def _ensure_book_exists(self, book_id: str) -> bool:
        """
        Check whether a book with `book_id` exists in the books DataFrame.

        Returns True if found, False otherwise.
        """
        return ((self.books_df["Book ID"] == book_id).any())

    def _ensure_member_exists(self, member_id: str) -> bool:
        """
        Check whether a member with `member_id` exists in the members DataFrame.

        Returns True if found, False otherwise.
        """
        return ((self.members_df["Member ID"] == member_id).any())

    # ---------------- Core operations ----------------
    def add_book(self, book_id: str, title: str, author: str, genre: str, availability: bool = True) -> bool:
        """
        Add a new book to the system.

        Returns True on success, False if a book with the same ID already exists.
        """
        if self._ensure_book_exists(book_id):
            logger.debug("Attempt to add existing book: %s", book_id)
            return False
        new_row = {"Book ID": book_id, "Title": title, "Author": author, "Genre": genre,
                   "Availability": "Available" if availability else "Issued", "available": availability}
        self.books_df = pd.concat([self.books_df, pd.DataFrame([new_row])], ignore_index=True)
        logger.info("Added book %s", book_id)
        return True

    def register_member(self, member_id: str, name: str, age: Optional[int] = None,
                        contact: Optional[str] = None) -> bool:
        """
        Register a new library member.

        Returns True on success, False if the member ID already exists.
        """
        if self._ensure_member_exists(member_id):
            logger.debug("Attempt to register existing member: %s", member_id)
            return False
        new_row = {"Member ID": member_id, "Name": name, "Age": age if age is not None else "",
                   "Contact Info": contact or ""}
        self.members_df = pd.concat([self.members_df, pd.DataFrame([new_row])], ignore_index=True)
        logger.info("Registered member %s", member_id)
        return True

    def update_book_availability(self, book_id: str, available: bool) -> bool:
        """
        Set the internal availability flag for a book.

        Returns True if the book exists and was updated, False otherwise.
        """
        mask = self.books_df["Book ID"] == book_id
        if not mask.any():
            logger.warning("Book not found: %s", book_id)
            return False
        self.books_df.loc[mask, "available"] = bool(available)
        logger.info("Updated availability for %s -> %s", book_id, "Available" if available else "Issued")
        return True

    def borrow_book(self, member_id: str, book_id: str, loan_days: Optional[int] = None) -> Tuple[bool, str]:
        """
        Borrow a book for a member.

        Checks member/book existence and availability, updates state and appends a borrow log entry.
        Returns (success, message) where message is human-readable.
        """
        if not self._ensure_member_exists(member_id):
            return False, f"Member not found: {member_id}"
        if not self._ensure_book_exists(book_id):
            return False, f"Book not found: {book_id}"
        # check availability
        book_row = self.books_df.loc[self.books_df["Book ID"] == book_id]
        if not bool(book_row.iloc[0]["available"]):
            return False, f"Book '{book_row.iloc[0]['Title']}' ({book_id}) is already issued."

        # update book availability
        self.books_df.loc[self.books_df["Book ID"] == book_id, "available"] = False

        # compute due_date
        now = datetime.datetime.now(datetime.timezone.utc)
        ld = int(loan_days) if (loan_days is not None) else self.loan_days
        due_date = (now + datetime.timedelta(days=ld)).date().isoformat()

        # append borrow log row
        new_log = {"timestamp": now.isoformat(), "member_id": member_id, "book_id": book_id, "action": "borrow",
                   "due_date": due_date}
        self.borrow_log_df = pd.concat([self.borrow_log_df, pd.DataFrame([new_log])], ignore_index=True)

        # recompute borrowed map
        self._recompute_current_borrowed_map()

        title = book_row.iloc[0]["Title"]
        logger.info("Borrowed %s to %s until %s", book_id, member_id, due_date)
        return True, f"Book '{title}' issued to {member_id}. Due on {due_date}."

    def return_book(self, member_id: str, book_id: str) -> Tuple[bool, str]:
        """
        Process a book return from a member.

        Validates that the member currently has the book, updates availability and appends a return log entry.
        Returns (success, message).
        """
        if not self._ensure_member_exists(member_id):
            return False, f"Member not found: {member_id}"
        if not self._ensure_book_exists(book_id):
            return False, f"Book not found: {book_id}"
        # check if member currently has this book borrowed (from borrowed_map)
        borrowed = self._borrowed_map.get(member_id, [])
        if book_id not in borrowed:
            return False, f"Member {member_id} does not have book {book_id} borrowed."

        # set availability True
        self.books_df.loc[self.books_df["Book ID"] == book_id, "available"] = True

        # append return log row
        now = datetime.datetime.now(datetime.timezone.utc)
        new_log = {"timestamp": now.isoformat(), "member_id": member_id, "book_id": book_id, "action": "return",
                   "due_date": ""}
        self.borrow_log_df = pd.concat([self.borrow_log_df, pd.DataFrame([new_log])], ignore_index=True)

        # recompute borrowed map
        self._recompute_current_borrowed_map()

        title = self.books_df.loc[self.books_df["Book ID"] == book_id, "Title"].iloc[0]
        logger.info("Book %s returned by %s", book_id, member_id)
        return True, f"Book '{title}' returned by {member_id}."

    # ---------------- Reports / Queries ----------------
    def search_books(self, query: str) -> List[Dict]:
        """
        Search books by title or author using a case-insensitive substring match.

        Returns a list of matching book records as dictionaries.
        """
        q = (query or "").strip()
        if q == "":
            return []
        mask_title = self.books_df["Title"].astype(str).str.contains(q, case=False, na=False)
        mask_author = self.books_df["Author"].astype(str).str.contains(q, case=False, na=False)
        res = self.books_df.loc[mask_title | mask_author]
        return res.to_dict(orient="records")

    def available_books_by_genre(self, genre: str) -> List[Dict]:
        """
        Return available books filtered by exact-compare genre (case-insensitive).

        Returns a list of book dicts matching the requested genre and availability.
        """
        g = (genre or "").strip()
        if g == "":
            return []
        mask = (self.books_df["available"] == True) & (
                    self.books_df["Genre"].astype(str).str.strip().str.lower() == g.lower())
        res = self.books_df.loc[mask]
        return res.to_dict(orient="records")

    def members_with_borrowed_books(self) -> List[Dict]:
        """
        Return a list of members who currently have one or more borrowed books.

        Each entry contains the member ID, name (if available) and the list of borrowed book IDs.
        """
        # Recompute the borrowed map from the borrow log to ensure up-to-date results
        # This avoids stale state if borrow_log_df has changed or prior operations didn't refresh.
        self._recompute_current_borrowed_map()

        members_list = []
        for member_id, borrowed in self._borrowed_map.items():
            if not borrowed:
                continue
            row = self.members_df.loc[self.members_df["Member ID"] == member_id]
            if row.empty:
                name = member_id
            else:
                name = row.iloc[0].get("Name", member_id)
            members_list.append({"Member ID": member_id, "Name": name, "BorrowedBooks": borrowed})
        return members_list

    def most_popular_genre(self) -> Optional[str]:
        """
        Compute the most frequently-borrowed genre from borrow history.

        Returns the genre string or None if there is insufficient data.
        """
        borrows = self.borrow_log_df[self.borrow_log_df["action"].astype(str).str.lower() == "borrow"]
        if borrows.empty:
            return None
        # Merge borrows with books_df to get genre
        merged = borrows.merge(self.books_df[["Book ID", "Genre"]], left_on="book_id", right_on="Book ID", how="left")
        merged["Genre"] = merged["Genre"].fillna("").astype(str).str.strip()
        counts = merged.groupby("Genre").size().reset_index(name="count")
        counts = counts[counts["Genre"] != ""]
        if counts.empty:
            return None
        top = counts.sort_values("count", ascending=False).iloc[0]
        return top["Genre"]

    # ---------------- Utilities ----------------
    def get_book(self, book_id: str) -> Optional[Dict]:
        """
        Retrieve a single book record by ID.

        Returns a dict of book fields or None if not found.
        """
        row = self.books_df.loc[self.books_df["Book ID"] == book_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_member(self, member_id: str) -> Optional[Dict]:
        """
        Retrieve a single member record by ID.

        Returns a dict of member fields or None if not found.
        """
        row = self.members_df.loc[self.members_df["Member ID"] == member_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def export_report_books(self) -> pd.DataFrame:
        """
        Produce a DataFrame suitable for reporting the books inventory.

        The returned DataFrame contains human-friendly Availability values.
        """
        out = self.books_df.copy()
        out["Availability"] = out["available"].map({True: "Available", False: "Issued"})
        return out[["Book ID", "Title", "Author", "Genre", "Availability"]]

    def export_report_members(self) -> pd.DataFrame:
        """
        Build a DataFrame summarizing members and their current borrowed books.

        Returns columns: Member ID, Name, BorrowedCount, BorrowedBooks (comma separated).
        """
        # Build DataFrame of members with their borrowed counts
        rows = []
        for _, r in self.members_df.iterrows():
            mid = r["Member ID"]
            borrowed = self._borrowed_map.get(mid, [])
            rows.append({
                "Member ID": mid,
                "Name": r.get("Name", ""),
                "BorrowedCount": len(borrowed),
                "BorrowedBooks": ",".join(borrowed)
            })
        return pd.DataFrame(rows)


# ---------------- CLI (keeps same behaviour) ----------------
def input_prompt(prompt: str) -> str:
    """
    Wrapper around built-in input() that returns a stripped string and handles interrupts.

    Returns an empty string on EOF/KeyboardInterrupt.
    """
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def print_menu():
    """
    Print the simple interactive CLI menu to stdout.

    This function only prints available options and does not return a value.
    """
    print("\n--- City Library Management (CLI) ---")
    print("1. List all books (sample)")
    print("2. Search book by title/author")
    print("3. Show only available books")
    print("4. Show available books by genre")
    print("5. Register member")
    print("6. Add book")
    print("7. Borrow book")
    print("8. Return book")
    print("9. Show members with borrowed books")
    print("10. Show most popular genre")
    print("11. Save state")
    print("0. Exit")


def cli_loop(lib: LibrarySystem):
    """
    Interactive command-loop for the library system.

    Presents a text menu, accepts user input and invokes `LibrarySystem` methods.
    """
    while True:
        print_menu()
        choice = input_prompt("Choose (0-11): ")
        if choice == "0":
            print("Exiting. You may save changes (option 11) before leaving.")
            break
        elif choice == "1":
            print(f"\nTotal books: {len(lib.books_df)}")
            df = lib.books_df.copy()
            # show first 50 rows
            for _, b in df.head(50).iterrows():
                print(
                    f"{b['Book ID']}: {b['Title']} | {b['Author']} | {b['Genre']} | {'Available' if b['available'] else 'Issued'}")
        elif choice == "2":
            q = input_prompt("Search query: ")
            res = lib.search_books(q)
            print(f"Found {len(res)} result(s):")
            for b in res:
                print(
                    f"{b['Book ID']}: {b['Title']} | {b['Author']} | {'Available' if b.get('available', False) else 'Issued'}")
        elif choice == "3":
            # Show only available books
            available = lib.books_df[lib.books_df.get('available', False) == True]
            print(f"Available books ({len(available)}):")
            for _, b in available.iterrows():
                print(f"{b['Book ID']}: {b['Title']} by {b.get('Author', '')}")
        elif choice == "4":
            g = input_prompt("Genre: ")
            res = lib.available_books_by_genre(g)
            print(f"Available books in '{g}' ({len(res)}):")
            for b in res:
                print(f"{b['Book ID']}: {b['Title']} by {b['Author']}")
        elif choice == "5":
            mid = input_prompt("Member ID: ")
            name = input_prompt("Name: ")
            age_raw = input_prompt("Age (optional): ")
            age = int(age_raw) if age_raw.isdigit() else None
            contact = input_prompt("Contact Info: ")
            ok = lib.register_member(mid, name, age, contact)
            print("Registered." if ok else "Failed (ID may exist).")
        elif choice == "6":
            bid = input_prompt("Book ID: ")
            title = input_prompt("Title: ")
            author = input_prompt("Author: ")
            genre = input_prompt("Genre: ")
            ok = lib.add_book(bid, title, author, genre, availability=True)
            print("Added." if ok else "Failed (ID may exist).")
        elif choice == "7":
            mid = input_prompt("Member ID: ")
            bid = input_prompt("Book ID: ")
            loan_days_raw = input_prompt(f"Loan days (default {lib.loan_days}) or press Enter: ")
            loan_days = int(loan_days_raw) if loan_days_raw.strip().isdigit() else None
            ok, msg = lib.borrow_book(mid, bid, loan_days)
            print(msg)
        elif choice == "9":
            # Show members with borrowed books (numeric 9 or 'm' alias)
            # Recompute and show short diagnostics to help surface any state mismatch
            lib._recompute_current_borrowed_map()
            try:
                tail = lib.borrow_log_df.tail(10)
                if not tail.empty:
                    print("\nRecent borrow-log entries:")
                    print(tail.to_string(index=False))
            except Exception:
                pass
            print("\nInternal borrowed map:", getattr(lib, '_borrowed_map', {}))
            members = lib.members_with_borrowed_books()
            print(f"\nMembers with borrowed books: {len(members)}")
            for m in members:
                print(f"{m['Member ID']}: {m['Name']} -> {m['BorrowedBooks']}")
        elif choice == "8":
            # Return book (numeric 8 or 'r' alias)
            mid = input_prompt("Member ID: ")
            bid = input_prompt("Book ID: ")
            ok, msg = lib.return_book(mid, bid)
            print(msg)
        elif choice == "10":
            pop = lib.most_popular_genre()
            print("Most popular genre:", pop or "N/A")
        elif choice == "11":
            lib.save_state()
            print("Saved state to CSVs.")
        else:
            print("Unknown choice. Try again.")


def demo_run():
    """
    Start a demo interactive session using the default CSV file locations.

    Loads any existing data, runs the CLI loop, and prompts to save state on exit.
    """
    lib = LibrarySystem()
    print("Welcome — demo run loaded library files (if present).")
    cli_loop(lib)
    ans = input_prompt("Save state before exit? (y/n): ")
    if ans.lower().startswith("y"):
        lib.save_state()
        print("Saved.")
    print("Goodbye.")


if __name__ == "__main__":
    demo_run()
