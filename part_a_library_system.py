"""
Part A: City Library Management System
--------------------------------------
Tasks:
1) Data model for Book and Member using @dataclass.
2) Core operations: add/update books, register members.
3) Borrow/return workflow with validation.
4) Search utilities (by title/author) and operational reports.
5) Sample run that demonstrates the workflow and prints outputs.

How to run:
$ python part_a_library_system.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import Counter


# ----------------------------- Data Model ----------------------------- #
@dataclass
class Book:
    book_id: str
    title: str
    author: str
    genre: str
    available: bool = True


@dataclass
class Member:
    member_id: str
    name: str
    age: int
    contact: str
    borrowed_books: List[str] = field(default_factory=list)


# --------------------------- Core System API -------------------------- #
class LibrarySystem:
    """
    In-memory prototype:
    - books, members: dictionaries keyed by IDs
    - borrow_log: list of actions for traceability
    - genre_issue_counter: tracks popularity by issues
    """

    def __init__(self):
        self.books: Dict[str, Book] = {}
        self.members: Dict[str, Member] = {}
        self.borrow_log: List[dict] = []
        self.genre_issue_counter: Counter = Counter()

    # ---- Books ----
    def add_book(self, book_id: str, title: str, author: str, genre: str) -> bool:
        """Add new book if ID unused. Returns True on success."""
        if book_id in self.books:
            return False
        self.books[book_id] = Book(book_id, title, author, genre, available=True)
        return True

    def update_book_availability(self, book_id: str, available: bool) -> bool:
        """Flip availability for an existing book."""
        book = self.books.get(book_id)
        if not book:
            return False
        book.available = available
        return True

    # ---- Members ----
    def register_member(self, member_id: str, name: str, age: int, contact: str) -> bool:
        """Register new member if ID unused."""
        if member_id in self.members:
            return False
        self.members[member_id] = Member(member_id, name, age, contact)
        return True

    # ---- Borrow/Return ----
    def borrow_book(self, member_id: str, book_id: str) -> bool:
        """
        Borrow book if: member exists, book exists, and book is available.
        Side-effects: mark book unavailable, append to member list, log action, count genre issue.
        """
        member = self.members.get(member_id)
        book = self.books.get(book_id)
        if not member or not book or not book.available:
            return False

        book.available = False
        member.borrowed_books.append(book_id)
        self.borrow_log.append({"member_id": member_id, "book_id": book_id, "action": "borrow"})
        self.genre_issue_counter[book.genre] += 1
        return True

    def return_book(self, member_id: str, book_id: str) -> bool:
        """Return book if member holds it; updates state and logs action."""
        member = self.members.get(member_id)
        book = self.books.get(book_id)
        if not member or not book or book_id not in member.borrowed_books:
            return False

        book.available = True
        member.borrowed_books.remove(book_id)
        self.borrow_log.append({"member_id": member_id, "book_id": book_id, "action": "return"})
        return True

    # ---- Queries/Reports ----
    def search_books(self, query: str) -> List[Book]:
        """Case-insensitive substring match on title or author."""
        q = query.lower()
        return [b for b in self.books.values() if q in b.title.lower() or q in b.author.lower()]

    def available_books_by_genre(self, genre: str) -> List[Book]:
        return [b for b in self.books.values() if b.genre.lower() == genre.lower() and b.available]

    def members_with_borrowed_books(self) -> List[Member]:
        return [m for m in self.members.values() if m.borrowed_books]

    def most_popular_genre(self) -> Optional[str]:
        return self.genre_issue_counter.most_common(1)[0][0] if self.genre_issue_counter else None


# ------------------------------- Demo -------------------------------- #
if __name__ == "__main__":
    lib = LibrarySystem()

    # Add books
    lib.add_book("B001", "The Pragmatic Programmer", "Andrew Hunt", "Technology")
    lib.add_book("B002", "Clean Code", "Robert C. Martin", "Technology")
    lib.add_book("B003", "1984", "George Orwell", "Fiction")
    lib.add_book("B004", "To Kill a Mockingbird", "Harper Lee", "Fiction")
    lib.add_book("B005", "The Name of the Wind", "Patrick Rothfuss", "Fantasy")

    # Register members
    lib.register_member("M001", "Aditi Rao", 29, "aditi@example.com")
    lib.register_member("M002", "Karan Shah", 35, "karan@example.com")

    # Borrow/return flows
    print("Borrow 1 (M001 -> B003):", lib.borrow_book("M001", "B003"))
    print("Borrow 2 (M002 -> B005):", lib.borrow_book("M002", "B005"))
    print("Borrow 3 (M001 -> B003 again):", lib.borrow_book("M001", "B003"), "(should be False)")
    print("Return 1 (M001 <- B003):", lib.return_book("M001", "B003"))
    print("Borrow 4 (M001 -> B003):", lib.borrow_book("M001", "B003"), "(should be True)")

    # Queries/reports
    print("\nAvailable Fiction:", [b.title for b in lib.available_books_by_genre("Fiction")] or "None")
    print("Members with loans:", [f"{m.name} ({len(m.borrowed_books)})" for m in lib.members_with_borrowed_books()] or "None")
    print("Search 'clean':", [b.title for b in lib.search_books("clean")] or "No matches")
    print("Most popular genre (by issues):", lib.most_popular_genre() or "N/A")
