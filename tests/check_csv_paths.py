import sys
import pathlib
# Add project root to sys.path so imports from repo root work when running this test
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from part_a_library_system import LibrarySystem

lib = LibrarySystem()
print("books_csv:", lib.books_csv, "exists:", lib.books_csv.exists())
print("members_csv:", lib.members_csv, "exists:", lib.members_csv.exists())
print("borrow_log_csv:", lib.borrow_log_csv, "exists:", lib.borrow_log_csv.exists())
print("books_df rows:", len(lib.books_df))
print("members_df rows:", len(lib.members_df))
print("borrow_log_df rows:", len(lib.borrow_log_df))
# show sample rows if present
if not lib.books_df.empty:
    print('\nSample books rows:\n', lib.books_df.head(3).to_string(index=False))
if not lib.members_df.empty:
    print('\nSample members rows:\n', lib.members_df.head(3).to_string(index=False))
