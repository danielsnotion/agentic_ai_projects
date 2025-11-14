import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from part_a_library_system import LibrarySystem

lib = LibrarySystem()
print('Initial members_with_borrowed_books():')
print(lib.members_with_borrowed_books())
print('\nAttempting borrow: member M001 book B001 (loan_days=10)')
ok, msg = lib.borrow_book('M001','B001', loan_days=10)
print('Borrow result:', ok, msg)
print('\nAfter borrow, members_with_borrowed_books():')
print(lib.members_with_borrowed_books())
print('\nBorrow log tail:')
print(lib.borrow_log_df.tail(10).to_string(index=False))
print('\nInternal borrowed_map:')
print(getattr(lib, '_borrowed_map', {}))

