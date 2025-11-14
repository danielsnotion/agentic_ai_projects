import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from part_a_library_system import LibrarySystem

lib = LibrarySystem()
# find an available book
avail = lib.books_df[lib.books_df.get('available', False) == True]
if avail.empty:
    print('No available books found to test.')
else:
    bid = avail.iloc[0]['Book ID']
    title = avail.iloc[0]['Title']
    print('Found available book:', bid, title)
    # ensure member M001 exists
    if not lib._ensure_member_exists('M001'):
        lib.register_member('M001', 'Member One', 25, 'm1@example.com')
        print('Registered M001')
    # borrow
    ok, msg = lib.borrow_book('M001', bid)
    print('Borrow attempt:', ok, msg)
    # show borrow log tail and members
    print('\nBorrow-log tail:')
    print(lib.borrow_log_df.tail(10).to_string(index=False))
    print('\nInternal borrowed_map:')
    print(lib._borrowed_map)
    print('\nmembers_with_borrowed_books():')
    print(lib.members_with_borrowed_books())

