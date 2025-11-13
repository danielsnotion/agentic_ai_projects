import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from part_a_library_system import LibrarySystem

lib = LibrarySystem()
print('Initial borrowed_map:', getattr(lib, '_borrowed_map', None))
# create test member and book
mid = 'MX001'
bid = 'BX001'
lib.register_member(mid, 'Debug Member', 40, 'debug@example.com')
lib.add_book(bid, 'Debug Book', 'Debug Author', 'DebugGenre', availability=True)
# Borrow
ok, msg = lib.borrow_book(mid, bid)
print('borrow result:', ok, msg)
print('borrow_log_df rows:', len(lib.borrow_log_df))
print(lib.borrow_log_df.tail(5).to_string(index=False))
print('internal _borrowed_map:', lib._borrowed_map)
print('members_with_borrowed_books():', lib.members_with_borrowed_books())

