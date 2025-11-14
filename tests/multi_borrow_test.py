import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from part_a_library_system import LibrarySystem

lib = LibrarySystem()
# Clean up any pre-existing test ids if present
for mid in ['M900','M901']:
    if lib._ensure_member_exists(mid):
        # remove from members_df
        lib.members_df = lib.members_df[lib.members_df['Member ID'] != mid]
for bid in ['B900','B901']:
    if lib._ensure_book_exists(bid):
        lib.books_df = lib.books_df[lib.books_df['Book ID'] != bid]

# Register two members
lib.register_member('M900','Member A',30,'a@example.com')
lib.register_member('M901','Member B',31,'b@example.com')
# Add two books
lib.add_book('B900','Book A','Author A','GenreA',availability=True)
lib.add_book('B901','Book B','Author B','GenreB',availability=True)
# Each member borrows one book
ok1, msg1 = lib.borrow_book('M900','B900')
ok2, msg2 = lib.borrow_book('M901','B901')
print('borrow results:', ok1, ok2)
print(msg1)
print(msg2)
# Inspect borrow_log and borrowed_map
print('\nBorrow log:')
print(lib.borrow_log_df.tail(10).to_string(index=False))
print('\nBorrowed map:')
print(lib._borrowed_map)
# Call members_with_borrowed_books
print('\nmembers_with_borrowed_books():')
print(lib.members_with_borrowed_books())

