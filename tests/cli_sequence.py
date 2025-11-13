import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from part_a_library_system import LibrarySystem

lib = LibrarySystem()
print('\n--- 1. List all books (sample) ---')
print(f'Total books: {len(lib.books_df)}')
for _, b in lib.books_df.head(3).iterrows():
    print(f"{b['Book ID']}: {b['Title']} | {b['Author']} | {b['Genre']} | {'Available' if b['available'] else 'Issued'}")

print('\n--- 2. Search book by title/author (query "Digital") ---')
res = lib.search_books('Digital')
print('Found:', len(res))
for r in res[:3]:
    print(r.get('Book ID'), r.get('Title'))

print('\n--- 3. Show available books by genre (History) ---')
res = lib.available_books_by_genre('History')
print('Found:', len(res))
for r in res[:3]:
    print(r.get('Book ID'), r.get('Title'))

print('\n--- 4. Register member (M300) ---')
ok = lib.register_member('M300', 'CLI Tester', 28, 'cli@test.com')
print('Registered:', ok)

print('\n--- 5. Add book (B300) ---')
ok = lib.add_book('B300', 'CLI Book', 'CLI Author', 'CLI-Genre', availability=True)
print('Added:', ok)

print('\n--- 6. Borrow book (M300 borrows B300) ---')
ok, msg = lib.borrow_book('M300', 'B300')
print('Borrow:', ok, msg)

print('\n--- 7. Return book (M300 returns B300) ---')
ok_ret, msg_ret = lib.return_book('M300', 'B300')
print('Return:', ok_ret, msg_ret)

print('\n--- 6b. Borrow again to test members list (M300 borrows B300) ---')
ok2, msg2 = lib.borrow_book('M300', 'B300')
print('Borrow:', ok2, msg2)

print('\n--- 8. Show members with borrowed books ---')
members = lib.members_with_borrowed_books()
print('Members with borrowed books:', len(members))
for m in members:
    print(m)

print('\n--- 9. Most popular genre ---')
print('Most popular genre:', lib.most_popular_genre())

print('\n--- 10. Save state ---')
lib.save_state()
print('Saved CSVs to:', lib.books_csv.parent)

# show borrow_log tail
print('\nBorrow log tail:')
print(lib.borrow_log_df.tail(5).to_string(index=False))

