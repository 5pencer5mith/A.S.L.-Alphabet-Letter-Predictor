import nltk
from nltk.corpus import words

def get_n_sized_words(n):

    try:
        word_list = words.words()
    except LookupError:
        print("Corpus not found")
        return None
    
    if n == 1:
        n_sized_words = [word for word in word_list if word.lower() in ("a", "i")] # Corrected line
    else:
        n_sized_words = [word for word in word_list if len(word) == n]
    
    return n_sized_words

str = "hihowareyou"

l_ptr = 0
r_ptr = 2

sub_str = str[l_ptr:r_ptr]

sub_size = len(sub_str)

available_words = get_n_sized_words(sub_size)

if not available_words:
    print("I broke :(")

if sub_str in available_words:
    print("Ohhhhh babyy")
else:
    print("Darn")



