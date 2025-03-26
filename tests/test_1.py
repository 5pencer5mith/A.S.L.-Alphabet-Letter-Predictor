import nltk
from nltk.corpus import words
import os
from dotenv import load_dotenv

def get_n_sized_words(n):

    load_dotenv()
    os.environ["NLTK_DATA"] = "/Volumes/SpencersDrive/undergraduate_research/nltk_data"


    try:
        word_list = words.words()
        print(type(word_list))
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

else:
    if sub_str in available_words:
        print("Ohhhhh babyy")
    else:
        print("Darn")



