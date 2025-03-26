from english_words import get_english_words_set

def load_dict():
    # load dictionary and filter out single letter "words" that aren't really words (everything but "a" and "i")
    dict = get_english_words_set(sources=['web2'])
    filtered = {word for word in dict if len(word) > 1 and word.lower() not in ["a", "i"]}

    return filtered

def get_possible_words(str, master_word_list):
    possible_words = []

    for i in range(len(str) - 1):
        starting_at_index_i = [] 
        for j in range(1, len(str) + 1):
            sub_str = str[i:j]
            if sub_str in master_word_list:
                starting_at_index_i.append([sub_str, j])
        possible_words.append(starting_at_index_i)

    return possible_words

def get_possible_sentences(index, possible_words, cache=None):
    if cache is None:
        cache = {}

    if index in cache:
        return cache[index]
    
    if index >= len(possible_words):
        return [""]
    
    sentences = []

    for word, start_of_next_index in possible_words[index]:
        groups = get_possible_sentences(start_of_next_index, possible_words, cache)

        for w in groups:
            sentences.append(word + " " + w)

    return sentences

def main():
    master_word_list = load_dict()
    
    str = "thequickbrownfox"

    possible_words = get_possible_words(str, master_word_list)

    possible_sentences = get_possible_sentences(0, possible_words)

    for sentence in possible_sentences:
        print(sentence)

    return

if __name__ == "__main__":
    main()
