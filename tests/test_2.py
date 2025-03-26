from english_words import get_english_words_set

dict = get_english_words_set(sources=['web2'])
filtered = {word for word in dict if len(word) > 1 and word.lower() not in ["a", "i"]}

# print(len(filtered))

big_string = "hellohowareyou"

p_l = 0
p_r = 1

possible_words = []

for i in range(len(big_string) - 1):
    starting_at_index_i = [] 
    for j in range(1, len(big_string) + 1):
        sub_string = big_string[i:j]
        if sub_string in filtered:
            starting_at_index_i.append([sub_string, j])
    possible_words.append(starting_at_index_i)

print(possible_words)

possible_sentences = []

# for i in range(len(possible_words[0]) - 1):
#     sentence = []
#     j = 0
#     while j < len(big_string):
#         if not possible_words[j]:
#             break
#         if i >= len(possible_words[j]):
#             break
#         current_word = possible_words[j][i][0]
#         start_of_next_word = possible_words[j][i][1]
#         sentence.append(current_word)
#         j = start_of_next_word

#         if start_of_next_word > len(big_string) - 1:
#             possible_sentences.append(sentence)
#             break
            

# print(possible_sentences)




# print("1", possible_words[0])
# print("2", possible_words[0][0])
# print("3", possible_words[0][0][0])

# print(len(possible_words[0]))



