# Given a 2D keyboard, and a “maximum jump distance” jump_distance,
# determine if a given word can be constructed using the characters in the keyboard,
# obeying the jump_distance. A jump can be up, down, right, or left.
# But NO diagonal, i.e. a diagonal jump would naturally consume 2 units of jump distance.

# Arbitrary keyboard given as list of lists, e.g.:

# [

# [‘Q’,’ X’, ‘P’, ‘L’, ‘E’],

# [‘W’, ‘A’, ‘C’, ‘I’, ’N’],

# ]

# Jumping distance given as an integer, e.g. 2.

# “PENCIL”, jump_distance = 2

# P, E -> [0,2], [0, 4],
# P → E → N → C → I → L True

# “PACE”, jump_distance = 2

# P → A → C → can’t go to E! False

# implement a method word_can_be_typed(keyboard, word, jump_distance) that returns True
# if the word can be constructed, else False.

# Edge Cases:
# - Characters in input words do not exist in input.
"""
Solution 1: O(V^3)

Generate all distances between all pairs of characters on the keyboard.

Solution 2:

1. Iterate through every row and column in our keyboard, storing rows and columns for each key.

2. Iterate through our word, for each consecutive pair of characters, we can compute their distance in O(1) by summing
   absolute differences in (r,c) for each. If any character does not eixst in our keyboard, return False.

"""
# Time Complexity: O(K + N)
# Space Complexity: O(K)

def solve(s: str, keyboard: list[list[str]], jump_distance: int) -> bool:
  if not keyboard:
    return not len(s)

  if not s:
    return True

  m, n = len(keyboard), len(keyboard[0])

  if len(set(s)) > m * n:
    return False

  adj = {}
  for i in range(m):
    for j in range(n):
      adj[keyboard[i][j]] = (i,j)

  s = s + s[0]
  for a,b in pairwise(s):
  # for a,b in zip(s,  s[1:]):
    if a not in adj or b not in adj:
      return False
    (x1,y1) = adj[a]
    (x2,y2) = adj[b]
    dist = abs(x1 - x2) + abs(y1 - y2)
    if dist > jump_distance:
      return False

  return True

# dict['A'] = 1
# TC: Word: B
# [['A', 'B', 'B', 'C', 'D']]
# Word: ABD, jump_distance = 2
# for each character (i,j):
#     for all other characters (i2,j2):
#          fetch distance between them, and update best found distances between character pairs.

# for each pair of characters in the input word: ('c', 'a').
#      consider every possible start.

# keyboard characters as vertices
# O(V + E)

# DFS, Backtracking, can I represent this as a graph? Should I do that now, or is it a waste of time?



def dfs(index, word, char_to_coords_list, cur_coord, jump_distance):
    if index == len(word) - 1:
        return True

    char = word[index]
    next_char = word[index + 1]
    both_chars_present = char in char_to_coords_list and next_char in char_to_coords_list
    if not both_chars_present:
        return False

    next_coords_list = char_to_coords_list[next_char]

    for next_coord in next_coords_list:
        if manhattan_distance(cur_coord, next_coord) <= jump_distance:
            print(char, next_char, cur_coord, next_coord)
            if dfs(index + 1, word, char_to_coords_list, next_coord, jump_distance):
                return True
    return False

def word_can_be_typed_follow_up(word, keyboard, jump_distance):
    if len(word) == 0:
        return True

    char_to_coords_list = {}
    for row_index in range(len(keyboard)):
        for col_index in range(len(keyboard[row_index])):
            char = keyboard[row_index][col_index]
            if char not in char_to_coords_list:
                char_to_coords_list[char] = []

            char_coord = (row_index, col_index)
            char_to_coords_list[char].append(char_coord)

    first_char = word[0]
    if first_char in char_to_coords_list:
        for coord in char_to_coords_list[first_char]:
            if dfs(0, word, char_to_coords_list, coord, jump_distance):
                return True
    return False
