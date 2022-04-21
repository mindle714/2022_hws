def trie(words, _end='_end_'):
  root = dict()
  for word in words:
    current_dict = root
    for letter in word:
      current_dict = current_dict.setdefault(letter, {})
    current_dict[_end] = _end
  return root

def in_trie(trie, word, _end='_end_'):
  current_dict = trie
  for letter in word:
    if letter not in current_dict: return False
    current_dict = current_dict[letter]
  return _end in current_dict

def expand_prefix(trie, word, max_prefs=100, _end='_end_'):
  current_dict = trie
  for letter in word:
    if letter not in current_dict: return []
    current_dict = current_dict[letter]

  prefixes = []
  dict_queue = [(word, current_dict)]

  while len(dict_queue) > 0:
    next_queue = []
    for pref_word, _dict in dict_queue:
      for key in _dict:
        if key == _end:
          prefixes.append(pref_word)
          if len(prefixes) == max_prefs:
            return prefixes
        else:
          next_queue.append((pref_word + key, _dict[key]))
    dict_queue = next_queue

  return prefixes
