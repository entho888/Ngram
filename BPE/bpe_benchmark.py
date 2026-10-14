import regex as re 
import heapq
import utils.linked_chained as lc


TAGS_SET = {"<BOS>", "<EOS>", "<SONG>", "</SONG>", "\n\n"}
NORMAL_PAT = re.compile(
    # Contractions: 't and 'm are tokens
    r"'s|'t|'re|'ve|'m|'ll|'d|"
    # Words: sequence of Unicode letters (after optional space)
    r" ?\p{L}+|"
    #Number: sequence of digits (after optional space)
    r" ?\p{N}+|"
    # Punctuation: sequence of non-alphanumeric/non-space
                #(after optional space)
    r" ?[^\s\p{L}\p{N}]+|"
    # whitespace
    r"\s+(?!\S)|\s+")
SPECIAL_PAT = re.compile("("+"|".join(re.escape(tag) for tag in TAGS_SET)+")")


def pretokenization(corpus) :
    """
    Args :
        corpus : iterable of strings
    Returns a list of strings (representing words of corpus)
    """
    pretokenized_corpus = []

    for line in corpus :
        parts = SPECIAL_PAT.split(line)
        for part in parts :
            if part == '' : continue
            elif part in TAGS_SET :
                pretokenized_corpus.append(part)
            else :
                pretokenized_corpus.extend(NORMAL_PAT.findall(part))

    return pretokenized_corpus


def encoder(pretokenized_corpus, token_to_id = None) :
    """
    Args :
        pretokenized_corpus : list of strings (representing words of corpus)

    If token_to_id is not declared, returns id_to_token, token_to_id and a list of lists (words) of ids of tokens (characters/tags)
    else returns a list of lists (words) of ids of tokens (characters/tags)
    """
    wordwise_tokenized_corpus = []

    if token_to_id is None :    
        token_to_id, id_to_token = {}, []

        new_id = 0
        for word in pretokenized_corpus :
            if word in TAGS_SET :
                if word not in token_to_id :
                    id_to_token.append(word)
                    token_to_id[word] = new_id
                    new_id += 1
                wordwise_tokenized_corpus.append([token_to_id[word]])
            else :   
                tokenized_word = []     
                for token in word :
                    if token not in token_to_id :
                        id_to_token.append(token)
                        token_to_id[token] = new_id
                        new_id += 1
                    tokenized_word.append(token_to_id[token])
                wordwise_tokenized_corpus.append(tokenized_word)

        return id_to_token, token_to_id, wordwise_tokenized_corpus

    else :
        for word in pretokenized_corpus :
            if word in TAGS_SET :
                wordwise_tokenized_corpus.append([token_to_id[word]])
            else :   
                tokenized_word = []     
                for token in word :
                    tokenized_word.append(token_to_id[token])
                wordwise_tokenized_corpus.append(tokenized_word)

        return wordwise_tokenized_corpus    
    

def BPE_encoder(text, merge_map, token_to_id = None):
    """
    Args :
        text : iterable of strings (messages / sentences)
        merge_map : given a pair of token's ids, give the id of the merge of the pair. The ids give information about the order of the merge.
        token_to_id : Dictionnary of (token, id) to translate a pretokenized corpus into ids

    Transforms text into a list of token's id representing characters and tags then merge pairs of token's id following the order of merge used in BPE. Returns the result.
    """
    # Decoupage du texte en mots et decoupage des mots en caracteres / balises, traduits en ids si voulu.
    if token_to_id is None :
        text = encoder(pretokenization(text), token_to_id=token_to_id)[2]
    else : 
        text = encoder(pretokenization(text), token_to_id=token_to_id)
    text = [lc.Linked_chain_list(word) for word in text]


    # On initialise le tas min, qui determine quelle paire on va fusionner, et pairs_location 
    heap = set()
    pairs_location = {}
    for i in range(len(text)) :
        word = text[i]
        
        l = len(word.prev)
        if l > 1 :
            for j in range(l-1) :
                pair = (word.tokens[j], word.tokens[j+1])
                if pair in merge_map : # on peut tomber sur des paires qui n'ont pas ete merge
                    heap.add((merge_map[pair], pair))
                    pairs_location.setdefault(pair, set()).add((i,j))
    heap = list(heap)
    heapq.heapify(heap)


    # On fusionne les paires jusqu'a ce que le tas de priorite de fusion soit vide
    while heap :
        # On recupere la paire a fusionner (on fusionne les paires dans le meme ordre que celui du BPE car les ids de merge sont dans l'ordre des fusions)
        merge_id, pair_to_merge = heapq.heappop(heap)

        # On fusionne les paires aux endroits reperes par pairs_location
        for word_index, tok_index in sorted(pairs_location.pop(pair_to_merge)) : # on veut traiter les paires a merge dans chaque mot dans l'ordre de lecture
            # On imagine x a b y et fusion de a b en c
            word = text[word_index]
            x_index = word.prev[tok_index]

            if word.is_valid(tok_index, pair_to_merge) : # On verifie que la paire est encore celle voulue
                if x_index != -1 : # Si x existe, on update les paires (x,a) et (x,c) NB : pas (x,a) car on update que les paires existantes
                    x = word.tokens[x_index]

                    pair = (x, merge_id)
                    # On ne peut fusionner une paire que si elle est dans merge_map
                    if pair in merge_map :
                        new_merge_id = merge_map[pair]
                        # On ne merge que si la paire a ete merge, dans le BPE, apres pair_to_merge
                        if new_merge_id > merge_id : 
                            if pair not in pairs_location :
                                pairs_location[pair] = {(word_index, x_index)}
                                heapq.heappush(heap, (new_merge_id, pair))
                            else : 
                                pairs_location[pair].add((word_index, x_index))

                word.merge(tok_index, merge_id)
                y_index = word.next[tok_index]

                if y_index != -1 : # Si y existe, on update les paires (b,y) et (c,y) NB : pas (x,a) car on update que les paires existantes
                    y = word.tokens[y_index]
            
                    pair = (merge_id, y)
                    if pair in merge_map :
                        new_merge_id = merge_map[pair]
                        # On ne merge que si la paire a ete merge, dans le BPE, apres pair_to_merge
                        if new_merge_id > merge_id : 
                            if pair not in pairs_location :
                                pairs_location[pair] = {(word_index, tok_index)}
                                heapq.heappush(heap, (new_merge_id, pair))
                            else : 
                                pairs_location[pair].add((word_index, tok_index))

    encoded_text = []
    for word in text :
        encoded_text.extend(word.reconstruct())

    return encoded_text


def BPE_decoder(encoded_text, id_to_token) :
    return [id_to_token[tok_id] for tok_id in encoded_text]


def BPE_tokenizer(text, merge_map, ids:bool = True, id_to_token = None, token_to_id = None):
    """
    Args :
        text : iterable of strings
        merge_map : given a pair of token's ids, give the id of the merge of the pair. The ids give information about the order of the merge.
        ids : if True, returns a list of ids after merging them, else, returns a list of strings (tokens) after merging them
        id_to_token : List of tokens, index in the list coincidates with the token's id
        token_to_id : Dictionnary of (token, id) to translate a pretokenized corpus into ids

    Returns a list of strings/ids (representing merged tokens of corpus)
    """
    # On teste la coherence des variables
    if token_to_id is None :
        raise ValueError("token_to_id should be declared")
    if id_to_token is not None and ids :
        raise ValueError("id_to_token shouldn't be declared or ids should be False")
    if id_to_token is None and not ids :
        raise ValueError("id_to_token should be declared or ids should be True")
    

    if ids :
        return BPE_encoder(text, merge_map, token_to_id=token_to_id)
    else : 
        return BPE_decoder(BPE_encoder(text, merge_map, token_to_id=token_to_id), id_to_token)
    

def BPE(k, corpus = None, id_to_token = None, token_to_id = None, merge_map = None,
        words_inventory = None, pairs_counter = None, heap_pairs_counter = None, pairs_location = None) :
    """
    Args :   
        k : the number of tokens to add to the initial vocabulary
        corpus : list of strings (letter, space, etc), the base corpus we work on, used to initially fill wordwise_tokenized_corpus and id_to_token, 
        token_to_id
        id_to_token : list of tokens
        token_to_id : dictionnary of (token,id)
        merge_map : Dictionnary of ((a,b), id of c:= id_to_token[a]id_to_token[b]) 
            as ids are incremental, ids and order of merge are equivalent.
        words_inventory : list / dictionnary (if so, the key will be the string of the word (might be harsh on memory)) of :
            'word in linked chain', number of instances of the word in the corpus, list of indexs where the words appears in the corpus (might be harsh on memory)
        pairs_counter : dictionnary with pairs of tokens' ids as keys and counter of them in the wordwise_tokenized_corpus as values
        heap_pairs_counter : 'lazy' max_heap of (counter, pair)        
        pairs_location : dictionnary with pairs of tokens as keys and list of (index of word, left node) where it appears in the wordwise_tokenized_corpus as 

    Returns enhanced id_to_token, token_to_id, merge_map, wordwise_tokenized_corpus after merging the k pairs of tokens to new tokens 
    and the updated version of pairs_counter, pairs_location.

    words_inventory : to train Ngrams afterwards. Will have to reconstruct the corpus to get a wordwise tokenized one.  
    token_to_id, id_to_token, merge_map : to encode and decode texts afterwards
    
    
    PARTICULARITY :
        heap
        ids of tokens and not tokens.
        linked chain of a word and not a list and full position of pairs in pairs_location
        "lazy" pairs_location 
        factorize the work on pairs by words : pairs_counter goes up or down by the number of occurence of the word in the corpus for each pair

    We do not need to use that much args nor do we need to return pairs_counter, but it's in order to iterate over k and estimate
    the average time of first loop in generated text and see if it's close to the theoretical result.
    """
    # Implementation des variables par defaut
    if corpus is None : corpus = []
    if id_to_token is None : id_to_token = []
    if token_to_id is None : token_to_id = {}
    if merge_map is None : merge_map = {}
    if words_inventory is None : words_inventory = {}
    if pairs_counter is None : pairs_counter = {}
    if heap_pairs_counter is None : heap_pairs_counter = []
    if pairs_location is None : pairs_location = {}

    # On teste si les variables sont coherentes
    if id_to_token == [] and token_to_id != {} :
        raise ValueError('id_to_token is empty while token_to_id is not')
    if id_to_token != [] and token_to_id == {} :
        for id in range(len(id_to_token)) :
            token_to_id[id_to_token[id]] = id
    if (words_inventory != {} and id_to_token == []) or ((words_inventory == {} and id_to_token != [])) :
        raise ValueError('id_to_token and words_inventory are not both empty or both not empty')
    if ((words_inventory == {} and merge_map != {})) :
        raise ValueError('merge_map is not empty while words_inventory is')
    if words_inventory == {} and corpus == [] :
        raise ValueError('words_inventory and corpus are both empty')
    if (pairs_counter == {} and pairs_location != {}) or (pairs_counter != {} and pairs_location == {}) :
        raise ValueError('pairs_counter and pairs_location are not both empty or both not empty')
    if (pairs_counter == {} and heap_pairs_counter != []) or (pairs_counter != {} and heap_pairs_counter == []) :
        raise ValueError('pairs_counter and heap_pairs_counter are not both empty or both not empty')
    if (words_inventory != {} and id_to_token != []) and (pairs_counter == {}) :
        raise ValueError("pairs_counter and pairs_location shouldn't be empty")
    

    # Initialisation de wordwise_tokenized_corpus, id_to_token et token_to_id
    new_id = None
    if words_inventory == {} :
        pretokenized_corpus = pretokenization(corpus)

        new_id = 0
        for i in range(len(pretokenized_corpus)) :
            word = pretokenized_corpus[i]
            if word in TAGS_SET :
                if word not in token_to_id :
                    id_to_token.append(word)
                    token_to_id[word] = new_id
                    new_id += 1
                    words_inventory[word] = {"chain" : lc.Linked_chain_list(List=[token_to_id[word]]), "frequence": 1, "location": [i]}
                else :
                    words_inventory[word]["frequence"] += 1
                    (words_inventory[word]["location"]).append(i)
            else :   
                tokenized_word = []     
                if word not in words_inventory :
                    for token in word :
                        if token not in token_to_id :
                            id_to_token.append(token)
                            token_to_id[token] = new_id
                            new_id += 1
                        tokenized_word.append(token_to_id[token])
                    words_inventory[word] = {"chain" : lc.Linked_chain_list(List=tokenized_word), "frequence": 1, "location": [i]}
                else :
                    words_inventory[word]["frequence"] += 1
                    (words_inventory[word]["location"]).append(i)


    # Initialisation de pairs_counter et pairs_location
    if pairs_counter == {} :
        for word, word_infos in words_inventory.items() :
            word_ids = word_infos["chain"].tokens
            l = len(word_ids)
            if l > 1 :
                for j in range(l-1) :
                    pair = (word_ids[j], word_ids[j+1])
                    pairs_counter[pair] = pairs_counter.get(pair, 0) + word_infos["frequence"]
                    pairs_location.setdefault(pair, set()).add((word,j))
            

        # Initialisation de heap_pairs_counter
        heap_pairs_counter = [(counter, pair) for pair, counter in pairs_counter.items()] # heapify works with the first element of a tuple
        heapq.heapify_max(heap_pairs_counter)   
    

    # On ajoute k tokens
    if new_id :
        new_id_start = new_id
    else :
        new_id_start = len(id_to_token)
    for new_id in range(new_id_start, k+new_id_start) :
        # on cherche les deux tokens a fusionner ie la paire la plus presente
        count, pair_to_merge = heapq.heappop_max(heap_pairs_counter)
        while pairs_counter.get(pair_to_merge, 0) != count :
            count, pair_to_merge = heapq.heappop_max(heap_pairs_counter)

        # on ajoute le nouveau token dans le vocabulaire
        a, b = pair_to_merge 
        new_token = id_to_token[a] + id_to_token[b]
        id_to_token.append(new_token)
        token_to_id[new_token] = new_id
        merge_map[pair_to_merge] = new_id

        # on modifie words_inventory, pairs_counter et pairs_location en fusionnant la paire
        affected_pairs = set() # on recupere les paires que l'on doit 'push' dans le heap
        # on veut traiter les paires a merge dans chaque mot dans l'ordre de lecture, ici on a juste besoin de trier par rapport aux indices ie deux elements du couple.
        for word, tok_index in sorted(pairs_location.pop(pair_to_merge), key=lambda x : x[1]) : 
            # On imagine x a b y et fusion de a b en c
            word_infos = words_inventory[word]
            tok = word_infos["chain"]
            freq = word_infos["frequence"]
            x_index = tok.prev[tok_index]

            if tok.is_valid(tok_index, pair_to_merge) :
                # A partir de la, on sait que le merge est licite
                if x_index != -1 : # Si x existe, on update les paires (x,a) et (x,c)
                    x = tok.tokens[x_index]

                    pair = (x, a)
                    # ce qui fait crash ceux sont les paires que l'on a efface de pairs_location et pairs_counter, ie celle dans pairs_counter
                    if pair in merge_map :
                        if pairs_counter[pair] == 0 : 
                            del pairs_counter[pair]
                        else :
                            affected_pairs.add(pair)
                            pairs_counter[pair] -= freq

                    pair = (x, new_id)
                    affected_pairs.add(pair)
                    pairs_counter[pair] = pairs_counter.get(pair, 0) + freq
                    pairs_location.setdefault(pair, set()).add((word, x_index))

                tok.merge(tok_index, new_id)
                y_index = tok.next[tok_index]

                if y_index != -1 : # Si y existe, on update les paires (b,y) et (c,y)
                    y = tok.tokens[y_index]

                    pair = (b, y)
                    if pair in merge_map :
                        if pairs_counter[pair] == 0 : 
                            del pairs_counter[pair]
                        else :
                            affected_pairs.add(pair)
                            pairs_counter[pair] -= freq
            
                    pair = (new_id, y)
                    affected_pairs.add(pair)
                    pairs_counter[pair] = pairs_counter.get(pair, 0) + 1
                    pairs_location.setdefault(pair, set()).add((word, tok_index))


        # Plus d'occurence valide de la paire fusionnee, on supprime.
        del pairs_counter[pair_to_merge]
        # On update le heap
        for pair in affected_pairs :
            if pair in pairs_counter :
                heapq.heappush_max(heap_pairs_counter, (pairs_counter[pair], pair))
    

    return id_to_token, token_to_id, merge_map, words_inventory, pairs_counter, heap_pairs_counter, pairs_location
