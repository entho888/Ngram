import numpy as np 
import re 
import time
import heapq
import math
import BPE.bpe_benchmark as BPE
import json
import random
import matplotlib.pyplot as plt


BOS, EOS, BSONG, ESONG = [None for _ in range(4)]

training_corpus_file = open("corpus\\training_corpus.jsonl","r")
training_corpus = []
for line in training_corpus_file :
    training_corpus.append(json.loads(line)["text"])


def BPE_wordwise_tokenized_corpus_reconstruction(words_inventory) :
    """
    Args :
        words_inventory : dictionnary of word: {"chain": linked chain of the word, "frequence": number of occurrences of the word in the corpus, "location": index of the word in the corpus}

    Returns a list (corpus) of lists (words) of ids/tokens         
    """
    
    number_of_words = 0
    for word, word_infos in words_inventory.items() :
        number_of_words += word_infos["frequence"]

    wordwise_tokenized_corpus = [None for _ in range(number_of_words)]

    for word, word_infos in words_inventory.items() :
        reconstructed_word = word_infos["chain"].reconstruct()
        for i in word_infos["location"] :
            wordwise_tokenized_corpus[i] = reconstructed_word

    return wordwise_tokenized_corpus


def messagewise_tokenized_corpus(wordwise_tokenized_corpus) :
    """
    Args :
        wordwise_tokenized_corpus : list of lists (representing words) of ids (representing tokens)
    
    Returns a list (corpus) of lists (messages) of ids (tokens) 
    """
    messagewise_tokenized_corpus = []
    message = []

    for word in wordwise_tokenized_corpus :
        message.extend(word)
        if word == [EOS] :
            messagewise_tokenized_corpus.append(message)
            message = []

    return messagewise_tokenized_corpus


def ALLgrams(N:int, flattened_messagewise_tokenized_corpus) :
    """
    Args : 
        flattened_messagewise_tokenized_corpus : list of lists (representing messages/lines) of strings / ids (representing tokens)

    Returns a dictionnary with context (tuple of tokens, empty tuple if no context) as keys and (following_tokens_id_list, following_tokens_cumulativ_counts_list, {token_id : count}) as values.
    The two lists purpose is sampling. 
    The {token_id : count} purpose is to calculate the probability of a message / text (ie multiple messages)
    In the dictionnary we have (context, value) for all context of length <= N-1 ie the dictionnary contains all kgrams for 1<=k<=N in order to compute them once and for all and so the interpolated models only need 1 dictionnary and not
    k many.
    NB : if we sort following_tokens_id_list, following_tokens_cumulativ_counts_list before computing the cumulativ counts, we only need ONE big dictionnary in memory to efficiently compute greedy, topk, etc sampling and probability
    """

    # Implementation of all the counters
    allGrams = {}
    for message in flattened_messagewise_tokenized_corpus :
        N_context = [BOS for _ in range(N-1)]
        for i in range(1,len(message)) :
            following_token = message[i]
            for k in range(1,N+1) :
                if k == 1 :
                    k_context = []
                else :
                    k_context = N_context[-(k-1):]
                k_context = tuple(k_context)

                if k_context in allGrams :
                    conditionnal_counter = allGrams[k_context][2]
                    conditionnal_counter[following_token] = conditionnal_counter.get(following_token, 0) + 1
                else : 
                    allGrams[k_context] = ([], [], {following_token: 1})

            N_context.pop(0)
            N_context.append(following_token)

    # Implementation of the lists, ordering and then cumativ_count_list
    for context, context_infos in allGrams.items() :
        dict_counter = context_infos[2]
        ids_list, counters_list = list(dict_counter.keys()), list(dict_counter.values())

        intricated_lists_quick_sort(counters_list, ids_list)

        # cumulated counts
        S = 0
        for i in range(len(counters_list)) :
            S += counters_list[i]
            context_infos[1].append(S)
            context_infos[0].append(ids_list[i])
        
    return allGrams


def intricated_lists_quick_sort(array1:list, array2:list):
    """
    In-place quick DECREASING sort array1 and apply the same changes to array2.
    Uses an explicit stack to avoid recursivity problems of Python.
    """
    if not array1 or not array2:
        return 
    if len(array1) != len(array2) :
        raise ValueError("Both arrays aren't the same length")

    stack = [0, len(array1) - 1]

    while stack:
        # Dépiler les bornes droite et gauche
        end = stack.pop()
        start = stack.pop()

        if start < end:
            # Partitionner le sous-tableau et obtenir l'index du pivot
            pivot_index = intricated_lists_quick_sort_partition(array1, array2, start, end)

            # Empiler les indices des deux sous-tableaux à traiter
            stack.extend([start, pivot_index - 1])
            stack.extend([pivot_index + 1, end])

    return


def intricated_lists_quick_sort_partition(array1, array2, start, end):
    pivot = array1[end]
    i = start - 1  # Index de la "frontière" entre plus petit et plus grand que le pivot

    for j in range(start, end):
        if array1[j] >= pivot:
            i += 1
            array1[i], array1[j] = array1[j], array1[i]
            array2[i], array2[j] = array2[j], array2[i]

    # Placer le pivot à sa position finale
    array1[i + 1], array1[end] = array1[end], array1[i + 1]
    array2[i + 1], array2[end] = array2[end], array2[i + 1]
    return i + 1


# N > 1, sinon ça n'a aucun intérêt...
def greedy_loop_searching(Allgrams, N:int, initial_context = None) : 
    if initial_context is not None :
        if N-1 > len(initial_context) :
            raise ValueError("initial_context is not long enough")
    else :
        initial_context = random_initial_context(Allgrams, N)
        while tuple(initial_context) not in Allgrams :
            initial_context = random_initial_context(Allgrams, N)

    tokens_counter = 0
    context = initial_context[-(N-1):]
    tpl_context = tuple(context)
    seen_context = set() # historique des contextes déjà vus
    tokens_counter = N-1


    # On génère un texte (ensemble de message) jusqu'à boucler.
    while tpl_context not in seen_context :
        seen_context.add(tpl_context)
        following_token = Allgrams[tpl_context][0][0]

        if following_token == EOS : # Si on a une fin de message, on en commence un autre
            context = random_initial_context(Allgrams, N)
            while tuple(context) not in Allgrams :
                context = random_initial_context(Allgrams, N)
            tokens_counter += N-1
        else :
            # une fois le contexte initial choisit, context agit comme une file : suppression du dernier élément de contexte, ajout du dernier mot dedans
            context.pop(0)
            context.append(following_token)
            tokens_counter += 1

        tpl_context = tuple(context)

    return tokens_counter


def uniform_loop_searching(k, N) :
    context = [random.randint(1,k) for i in range(N-1)]
    seen_k = set()
    u = random.randint(1,k)

    c = N-1
    while tuple(context) not in seen_k :
        seen_k.add(tuple(context))

        context.pop(0)
        context.append(u)
        u = random.randint(1,k)
        c+=1

    return c


# N > 1, sinon ça n'a aucun intérêt...
def greedy_sampling(Allgrams, N:int, id_to_token, tokenized:bool = False, initial_context = None) :
    """
    Initialisation du contexte : 3 méthodes :
        0) on ajoute suffisamment de <BOS> pour ne pas se poser de question --> I am I am ... en boucle ...
        1) on récupère tous les débuts de phrases à N tokens et on choisit aléatoirement (proportionnellement à leur fréquence) parmi ceux là
        2) on choisit aléatoirement (proportionnellement à leur fréquence) le premier token (ie grâce au model stochatic N=1), puis on choisit le 
        deuxième à l'aide du model stochatic N=2, etc jusqu'à avoir N tokens dans le contexte
    """
    if initial_context is not None :
        if N-1 > len(initial_context) :
            raise ValueError("initial_context in not long enough")
    else :
        initial_context = random_initial_context(Allgrams, N)
    

    context = initial_context[-(N-1):]
    tpl_context = tuple(context)
    following_token = Allgrams[tpl_context][0][0]
    seen_context = set() # historique des contextes déjà vus
    encoded_message = initial_context 

    while following_token != EOS and tpl_context not in seen_context : # condition pour ne pas boucler
        seen_context.add(tpl_context)
        if following_token not in {"BSONG", "ESONG"} :
            encoded_message.append(following_token)

        # suppression du dernier élément de contexte, ajout du dernier mot dedans
        context.pop(0)
        context.append(following_token)
        tpl_context = tuple(context)

        following_token = Allgrams[tpl_context][0][0]


    if following_token != EOS : # Pour montrer où est la boucle
        encoded_message.append(following_token)

    if not tokenized :
        message = "".join(BPE.BPE_decoder(encoded_message, id_to_token))
    else :
        message = BPE.BPE_decoder(encoded_message, id_to_token)

    return message
    

def random_choice(cumu_counters:list) :
    u = random.randint(0, cumu_counters[-1])
    i,j = 0, len(cumu_counters)-1
    while j-i > 1 :
        k = (i+j)//2
        if u > cumu_counters[k] :
            i = k
        else :
            j = k
    
    return i


def random_initial_context(Allgrams, N) :
    # initialisation du contexte
    context = [BOS for _ in range(N-1)]
    initial_context = []
    c = 0

    while c < N-1 : 
        following_tokens_correspondance = Allgrams[tuple(context)]
        following_token_index = random_choice(following_tokens_correspondance[1])
        following_token = following_tokens_correspondance[0][following_token_index]
        
        # Si on tombe sur une fin de message. On persiste ... en commencant un nouveau message
        if following_token == EOS :
            context = [BOS for _ in range(N-1)]
        else :
            context.pop(0)
            context.append(following_token)

            initial_context.append(following_token)
            c += 1

    return initial_context




#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
# Test du théorème de temps de première répétition
# ATTENTION : le temps d'arrêt est un résultat asymptotique sur le nombre de tokens à générer pour atteindre une première répétition. 
# Le nombre de tokens ajoutés par fusion (k) n'est pas le nombre de tokens qu'il est possible de générer avec le modèle !!
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
random.seed(a="fleur", version=2) # Pour que ça soit reproductible
N = 4

k_list = list(set([int(k) for k in np.floor(10**(np.linspace(np.log10(1), np.log10(30000), 100)))]))
k_list.sort()

id_to_token, token_to_id, merge_map, words_inventory, pairs_counter, heap_pairs_counter, pairs_location = BPE.BPE(0, corpus=training_corpus)
BOS, EOS, BSONG, ESONG = token_to_id["<BOS>"], token_to_id["<EOS>"], token_to_id["<SONG>"], token_to_id["</SONG>"]  
R = 500 # moyenne empirique sur R=500 simulations
greedy_loop_list_N2, greedy_loop_list_N3, greedy_loop_list_N4 = [], [], []
number_of_possible_tokens_N2_list, number_of_possible_tokens_N3_list, number_of_possible_tokens_N4_list = [], [], []


for i in range(len(k_list)) :
    print(i)
    if i == 0 : nb_of_new_tokens = k_list[i]
    else : nb_of_new_tokens = k_list[i] - k_list[i-1] # l'argument k de BPE correspondant au nombre de tokens supplémentaires à ajouter par fusion
    id_to_token, token_to_id, merge_map, words_inventory, pairs_counter, heap_pairs_counter, pairs_location = BPE.BPE(nb_of_new_tokens,
                                                                                                                id_to_token=id_to_token,
                                                                                                                token_to_id=token_to_id,
                                                                                                                merge_map=merge_map,
                                                                                                                words_inventory=words_inventory,
                                                                                                                pairs_counter=pairs_counter,
                                                                                                                heap_pairs_counter=heap_pairs_counter,
                                                                                                                pairs_location=pairs_location)  
    wordwise_tokenized_corpus = BPE_wordwise_tokenized_corpus_reconstruction(words_inventory)
    messagewise_tokenized_corpus_flattened = messagewise_tokenized_corpus(wordwise_tokenized_corpus)
    Allgrams = ALLgrams(N, messagewise_tokenized_corpus_flattened)

    # Le nombre de tokens ajoutés par fusion (k) n'est pas le nombre de tokens qu'il est possible de générer avec le modèle !!
    possible_tokens_N2, possible_tokens_N3, possible_tokens_N4 = set(), set(), set()
    for context, context_infos in Allgrams.items() :
        match len(context) : 
            case 1 : # on récupère uniquement les tokens qui sont générables par le greedy_model N=2
                possible_tokens_N2.add(context[0]) # id du token contexte
                possible_tokens_N2.add(context_infos[0][0]) # id du token choisi par le modèle greedy
            case 2 : # on récupère uniquement les 2grams qui sont générables par le greedy_model N=3
                possible_tokens_N3.add(context) # id du token choisi par le modèle greedy
            case 3 : # on récupère uniquement les 3grams qui sont générables par le greedy_model N=4
                possible_tokens_N4.add(context) # id du token choisi par le modèle greedy
    number_of_possible_tokens_N2_list.append(len(possible_tokens_N2))
    number_of_possible_tokens_N3_list.append(len(possible_tokens_N3))
    number_of_possible_tokens_N4_list.append(len(possible_tokens_N4))

    # Calcul moyenne empirique modèle greedy
    mean_gready_N2, mean_gready_N3, mean_gready_N4 = 0, 0, 0
    for i in range(R) :
        mean_gready_N2 += greedy_loop_searching(Allgrams, N=2)
        mean_gready_N3 += greedy_loop_searching(Allgrams, N=3)
        mean_gready_N4 += greedy_loop_searching(Allgrams, N=4)
    greedy_loop_list_N2.append(mean_gready_N2/R)
    greedy_loop_list_N3.append(mean_gready_N3/R)
    greedy_loop_list_N4.append(mean_gready_N4/R)


#Vérification empirique du théorème sur les variables uniformes
X = list(set([int(k) for k in np.floor(10**(np.linspace(np.log10(1), np.log10(10**7), 300)))]))
X.sort()
uniform_loop_list_N2 = []
R = 1000

for x in X :
    mean_uniform_N2 = 0
    for i in range(R) :
        mean_uniform_N2 += uniform_loop_searching(x, 2)
    uniform_loop_list_N2.append(mean_uniform_N2/R)
    

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
# Vérification de l'existence de boucle 
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
id_to_token, token_to_id, merge_map, words_inventory, pairs_counter, heap_pairs_counter, pairs_location = BPE.BPE(0, corpus=training_corpus)

loop_existence_list_stochatic_N2, loop_existence_list_stochatic_N3, loop_existence_list_stochatic_N4 = [], [], []
loop_existence_list_greedy_N2, loop_existence_list_greedy_N3, loop_existence_list_greedy_N4 = [], [], []


for i in range(len(k_list)) :
    print(i)
    if i == 0 : nb_of_new_tokens = k_list[i]
    else : nb_of_new_tokens = k_list[i] - k_list[i-1] # l'argument k de BPE correspondant au nombre de tokens supplémentaires à ajouter par fusion
    id_to_token, token_to_id, merge_map, words_inventory, pairs_counter, heap_pairs_counter, pairs_location = BPE.BPE(nb_of_new_tokens,
                                                                                                            id_to_token=id_to_token,
                                                                                                            token_to_id=token_to_id,
                                                                                                            merge_map=merge_map,
                                                                                                            words_inventory=words_inventory,
                                                                                                            pairs_counter=pairs_counter,
                                                                                                            heap_pairs_counter=heap_pairs_counter,
                                                                                                            pairs_location=pairs_location)  
    wordwise_tokenized_corpus = BPE_wordwise_tokenized_corpus_reconstruction(words_inventory)
    messagewise_tokenized_corpus_flattened = messagewise_tokenized_corpus(wordwise_tokenized_corpus)
    Allgrams = ALLgrams(N, messagewise_tokenized_corpus_flattened)

    for n in range(2,N+1) :
        #greedy sampling
        context_seen = set()
        stack = []

        start = tuple([BOS for _ in range(n-1)])
        stack.append(start)

        is_det, det_history = False, set()
        loop_existence = False

        # parcours de la chaine de Markov vu comme graphe en pseudo DFS, arrêt à la première boucle
        while len(stack) != 0 and not loop_existence : 
            context = stack.pop()
            context_seen.add(context) 
            
            if context in Allgrams :
                idx = Allgrams[context][0][0] #on prend seulement le token prochain le plus probable

                new_context = context[1:] + (idx,)
                if is_det :
                    if new_context in det_history :
                        loop_existence = True
                        break
                    elif new_context not in context_seen :
                        det_history.add(new_context)
                        stack.append(new_context)
                    else :
                        is_det, det_history = False, set()
                else :
                    is_det = True
                    if new_context not in context_seen :
                        det_history.add(context)
                        det_history.add(new_context)
                        stack.append(new_context)
            else : 
                is_det, det_history = False, set()

        # pour l'affichage de l'existence de boucle
        if loop_existence : loop_existence = 1
        else : loop_existence = 0
            
        match n :
            case 2:
                loop_existence_list_greedy_N2.append(loop_existence)
            case 3:
                loop_existence_list_greedy_N3.append(loop_existence)
            case 4:
                loop_existence_list_greedy_N4.append(loop_existence)

        #stochatic sampling
        context_seen = set()
        stack = []

        start = tuple([BOS for _ in range(n-1)])
        stack.append(start)

        is_det, det_history = False, set()
        loop_existence = False

        # parcours de la chaine de Markov vu comme graphe en pseudo DFS, arrêt à la première boucle
        while len(stack) != 0 and not loop_existence : 
            context = stack.pop()
            context_seen.add(context)
            
            if context in Allgrams :
                ids_list = Allgrams[context][0]

                if len(ids_list) == 1 :
                    new_context = context[1:] + (ids_list[0],)
                    if is_det :
                        if new_context in det_history :
                            loop_existence = True
                            break
                        elif new_context not in context_seen :
                            det_history.add(new_context)
                            stack.append(new_context)
                        else :
                            is_det, det_history = False, set()
                    else :
                        is_det = True
                        if new_context not in context_seen :
                            det_history.add(context)
                            det_history.add(new_context)
                            stack.append(new_context)
                else :
                    if is_det :
                        is_det, det_history = False, set()
                    for idx in ids_list :
                        new_context = context[1:] + (idx,)
                        if new_context not in context_seen :
                            stack.append(new_context)
            else : 
                is_det, det_history = False, set()
                
        # pour l'affichage de l'existence de boucle
        if loop_existence : loop_existence = 1
        else : loop_existence = 0
            
        match n :
            case 2:
                loop_existence_list_stochatic_N2.append(loop_existence)
            case 3:
                loop_existence_list_stochatic_N3.append(loop_existence)
            case 4:
                loop_existence_list_stochatic_N4.append(loop_existence)



"""
plt.figure(figsize=(8, 8*9/16))
plt.plot([-2000, 32000], [1,1], dashes=[6, 18], label="Existence", color='black')
plt.plot([-2000, 32000], [0,0], dashes=[6, 18], label="Non-existence", color='green')
plt.plot(k_list, loop_existence_list_greedy_N2, label="greedy", linestyle = "none", marker = '+')
plt.plot(k_list, loop_existence_list_stochatic_N2, label="stochastic", linestyle = "none", marker = 'x')
plt.xlabel("Nombre de tokens ajoutés : k")
plt.title("Existence de boucle 2-grams")
plt.legend(fontsize = 15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("presentation\\images\\Existence de boucle N=2")
plt.show()

plt.figure(figsize=(8, 8*9/16))
plt.plot([-2000, 32000], [1,1], dashes=[6, 18], label="Existence", color='black')
plt.plot([-2000, 32000], [0,0], dashes=[6, 18], label="Non-existence", color='green')
plt.plot(k_list, loop_existence_list_greedy_N3, label="greedy", linestyle = "none", marker = '+')
plt.plot(k_list, loop_existence_list_stochatic_N3, label="stochastic", linestyle = "none", marker = 'x')
plt.xlabel("Nombre de tokens ajoutés : k")
plt.title("Existence de boucle 3-grams")
plt.legend(fontsize = 15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("presentation\\images\\Existence de boucle N=3")
plt.show()

plt.figure(figsize=(8, 8*9/16))
plt.plot([-2000, 32000], [1,1], dashes=[6, 18], label="Existence", color='black')
plt.plot([-2000, 32000], [0,0], dashes=[6, 18], label="Non-existence", color='green')
plt.plot(k_list, loop_existence_list_greedy_N4, label="greedy", linestyle = "none", marker = '+')
plt.plot(k_list, loop_existence_list_stochatic_N4, label="stochastic", linestyle = "none", marker = 'x')
plt.xlabel("Nombre de tokens ajoutés : k")
plt.title("Existence de boucle 4-grams")
plt.legend(fontsize = 15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("presentation\\images\\Existence de boucle N=4")
plt.show()

plt.figure(figsize=(8, 8*9/16))
plt.plot(k_list, loop_existence_list_greedy_N2, label="N=2", linestyle = "none", marker = '+')
plt.plot(k_list, loop_existence_list_greedy_N3, label="N=3", linestyle = "none", marker = '.')
plt.plot(k_list, loop_existence_list_greedy_N4, label="N=4", linestyle = "none", marker = 'o')
plt.xlabel("Nombre de tokens : k")
plt.ylabel("1 = Existence de boucle, 0 = non existence de boucle")
plt.title("Existence de boucle\n"
          "avec greedy sampling de N-grams")
plt.legend(fontsize = 15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("presentation\\images\\Existence de boucle déterministe")
plt.show()

plt.figure(figsize=(8, 8*9/16))
plt.plot(k_list, loop_existence_list_stochatic_N2, label="N=2", linestyle = "none", marker = '+')
plt.plot(k_list, loop_existence_list_stochatic_N3, label="N=3", linestyle = "none", marker = '.')
plt.plot(k_list, loop_existence_list_stochatic_N4, label="N=4", linestyle = "none", marker = 'o')
plt.xlabel("Nombre de tokens : k")
plt.ylabel("1 = Existence de boucle, 0 = non existence de boucle")
plt.title("Existence de boucle\n"
          "avec stochatic sampling de N-grams")
plt.legend(fontsize = 15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("presentation\\images\\Existence de boucle stochatic")
plt.show()
"""