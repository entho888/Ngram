import regex as re
import json
import random

space = " "
TAGS_LIST = ["<BOS>", "<EOS>", "<SONG>", "</SONG>", "\n\n"]




"""
Bonne idée en soit, mais non transposable : 
    # Pour tokenizer plus facilement le corpus plus tard
    TAGS_DICT = {TAGS_LIST[i]:i for i in range(len(TAGS_LIST))} # On peut mettre des chiffres car il n'y en a pas dans le texte (seulement dans les noms des personnages AVANT réplique, au personnage n'est appelé "1. Play" par un autre)
"""

# Gestions des chansons, lectures, sonnets etc dans les répliques
"""
On ajoute, à la main, des balises <SONG> </SONG> quand il y a une chanson ou la lecture d'une texte dans l'objectif de garder la structure,
on essaie de distinguer la prose des vers et du chant :
[\s\[\(][sS][oi]((ng)|(NG))(s|(ing)|())([\n\]\)\s\.;]|())
[\s\[][rR]ead(s|(ing))
[sS]croll

Ensuite on décide qu'entre les balises, il y ait 4 espaces entre le texte et le début de la ligne (renormalisation)

Remarque : 
  Peto. [reads] 
    <SONG>
                 'Item. A capon. . . . . . . . . . . . .  ii s. ii d.
                 Item, Sauce. . . . . . . . . . . . . .      iiii d.
                 Item, Sack two gallons . . . . . . . . v s. viii d.
                 Item, Anchovies and sack after supper.  ii s. vi d.
                 Item, Bread. . . . . . . . . . . . . .          ob.'
    </SONG>

a été manuellement remplacé par :
  Peto. [reads] 
    <SONG>
                 'Item. A capon. . . . . . . . . . . . . . ii s. ii d.
                 Item, Sauce. . . . . . . . . . . . . . . . . iiii d.
                 Item, Sack two gallons. . . . . . . . . v s. viii d.
                 Item, Anchovies and sack after supper. . ii s. vi d.
                 Item, Bread. . . . . . . . . . . . . . . . . . . ob.'
    </SONG>
pour ne pas être destructuré plus tard par les autres fonctions (gestion des 'gros' espaces).


Une question se pose : est-ce qu'on apprend au modèle le retour à la ligne avant <SONG> et après </SONG> ou est-ce qu'on l'impose dans la génération de texte ? 
Je choisis de laisser le modèle apprendre MAIS, plus tard dans le nettoyage du corpus, de ne mettre des retours à la ligne avant <SONG> que si la ligne précedente n'est pas un nom de personnages (ie la réplique commence par <SONG>)
et même pour </SONG> et fin de réplique.
"""
def song_normalization(initial_corpus) : 
    inside_song = False
    first_line_song = False
    corpus = []

    for ligne in initial_corpus :
        if inside_song :
            # Balise de fin, on modifie la dernière ligne envoyée pour lui ajouter la balise de fin.
            if '</SONG>' in ligne :
                previous_line_song = corpus.pop()
                inside_song = False
                if '\n' in previous_line_song :
                    previous_line_song = previous_line_song[:-1]
                corpus.append(previous_line_song + "</SONG>" + '\n')
            
            else :
                # on veut garder les lignes vides dans les musiques, pour la structure, et qu'elles ne passent pas à la trappe lorsqu'on enlève les autres lignes vides
                if re.search(r'^\s*$', ligne) : 
                    previous_line_song = corpus.pop()
                    corpus.append(previous_line_song + '\n') # donc \n\n devient un token (en tout cas, on le veut.)
                # On enlève les espaces de début et de fin de ligne 
                else :
                    ligne = re.sub(r'^\s*', '', ligne)
                    ligne = re.sub(r'\s*\n$', '\n', ligne)

                    # Si c'est la première ligne de la musique, on y ajoute la balise de début.
                    if first_line_song : 
                        first_line_song = False
                        ligne = "<SONG>" + ligne

                    corpus.append(4*space + ligne)
        
        else : 
            # Permet d'identifier qu'on est dans une musique et que la prochaine ligne en est la première
            if '<SONG>' in ligne :
                inside_song = True
                first_line_song = True 
            else :
                # Les lignes qui ne sont pas des musiques ne sont pas concernées, encore ...
                corpus.append(ligne)
    
    return corpus


# On enlève une grosse partie des éléments de contexte des scènes et les distinctions entre oeuvres (titre, texte écrit différemment (pas le même nombre d'espaces, noms personnages pas en majuscules))
"""
Recherche des derniers 'mid' espaces qui ne sont pas des didascalies: 
^\s{6}[^\s]
^\s{7}[^\s]
^\s{8}[^\s]
^\s{9,}[^\s]
-> a permi de détecter des nouveaux <SONG>, des bouts de répliques un trop décalés, etc, ce qui reste dans 'with_songs' n'est plus que didascalies.

[A-Za-z\.;!?':,-]\s{2,7}\[{0,1}\[{0,1}[A-Za-z] -> a détecté des espaces <4 dans des phrases. Attention aux signatures de lettre !
[A-Za-z\.;!?':,-]\s{8,}\[{0,1}\[{0,1}[A-Za-z] -> si on enlève les espaces entre les noms des personnages et la réplique, il n'y a que des didascalies !

Enlever les espaces entre le nom du personnage et la réplique (et en même vérifier s'il reste des endroits avec sonnets, chansons, etc) :
^\s{2}[A-Z]+\.\s{2,}  OU ^\s{2}(\d\.\s){0,1}[A-Z]+\s{0,1}[A-Z']*\s{0,1}[A-Z']*\s{0,1}[A-Z']*\.\s{2,} pour inclure plus de nom de personnages
^\s{2}(\d\.\s){0,1}[a-z]+\s{0,1}[a-z']*\s{0,1}[a-z']*\s{0,1}[a-z']*\s{0,1}\.\s{2,} pour les minuscules, mais y'a rien.

Verifier si les noms des personnages sont bien à 2 espaces : 
^\s{1}[^\s]

Verifier si les seules lignes à 5 espaces sont dans "THE TRAGEDY OF KING LEAR" :
^\s{5}[^\s]
"""

CHARACTERS_NAMES_NO_CAPS = {
    "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK\n", "THE FIRST PART OF KING HENRY THE FOURTH\n", "THE TRAGEDY OF KING LEAR\n", "THE TRAGEDY OF ROMEO AND JULIET\n"
}

def upper_repl(match):
    return match.group(1).upper()

def dialogue_only_corpus(initial_corpus) :
    """
    - Entre date et by William Shapkespeare, on peut tout supprimer et récupérer le nom de l'oeuvre (seule ligne avec du texte entre les deux) pour
    faire des distinctions de cas.
    - Entre le début de l'oeuvre et le premier personnage qui parle, on supprime. 
    - Idem entre le début de scene (scene, epilogue, prologue / induction (toutes les 'inductions' suivent une SCENE)) et premier personnage 
    qui parle ensuite.

    On renormalise la structure de dialogues : 
    - noms de personnages qui parlent : 2 espaces après le début de la ligne
    - texte d'une réplique : 4 espaces après le début de la ligne

    On distingue quelques cas particuliers :
    - THE COMEDY OF ERRORS, où il n'y a pas d'espace pour le nom du personnage et 2 espaces pour le texte d'une réplique
    - les oeuvres dans CHARACTERS_NAMES_NO_CAPS dont les noms de personnages sont écrits avec une lettre majuscule puis des minuscule 
    et peuvent avoir des nombres dedans. Exemple : 1. Play. ou 2. Play.
    - THE TRAGEDY OF KING LEAR où il y a, en plus, 5 espaces entre le début de la ligne et le texte d'une réplique
    - THE TRAGEDY OF OTHELLO, MOOR OF VENICE où il y a parfois de 'gros' espaces entre le nom du personnage et le reste de la réplique sur la
    même ligne, sans que ça soit une didascalie. Pour éviter de traiter le texte comme une didascalie par la suite, on le traite ici
    
    Pour les autres oeuvres, on est dans le cas nom de personnages en majuscules et "espaces normalisés".
    """
    corpus = []
    date, shakespeare, act, recopy = False, False, False, False
    book_name = ""

    for ligne in initial_corpus : 
        # On repart de zéro à la fin d'une oeuvre
        if "THE END" in ligne : 
            date, shakespeare, act, recopy = False, False, False, False
        
        # La date est le premier élément d'une oeuvre
        elif re.search(r'\d{4}', ligne) and not shakespeare : 
            date = True

        # Entre la date et la signature de l'auteur, il y a la capture du nom de l'oeuvre et la signature de l'auteur
        elif date and not shakespeare :
            # capture du nom de l'oeuvre
            if re.search(r'^[A-Z]+', ligne) :
                book_name = ligne
            # signature de l'auteur
            elif re.search(r'by\sWilliam\sShakespeare', ligne) :
                date, shakespeare = False, True

        # Après la signature de l'auteur
        elif not date and shakespeare :
            # Soit on a pas encore rencontré le début d'une scène et on le cherche soit on l'a déjà rencontré (recopy == True)
            if not act :
                if re.search(r"(((ACT\s[A-Z]{1,}\.{0,1}\s)|())(SCENE|Scene)\s{0,1}((\d{1,2})|([A-Z]+))\.{0,1})|(((ACT_\d\|)|())(SC_\d+|Sc_\d+))|(SCENE:)|(SCENE\.)|(ACT\s[A-Z]+\.\s(PROLOGUE)\.)", ligne) :
                    act, recopy = True, False
                # pour essayer de garder la structure CHAR_NAME. réplique. et car il n'y a pas de SCENE avant.
                elif "EPILOGUE" in ligne :
                        act, recopy = True, True
                        corpus.append(2*space + "EPILOGUE.\n") 
                else :
                    if recopy :
                        if book_name == "THE COMEDY OF ERRORS\n" :
                            # On ajoute 2 espaces avant le nom du personnage
                            if re.search(r"^(?:\d\.\s)?[A-Z]+(?:\s?[A-Z']*){0,3}\.", ligne) :
                                ligne = re.sub(r"^((?:\d\.\s)?[A-Z]+(?:\s?[A-Z']*){0,3}\.)", 2*space + r"\1", ligne)
                            # On ajoute 2 espaces avant le début de la réplique (donc 4 espaces en tout)
                            elif re.search(r"^(\s{2})[^\s]", ligne) :
                                ligne = re.sub(r"^(\s{2}[^\s])", 2*space + r"\1", ligne)
                        elif book_name in CHARACTERS_NAMES_NO_CAPS :
                            # On met en majuscule le nom du personnage grâce à la fonction upper_repl
                            if re.search(r"^\s{2}(?:\d\.\s)?[A-Z][a-z]+(?:\s?[A-Z]?[a-z']*){0,3}\.", ligne) :
                                ligne = re.sub(r"^(\s{2}(?:\d\.\s)?[A-Z][a-z]+(?:\s?[A-Z]?[a-z']*){0,3}\.)", upper_repl, ligne)
                            # On passe de 5 à 4 espaces entre le début de la ligne et le texte de la réplique
                            elif book_name == "THE TRAGEDY OF KING LEAR\n" and re.search(r"^\s{5}[^\s]", ligne) :
                                ligne = re.sub(r"^\s{5}([^\s])", 4*space + r"\1", ligne)
                        # On enlève les 'gros' espaces après nom de personnages pré-réplique, présents uniquement dans cet oeuvre
                        elif book_name == "THE TRAGEDY OF OTHELLO, MOOR OF VENICE\n" :
                            if re.search(r"^(\s{2}(?:\d\.\s)?[A-Z]+(?:\s?[A-Z']*){0,3}\.)\s{2,}", ligne) :
                                ligne = re.sub(r"^(\s{2}(?:\d\.\s)?[A-Z]+(?:\s?[A-Z']*){0,3}\.)\s{2,}", r"\1" + space, ligne)
                         
                        corpus.append(ligne)

            else :
                # On a rencontré le début de la scène, mais pas encore les premières répliques (et donc les noms des personnages qui parlent).
                if not recopy :
                    if book_name == "THE COMEDY OF ERRORS\n" :
                        if re.search(r"^(?:\d\.\s)?[A-Z]+(?:\s?[A-Z']*){0,3}\.", ligne) :
                            act, recopy = False, True
                            # On ajoute 2 espaces avant le nom du personnage
                            corpus.append(re.sub(r"^((?:\d\.\s)?[A-Z]+(?:\s?[A-Z']*){0,3}\.)", 2*space + r"\1", ligne))
                    # On met en majuscule le nom du personnage grâce à la fonction upper_repl
                    elif book_name in CHARACTERS_NAMES_NO_CAPS :
                        if re.search(r"^\s{2}(?:\d\.\s)?[A-Z][a-z]+(?:\s?[A-Z]?[a-z']*){0,3}\.", ligne) :
                            act, recopy = False, True
                            corpus.append(re.sub(r"^(\s{2}(?:\d\.\s)?[A-Z][a-z]+(?:\s?[A-Z]?[a-z']*){0,3}\.)", upper_repl, ligne))
                    # On enlève les 'gros' espaces après nom de personnages pré-réplique, présents uniquement dans cet oeuvre
                    elif book_name == "THE TRAGEDY OF OTHELLO, MOOR OF VENICE\n" :
                        if re.search(r"^(\s{2}(?:\d\.\s)?[A-Z]+(?:\s?[A-Z']*){0,3}\.)\s{2,}", ligne) :
                            act, recopy = False, True
                            corpus.append(re.sub(r"^(\s{2}(?:\d\.\s)?[A-Z]+(?:\s?[A-Z']*){0,3}\.)\s{2,}", r"\1" + space, ligne))
                    else :
                        if re.search(r"^\s{2}(?:\d\.\s)?[A-Z]+(?:\s?[A-Z']*){0,3}\.", ligne) :
                            act, recopy = False, True
                            corpus.append(ligne)
                
                # Ce cas est spécialement pour les epilogues, permet de recopier le texte dans un epilogue.
                else :
                    corpus.append(ligne)

    return corpus

"""
On vérifie qu'il ne reste pas de nom de personnages en miniscules : 
^\s{2}(?:\d\.\s)?[A-Z][a-z]+(?:\s?[A-Z]?[a-z']*){0,3}\. (il en restait 2, dans d'autres oeuvres)
"""


# On enlève le texte après des gros (5>=) espaces car il n'y a que des didascalies
def large_space_stage_directions(initial_corpus) :
    corpus = []

    for ligne in initial_corpus :
        if 5*space in ligne :
            # on enlève tout ce qu'il y a après les 5 espaces et plus (et les espaces aussi)
            corpus.append(re.sub(r"\s{5,}.*", "\n", ligne))
        else :
            corpus.append(ligne)
    return corpus



# On enlève les didascalies entre crochets et entre parenthèses
"""
On vérifie les didascalies "évidentes" dans des crochets (attention parfois il y a des uniquement un bout dedans ou des crochets non fermé ou des crochets étallés sur plusieurs lignes)
Exemple embêtant : [Kneels] Rise up, Sir John Mortimer. [Rises] ; [Aside]  It works.  [To FERDINAND] ; [To FERDINAND]  Come, follow.  [To MIRANDA] ; [To SERVANT] Get you gone, sirrah. [Exit SERVANT]
Exemple embêtant numéro 2 : [Within: Good master porter, I belong to th' larder.] (que l'on ignorera)
 donc \[.*\] n'est pas si simple... solution -> \[[^\[]*\]
Il faut enlever intelligemment les espaces aux alentours pour les cas suivants :
 1) CALIBAN.   [ Within]  There's wood enough within.
 2)     [ Within]  There's wood enough within.
 3)     There's wood enough within.   [ Within]
 4) The which, [To AGAMEMNON] most mighty, for thy place and sway,    
 5) [Sings]  It was the friar of orders grey, (à l'intérieur d'un <SONG></SONG>)

Pour les cas où les crochets sont sur plusieurs lignes : 
Exemple numéro 1 : 
    before us. Away with her to prison. Go to; no more words. [Exeunt
    OFFICERS with MISTRESS OVERDONE]  Provost, my brother Angelo will
Exemple intriguant : [To ISABELLA) 
Remarques : 
- ça s'étale sur 3 lignes max
- il n'y a pas fin (stricte) de didascalie terminant par ] puis un début (strict) de didascalie commencant par [ 
- il y a des fin de didascalie ou des débuts avec, sur la même ligne, une didascalie entière.
- il y a aussi les didascalies parenthèses qui nécessitent d'être regardées en détail car il y a parfois du vrai texte dedans.
  On pourrait penser que seules celles à côté du nom du personnage sont des didascalies, mais non, et je ne trouve pas de motif.
  Finalement, on traite les parenthèses à la main dans le fichier 'with_songs'
"""
def bracket_stage_directions(initial_corpus) :
    corpus = []
    left_bracket = False
    
    for ligne in initial_corpus :
        # D'abord les didascalies bien crochetées
        if re.search(r"\[[^\[]*\]", ligne) :
            # 1) CALIBAN.   [ Within]  There's wood enough within. OU 4) The which, [To AGAMEMNON] most mighty, for thy place and sway,
            if re.search(r"[^\s]\s*\[[^\[]*\]\s*", ligne) :
                ligne = re.sub(r"([^\s])\s*\[[^\[]*\]\s*([^\s])", r"\1" + space + r"\2", ligne)
            # 3)     There's wood enough within.   [ Within]   Remarque : Traite aussi les disdascalies seules sur leur ligne
            if re.search(r"\[[^\[]*\]\s*\n$", ligne) :
                ligne = re.sub(r"\s*\[[^\[]*\]\s*\n$", "\n", ligne)
            # 2)     [ Within]  There's wood enough within.
            if re.search(r"^\s{4}\[[^\[]*\]", ligne) :
                ligne = re.sub(r"^\s{4}\[[^\[]*\]\s*", 4*space, ligne)
            # 5) [Sings]  It was the friar of orders grey, (à l'intérieur d'un <SONG></SONG>)
            #if re.search(r"^\[[^\[]*\]", ligne) :
            #    ligne = re.sub(r"^\[[^\[]*\]\s*([^\s])", r"\1", ligne)

        # LUCIO. [To ISABELLA) You had marr'd all else. Remarque : ici on connait toutes les données, sinon on devrait également traiter tous les cas précédents...
        if re.search(r"\[[^(\[\]\()]*\)", ligne) :
            ligne = re.sub(r"\s*\[[^(?:\[\]\()]*\)\s*", space, ligne)

        """
        # fin stricte de didascalie avec du texte après
        if re.search(r"\]\s*([^\s])", ligne) :
            ligne = re.sub(r"\]\s*([^\s])", 4*space + r"\1", ligne)
            left_bracket = False
        # fin stricte de didascalie sans texte après (\n ne compte pas comme du texte)
        if re.search(r"\]", ligne) :
            ligne = re.sub(r"[^\[]*\]\s*$", 4*space + "\n", ligne)
            left_bracket = False
        """
        if re.search(r"\]\s*", ligne) :
            ligne = re.sub(r"\]\s*", 4*space, ligne)
            left_bracket = False

        """
        # début strict de didascalie avec du texte avant
        if re.search(r"([^\s])\s*\[[^\]]*", ligne) :
            left_bracket = True
            ligne = re.sub(r"([^\s])\s*\[[^\]]*", r"\1" + "\n", ligne)
            corpus.append(ligne)
        # début strict de didascalie sans texte avant
        if re.search(r"^\s*\[[^\]]*", ligne) :
            left_bracket = True
            # On ne garde pas la ligne
        """
        if re.search(r"\s*\[[^\]]*", ligne) :
            left_bracket = True
            ligne = re.sub(r"\s*\[[^\]]*", "\n", ligne)
            corpus.append(ligne)


        # Si on est dans une didascalie non encore fermée, la ligne est supprimée
        if not left_bracket :
            corpus.append(ligne)

    return corpus


# Gestion des didascalies qui peuvent rester (on les repère avec des mots types des didascalies)
"""
On vérifie les mots clés classiques des didascalies (exit, exeunt, enter, flourish, etc)
1) Exit / Exeunt
On distingue :
    - Exit ou <Exit en début de ligne avec que des espaces (éventuellement) avant -> toutes la ligne est une didascalie
    - Exit après du texte -> on enlève tout ce qu'il y a après (on oublie pas le \n)
2) Enter / Flourish (moins utilisé, mais quand même)
Difficile de faire des distinctions algorithmiques car c'est un mot couramment utilisé... mais c'était faisable à la main.
3) Re-enter / Alarum / Alarm / Kills / Draws / Stabs / Beats / Trumpet / Hautboys / Music / Knock / Thunder / Tucket / Drums / S[io]ngs / Noise /
Above / Below / Aloft / Within / Aside / Apart
Il n'y a plus / pas de didascalies avec
"""
def special_words_stage_directions(initial_corpus) :
    corpus = []
    for ligne in initial_corpus :
        if "Exit" in ligne :
            """
            if re.search(r"^\s*\<?Exit", ligne) : continue
            elif re.search(r"[^\s]\s*Exit", ligne) :
                corpus.append(re.sub(r"([^\s])\s*Exit.*$", r"\1" + "\n", ligne))
            """
            ligne = re.sub(r"\s*\<{0,1}Exit.*$", "\n", ligne)
        elif "Exeunt" in ligne :
            """
            if re.search(r"^\s*\<?Exeunt", ligne) : continue
            elif re.search(r"[^\s\<]\s*Exeunt", ligne) :
                corpus.append(re.sub(r"([^\s])\s*Exeunt.*$", r"\1" + "\n", ligne))
            """
            ligne = re.sub(r"\s*\<{0,1}Exeunt.*$", "\n", ligne)
        
        corpus.append(ligne)
    
    return corpus
        

# Gestion des caractères spéciaux qui n'ont rien à faire là
"""
Exemples : 
  MRS. FORD. } Away, away.
  MRS. PAGE. } Away, away. 

  ,,
"""


# Tout ce qui n'est pas dans des balises <SONG> ou dans un EPILOGUE (car déjà fait) est retravaillé en prose et on gère les espaces >1
# ie on recolle l'intra réplique en faisant attention aux majuscules (après un point uniquement), espaces après virgule. 
def lower_repl(match):
    return match.group(1).lower()


def collapse_linebreaks(initial_corpus):
    corpus = []
    current_line = ""
    inside_song, inside_epilogue, is_previous_line_song = False, False, False

    for ligne in initial_corpus :
        if not inside_song :
            # Début de réplique
            if re.search(r"^(\s{2}(?:\d\.\s)?[A-Z]+(?:\.?\s?[A-Z']*){0,3}\.)", ligne) :
                # On ajoute la réplique maintenant finie au corpus
                if current_line != "" : 
                    if is_previous_line_song :
                        if current_line[-1] == '\n' : 
                            current_line = current_line[:-1]
                        is_previous_line_song = False

                    current_line += "<EOS>"
                    current_line = re.sub(' +', ' ', current_line).strip() # on remplace les espaces >1 par un seul espace
                    corpus.append(current_line)
                    current_line = "" # Reset
                    inside_epilogue = False
    
                if "EPILOGUE" in ligne : 
                    inside_epilogue = True
                    ligne = ""
                elif re.search(r"^\s{2}(?:\d\.\s)?[A-Z]+(?:\.?\s?[A-Z']*){0,3}\.\s*([^\s])", ligne) :
                    ligne = re.sub(r"^\s{2}(?:\d\.\s)?[A-Z]+(?:\.?\s?[A-Z']*){0,3}\.\s*([^\s])", r"\1", ligne)     # On enlève le nom du personnage qui parle et on gère les espaces du début
                    ligne = re.sub(r"\s*$", "", ligne)    # On gère les espaces de fin de ligne (et l'éventuel retour à la ligne) pour récupérer plus simplement le dernier caractère de cette dernière
                else :
                    ligne = "" 
                ligne = "<BOS>" + ligne

            # On tombe sur un <SONG> : on met un retour à la ligne avant, uniquement s'il y a déjà du texte dans la réplique.
            elif "<SONG>" in ligne :
                ligne = ligne[4:]
                if current_line != "" :
                    ligne = '\n' + ligne

                if "</SONG>" in ligne : # il peut y avoir <SONG> puis </SONG> sur la même ligne
                    is_previous_line_song, inside_song = True, False 
                else: 
                    inside_song = True

            # Texte de réplique
            elif re.search(r"^\s{4}[^\s]", ligne) :
                if not inside_epilogue :
                    if current_line[-1] in {".", "!", "?"} :
                        ligne = space + (ligne[4]).upper() + ligne[5:] 
                    elif current_line[-1] == '\n' : # signifie que c'est après un </SONG> (auquel cas, on garde tel quel)
                        ligne = ligne[4:]
                    else :
                        ligne = space + (ligne[4]).lower() + ligne[5:] 
                    ligne = re.sub(r"\s*$", "", ligne)    # On gère les espaces de fin de ligne (et l'éventuel retour à la ligne) pour récupérer plus simplement le dernier caractère de cette dernière
                else :
                    ligne = ligne[4:]

            # Ligne vide, qu'on ignore / supprime
            elif re.search(r"^\s*$", ligne) :
                continue

            current_line += ligne

        else : 
            # fin de la musique     
            if "</SONG>" in ligne :
                is_previous_line_song, inside_song = True, False 

            # Ligne vide, qu'on ignore / supprime (mais dans <SONG>, elles ont été enregistrées au préalable dans song_normalization avec '\n\n')
            elif re.search(r"^\s*$", ligne) :
                continue

            current_line += ligne[4:]

    return corpus


"""
initial_corpus = open("corpus\\t8.shakespeare_with_songs.txt","r")
training_corpus = open("corpus\\training_corpus.jsonl", "w", encoding="utf-8")
dev_corpus = open("corpus\\dev_corpus.jsonl", "w", encoding="utf-8")
test_corpus = open("corpus\\test_corpus.jsonl", "w", encoding="utf-8")


new_corpus = song_normalization(initial_corpus)
initial_corpus.close()
new_corpus = dialogue_only_corpus(new_corpus)
new_corpus = large_space_stage_directions(new_corpus)
new_corpus = bracket_stage_directions(new_corpus)
new_corpus = special_words_stage_directions(new_corpus)
new_corpus = collapse_linebreaks(new_corpus)

#Ici new corpus est nettoyé, il faut maintenant le couper en plusieurs parties : 1 partie entrainer le modèle, 1 partie pour ajuster les hyperparamètres et 1 partie pour évaluer le modèle
random.seed(a="fleur", version=2) # Pour que ça soit reproductible (cf tuto de Stanford)
corpus_length = len(new_corpus)

index_list = [i for i in range(corpus_length)]
random.shuffle(index_list)
split_1 = int(0.8 * corpus_length)
split_2 = int(0.9 * corpus_length)

for i in index_list[:split_1] :
    ligne = new_corpus[i]
    json.dump({"text":ligne}, training_corpus)
    training_corpus.write('\n')
training_corpus.close()

for i in index_list[split_1:split_2] :
    ligne = new_corpus[i]
    json.dump({"text":ligne}, dev_corpus)
    dev_corpus.write('\n')
dev_corpus.close()

for i in index_list[split_2:] :
    ligne = new_corpus[i]
    json.dump({"text":ligne}, test_corpus)
    test_corpus.write('\n')
test_corpus.close()
"""




