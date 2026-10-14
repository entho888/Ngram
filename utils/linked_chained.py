from array import array


class Linked_chain_list :

    def __init__(self, List:list):
        length = len(List)
        
        self.tokens = List.copy()
        self.next = array('b', range(1,length))
        self.next.append(-1)
        self.prev = array('i', [-1] + list(range(length - 1)))


    def merge(self, index, value): 
        b = self.next[index]

        if b == -1 :
            return

        else :
            nxt = self.next[b]
            self.tokens[index] = value
            self.next[index] = nxt
            self.next[b] = -1 # Pour simplifier un raisonnement dans BPE
            if nxt != -1 :
                self.prev[nxt] = index


        # NE PAS OUBLIER : si on merge (a,b)->c, on decremente les paires (self.tokens[self.prev], a) (si self.prev !=-1) et (b, self.tokens[self.next]) (si self.next != -1)
        # apres merge, on incremente les paires (self.tokens[self.prev], c) (si self.prev !=-1) et (c, self.tokens[self.next]) (si self.next != -1) ainsi que leur location

    def is_valid(self, index, pair) :
        b_index = self.next[index]

        if (b_index != -1 and (self.tokens[index], self.tokens[b_index]) == pair) : # on verifie s'il y a toujours un b et si la paire (a,b) est bien pair.
            return True 
        else :
            return False
        

    def reconstruct(self) :
        word = [self.tokens[0]]
        x = self.next[0]

        while x != -1 :
            word.append(self.tokens[x])
            x = self.next[x]

        return word