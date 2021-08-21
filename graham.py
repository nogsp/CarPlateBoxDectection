class pt(object):
    x = 0
    y = 0
    def __init__(self,x,y):
        self.x = float(x)
        self.y = float(y)
    
    def __repr__(self):
        return '(' + str(self.x) + ',' + str(self.y) + ')'

    def cross_product(self,a,b):
        return ((a.x - self.x)*(b.y - self.y)) - ((b.x - self.x)*(a.y - self.y))
    
    def norm(self,a):
        return (((a.x-self.x)*(a.x-self.x)) + ((a.y-self.y)*(a.y-self.y))) ** (1.0/2.0)

    #operators
    def __eq__(self, other):
        ans = (self.x == other.x) and (self.y == other.y)
        return ans

    def __sub__(self, other):
        ans = pt(self.x - other.x, self.y - other.y)
        return ans

    def __add__(self,other):
        ans = pt(self.x + other.x, self.y + other.y)
        return ans
    
    def __NEG__(self):
        ans = pt(-self.x, -self.y)
        return ans

    '''@staticmethod
    def cross_product(a, b):
        return (a.x*b.y) - (b.x*a.y)
    '''


def graham_scan(lista):
    ind = 0
    i = 0
    for p in lista:
        if(p.y < lista[ind].y):
            ind = i
        if(p.y == lista[ind].y):
            if(p.x < lista[ind].x):
                ind = i
        i+=1 
    pivot = lista[ind]
    del(lista[ind])

    def cmp(a,b):
        cp = pivot.cross_product(a,b)
        if cp > 0: return -1
        if cp < 0: return 1
        norma = pivot.norm(a)
        normb = pivot.norm(b)
        return normb-norma

    def key_cmp(mycmp):
        class k:
            def __init__(self, obj, *args):
                self.obj = obj
            def __lt__(self, other):
                return mycmp(self.obj, other.obj) < 0
            def __gt__(self, other):
                return mycmp(self.obj, other.obj) > 0
            def __eq__(self, other):
                return mycmp(self.obj, other.obj) == 0
            def __le__(self, other):
                return mycmp(self.obj, other.obj) <= 0
            def __ge__(self, other):
                return mycmp(self.obj, other.obj) >= 0
            def __ne__(self, other):
                return mycmp(self.obj, other.obj) != 0
        return k

    lista.sort(key = key_cmp(cmp))
    
    #for p in lista:
    #    print(p)
    #print('resp:')
    
    convex = [pivot, lista[0]]
    del(lista[0])

    for p in lista:
        tam = len(convex)
        p1 = convex[tam-2]
        p2 = convex[tam-1]
        cp = p1.cross_product(p2,p)

        if cp == 0:
            convex.pop() #Se eu quiser pontos na mesma reta eh so comentar
            convex.append(p)
        elif cp > 0:
            convex.append(p)
        else:
            while cp <= 0 and len(convex) > 2: #Se eu quiser pontos na mesma reta eh so tirar o =
                convex.pop()
                tam = len(convex)
                p1 = convex[tam-2]
                p2 = convex[tam-1]

                cp = p1.cross_product(p2,p)
            
            convex.append(p)
    
    return convex