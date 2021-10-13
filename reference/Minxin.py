class A(object):
    her=None
    def ppp(self):
        print(self.her)

class B():
    def __init__(self,he):
        self.he = he

class C(A, B):
    pass

class D(C):
    def __init__(self,her):
        self.her = her
D_i = D("Lili")
D_i.ppp()