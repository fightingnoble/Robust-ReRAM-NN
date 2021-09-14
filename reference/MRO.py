class A(object):
    def __init__(self, *args, **kwargs):
        print("A")

class B(object):
    def __init__(self, *args, **kwargs):
        print("B")

class C(A):
    def __init__(self, arg, *args, **kwargs):
        print ("C","arg=",arg)
        super(C, self).__init__(arg, *args, **kwargs)

print(C.__mro__)