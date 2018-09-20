def lazy_sum(*args):
    def sum():
        x=0
        for n in args:
            x=x+n
        return x
    return sum

print(lazy_sum(1,2,3,4,5,6,7,8,9))
f = lazy_sum(1,2,3,4,5,6,7,8,9)
print(type(f))
print(f())
