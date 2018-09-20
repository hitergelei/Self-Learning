class Rational0:
    def __init__(self, num, den=1):
        self.num = num
        self.den = den

    def plus(self, another):
        den = self.den * another.den
        num = (self.num * another.den + self.den * another.num)
        self.num = num
        self.den = den
        return self

    def print(self):
        print(str(self.num) + "/" + str(self.den))
