class Rational0:
    def __init__(self, num, den=1):
        self.num = num
        self.den = den

    def plus(self, another):
        den = self.den * another.den
        num = (self.num * another.den + self.den * another.num)
        # self.den = den
        # self.num = num
        # return self
        return Rational0(num, den)

    # def sum(self):
    # 	value = self.den + self.num
    # 	print(value)

    def print(self):
        print(str(self.num) + "/" + str(self.den))


r1 = Rational0(3, 5)
r2 = r1.plus(Rational0(7,15))
r2.print()
# r2.sum()
