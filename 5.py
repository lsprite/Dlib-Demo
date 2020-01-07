class Student():

    def __init__(self, __name='default_name', age=0):
        self.__name = __name
        self.age = age


s = Student()
s.age = 18
print(s._Student__name)

print(dir(Student))