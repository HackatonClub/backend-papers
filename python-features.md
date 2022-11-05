# Python. Продвинутый уровень

- [Python. Продвинутый уровень](#python-продвинутый-уровень)
  - [Модуль Functools^](#модуль-functools)
    - [cached_property](#cached_property)
    - [partial](#partial)
    - [reduce](#reduce)
    - [singledispatch](#singledispatch)
    - [wraps](#wraps)
  - [Модуль Collections^](#модуль-collections)
    - [namedtuple](#namedtuple)
    - [Counter](#counter)
    - [defaultdict](#defaultdict)
    - [OrderedDict](#ordereddict)
    - [ChainMap](#chainmap)
    - [deque](#deque)
    - [collections.abc](#collectionsabc)
  - [Производительность (стандартная библиотека)^](#производительность-стандартная-библиотека)
  - [Производительность (сторонние модули)^](#производительность-сторонние-модули)
  - [Обработка больших файлов^](#обработка-больших-файлов)
  - [Сахарок^](#сахарок)
  - [Что нового в 3.10^](#что-нового-в-310)

## Модуль Functools[^](#functools)

### cached_property

Данный декоратор преобразует метод класса в свойство, значение которого вычисляется один раз, а затем кэшируется как обычный атрибут на все время жизни экземпляра. Аналогично использованию `@property`, но с добавлением кэширования. Полезно для дорогостоящих вычисляемых свойств экземпляров, которые в противном случае фактически неизменяемы.
```python
from functools import cached_property
from math import pi


class Circle:
    def __init__(self, radius):
        self.radius = radius

    @cached_property
    def square(self):
        return pi * self.radius**2


new_circle = Circle(5)
print(new_circle.square)
# 78.53981633974483
new_circle.radius = 10
print(new_circle.square)
# 78.53981633974483
print(new_circle.radius)
# 10
```
Как мы видим значение площади не меняется так как она закэширована, чтобы это исправить нужно удалить кеш.
```python
del new_circle.square
print(new_circle.square)
# 314.1592653589793
```

### partial

С помощью данной функции мы можем заморозить часть аргументов, вызываемой функции.

Пример:

* Вывод квадратного уравнения по заданным параметрам

```python
def quadratic_equation(a,b,c):
    print("{}x^2+{}x+{}".format(a,b,c))

print(quadratic_equation(2,3,4))
```
Допустим нам нужно реализовать функцию, которая бы всегда выводила, квадратное уравнение с фиксированным параметром `a` , чтобы мы сделали?

```python
def quad_equation_fix_a_2(b,c):
    return quadratic_equation(2,b,c)
print(quad_equation_fix_a_2(6,10))
# 2x^2+6x+10

def quad_equation_fix_a_5(b,c):
    return quadratic_equation(5,b,c)
print(quad_equation_fix_a_5(6,10))
# 5x^2+6x+10
```
Когда таких функций переписанных мало, то этот метод сойдет, но когда их много, на помощь придет `partial`
```python
from functools import partial

quad_equation_fix_2 = partial(quadratic_equation, 2)
print(quad_equation_fix_2(4, 3))
# 2x^2+4x+3

# Фиксируем элемент c изначальной функции
quad_equation_fix_5 = partial(quadratic_equation, 5, c=5)
print(quad_equation_fix_5(1))
# 5x^2+1x+5

```
__Что по сути делает `partial`?__

Она создает новую функцию, в которую записываются **args, переданные в `partial`, давайте посмотрим на аргументы функции `partial`
```python
def partial(func, /, *args, **keywords):
```
Параметры:
* func - любой вызываемый объект

* *args - позиционные аргументы func

* **keywords - ключевые аргументы func.

Возвращает новый объект, который ведет себя как func

### Объект `partial()` имеет три атрибута только для чтения:
* partial.func:

    Атрибут partial.func возвращает исходную функцию, переданную в качестве первого аргумента partial(). Другими словами, вызовы будут перенаправлены в исходную функцию func с новыми аргументами и ключевыми словами.

* partial.args:

    Атрибут partial.func возвращает крайние левые позиционные аргументы, с которыми будет вызвана исходная функция func.

* partial.keywords:

    Атрибут partial.keywords возвращает ключевые аргументы, с которыми будет вызвана исходная функция func.

```python
print(quad_equation_fix_5.func)
# <function quadratic_equation at 0x0000020E29C93400>

print(quad_equation_fix_5.args)
# (5,)

print(quad_equation_fix_5.keywords)
# {'c': 5}

```
### reduce

Reduce принимает функцию и набор пунктов. Возвращает значение, получаемое комбинированием всех пунктов

### reduce(f, *[i1, i2, i3, i4]*) = *f(i1, f(i2, f(i3)))*

Для начала давайте посмотрим на её аргументы в Python:
```python
 def reduce(function, iterable, initializer=None):
```
Первым аргументом является функция, которая будет производить действия над `iterable object`(Итерируемый объект, над которым возможна операция `for i in itrable_object`)
Последний аргумент необязательный - `initializer`. 
Eсли он присутствует, он помещается перед элементами итерируемого объекта в вычислении и используется по умолчанию, когда итерируемый объект пуст. Если инициализатор не указан и итерируемый содержит только один элемент, возвращается первый элемент.

```python 
from functools import reduce

a = [1, 2, 3]
reduce_sum = reduce(lambda x, a: x + a, a)
print("Сумма с reduce:", reduce_sum)
# Сумма с reduce: 6
reduce_initializer_sum = reduce(lambda x, a: x + a, a, 4)
print("Сумма с reduce and initializer:" reduce_initializer_sum)
# Сумма с reduce and initializer: 10
```

Еще пример использования reduce:

- Задача: дан лист целых чисел найти одно единственное значение которое не повторяется в списке

```python 
from functools import reduce

a = [1, 2, 3, 3, 2, 8, 5, 6, 5, 6, 1]
seekonlyone = reduce(lambda x, y: x ^ y, a)
print(seekonlyone)
```
### singledispatch

Декоратор `@singledispatch` модуля `functools` создает из обычной функции - универсальную функцию.(По сути это перегрузка функции)
Чтобы определить универсальную функцию, оберните ее с помощью декоратора `@singledispatch`. Обратите внимание, что в перегруженные реализации передается тип первого аргумента:

```python
from functools import singledispatch
from datetime import date, datetime, time

@singledispatch
def format(arg):
    return arg

@format.register
def _(arg: date):
    return f"{arg.day}-{arg.month}-{arg.year}"

@format.register
def _(arg: datetime):
    return f"{arg.day}-{arg.month}-{arg.year} {arg.hour}:{arg.minute}:{arg.second}"

@format.register(time)
def _(arg):
    return f"{arg.hour}:{arg.minute}:{arg.second}"

print(format("today"))
# today
print(format(date(2021, 5, 26)))
# 26-5-2021
print(format(datetime(2021, 5, 26, 17, 25, 10)))
# 26-5-2021 17:25:10
print(format(time(19, 22, 15)))
# 19:22:15
```
Чтобы добавить перегруженные реализации в функцию, используйте атрибут `.register()` обобщенной функции `fun`. Выражение `fun.register()` то же является декоратором.
Затем мы определяем отдельные функции для каждого типа, который мы хотим перегрузить — в данном случае дату и время — каждый из них имеет имя `_` (подчеркивание), потому что они все равно будут вызываться (отправляться) через метод форматирования, поэтому нет необходимости давать им полезные имена. Каждый из них также украшен `@format.register`, который связывает их с ранее упомянутой функцией форматирования. 

Также мы можем перегружать с помощью  `@singledispatch` методы классов:
```python
from functools import singledispatchmethod


class Formatter:
    @singledispatchmethod
    def format(self, arg):
        raise NotImplementedError(f"Cannot format value of type {type(arg)}")

    @format.register
    def _(self, arg: date):
        return f"{arg.day}-{arg.month}-{arg.year}"

    @format.register
    def _(self, arg: time):
        return f"{arg.hour}:{arg.minute}:{arg.second}"


f = Formatter()
print(f.format(date(2021, 5, 26)))
# 26-5-2021
print(f.format(time(19, 22, 15)))
# 19:22:15
```
### wraps

Для начала вспомним, что такое вообще декораторы. Это функция, которая принимает функцию в качестве аргумента и модифицирует ее.
```python
def logged(func):
    def with_logging(*args, **kwargs):
        "Логируем функцию"
        print(" Я работаю до функции " + func.__name__)
        return func(*args, **kwargs)

    return with_logging


@logged
def hello(name):
    "Говорит привет кто-то"
    return f"Hello,{name}!"


print(hello("Jack"))
# Я работаю до функции hello
# Hello, Jack!
```
Единственный минус это:
```python
print(hello.__name__)
# with_logging
print(hello.__doc__)
# Логируем функцию
```
Когда мы вызываем имя функции и документацию декорированной функции, мы понимаем, что они были заменены значениями из функции декоратора. Это будет затруднять логирование и отладку, поэтому есть `wraps`:
```python
from functools import wraps


def logged_wraps(func):
    @wraps(func)
    def with_logging_wraps(*args, **kwargs):
        "Логируем функцию"
        print("Я работаю до функции "+ func.__name__ )
        return func(*args, **kwargs)
    return with_logging_wraps


@logged_wraps
def wraps_hello(name):
    "Говорит привет кто-то"
    return f"Hello, {name}!"


print(wraps_hello("Jack"))
# Я работаю до функции wraps_hello
# Hello, Jack!

print(wraps_hello.__name__)
# wraps_hello
print(wraps_hello.__doc__)
# Говорит привет кто-то
```

## Модуль Collections[^](#collections)

### namedtuple
Это `tuple`, который каждому элементу кортежа присваивает имя.
К `namedtuple` можно обращаться как к обычному `tuple`, а также к каждому элементу можно обращаться по атрибутам.
Использование `namedtuple`, позволяет писать более читаемый и самодокументированный код.
Если требуется структура данных напоминающая `namedtuple`, но при этом с возможностью изменения полей, то тогда стоит обратить внимание на `dataclasses`.

#### Примеры использования:
1. Создание объекта `namedtuple`:
```python
from collections import namedtuple
from typing import NamedTuple

# 1.1. Стандартный способ:

Point = namedtuple("Point", ["x", "y"])
p = Point(11, y=22)

# 1.2. Более лаконичный способ:

class Point(NamedTuple):
    x: int
    y: int

p = Point(11, y=22)
```
2. Базовый функционал:
```python
# 2.1. Индексация объектов как в обычном tuple
p[0] + p[1]

# 2.2. Аналогично распаковка
x, y = p

# 2.3. Обращение к элементам по названиям полей
p.x + p.y

# 2.4. __repr__ метод переопределен в следующим виде:
print(p)
# will print: "Point(x=11, y=22)"
```

### Counter
Это подкласс `dict`, который служит для подсчета хешируемых объектов (чаще всего внутри итерируемых). Он представляет собой коллекцию, где элементы хранятся как ключи, а в качестве значений
#### Примеры использования:
1. Создание объекта `Counter`:
```python
from collections import Counter

# 1.1. Создание по итерируемому объекту
cnt1 = Counter(["a", "b", "c", "a", "b", "b"])

# 1.2. Создание через dict
cnt2 = Counter({"a": 2, "b": 3, "c": 1})

# 1.3. Создание через kwargs
cnt3 = Counter(a=2, b=3, c=1)

# 1.4. Counter ведет себя как dict
print(cnt1)
# will print: Counter({'b': 3, 'a': 2, 'c':1})
```
2. Базовый функционал:
```python
from collections import Counter

# 2.1. Не кидает KeyError
c = Counter(["a", "bb", "a", "d"])
print(c["c"])
# will print: 0

# 2.2. Может возвратить sorted multiset:
print(
    sorted(c.elements())
)  # Без sorted, elements() вернет в том порядке в котором они встретились
# will print: ['a', 'a', 'bb', 'd']

# 2.3. Может вернуть n самых частых/редких элементов
n = 2
print(c.most_common(n))
# will print: [('a', 2), ('bb', 1)]
print(c.most_common()[: -n - 1 : -1])
# will print: [('d',1), ('bb',1)]

# 2.4. Частотные характеристики можно вычитать
d = Counter("ab")
print(c.subtract(d))

# 2.5. Унарные операции сложения/вычитания
e = Counter(a=2, b=-4)
print(+e)  # Складывает с пустым Counter
# will print: Counter({'a': 2})
print(-e)  # Вычитает из пустого Counter
# will print: Counter({'b': 4})
```
3. Задачи:
```python
# 3.1. Найти все подстроки-анаграммы в другой подстроке
def find_anagrams(self, s: str, p: str) -> list[str]:
    cnt1 = Counter(s[: len(p)])
    cnt2 = Counter(p)
    res = list()
    for i in range(n2, n1):
        cnt1[s[i]] += 1
        cnt1[s[i - n2]] -= 1
        if +cnt1 == +cnt2:
            res.append(i - n2 + 1)
    return res
```

### defaultdict
Это подкласс `dict`, который переопределяет метод добавления значений в словарь. Когда ключ не был найден в словаре, тогда в словаре создается этот ключ с заданным дефолтным значением.
Данный объект работает быстрее чем словарь образованный методом `setdefault()`.

#### Примеры использования:
1. Создание `defaultdict`:
```python
from collections import defaultdict

# 1.1. Создание счетчика
d = defaultdict(int)
d[3] += 1
print(d[2], d[3])
# will print: 0 1

# 1.2. Создание группировщика
l = defaultdict(list)

# 1.3. Создание группировщика уникальных значений
l = defaultdict(set)
```
2. Базовый функционал:
```python
# 1.1. Группировка по ключу
s = [("a", 1), ("b", 2), ("b", 3), ("a", 3), ("c", 1)]
d = defaultdict(list)
for k, v in s:
    d[k].append(v)
print(sorted(d.items()))
# will print: [('a', [1,3]),('b', [2,3]),('c',[1])]
```
### OrderedDict
Отсортированный словарь, ведет себя как обычный `dict`, но у него имеются дополнитеьные возможности для порядковых операций.

Отличия от `dict`:
- `OrderedDict` разработан для эффективности по памяти, скорости итерирования по элементам и скорости выполнения операций обновления значений.
- Операция сравнения учитывает порядок ключей
- Может обрабатывать операции связанные с порядком ключей быстрее чем `dict`.
- `move_to_end()` - эффективный метод перестановки позиции ключа в конец
#### Примеры использования:
1. Создание `OrderedDict`:
```python
from collections import OrderedDict

# 1.1. Создание базовое
od = OrderedDict()

# 1.2. Более эффективная реализация dict
class LastUpdatedOrderedDict(OrderedDict):
    def __setitem__(self, key, value):
        super().__setitem__(key,value)
        self.move_to_end(key)
        
a = LastUpdatedOrderedDict()
```
2. Базовый функционал:
```python
from collections import OrderedDict

# 2.1. Быстрое перемещение в начало/конец списка
d = OrderedDict.fromkeys("abcde")
d.move_to_end("b")
print("".join(d))
# will print: 'acdeb'
d.move_to_end("b", last=False)
print("".join(d))
# will print: 'bacde'

# 2.2. Реализация LRU cache ограниченного по времени
class TimeBoundedLRU:  # Декоратор
    "LRU Cache that invalidates and refreshes old entries."

    def __init__(self, func, maxsize=128, maxage=30):
        self.cache = OrderedDict()  # { args : (timestamp, result)}
        self.func = func
        self.maxsize = maxsize
        self.maxage = maxage

    def __call__(self, *args):
        if args in self.cache:
            self.cache.move_to_end(args)
            timestamp, result = self.cache[args]
            if time() - timestamp <= self.maxage:
                return result
        result = self.func(*args)
        self.cache[args] = time(), result
        if len(self.cache) > self.maxsize:
            self.cache.popitem(0)
        return result
```

### ChainMap
Класс похожий на словарь используемый для соединения множества словарей в единый вид.
Также если один из составляющих `ChainMap` словарей изменяется, то и сам `ChainMap` обновляет свои значения.
Если несколько в полученной модели несколько ключей ссылающихся на разные объекты, то возвращается тот, чей словарь был записан раньше. 
При этом если операции обновления будут работать только при обращении к ключу первого словаря. Но операции чтения работают по всей цепочке словарей.
#### Примеры кода:
1. Создание `ChainMap`:
```python
from collections import ChainMap
# 1.1. Базовое создание
baseline = {'music': 'bach', 'art': 'rembrandt'}
adjustments = {'art': 'van gogh', 'opera': 'carmen'}
print(list(ChainMap(adjustments, baseline)))
# will print: ['music', 'art', 'opera']
```
2. Примеры использования:
```python
from collections import ChainMap
import os, argparse

# 2.1. Парсинг аргументов c приоритетом
defaults = {'color': 'red', 'user': 'guest'}
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user')
parser.add_argument('-c', '--color')
namespace = parseer.parse_args()
command_line_args = {k: v for k, v in vars(namespace).items() if v is not None}

combined = ChainMap(command_line_args, os.environ, defaults)
print(combined['color'])
print(combined['user'])

# 2.2. Словарь позволяющий обновлять элементы и у других словарей в цепочке
class DeepChainMap(ChainMap):
    def __setitem__(self, key, value):
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value
    def __delitem__(self, key):
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)
```
### deque
Дэк - структура данных, которая сочетает в себе стек и очередь или же просто двусторонняя очередь.
В дэк можно добавлять можно делать вставки в начало и в конец. Операции вставки и удаления с начала/конца выполняются за O(1). 
`deque` является thread-safe и эффективен по использованию памяти.
Если `maxlen` не указан, то дек может расти до любых размеров. Если дек заполнится, то добавление новых элементов с одного конца будет приводить к удалению элементов с другого. Это удобно для потоковых(pipe) алгоритмов.
Они используются например для сохранения истории каких либо действий (например операций undo), которые со временем нужно будет очищать. 
#### Примеры кода:
1. Создание `deque`:
```python
from collections import deque

# 1.1. Базовое создание
d = deque('ghi') # через итерируемый объект
print(d)
# will print: deque(['g', 'h', 'i])
```
2. Базовый функционал:
```python
from collections import deque

# 2.1. Вставка/удаление
d.append('j')
d.appendleft('f')
print(d)
# will print: deque(['f', 'g', 'h', 'i', 'j'])

d.pop()
d.popleft()
print(d)
# will print: deque(['g', 'h', 'i'])

# 2.2. Индексация
print(d[0], d[-1])
# will print: g i

# 2.3. Увеличение дека
d.extend('jkl')
d.extendleft('abc')
print(d)
# will print: deque(['a', 'b', 'c', 'g', 'h', 'i', 'j', 'k', 'l'])

# 2.4. Циклический сдвиг (эффективнее чем у list)
f = deque([1,2,3])
f.rotate(1)
f.rotate(-2)
print(d)
# will print: deque([2, 3, 1]) 
```
3. Примеры использования:
```python
from collections import deque

# 3.1. Реализация pipe
N = 5
pipe = deque(maxlen=N)
pipe.extend('123456789')
print(pipe)
# will print: deque(['5', '6', '7', '8', '9'], maxlen=5)

# 3.2. Эффективное считывание строчек файла с конца
def tail(filename, n=10):
    with open(filename) as f:
        return deque(f, n)
        
# 3.3. Реализация балансировщика по типу Round-Robin
def roundrobin(*iterables):
    iterators = deque(map(iter, iterables))
    while iterators:
        try:
            while True:
                yield next(iterators[0])
                iterators.rotate(-1)
        except StopIteration:
            # Remove an exhausted iterator.
            iterators.popleft()
            
for el in roundrobin('abc','d','ef'):
    print(el, end=' ')
# will print: a d e b f c
```
4. Задачи:
```python
# 4.1. Moving average
def moving_average(iterable, n=3):
    it = iter(iterable)
    d = deque(itertools.islice(it, n-1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / n
```
### collections.abc
Этот модуль предоставляет базовые абстрактные классы которые могут быть использованы как для реализации своих контейнеров, так и для проверки на то, предоставляет ли рассматриваемый класс необходимый интерфейс (например можно ли по его элементам итерироваться). 
Проверка на реализацию интерфейса осуществляется через вызов функций: `issubclass()` `isinstance()`. Некоторые классы могут принадлежать конкретным интерфейсам и без наследования и регистрации. Для этого достаточно реализовать абстрактные методы. 
Собственный класс может наследоваться от этих абстрактных классов и реализовывать их, при наследовании также добавятся миксины сопровождающиеся вместе с наследуемыми классами.
Примеры частоиспользуемых классов:
- Container
- Hashable
- Iterable
- Iterator
- Collection
- Sequence
- Mapping
#### Примеры кода:
1. Базовый функционал:
```python
from collections.abc import Sequence

# 1.1. Наследование
class C(Sequence): # Реализация класса с методами получения элементов по индексам
    def __init__(self): pass
    def __getitem__(self, index):  pass
    def __len__(self):  pass
    def count(self, value): pass
    
# 1.2. Регистрация
class D:
    def __init__(self): pass
    def __getitem__(self, index):  pass
    def __len__(self):  pass
    def count(self, value): pass
    def index(self, value): pass
    
Sequence.register(D)

# 1.3. Проверка на реализацию интерфейсов
print(issubclass(D, Sequence), isinstance(D(), Sequence))
# will print: True True
```
2. Примеры использования:
```python
from collections.abc import Set, Mapping

# 2.1. Реализация словаря, с доступом только для чтения
class ReadOnlyDict(Mapping):
    data: dict()
    def __init__(self, input_dict: dict):
        assert isinstance(input_dict, dict)
        self.data = input_dict
    def __getitem__(self, item):
        return self.data[item]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data.keys())
        
d = ReadOnlyDict({1:1, 2:2})
print(d[2])
# will print: 2
d[3] = 3 # will raise TypeError

# 2.2. Альтернативная реализация set, не требующая хеширования, эффективная по памяти, но не эффективная по скорости
class ListSet(Set):
    def __init__(self, iterable):
        self.elements = lst = [] # Эффективность по памяти заключается в использовании связных списков, которые при добавлении элементов линейно увеличивают выделенную память. Обычные Set и dict увеличивают размер выделенной памяти в 2 или 4 раза при достижении предела.
        for value in iterable:
            if value not in lst:
                lst.append(value)
    def __iter__(self):
        return iter(self.elements)
    def __contains__(self, value):
        return value in self.elements
    def __len__(self):
        return len(self.elements)
        
s1 = ListSet('abcdef')
s2 = ListSet('defghi')
print(list(s1 & s2))
# will print: ['d', 'e', 'f']
```
## Производительность (стандартная библиотека)[^](#functools)

Практически всегда для набора данных мы используем list. Он может содержать различные типы данных в одном контейнере и имеет много метод для обработки. Но если нам нужно обработать однородные данные, например, массив чисел, то можно использовать array.
Array имеет большую скорость и меньше затрат на память, так как явно выделяет нужное количество элементов.


```python
_list = [3, 6, 9, 12]
print(_list)
print(type(_list))
```

```bash
[3, 6, 9, 12]
<class 'list'>
```

```python
import array as arr

_array = arr.array("i", [3, 6, 9, 12])
print(_array)
print(type(_array))
```

```bash
array('i', [3, 6, 9, 12])
<class 'array.array'>
```

Для `array` надо явно указывать тип данных в отличие от list. В остальных случаях, доступ по индексу, итерирование, добавление новых элементов работает так же. 

Сравнение скорости

```python
# Python list with append()
np.mean(timeit.repeat(setup="a = []", stmt="a.append(1.0)", number=1000, repeat=5000)) * 1000
# 0.054 +/- 0.025 msec

# Python array with append()
np.mean(timeit.repeat(setup="import array; a = array.array('f')", stmt="a.append(1.0)", number=1000, repeat=5000)) * 1000
# 0.104 +/- 0.025 msec

# Numpy array with append()
np.mean(timeit.repeat(setup="import numpy as np; a = np.array([])", stmt="np.append(a, [1.0])", number=1000, repeat=5000)) * 1000
# 5.183 +/- 0.950 msec

# Python list using +=
np.mean(timeit.repeat(setup="a = []", stmt="a += [1.0]", number=1000, repeat=5000)) * 1000
# 0.062 +/- 0.021 msec

# Python array using += 
np.mean(timeit.repeat(setup="import array; a = array.array('f')", stmt="a += array.array('f', [1.0]) ", number=1000, repeat=5000)) * 1000
# 0.289 +/- 0.043 msec

# Python list using extend()
np.mean(timeit.repeat(setup="a = []", stmt="a.extend([1.0])", number=1000, repeat=5000)) * 1000
# 0.083 +/- 0.020 msec

# Python array using extend()
np.mean(timeit.repeat(setup="import array; a = array.array('f')", stmt="a.extend([1.0]) ", number=1000, repeat=5000)) * 1000
# 0.169 +/- 0.034
```

__Встроенные функции для вычислительных операций__

Например: abs, min, max, len, sum

```python
# 1. Кастомная реализация sum
sum_value = 0
for n in a_long_list:
    sum_value += n

# CPU times: user 9.91 ms, sys: 2.2 ms, total: 101 ms
# Wall time: 100 ms

# 2. Встроенная функция sum
sum_value = sum(a_long_list)

# CPU times: user 4.74 ms, sys: 277 μs, total: 5.02 ms
# Wall time: 4.79 ms
```

__Генераторы списков__

```python
import random
random.seed(666)
another_long_list = [random.randint(0,500) for i in range(1000000)]

# 1. Создание нового списка с помощью цикла for
even_num = []
for number in another_long_list:
  if number % 2 == 0:
    even_num.append(number)

# CPU times: user 113 ms, sys: 3.55 ms, total: 117 ms
# Wall time: 117 ms

# 2. Создание нового списка с помощью генератора списка
even_num = [number for number in another_long_list if number % 2 == 0]

# CPU times: user 56.6 ms, sys: 3.73 ms, total: 60.3 ms
# Wall time: 64.8 ms
```

__enumerate для цикла__

```python
import random
random.seed(666)
a_short_list = [random.randint(0,500) for i in range(5)]

# 1. Получение индексов с помощью использования длины списка
%%time
for i in range(len(a_short_list)):
  print(f'number {i} is {a_short_list[i]}')

# CPU times: user 189 μs, sys: 123 μs, total: 312 μs
# Wall time: 214 μs

# 2. Получение индексов с помощью enumerate()
for i, number in enumerate(a_short_list):
    print(f'number {i} is {number}')

# CPU times: user 72 μs, sys: 15 μs, total: 87 μs
# Wall time: 90.1 μs
```

__Подсчет уникальных значений__

```python
num_counts = {}
for num in a_long_list:
    if num in num_counts:
        num_counts[num] += 1
    else:
        num_counts[num] = 1

# CPU times: user 448 ms, sys: 1.77 ms, total: 450 ms
# Wall time: 450 ms

num_counts2 = Counter(a_long_list)

# CPU times: user 40.7 ms, sys: 329 μs, total: 41 ms
# Wall time: 41.2 ms
```

__Цикл for внутри функции__

```python
def compute_cubic1(number):
    return number**3

new_list_cubic1 = [compute_cubic1(number) for number in a_long_list]

# CPU times: user 335 ms, sys: 14.3 ms, total: 349 ms
# Wall time: 354 ms

def compute_cubic2():
    return [number**3 for number in a_long_list]

new_list_cubic2 = compute_cubic2()

# CPU times: user 261 ms, sys: 15.7 ms, total: 277 ms
# Wall time: 277 ms
```

__Конкатенация строки через join__

```python
abc = ''
abc = ''.join((abc, 'abc'))
```

Такой метод работает медленее в ~ 1.5 раза

```python
abc = ''
abc += 'abc'
```


__Разделение строки__

Использовать _partition_ вместо split
Когда мы знаем что разделяющий символ в конце, лучше использовать rsplit


```python
>>> from timeit import timeit
>>> timeit('"abcdefghijklmnopqrstuvwxyz,1".split(",", 1)')
0.23717808723449707
# ['abcdefghijklmnopqrstuvwxyz', '1']

>>> timeit('"abcdefghijklmnopqrstuvwxyz,1".rsplit(",", 1)')
0.20203804969787598
# ['abcdefghijklmnopqrstuvwxyz', '1']

>>> timeit('"abcdefghijklmnopqrstuvwxyz,1".partition(",")')
0.11137795448303223
# ('abcdefghijklmnopqrstuvwxyz', ',', '1')

>>> timeit('"abcdefghijklmnopqrstuvwxyz,1".rpartition(",")')
0.10027790069580078
# ('abcdefghijklmnopqrstuvwxyz', ',', '1')
```

Отличие partition от split: разделяющий символ будет в выводе

## Производительность (сторонние модули)[^](#functools)

__ujson__

Быстрый аналог модуля json

Аналоги: orjson, nujson, simplejson

```python
>>> import ujson
>>> ujson.dumps([{"key": "value"}, 81, True])
'[{"key":"value"},81,true]'
>>> ujson.loads("""[{"key": "value"}, 81, true]""")
[{'key': 'value'}, 81, True]
```

__immutables__

Аналог dict, изменяемый через специальные мутации

![](https://github.com/MagicStack/immutables/raw/master/bench.png)

```python
import immutables

map = immutables.Map(a=1, b=2)

print(map['a'])
# will print '1'

print(map.get('z', 100))
# will print '100'

print('z' in map)
# will print 'False'
```

Изменение
```python
map2 = map.set('a', 10)
print(map, map2)
# will print:
#   <immutables.Map({'a': 1, 'b': 2})>
#   <immutables.Map({'a': 10, 'b': 2})>

map3 = map2.delete('b')
print(map, map2, map3)
# will print:
#   <immutables.Map({'a': 1, 'b': 2})>
#   <immutables.Map({'a': 10, 'b': 2})>
#   <immutables.Map({'a': 10})>
```

После изменений нужно вызвать метод finish

```python
with map.mutate() as mm:
    mm['a'] = 100
    del mm['b']
    mm.set('y', 'y')
    map2 = mm.finish()

print(map, map2)
# will print:
#   <immutables.Map({'a': 1, 'b': 2})>
#   <immutables.Map({'a': 100, 'y': 'y'})>
```

## Обработка больших файлов[^](#big-files)

1. Текстовые данные

Если текстовый файл разделен на строки, то всё достаточно просто.
```python
for line in open('really_big_file.dat'):
    process_data(line)
```

Но, при использовании `f.readlines()` весь файл будет выгружаться в ОЗУ
```python
with open("input.txt") as f:
   data = f.readlines()
   for line in data:
       process(line)
```

2. Бинарные файлы
Это обычный вариант обработки файла.

```python
with open('really_big_file.dat') as f:
    process_data(piece)
```

Чем плох этот вариант?
Все данные из файла выгружаются в оперативную память. Причем объём выделенной оперативной памяти в несколько раз превышает размер изначального файла, также только начало обработки может занимать несколько минут.

1) Генераторы
```python
def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


with open('really_big_file.dat') as f:
    for piece in read_in_chunks(f):
        process_data(piece)
```

Функция `read_in_chucks` открывает файл и читает 1024 байта из файла. При следующих вызовах генератор не идёт заново по файлу, а двигается от изначального состояние и выдаёт следующие 1024 байта.

2) Итераторы
```python
f = open('really_big_file.dat')
def read1k():
    return f.read(1024)

for piece in iter(read1k, ''):
    process_data(piece)
```

Аналогичный способ через метод `iter`

3. Многопроцессорность
Если нам не вашен порядок обработки строк мы можем задействовать несколько ядер.

```python
import multiprocessing as mp

pool = mp.Pool(cores)
jobs = []

with open("input.txt") as f:
    for line in f:
        jobs.append(pool.apply_async(process, [line]))

# дождаться окончания всех работ
for job in jobs:
    job.get()

pool.close()
```

Здесь происходит генерация пула обработчиков для каждого ядра перед созданием группы задач.

Хотя вышеперечисленное теперь использует все эти ядра, к сожалению, снова возникают проблемы с памятью. Мы специально используем функцию apply_async, чтобы пул не блокировался во время обработки каждой строки. Однако при этом все данные снова считываются в память; это время сохраняется в виде отдельных строк, связанных с каждым заданием, ожидая обработки в строке. Таким образом, память снова будет переполнена. В идеале метод считывает строку в память только тогда, когда подходит ее очередь на обработку.

```python
import multiprocessing as mp


def process_wrapper(lineID):
    with open("input.txt") as f:
        for i, line in enumerate(f):
            if i != lineID:
                continue
            else:
                process(line)
                break


pool = mp.Pool(cores)
jobs = []

with open("input.txt") as f:
    for ID, line in enumerate(f):
        jobs.append(pool.apply_async(process_wrapper, [ID]))

# дождаться окончания всех работ
for job in jobs:
    job.get()

pool.close()
```

Выше мы изменили функцию, переданную в пул обработчика, чтобы она включала в себя открытие файла, поиск указанной строки, чтение ее в память и последующую обработку. Единственный вход, который теперь сохраняется для каждой порожденной задачи - это номер строки, что предотвращает переполнение памяти. К сожалению, накладные расходы, связанные с необходимостью найти строку путем итеративного чтения файла для каждого задания, являются несостоятельными, поскольку по мере того, как вы углубляетесь в файл, процесс занимает все больше времени. Чтобы избежать этого, мы можем использовать функцию поиска файловых объектов, которая пропускает вас в определенное место в файле. Сочетание с функцией tell, которая возвращает текущее местоположение в файле, дает:

```python
import multiprocessing as mp

def process_wrapper(line_byte):
   with open("input.txt") as f:
       f.seek(line_byte)
       line = f.readline()
       process(line)

pool = mp.Pool(cores)
jobs = []

with open("input.txt") as f:
   next_line_byte = f.tell()
   for line in f:
       jobs.append(pool.apply_async(process_wrapper, [next_line_byte]) )
       next_line_byte = f.tell()

for job in jobs:
   job.get()

pool.close()
```

Используя поиск, мы можем перейти непосредственно к правильной части файла, после чего мы читаем строку в память и обрабатываем ее. Мы должны быть осторожны, чтобы правильно обрабатывать первую и последнюю строки, но в противном случае это будет именно то, что мы излагаем, а именно использование всех ядер для обработки данного файла без переполнения памяти.

Есть накладные расходы, связанные с открытием и закрытием файла для каждой отдельной строки. Если мы обрабатываем несколько строк файла за один раз, мы можем сократить эти операции. Самая большая техническая сложность при этом заключается в том, что при переходе к месту в файле вы, скорее всего, не находитесь в начале строки. Для простого файла, как в этом примере, это просто означает, что вам нужно вызвать readline, который читает следующий символ новой строки. Более сложные типы файлов, вероятно, требуют дополнительного кода, чтобы найти подходящее место для начала / конца чанка.

```python
import multiprocessing as mp
import os


def process_wrapper(chunk_start, chunk_size):
    with open("input.txt") as f:
        f.seek(chunk_start)
        lines = f.read(chunk_size).splitlines()
        for line in lines:
            process(line)


def chunkify(fname, size=1024 * 1024):
    file_end = os.path.getsize(fname)
    with open(fname, "r") as f:
        chunk_end = f.tell()
    while chunk_end:
        chunk_start = chunk_end
        f.seek(size, 1)
        f.readline()
        chunk_end = f.tell()
        yield chunk_start, chunk_end - chunk_start
        if chunk_end > file_end:
            break


pool = mp.Pool(cores)
jobs = []

for chunk_start, chunk_size in chunkify("input.txt"):
    jobs.append(pool.apply_async(process_wrapper, (chunk_start, chunk_size)))

for job in jobs:
    job.get()

pool.close()
```

## Сахарок[^](#sugar)

Двойное сравнивание

New
```python
if min_price <= price <= max_price:
    ...
```
Old
```python
if price >= min_price and price <= max_price:
    ...
```

__Оператор морж__

Объединение операторов присваивания и сравнения

_До_
```python
block = f.read(256)
while block != '':
    process(block)
```

_После_
```python
while (block := f.read(256)) != '':
    process(block)
```

_До_
```python
env_base = os.environ.get("PYTHONUSERBASE", None)
if env_base:
    return env_base
```

_После_

```python
if env_base := os.environ.get("PYTHONUSERBASE", None):
    return env_base
```
_До_
```python
if self._is_special:
    ans = self._check_nans(context=context)
    if ans:
        return ans
```
_После_
```python
if self._is_special and (ans := self._check_nans(context=context)):
    return ans
```

## Что нового в 3.10[^](#3-10)

__Упрощённый оператор Union__

```python
# Function that accepts either `int` or `float`
# Old:
def func(value: Union[int, float]) -> Union[int, float]:
    return value

# New:
def func(value: int | float) -> int | float:
    return value
```


__Обновление контекстного менеджера__


Несколько контекстных менеджеров в одном

```python
with (
    open("somefile.txt") as some_file,
    open("otherfile.txt") as other_file,
):
    ...

from contextlib import redirect_stdout

with (open("somefile.txt", "w") as some_file,
      redirect_stdout(some_file)):
    ...

```

__Pattern Matching__

Конструкция похожая на switch-case, имеющая больше возможностей. Сам pattern-matching взят из функциональных языков (Haskell).

_ - любое выражение

```python
def http_error(status):
    match status:
        case 400:
            return "Bad request"
        case 404:
            return "Not found"
        case 418:
            return "I'm a teapot"
        case _:
            return "Something's wrong with the internet"
```

Можно объединять общие случаи

```python
case 401 | 403 | 404:
    return "Not allowed"
```

```python
def func(person):  # person = (name, age, gender)
    match person:
        case (name, _, "male"):
            print(f"{name} is man.")
        case (name, _, "female"):
            print(f"{name} is woman.")
        case (name, age, gender):
            print(f"{name} is {age} old.")
        
func(("John", 25, "male"))
# John is man.
```

__Улучшение сообщений об ошибках__

Теперь сообщение об ошибке более точно указывает возможную причину ошибки.

До

```python
File "example.py", line 3
    some_other_code = foo()
                    ^
SyntaxError: invalid syntax
```

После

```python
File "example.py", line 1
    expected = {9: 1, 18: 2, 19: 2, 27: 3, 28: 3, 29: 3, 36: 4, 37: 4,
               ^
SyntaxError: '{' was never closed
```

__Новые параметры в датаклассах__

* kw_only (по умолчанию False) - все атрибуты становятся доступны только по ключу.

* slots (по умолчанию False) - автоматически генерирует __slots__, что ускоряет доступ к аттрибутам, но вызывает проблемы с наследованием.

```python
@dataclass(slots=True)
class D:
    x: List = []
    def add(self, element):
        self.x += element
```

__functools.singledispatch__

functools.singledispatch (изпользуется в перегрузке функций) поддерживает Union и UnionType аннотации.

```python
from functools import singledispatch
@singledispatch
def fun(arg, verbose=False):
    if verbose:
        print("Let me just say,", end=" ")
    print(arg)

@fun.register
def _(arg: int | float, verbose=False):
    if verbose:
        print("Strength in numbers, eh?", end=" ")
    print(arg)

from typing import Union
@fun.register
def _(arg: Union[list, set], verbose=False):
    if verbose:
        print("Enumerate this:")
    for i, elem in enumerate(arg):
        print(i, elem)
```


__Улучшение производительности__

По сравнению с 3.10 производительность улучшилась в среднем в 1.22 раза.

* Исключения с "нулевыми затратами". Затраты на ``try`` практически убраны, когда исключение не вызывается.

* Словари не хранят значение хэша, когда все ключи - ASCII строки.

* модуль `re` работает на 10% быстрее.
