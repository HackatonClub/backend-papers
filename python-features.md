# Python. Продвинутый уровень

- [Python. Продвинутый уровень](#python-продвинутый-уровень)
  - [Модуль Functools^](#модуль-functools)
  - [Модуль Collections^](#модуль-collections)
  - [Производительность (стандартная библиотека)^](#производительность-стандартная-библиотека)
  - [Производительность (сторонние модули)^](#производительность-сторонние-модули)
  - [Обработка больших файлов^](#обработка-больших-файлов)
  - [Сахарок^](#сахарок)
  - [Что нового в 3.10^](#что-нового-в-310)

## Модуль Functools[^](#functools)
TODO: описать основные методы, по желанию добавить свои 
* cached_property
* partial
* reduce
* singledispatch
* wraps

## Модуль Collections[^](#collections)

TODO: описать основные структуры, по желанию добавить свои
* namedtuple
* Counter
* defaultdict
* OrderedDict
* ChainMap
* deque
* collections.abc

### namedtuple
Это `tuple`, который каждому элементу кортежа присваивает имя.
К `namedtuple` можно обращаться как к обычному `tuple`, а также к каждому элементу можно обращаться по атрибутам.
Использование `namedtuple`, позволяет писать более читаемый и само-документирующийся код.

#### Примеры использования:
1. Создание объекта `namedtuple`:
```python
# 1.1. Стандартный способ:
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(11, y=22)
# 1.2. Более лаконичный способ:
from typing import NamedTuple
class Point(NamedTuple):
    x: int
    y: int
p = Point(11,y=22)
```
2. Базовый функционал:
```python
# 2.1. Индексация обхектов как в обычном tuple
p[0] + p[1]
# 2.2. Аналогично распаковка
x, y = p
# 2.3. Обращение к элементам по названиям полей
p.x + p.y
# 2.4. __repr__ метод переобределен в следующим виде:
print(p)
# will print: "Point(x=11, y=22)"
```

_TODO_: сделать ссылку на `dataclasses` как логичное продолжение*

### Counter
Это подкласс `dict`, который служит для подсчета хешируемых обхектов (чаще всего внутри итерируемых). Он представляет из себя коллекцию, где элементы хранятся как ключи, а в качестве значений
#### Примеры использования:
1. Создание объекта `Counter`:
```python
from collections import Counter
# 1.1. Создание по итерируемому объекту
cnt1 = Counter(['a', 'b', 'c', 'a', 'b', 'b']
# 1.2. Создание через dict
cnt2 = Counter({'a':2, 'b':3, 'c':1}
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
c = Counter(['a','bb','a','d'])
print(c['c'])
# will print: 0
# 2.2. Может возвратить sorted multiset:
print(sorted(c.elements())) # Без sorted, elements() вернет в том порядке в котором они встретились
# will print: ['a', 'a', 'bb', 'd']
# 2.3. Может вернуть n самых частых/редких элементов
n = 2
print(c.most_common(n))
# will print: [('a', 2), ('bb', 1)]
print(c.most_common()[:-n-1:-1])
# will print: [('d',1), ('bb',1)]
# 2.4. Частотные характеристики можно вычитать
d = Counter('ab')
print(c.subtract(d))
# 2.5. Унарные операции сложения/вычитания
e = Counter(a=2, b=-4)
print(+e) # Складывает с пустым Counter
# will print: Counter({'a': 2})
print(-e) # Вычитает из пустого Counter
# will print: Counter({'b': 4})
```
3. Задачи:
```python
# 3.1. Найти все подстроки-анаграммы в другой подстроке
...
```
_TODO_: Добавить решение задачки 

### defaultdict
Это подкласс `dict`, который переопределяет метод добавления значений в словарь. Когда ключ не был найден в словаре, тогда в словаре создается этот ключ с заданным дефолтным значением.
Данный объект работает быстрее чем словарь образованный методом `setdefault()`.

#### Примеры использования:
1. Создание `defaultdict`:
```python
from collections import defaultdict
# 1.1. Создание счетчика
d = defaultdict(int)
d[3]+=1
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
s = [('a', 1),('b',2),('b',3),('a', 3),('c', 1)]
d = defaultdict(list)
for k,v in s:
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
d = OrderedDict.fromkeys('abcde')
d.move_to_end('b')
print(''.join(d))
# will print: 'acdeb'
d.move_to_end('b', last=False)
print(''.join(d))
# will print: 'bacde'
# 2.2. Имплементация LRU cache ограниченного по времени
class TimeBoundedLRU: # Декоратор
    "LRU Cache that invalidates and refreshes old entries."

    def __init__(self, func, maxsize=128, maxage=30):
        self.cache = OrderedDict()      # { args : (timestamp, result)}
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
Если несколько в полученной модели несколько ключей ссылаюихся на разные объекты, то возвращается тот, чей словарь был записан раньше. 
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
Дэк - структура данных, которая сочетает в себе стек и очередь. `deque` является thread-safe и эффективен по использованию памяти. Операции вставки и удаления с начала/конца выполняются за O(1). 
Если `maxlen` не указан, то дек может расти до любых размеров. Если дек заполнится то добавление новых элементов с одного конца будет приводить к удалению элементов с другого. Это удобно для потоковых алгоритмов. 
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
d = deque('ghi')
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
# 2.4. Циклический сдвиг (эффективвнее чем у list)
f = deque([1,2,3])
f.rotate(1)
f.rotate(-2)
print(d)
# will print: deque([2, 3, 1]) 
```
3. Примеры использования:
```python
from collections import deque
# 3.1. Реализация tail для считывания из файла
def tail(filename, n=10):
    with open(filename) as f:
        return deque(f, n)
# 3.2. Имплементация балансировщика по типу Round-Robin
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
Проверка на реализацию интерфейса осуществляется через вызов функций: `issubclass(()` `isinstance()`. Некоторые классы могут принадлежать конкретным интерфейсам и без наследования и регистрации. Для этого достаточно реализовать абстрактные методы. 
Собственный класс может наследоваться от этих асбтрактных классов и реализовывать их, при наследовании также приобретятся миксины сопровождающиеся вместе с наследуемыми классами.
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
class C(Sequence):
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
from collections.abc import Set
# 2.1. Альтернативная реализация set, не требующая хеширования, эффективная по памяти, но не эффективная по скорости
class ListSet(Set):
    def __init__(self, iterable):
        self.elements = lst = []
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

_TODO_: добавить про array (насколько быстрее list)

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

TODO: чанки и обработка многопроцессорностью

https://dev-gang.ru/article/obrabotka-bolshih-failov-s-ispolzovaniem-python-8btakx0nzr/

https://stackoverflow.com/questions/519633/lazy-method-for-reading-big-file-in-python

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
