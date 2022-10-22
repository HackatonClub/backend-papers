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

#### Примеры:
1. Базовое создание `namedtuple`:
```python
 
```
2. Более лаконичный способ создания:
```python
```
3. 


## Производительность (стандартная библиотека)[^](#functools)

TODO: добавить про array (насколько быстрее list)

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