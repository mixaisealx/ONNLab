# Obfuscated Neural Network Lab

В этом репозитории представлен проект `ONNLab`. Этот проект, по сути, - сборник экспериментов, построенный на базе общей структуры. Все эксперименты имеют расширение `.cpp`, реализации структурных элементов и др. - `.h`. 
*В некотором смысле, концепция проекта вдохновлена устройством большинства лабораторий оптики - как можно большая гибкость для самых разных экспериментов.*

Основной компилируемый файл - `onnlab/onnlab.cpp`.

`exp_m1ReLU_svsg1` - реализация задачи из `exp_ReLU_svsg1` на `m1ReLU`. 
Из ключевых особенностей - сеть сходится, при этом, на промежуточном (по сути, искусственном препятствии) "сжимающем" слое для сходимости потребовалось на 1 нейрон меньше, чем в `exp_ReLU_svsg1` на обычном ReLU, что составляет около 17%, и это не предел, так как поставленная в `exp_m1ReLU_svsg1` задача, вероятно, не является "самой лучшей" для `m1ReLU`.

`exp_m1ReLU_svsg3` - реализация задачи из `exp_ReLU_svsg1` (и `exp_m1ReLU_svsg1`) на `m1ReLU` без использования искусственного слоя, но с применением [потенциально] более качественной эвристики (средняя сходимость сетей должна возрасти).

Некоторые элементы выглядят недоделанными. По скольку проект зачастую сильно меняется, что-то может быть удалено или переделано, а в других файлах остаться незаконченные элементы. Я сохраняю всю историю в GIT, но не выгружаю в Public проекты GitHub.