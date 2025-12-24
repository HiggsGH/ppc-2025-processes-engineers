# Быстрая сортировка с простым слиянием

- Студент: Батушин Илья Александрович, группа 3823Б1ПР2
- Технологии: SEQ, MPI
- Вариант: 14

## 1. Введение

Сортировка данных является одной из фундаментальных задач, находящей применение в базах данных, алгоритмах поиска, машинном обучении и многих других областях. В условиях роста объёмов обрабатываемых данных параллельные алгоритмы сортировки становятся всё более актуальными. Данная работа посвящена разработке и реализации параллельного алгоритма сортировки, сочетающего эффективную локальную быструю сортировку с последующим централизованным слиянием отсортированных блоков. Реализация выполнена с использованием технологии MPI (Message Passing Interface), обеспечивающей масштабируемость на распределённых системах.

## 2. Постановка задачи

Дан одномерный массив целых чисел. Требуется отсортировать массив по возрастанию.

**Входные данные:**
- `input` - одномерный вектор целых чисел произвольной длины

**Выходные данные:**
- `output` - отсортированный по возрастанию вектор целых чисел

**Ограничения:**
- Элементы массива — 32-битные целые числа (`int`)
- Размер массива может быть равен нулю (пустой массив)
- Алгоритм должен корректно обрабатывать массивы с дубликатами, отрицательными числами и уже отсортированные последовательности

**Пример:**

Входные данные: `input` = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0]
Выходные данные: `output` = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

## 3. Описание базового алгоритма (последовательная версия)

Последовательная реализация основана на итеративной быстрой сортировке с выбором опорного элемента из середины подмассива.

Алгоритм состоит из следующих шагов:

1. **Инициализация:** Создание стека для хранения сегментов, требующих сортировки
2. **Обработка сегментов:** Пока стек не пуст:
   - Извлечение сегмента из стека
   - Пропуск сегментов длины ≤ 1
   - Выбор опорного элемента из середины сегмента и перемещение его в начало
   - Разбиение массива по схеме Хоара
   - Добавление левой и правой частей в стек
3. **Завершение:** Возврат отсортированного массива

**Сложность алгоритма**

**Время:** O(N log N) в среднем, O(N²) в худшем случае

**Память:** O(log N) для стека (итеративная реализация)

## 4. Схема распараллеливания

Параллельная реализация использует гибридный подход:
- Разделение данных между процессами по блочному принципу с учётом остатка
- Локальная сортировка каждого процесса независимо
- Централизованное слияние отсортированных блоков в процессе с rank 0

**Распределение данных:**
```cpp
base = total / size                    // Базовое количество элементов на процесс
extra = total % size                   // Остаточные элементы для распределения

// Процессы с rank < extra получают дополнительный элемент
start = (rank * base) + std::min(rank, extra)
count = base + (rank < extra ? 1 : 0)
```

**Роли процессов**
- **rank 0:** Координирует выполнение, распределяет данные, собирает и сливает результаты, рассылает финальный ответ
- **rank 1..P-1:** Выполняют локальную сортировку своих блоков и отправляют результаты rank 0

**Схема коммуникации**
- **Фаза распределения:** Rank 0 отправляет данные другим процессам
- **Фаза вычислений:** Все процессы независимо сортируют свои блоки
- **Фаза сбора:** Все процессы отправляют отсортированные блоки rank 0
- **Фаза слияния:** Rank 0 последовательно сливает все блоки в один отсортированный массив
- **Фаза синхронизации:** Rank 0 рассылает финальный результат всем процессам

## 5. Детали реализации

**Файлы:**
- `common/include/common.hpp` - определение типов данных
- `seq/include/ops_seq.hpp`, `seq/src/ops_seq.cpp` - последовательная реализация
- `mpi/include/ops_mpi.hpp`, `mpi/src/ops_mpi.cpp` - параллельная реализация
- `tests/functional/main.cpp` - функциоанльные тесты
- `tests/performance/main.cpp` - тесты производительности

**Ключевые классы:**
- `BatushinIQuickSortWithSimpleMergeSEQ` - последовательная реализация
- `BatushinIQuickSortWithSimpleMergeMPI` - параллельная MPI реализация

**Основные методы:**
- `ValidationImpl()` - проверка входных данных
- `PreProcessingImpl()` - подготовительные вычисления  
- `RunImpl()` - основной алгоритм
- `PostProcessingImpl()` - завершающая обработка

**Вспомогательные функции:**
- `ComputeLocalRange()` - вычисление диапазона элементов для процесса
- `DistributeData()` - распределение исходных данных между процессами
- `IterativeQuickSort()` - итеративная быстрая сортировка с оптимизацией
- `SortSmallArray()` - сортировка вставками для малых подмассивов
- `GatherAndMerge()` - сбор и последовательное слияние отсортированных блоков
- `BroadcastResult()` - рассылка финального результата всем процессам

**Допущения:**
- Все процессы имеют доступ ко всему исходному массиву (в MPI версии)
- Размер массива может быть произвольным (включая 0)
- Элементы массива — 32-битные целые числа

**Обрабатываемые граничные случаи:**
- Пустой массив
- Массив из одного элемента
- Массивы с дубликатами
- Массивы с отрицательными числами
- Уже отсортированные массивы
- Обратно отсортированные массивы
- Неравномерное распределение элементов между процессами

## 6. Экспериментальное окружение

### 6.1 Аппаратное обеспечение/ОС:

- **Процессор:** Intel Core i5-1135G7
- **Ядра:** 4 физических ядра (8 логических потоков)  
- **ОЗУ:** 8 ГБ DDR4
- **ОС:** WSL Ubuntu 24.04.3 LTS (Linux kernel 5.15)

### 6.2 Программный инструментарий

- **Компилятор:** g++ 13.3.0
- **Тип сборки:** Release
- **Стандарт C++:** C++20
- **MPI:** OpenMPI 4.1.6

### 6.3 Тестовое окружение

```bash
PPC_NUM_PROC=1,2,3,4,5,6,7,8
```
**Размер тестового массива:** 5 000 000 элементов

## 7. Результаты

### 7.1. Корректность работы

Все функциональные тесты пройдены успешно:
- Пустой массив
- Массивы различных размеров (от 1 до 10+ элементов)
- Случаи с одним элементом
- Массивы с отрицательными числами
- Массивы с дубликатами
- Уже отсортированные и обратно отсортированные массивы
- Неравномерное распределение элементов между процессами
- SEQ и MPI версии выдают идентичные результаты для всех тестовых случаев

### 7.2. Производительность

**Время выполнения (секунды) для матрицы 5000×5000:**

| Версия | Количество процессов | Task Run время |
|--------|---------------------|----------------|
| SEQ    | 1                   | 0.2066         |
| MPI    | 1                   | 0.3681         |
| MPI    | 2                   | 0.2113         |
| MPI    | 3                   | 0.1664         |
| MPI    | 4                   | 0.1535         |
| MPI    | 5                   | 0.3833         |
| MPI    | 6                   | 0.3695         |
| MPI    | 7                   | 0.3223         |
| MPI    | 8                   | 0.3501         |

**Ускорение относительно SEQ версии:**

| Количество процессов | Ускорение | Эффективность |
|---------------------|-----------|---------------|
| 1                   | 0.56×     | 56%           |
| 2                   | 0.98×     | 49%           |
| 3                   | 1.24×     | 41%           |
| 4                   | 1.35×     | 34%           |
| 5                   | 0.54×     | 11%           |
| 6                   | 0.56×     | 9%           |
| 7                   | 0.64×     | 9%           |
| 8                   | 0.59×     | 7%           |


**Формула ускорения:** Ускорение = Время SEQ / Время MPI

**Формула эффективности:** Эффективность = (Ускорение / Количество процессов) × 100%

### 7.3. Анализ эффективности

- **Лучшее ускорение:** 1.35× достигнуто при использовании 4 процессов
- **Оптимальная конфигурация:** 3-4 процесса
- **Эффективность MPI:** снижается при увеличении числа процессов сверх 4 из-за oversubscribe

### 7.4. Наблюдения

1. **MPI с 1 процессом** медленнее SEQ из-за накладных расходов на коммуникацию
2. **MPI с 2 процессами** почти достигает производительности SEQ (0.98×)
3. **MPI с 3-4 процессами** демонстрирует реальное ускорение (1.24–1.35×)
4. **MPI с 5+ процессами** показывает снижение производительности из-за конкуренции за ресурсы на 4-ядерной системе

## 8. Выводы

### 8.1. Достигнутые результаты

- **Корректность:** Разработанный алгоритм успешно прошёл все функциональные тесты, включая граничные случаи
- **Эффективность параллелизации:** Достигнуто ускорение до 1.35× на 4 процессах для массива из 5 млн элементов
- **Оптимальная конфигурация:** Наилучшие результаты получены при использовании 3-4 процессов, что соответствует числу физических ядер процессора

### 8.2. Ограничения и проблемы

- **Накладные расходы MPI:** ППри малом числе процессов (1-2) накладные расходы превышают выгоду от параллелизации
- **Ограничения аппаратуры:** Максимальное ускорение ограничено 4 физическими ядрами процессора
- **Oversubscribe:** При использовании более 4 процессов наблюдается деградация производительности из-за конкуренции за ресурсы
- **Размер данных:** Для матриц меньшего размера накладные расходы MPI могут превышать выгоду от параллелизации

## 9. Источники
1. Лекции по параллельному программированию Сысоева А. В
2. Материалы курса: https://github.com/learning-process/ppc-2025-processes-engineers

## 10. Приложение

```cpp
namespace {

void SortSmallArray(std::vector<int> &data, int begin, int end) {
  for (int i = begin + 1; i <= end; ++i) {
    int key = data[i];
    int j = i - 1;
    while (j >= begin && data[j] > key) {
      data[j + 1] = data[j];
      --j;
    }
    data[j + 1] = key;
  }
}

void PartitionArray(std::vector<int> &data, int begin, int end, int &left_end, int &right_begin) {
  if (begin >= end) {
    left_end = begin - 1;
    right_begin = end + 1;
    return;
  }

  int mid = begin + ((end - begin) / 2);
  std::swap(data[mid], data[begin]);
  int pivot = data[begin];

  int i = begin;
  int j = end;
  while (i <= j) {
    while (data[i] < pivot) {
      ++i;
    }
    while (data[j] > pivot) {
      --j;
    }
    if (i <= j) {
      std::swap(data[i], data[j]);
      ++i;
      --j;
    }
  }
  left_end = j;
  right_begin = i;
}

void IterativeQuickSort(std::vector<int> &data) {
  if (data.size() <= 1) {
    return;
  }

  const int threshold = 16;
  struct Segment {
    int begin, end;
  };
  std::stack<Segment> stk;
  stk.push({0, static_cast<int>(data.size() - 1)});

  while (!stk.empty()) {
    Segment seg = stk.top();
    stk.pop();
    if (seg.begin >= seg.end) {
      continue;
    }

    if (seg.end - seg.begin + 1 <= threshold) {
      SortSmallArray(data, seg.begin, seg.end);
      continue;
    }

    int left_end = 0;
    int right_begin = 0;
    PartitionArray(data, seg.begin, seg.end, left_end, right_begin);

    if (seg.begin <= left_end) {
      stk.push({seg.begin, left_end});
    }
    if (right_begin <= seg.end) {
      stk.push({right_begin, seg.end});
    }
  }
}

std::pair<int, int> ComputeLocalRange(int rank, int size, int total) {
  int base = total / size;
  int extra = total % size;
  int start = (rank * base) + std::min(rank, extra);
  int end = start + base + (rank < extra ? 1 : 0) - 1;
  return std::make_pair(start, end);
}

void DistributeData(int rank, int size, const std::vector<int> &global_input, std::vector<int> &local_data) {
  auto range = ComputeLocalRange(rank, size, static_cast<int>(global_input.size()));
  int local_start = range.first;
  int local_end = range.second;
  int local_count = (local_start <= local_end) ? (local_end - local_start + 1) : 0;

  if (rank == 0) {
    if (local_count > 0) {
      local_data.resize(local_count);
      std::copy(global_input.begin() + local_start, global_input.begin() + local_start + local_count,
                local_data.begin());
    }
    for (int proc_rank = 1; proc_rank < size; ++proc_rank) {
      auto proc_range = ComputeLocalRange(proc_rank, size, static_cast<int>(global_input.size()));
      int s = proc_range.first;
      int e = proc_range.second;
      int cnt = (s <= e) ? (e - s + 1) : 0;
      if (cnt > 0) {
        std::vector<int> send_buffer(global_input.begin() + s, global_input.begin() + s + cnt);
        MPI_Send(send_buffer.data(), cnt, MPI_INT, proc_rank, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    if (local_count > 0) {
      local_data.resize(local_count);
      MPI_Recv(local_data.data(), local_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

void BroadcastResult(int rank, std::vector<int> &result, std::vector<int> &output) {
  int result_size = 0;
  if (rank == 0) {
    output = std::move(result);
    result_size = static_cast<int>(output.size());
    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (result_size > 0) {
      MPI_Bcast(output.data(), result_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (result_size > 0) {
      output.resize(result_size);
      MPI_Bcast(output.data(), result_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
}

std::vector<int> GatherAndMerge(int rank, int size, const std::vector<int> &local_data) {
  if (rank != 0) {
    int count = static_cast<int>(local_data.size());
    MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (count > 0) {
      MPI_Send(local_data.data(), count, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    return {};
  }

  std::vector<std::vector<int>> all_blocks;
  all_blocks.reserve(size);

  if (!local_data.empty()) {
    all_blocks.emplace_back(local_data.begin(), local_data.end());
  } else {
    all_blocks.emplace_back();
  }

  for (int src = 1; src < size; ++src) {
    int count = 0;
    MPI_Recv(&count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    all_blocks.emplace_back();
    if (count > 0) {
      all_blocks.back().resize(count);
      MPI_Recv(all_blocks.back().data(), count, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  std::vector<int> result;
  for (const auto &block : all_blocks) {
    if (block.empty()) {
      continue;
    }
    if (result.empty()) {
      result.assign(block.begin(), block.end());
    } else {
      std::vector<int> merged;
      merged.reserve(result.size() + block.size());
      
      // Ручное слияние вместо std::ranges::merge
      auto it1 = result.begin();
      auto it2 = block.begin();
      while (it1 != result.end() && it2 != block.end()) {
        if (*it1 <= *it2) {
          merged.push_back(*it1);
          ++it1;
        } else {
          merged.push_back(*it2);
          ++it2;
        }
      }
      while (it1 != result.end()) {
        merged.push_back(*it1);
        ++it1;
      }
      while (it2 != block.end()) {
        merged.push_back(*it2);
        ++it2;
      }
      
      result = std::move(merged);
    }
  }
  return result;
}

}  // namespace

bool BatushinIQuickSortWithSimpleMergeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &global_input = GetInput();
  int total_size = static_cast<int>(global_input.size());

  if (total_size == 0) {
    GetOutput().clear();
    int dummy = 0;
    MPI_Bcast(&dummy, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return true;
  }

  std::vector<int> local_data;
  DistributeData(rank, size, global_input, local_data);
  
  if (!local_data.empty()) {
    IterativeQuickSort(local_data);
  }

  std::vector<int> result = GatherAndMerge(rank, size, local_data);
  BroadcastResult(rank, result, GetOutput());

  return true;
}
```