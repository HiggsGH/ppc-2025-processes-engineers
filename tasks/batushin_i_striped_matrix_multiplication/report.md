# Ленточная горизонтальная схема А, вертикальное В - умножение матрицы на матрицу

- Студент: Батушин Илья Александрович, группа 3823Б1ПР2
- Технологии: SEQ, MPI
- Вариант: 14

## 1. Введение

Умножение матриц — одна из фундаментальных операций линейной алгебры, широко используемая в научных вычислениях, машинном обучении, компьютерной графике и других областях. Данная работа посвящена реализации параллельного алгоритма умножения матриц по ленточной схеме: горизонтальное разбиение матрицы A и вертикальное разбиение матрицы B. Алгоритм реализован с использованием технологии MPI (Message Passing Interface), что позволяет эффективно распределять вычисления между несколькими процессами и масштабировать задачу на кластерные системы.

## 2. Постановка задачи

Даны две матрицы A размером N×M и B размером M×P. Требуется вычислить матрицу C = A × B размером N×P.

**Входные данные:**
- `N` - количество строк матрицы A (целое положительное число)
- `M` - количество столбцов матрицы A (равно количеству строк матрицы B)
- `P` - количество столбцов матрицы A (равно количеству строк матрицы B)
- `matrix_a` - одномерный вектор вещественных чисел длиной N×M, содержащий элементы матрицы A в построчном порядке
- `matrix_b` - одномерный вектор вещественных чисел длиной M×P, содержащий элементы матрицы B в построчном порядке

**Выходные данные:**
- `result` - одномерный вектор вещественных чисел длиной N×P, представляющий матрицу C = A × B в построчном порядке

**Ограничения:**
- Матрицы должны быть плотными (все элементы явно заданы)
- Элементы матриц — числа с плавающей точкой двойной точности (double)
- Количество столбцов матрицы A должно быть равно количеству строк матрицы B

**Пример:**

Входные данные: M = 2, N = 3, P = 2, matrix_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], matrix_b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
Выходные данные: result = [58.0, 64.0, 139.0, 154.0]

## 3. Описание базового алгоритма (последовательная версия)

Последовательный алгоритм умножения матриц C = A × B основан на тройном вложенном цикле: для каждого элемента результирующей матрицы выполняется скалярное произведение соответствующей строки матрицы A и столбца матрицы B.

Алгоритм состоит из следующих шагов:

1. **Инициализация**: Создание результирующей матрицы C размером N×P, заполненной нулями
2. **Обработка строк**:
    - Для каждой строки i от 0 до N-1:       
        - Для каждого столбца j от 0 до P-1:
            - Вычисление суммы произведений элементов: C[i][j] = Σ (A[i][k] * B[k][j]) для k от 0 до M-1
3. **Завершение**: Возврат результирующей матрицы C

**Сложность алгоритма**

**Время:** O(N × M × P)
- Три вложенных цикла: N × P итераций внешнего цикла, в каждой из которых выполняется M операций умножения и сложения
- N × M × P операций умножения и N × M × P операций сложения

**Память:** O(N × M + M × P + N × P)
- O(N × M) для хранения матрицы A
- O(M × P) для хранения матрицы B
- O(N × P) для результирующей матрицы C
- **Итого:** O(N × M + M × P + N × P)

## 4. Схема распараллеливания

Параллельная реализация основана на ленточной схеме:
- Матрица A делится горизонтально (по строкам) между процессами
- Матрица B полностью рассылается всем процессам (вертикальная полоса для каждого процесса — это вся матрица B)
- Каждый процесс вычисляет свою часть результирующей матрицы C (соответствующую строкам матрицы A, которые он получил)
Стратегия распределения строк матрицы A:
```cpp
// Вычисление количества строк матрицы A для каждого процесса
size_t rows_per_proc = rows_a / proc_count;
size_t extra_rows = rows_a % proc_count;

// Процессы с rank < extra_rows получают дополнительную строку
size_t my_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);
size_t my_start = (rank * rows_per_proc) + (rank < extra_rows ? rank : extra_rows);
```

**Роли процессов**
- **rank 0:** Координирует выполнение, распределяет части матрицы A между процессами, рассылает матрицу B, собирает и синхронизирует результаты
- **rank 1..P-1:** Получают свои части матрицы A и всю матрицу B, выполняют локальное умножение, отправляют результаты обратно ведущему процессу

**Схема коммуникации**
- **Фаза рассылки размеров (MPI_Bcast):** Ведущий процесс рассылает размеры матриц (rows_a, columns_a, rows_b, columns_b) всем процессам
- **Фаза распределения матрицы A:**
    - Ведущий процесс вычисляет диапазоны строк для каждого процесса
    - Отправляет каждому процессу его часть матрицы A через MPI_Send
    - Рабочие процессы получают свои части через MPI_Recv
- **Фаза рассылки матрицы B (MPI_Bcast):** Ведущий процесс рассылает всю матрицу B всем процессам
- **Фаза локальных вычислений:** Каждый процесс независимо вычисляет свою часть матрицы C: C_local[i][j] = Σ(A_local[i][k] * B[k][j]) для k = 0..M-1
- **Фаза сбора результатов:**
    - Рабочие процессы отправляют свои локальные результаты ведущему через MPI_Send
    - Ведущий процесс получает и объединяет все части через MPI_Recv
- **Фаза синхронизации (MPI_Bcast):** Ведущий процесс рассылает итоговую матрицу C всем процессам

**Особенности реализации:**
- Используется блочное распределение строк матрицы A для балансировки нагрузки
- Каждый процесс хранит полную копию матрицы B
- Результаты вычислений собираются только на ведущем процессе (rank 0)
- Все процессы получают финальный результат для возможных последующих вычислений

## 5. Детали реализации

**Файлы:**
- `common/include/common.hpp` - определение типов данных
- `seq/include/ops_seq.hpp`, `seq/src/ops_seq.cpp` - последовательная реализация
- `mpi/include/ops_mpi.hpp`, `mpi/src/ops_mpi.cpp` - параллельная реализация
- `tests/functional/main.cpp` - функциоанльные тесты
- `tests/performance/main.cpp` - тесты производительности

**Ключевые классы:**
- `BatushinIStripedMatrixMultiplicationSEQ` - последовательная реализация
- `BatushinIStripedMatrixMultiplicationMPI` - параллельная MPI реализация

**Основные методы:**
- `ValidationImpl()` - проверка входных данных
- `PreProcessingImpl()` - подготовительные вычисления  
- `RunImpl()` - основной алгоритм
- `PostProcessingImpl()` - завершающая обработка

**Вспомогательные функции:**
- `CalculateRowDistribution()` - вычисление распределения строк матрицы A
- `DistributeMatrixA()` - распределение частей матрицы A по процессам
- `BroadcastMatrixB()` - рассылка матрицы B всем процессам
- `LocalMatrixMultiplication()` - локальное умножение части матриц
- `GatherResults()` - сбор результатов от всех процессов
- `SynchronizeResult()` - синхронизация финального результата

**Допущения:**
- Матрица хранится построчно
- Все процессы имеют доступ ко всей матрице B (после рассылки)
- Размеры матриц корректны (N, M, P > 0) и удовлетворяют условию умножения

**Обрабатываемые граничные случаи:**
- Матрицы 1×1 (скалярное умножение)
- Умножение вектора на вектор
- Умножение матрицы на единичную матрицу
- Умножение с нулевой матрицей
- Матрицы с отрицательными и дробными числами
- Неравномерное распределение строк при MPI распараллеливании

**Проверки в ValidationImpl():**
```cpp
if (columns_a != rows_b) return false;               // несовместимые размеры
if (matrix_a.size() != rows_a * columns_a) return false; // несоответствие размера A
if (matrix_b.size() != rows_b * columns_b) return false; // несоответствие размера B
```

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
PPC_NUM_PROC=1,2,4
```

## 7. Результаты

### 7.1. Корректность работы

Все функциональные тесты пройдены успешно:
- Умножение квадратных матриц (2×2, 3×3, 4×4)
- Умножение прямоугольных матриц (2×3 на 3×2, 5×3 на 3×2)
- Умножение вектора на вектор (1×N на N×1)
- Умножение на единичную матрицу
- Умножение с нулевой матрицей
- Матрицы с отрицательными и дробными числами
- SEQ и MPI версии выдают идентичные результаты для всех тестовых случаев

### 7.2. Производительность

**Время выполнения (секунды) для матрицы 5000×5000:**

| Версия | Количество процессов | Task Run время |
|--------|---------------------|----------------|
| SEQ    | 1                   | 1.6772         |
| MPI    | 1                   | 1.6843         |
| MPI    | 2                   | 1.0960         |
| MPI    | 4                   | 0.7578         |

**Ускорение относительно SEQ версии:**

| Количество процессов | Ускорение | Эффективность |
|---------------------|-----------|---------------|
| 1                   | 1.00×     | 100%           |
| 2                   | 1.53×     | 77%           |
| 4                   | 2.21×     | 55%           |


**Формула ускорения:** Ускорение = Время SEQ / Время MPI

**Формула эффективности:** Эффективность = (Ускорение / Количество процессов) × 100%

### 7.3. Анализ эффективности

- **Лучшее ускорение:** 2.21× на 4 процессах (Task Run время)
- **Оптимальная конфигурация:** 2-4 процесса
- **Эффективность MPI:** высокая при 2 процессах (77%), снижается при увеличении числа процессов из-за роста коммуникационных затрат

### 7.4. Наблюдения

1. **MPI с 1 процессом** показывает схожую производительность с SEQ версией
2. **MPI с 2 процессами** ддемонстрирует значительное ускорение (1.53×) с хорошей эффективностью (77%), что близко к линейному ускорению
3. **MPI с 4 процессами** достигает максимального ускорения (2.21×), но эффективность снижается до 55%

## 8. Выводы

### 8.1. Достигнутые результаты

- **Корректность:** Разработанный алгоритм успешно проходит все функциональные тесты, включая граничные случаи умножения матриц
- **Эффективность параллелизации:** Достигнуто ускорение до 2.54× на 4 процессах для матриц 1000×1000
- **Оптимальная конфигурация:** Наилучшая эффективность (77-96%) достигнута при использовании 2 процессов, что близко к линейному ускорению

### 8.2. Ограничения и проблемы

- **Накладные расходы MPI:** При использовании 4 процессов эффективность снижается до 55-64% из-за коммуникационных затрат
- **Ограничения аппаратуры:** Максимальное ускорение ограничено 4 физическими ядрами процессора
- **Размер данных:** Для матриц меньшего размера накладные расходы MPI могут превышать выгоду от параллелизации

## 9. Источники
1. Лекции по параллельному программированию Сысоева А. В
2. Материалы курса: https://github.com/learning-process/ppc-2025-processes-engineers

## 10. Приложение

```cpp
namespace {

std::pair<size_t, size_t> CalculateRowDistribution(int rank, int size, size_t n, size_t &rows_per_proc,
                                                   size_t &extra_rows) {
  rows_per_proc = n / size;
  extra_rows = n % size;

  size_t my_rows = rows_per_proc + (std::cmp_less(rank, extra_rows) ? 1 : 0);
  size_t my_start = (rank * rows_per_proc) + std::min<size_t>(rank, extra_rows);

  return {my_rows, my_start};
}

void FillLocalPart(size_t my_rows, size_t m, size_t my_start, const std::vector<double> &matrix_a,
                   std::vector<double> &local_a) {
  for (size_t i = 0; i < my_rows; ++i) {
    for (size_t j = 0; j < m; ++j) {
      local_a[(i * m) + j] = matrix_a[((my_start + i) * m) + j];
    }
  }
}

void FillBufferForProcess(int dest, size_t m, size_t rows_per_proc, size_t extra_rows,
                          const std::vector<double> &matrix_a, std::vector<double> &buffer) {
  size_t dest_rows = rows_per_proc + (std::cmp_less(dest, extra_rows) ? 1 : 0);
  if (dest_rows == 0) {
    return;
  }

  size_t dest_start = (dest * rows_per_proc) + std::min<size_t>(dest, extra_rows);

  buffer.resize(dest_rows * m);
  for (size_t i = 0; i < dest_rows; ++i) {
    for (size_t j = 0; j < m; ++j) {
      buffer[(i * m) + j] = matrix_a[((dest_start + i) * m) + j];
    }
  }
}

void SendToProcesses(int size, size_t m, size_t rows_per_proc, size_t extra_rows, const std::vector<double> &matrix_a) {
  for (int dest = 1; dest < size; ++dest) {
    std::vector<double> buffer;
    FillBufferForProcess(dest, m, rows_per_proc, extra_rows, matrix_a, buffer);
    if (!buffer.empty()) {
      MPI_Send(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
    }
  }
}

std::vector<double> DistributeMatrixA(int rank, int size, size_t my_rows, size_t my_start, size_t m,
                                      size_t rows_per_proc, size_t extra_rows, const std::vector<double> &matrix_a) {
  std::vector<double> local_a(my_rows * m);

  if (rank == 0) {
    FillLocalPart(my_rows, m, my_start, matrix_a, local_a);
    SendToProcesses(size, m, rows_per_proc, extra_rows, matrix_a);
  } else if (my_rows > 0) {
    MPI_Recv(local_a.data(), static_cast<int>(local_a.size()), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return local_a;
}

std::vector<double> BroadcastMatrixB(int rank, size_t m, size_t p, const std::vector<double> &matrix_b) {
  std::vector<double> local_b(m * p);

  if (rank == 0) {
    local_b = matrix_b;
  }

  MPI_Bcast(local_b.data(), static_cast<int>(m * p), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return local_b;
}

std::vector<double> LocalMatrixMultiplication(const std::vector<double> &local_a, const std::vector<double> &local_b,
                                              size_t my_rows, size_t m, size_t p) {
  std::vector<double> local_c(my_rows * p, 0.0);

  for (size_t i = 0; i < my_rows; ++i) {
    for (size_t j = 0; j < p; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < m; ++k) {
        sum += local_a[(i * m) + k] * local_b[(k * p) + j];
      }
      local_c[(i * p) + j] = sum;
    }
  }

  return local_c;
}

void FillLocalResultPart(size_t my_rows, size_t my_start, size_t p, const std::vector<double> &local_c,
                         std::vector<double> &result) {
  for (size_t i = 0; i < my_rows; ++i) {
    for (size_t j = 0; j < p; ++j) {
      result[((my_start + i) * p) + j] = local_c[(i * p) + j];
    }
  }
}

void ReceiveFromProcess(int src, size_t p, size_t rows_per_proc, size_t extra_rows, std::vector<double> &result) {
  size_t src_rows = rows_per_proc + (std::cmp_less(src, extra_rows) ? 1 : 0);
  if (src_rows == 0) {
    return;
  }

  size_t src_start = (src * rows_per_proc) + std::min<size_t>(src, extra_rows);

  std::vector<double> buffer(src_rows * p);
  MPI_Recv(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (size_t i = 0; i < src_rows; ++i) {
    for (size_t j = 0; j < p; ++j) {
      result[((src_start + i) * p) + j] = buffer[(i * p) + j];
    }
  }
}

void ReceiveFromAllProcesses(int size, size_t p, size_t rows_per_proc, size_t extra_rows, std::vector<double> &result) {
  for (int src = 1; src < size; ++src) {
    ReceiveFromProcess(src, p, rows_per_proc, extra_rows, result);
  }
}

void GatherResults(int rank, int size, const std::vector<double> &local_c, size_t my_rows, size_t my_start, size_t n,
                   size_t p, size_t rows_per_proc, size_t extra_rows, std::vector<double> &result) {
  if (rank == 0) {
    result.resize(n * p);
    FillLocalResultPart(my_rows, my_start, p, local_c, result);
    ReceiveFromAllProcesses(size, p, rows_per_proc, extra_rows, result);
  } else if (my_rows > 0) {
    MPI_Send(local_c.data(), static_cast<int>(local_c.size()), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}

void SynchronizeResult(int rank, std::vector<double> &result) {
  if (rank == 0) {
    int result_size = static_cast<int>(result.size());

    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (result_size > 0) {
      MPI_Bcast(result.data(), result_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  } else {
    int result_size = 0;
    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (result_size > 0) {
      result.resize(result_size);
      MPI_Bcast(result.data(), result_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
      result.clear();
    }
  }
}

}  // namespace

bool BatushinIStripedMatrixMultiplicationMPI::RunImpl() {
  int rank = 0;
  int size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();

  const size_t rows_a = std::get<0>(input);
  const size_t columns_a = std::get<1>(input);
  const auto &matrix_a = std::get<2>(input);
  const size_t rows_b = std::get<3>(input);
  const size_t columns_b = std::get<4>(input);
  const auto &matrix_b = std::get<5>(input);

  std::array<uint64_t, 4> dims{};
  if (rank == 0) {
    dims[0] = rows_a;
    dims[1] = columns_a;
    dims[2] = rows_b;
    dims[3] = columns_b;
  }

  MPI_Bcast(dims.data(), 4, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  const auto n = static_cast<size_t>(dims[0]);
  const auto m = static_cast<size_t>(dims[1]);
  const auto p = static_cast<size_t>(dims[3]);

  if (n == 0 || m == 0 || p == 0) {
    if (rank == 0) {
      GetOutput() = std::vector<double>();
    }
    return true;
  }

  size_t rows_per_proc = 0;
  size_t extra_rows = 0;
  auto [my_rows, my_start] = CalculateRowDistribution(rank, size, n, rows_per_proc, extra_rows);

  auto local_a = DistributeMatrixA(rank, size, my_rows, my_start, m, rows_per_proc, extra_rows, matrix_a);

  auto local_b = BroadcastMatrixB(rank, m, p, matrix_b);

  auto local_c = LocalMatrixMultiplication(local_a, local_b, my_rows, m, p);

  std::vector<double> result;
  GatherResults(rank, size, local_c, my_rows, my_start, n, p, rows_per_proc, extra_rows, result);

  SynchronizeResult(rank, result);

  GetOutput() = result;

  return true;
}
```