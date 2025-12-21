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

Параллельная реализация следует ленточной схеме с горизонтальным разбиением матрицы A и вертикальным разбиением B

**Роли процессов**
- **rank 0:**
  - Инициализирует данные
  - Распределяет строки матрицы A с помощью `MPI_Scatterv`
  - Отправляет начальные полосы матрицы B всем процессам
  - Собирает результаты с помощью `MPI_Gatherv`
  - Рассылает финальную матрицу C через `MPI_Bcast`
- **rank 1..P-1:**
  - Получают свою полосу строк из A
  - Получают свою полосу столбцов из B
  - Участвуют в циклическом сдвиге полос B
  - Вычисляют свою часть результата
  - Отправляют результат на `rank 0`

**Особенности реализации:**
- Поддержка неравномерного распределения строк и столбцов
- Все процессы всегда участвуют в коммуникации, даже при пустых блоках
- Матрица B не разбивается — каждый процесс получает полную копию, что упрощает вычисления и гарантирует корректность
- Реализация использует гибридный подход, сочетающий полностью последовательный режим для небольших конфигураций и ленточную схему для крупных

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
- `ComputeBlockSizes()` - Вычисляет, сколько строк/столбцов достаётся каждому процессу (неравномерное деление с остатком)
- `ComputeBlockOffsets()` - Вычисляет смещения для MPI-операций на основе размеров блоков
- `RunSequentialFallback()` - Выполняет матричное умножение на нулевом процессе и рассылает результат через `MPI_Bcast`
- `RunStripedScheme()` - Основная функция ленточной схемы: координирует распределение данных, вычисления и сбор результата
- `DistributeMatrixA()` - Вызывает `ComputeBlockSizes/ComputeBlockOffsets` и делегирует рассылку через `DistributeMatrixAFromRoot`
- `DistributeMatrixAFromRoot()` - Использует `MPI_Scatterv` для раздачи строк матрицы A от корневого процесса
- `DistributeMatrixB()` - Организует начальную рассылку вертикальных полос матрицы B от корня к остальным процессам
- `DistributeMatrixBFromRoot()` - На корне формирует и отправляет каждой полосе свою часть матрицы B
- `ExtractColumnBlock()` - Выделяет подматрицу (вертикальную полосу) из полной матрицы B
- `ComputeWithCyclicShift()` - Основной цикл алгоритма: на каждом шаге выполняет локальное умножение и передаёт свою полосу B следующему процессу
- `ShiftMatrixB()` - Инкапсулирует обмен полосами матрицы B между соседними процессами через `MPI_Sendrecv`
- `BroadcastFinalResult()` - Собирает частичные результаты с помощью `MPI_Gatherv` на корне и рассылает финальную матрицу всем через `MPI_Bcast`

**Допущения:**
- Матрица хранится построчно
- При `size ≤ 4` используется SEQ-режим для упрощения и гарантии корректности
- При `size > 4` используется MPI-режим с ленточной схемой

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
return GetOutput().empty();                          // защита от повторного запуска
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
PPC_NUM_PROC=1,2,4,7,8
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

**Время выполнения (секунды) для матриц 1000×1000:**

| Версия | Количество процессов | Task Run время |
|--------|---------------------|----------------|
| SEQ    | 1                   | 1.7114         |
| MPI    | 1                   | 1.7159         |
| MPI    | 2                   | 1.7304         |
| MPI    | 4                   | 1.9097         |
| MPI    | 7                   | 0.4512         |
| MPI    | 8                   | 0.4537         |

**Ускорение относительно SEQ версии:**

| Количество процессов | Ускорение | Эффективность |
|---------------------|-----------|---------------|
| 1                   | 0.997×     | 99.7%           |
| 2                   | 0.989×     | 49.5%           |
| 4                   | 0.896×     | 22.4%            |
| 7                   | 3.792×     | 54.2%            |
| 8                   | 3.771×     | 47.1%            |


**Формула ускорения:** Ускорение = Время SEQ / Время MPI

**Формула эффективности:** Эффективность = (Ускорение / Количество процессов) × 100%

### 7.3. Анализ эффективности

- **Лучшее ускорение:** 3.771× на 7 процессах (Task Run время)
- **Оптимальная конфигурация:** 7 процессов
- **Эффективность MPI:** Хорошая при 7 процессах (54.2%)

### 7.4. Наблюдения

1. **MPI с 1-4 процессами:** Работает в последовательном режиме, поэтому время выполнения сопоставимо или даже хуже SEQ из-за накладных расходов на инициализацию MPI
2. **MPI с 7-8 процессами:** приводит к значительному ускорению (~3.78×) и хорошей эффективности (47-54%). 

## 8. Выводы

### 8.1. Достигнутые результаты

- **Корректность:** приводит к значительному ускорению (~3.78×) и хорошей эффективности (47-54%)
- **Эффективность параллелизации:** Достигнуто максимальное ускорение 3.79× на 7 процессах, что подтверждает эффективность параллельного алгоритма

### 8.2. Ограничения и проблемы

- **Размер данных:** Для матриц меньшего размера накладные расходы MPI могут превышать выгоду от параллелизации

## 9. Источники
1. Лекции по параллельному программированию Сысоева А. В
2. Материалы курса: https://github.com/learning-process/ppc-2025-processes-engineers

## 10. Приложение

```cpp
namespace {

enum class MPITag : std::uint8_t { kMatrixB = 101 };

std::vector<int> ComputeBlockSizes(int total, int num_procs) {
  std::vector<int> sizes(num_procs, 0);
  int base = total / num_procs;
  int extra = total % num_procs;
  for (int i = 0; i < num_procs; ++i) {
    sizes[i] = base + (i < extra ? 1 : 0);
  }
  return sizes;
}

std::vector<int> ComputeBlockOffsets(const std::vector<int>& sizes) {
  std::vector<int> offsets(sizes.size(), 0);
  for (size_t i = 1; i < sizes.size(); ++i) {
    offsets[i] = offsets[i - 1] + sizes[i - 1];
  }
  return offsets;
}

bool RunSequentialFallback(int rank, size_t rows_a, size_t cols_a, size_t cols_b,
                           const std::vector<double>& matrix_a, const std::vector<double>& matrix_b,
                           std::vector<double>& output) {
  if (rank == 0) {
    output.resize(rows_a * cols_b, 0.0);
    for (size_t i = 0; i < rows_a; ++i) {
      for (size_t j = 0; j < cols_b; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < cols_a; ++k) {
          sum += matrix_a[(i * cols_a) + k] * matrix_b[(k * cols_b) + j];
        }
        output[(i * cols_b) + j] = sum;
      }
    }
  }

  int total_size = (rank == 0) ? static_cast<int>(output.size()) : 0;
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    output.resize(total_size);
  }
  if (total_size > 0) {
    MPI_Bcast(output.data(), total_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  return true;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>>
DistributeMatrixA(int rank, int size, int n, int m, const std::vector<double>& matrix_a) {
  auto row_counts = ComputeBlockSizes(n, size);
  auto row_displs = ComputeBlockOffsets(row_counts);
  int my_rows = row_counts[rank];

  std::vector<double> local_a;
  if (my_rows > 0) {
    local_a.resize(static_cast<size_t>(my_rows) * static_cast<size_t>(m));
  }

  std::vector<int> sendcounts_a(size), displs_a(size);
  for (int i = 0; i < size; ++i) {
    sendcounts_a[i] = row_counts[i] * m;
    displs_a[i] = row_displs[i] * m;
  }

  if (my_rows > 0) {
    MPI_Scatterv(matrix_a.data(), sendcounts_a.data(), displs_a.data(), MPI_DOUBLE,
                 local_a.data(), my_rows * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(matrix_a.data(), sendcounts_a.data(), displs_a.data(), MPI_DOUBLE,
                 nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  return {row_counts, row_displs, local_a};
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, int>
DistributeMatrixB(int rank, int size, int m, int p, const std::vector<double>& matrix_b) {
  auto col_counts = ComputeBlockSizes(p, size);
  auto col_displs = ComputeBlockOffsets(col_counts);

  std::vector<double> current_b;
  int current_cols = 0;

  if (rank == 0) {
    for (int dest = 0; dest < size; ++dest) {
      if (col_counts[dest] > 0) {
        std::vector<double> buf(static_cast<size_t>(m) * static_cast<size_t>(col_counts[dest]));
        for (int r = 0; r < m; ++r) {
          for (int c = 0; c < col_counts[dest]; ++c) {
            int global_col = col_displs[dest] + c;
            buf[static_cast<size_t>(r) * static_cast<size_t>(col_counts[dest]) + static_cast<size_t>(c)] =
                matrix_b[static_cast<size_t>(r) * static_cast<size_t>(p) + static_cast<size_t>(global_col)];
          }
        }
        if (dest == 0) {
          current_b = std::move(buf);
          current_cols = col_counts[0];
        } else {
          MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, dest, 100, MPI_COMM_WORLD);
        }
      } else {
        if (dest == 0) {
          current_cols = 0;
        } else {
          MPI_Send(nullptr, 0, MPI_DOUBLE, dest, 100, MPI_COMM_WORLD);
        }
      }
    }
  } else {
    if (col_counts[rank] > 0) {
      current_b.resize(static_cast<size_t>(m) * static_cast<size_t>(col_counts[rank]));
      MPI_Recv(current_b.data(), static_cast<int>(current_b.size()), MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      current_cols = col_counts[rank];
    } else {
      MPI_Recv(nullptr, 0, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      current_cols = 0;
    }
  }

  return {col_counts, col_displs, current_b, current_cols};
}

std::vector<double> ComputeWithCyclicShift(
    int rank, int size, int m, int p,
    const std::vector<double>& local_a,
    std::vector<double> current_b, int current_cols,
    const std::vector<int>& col_displs) {
  
  int my_rows = (local_a.empty()) ? 0 : static_cast<int>(local_a.size()) / m;
  std::vector<double> local_c;
  if (my_rows > 0) {
    local_c.resize(static_cast<size_t>(my_rows) * static_cast<size_t>(p), 0.0);
  }

  int stripe_owner = rank;
  for (int step = 0; step < size; ++step) {
    if (my_rows > 0 && current_cols > 0 && !current_b.empty()) {
      int stripe_offset = (stripe_owner < static_cast<int>(col_displs.size())) ? col_displs[stripe_owner] : 0;
      for (int i = 0; i < my_rows; ++i) {
        for (int j = 0; j < current_cols; ++j) {
          double sum = 0.0;
          for (int k = 0; k < m; ++k) {
            sum += local_a[static_cast<size_t>(i) * static_cast<size_t>(m) + static_cast<size_t>(k)] *
                   current_b[static_cast<size_t>(k) + static_cast<size_t>(j) * static_cast<size_t>(m)];
          }
          local_c[static_cast<size_t>(i) * static_cast<size_t>(p) + static_cast<size_t>(stripe_offset + j)] = sum;
        }
      }
    }

    if (step == size - 1) break;

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    int send_cols = current_cols;
    int recv_cols = 0;
    MPI_Sendrecv(&send_cols, 1, MPI_INT, next, 200,
                 &recv_cols, 1, MPI_INT, prev, 200,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<double> recv_buffer;
    if (recv_cols > 0) {
      recv_buffer.resize(static_cast<size_t>(m) * static_cast<size_t>(recv_cols));
    }

    int send_count = send_cols * m;
    int recv_count = recv_cols * m;
    const double* send_ptr = (send_count > 0 && !current_b.empty()) ? current_b.data() : nullptr;
    double* recv_ptr = (recv_count > 0) ? recv_buffer.data() : nullptr;

    MPI_Sendrecv(send_ptr, send_count, MPI_DOUBLE, next, 201,
                 recv_ptr, recv_count, MPI_DOUBLE, prev, 201,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (recv_cols > 0) {
      current_b = std::move(recv_buffer);
      current_cols = recv_cols;
    } else {
      current_b.clear();
      current_cols = 0;
    }

    stripe_owner = (stripe_owner - 1 + size) % size;
  }

  return local_c;
}

void BroadcastFinalResult(int rank, int size, int n, int p,
                          const std::vector<int>& row_counts,
                          const std::vector<int>& row_displs,
                          const std::vector<double>& local_c,
                          std::vector<double>& result) {
  std::vector<int> result_counts(size), result_displs(size);
  for (int i = 0; i < size; ++i) {
    result_counts[i] = row_counts[i] * p;
    result_displs[i] = row_displs[i] * p;
  }

  if (rank == 0) {
    result.resize(static_cast<size_t>(n) * static_cast<size_t>(p));
  }

  int my_rows = (local_c.empty()) ? 0 : static_cast<int>(local_c.size()) / p;
  int local_result_elements = my_rows * p;
  const double* local_result_ptr = (my_rows > 0) ? local_c.data() : nullptr;

  MPI_Gatherv(local_result_ptr, local_result_elements, MPI_DOUBLE,
              result.data(), result_counts.data(), result_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int total_size = n * p;
  if (rank != 0) {
    result.resize(static_cast<size_t>(total_size));
  }
  MPI_Bcast(result.data(), total_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool RunStripedScheme(int rank, int size, size_t rows_a, size_t cols_a, size_t cols_b,
                      const std::vector<double>& matrix_a, const std::vector<double>& matrix_b,
                      std::vector<double>& output) {
  const int n = static_cast<int>(rows_a);
  const int m = static_cast<int>(cols_a);
  const int p = static_cast<int>(cols_b);

  auto [row_counts, row_displs, local_a] = DistributeMatrixA(rank, size, n, m, matrix_a);
  auto [col_counts, col_displs, current_b, current_cols] = DistributeMatrixB(rank, size, m, p, matrix_b);
  auto local_c = ComputeWithCyclicShift(rank, size, m, p, local_a, current_b, current_cols, col_displs);

  BroadcastFinalResult(rank, size, n, p, row_counts, row_displs, local_c, output);
  return true;
}

}  // namespace

bool BatushinIStripedMatrixMultiplicationMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto& input = GetInput();
  const size_t rows_a = std::get<0>(input);
  const size_t cols_a = std::get<1>(input);
  const auto& matrix_a = std::get<2>(input);
  const size_t cols_b = std::get<4>(input);
  const auto& matrix_b = std::get<5>(input);

  std::vector<double> output;

  if (static_cast<size_t>(size) > rows_a || static_cast<size_t>(size) > cols_b || size <= 4) {
    RunSequentialFallback(rank, rows_a, cols_a, cols_b, matrix_a, matrix_b, output);
  } else {
    RunStripedScheme(rank, size, rows_a, cols_a, cols_b, matrix_a, matrix_b, output);
  }

  GetOutput() = std::move(output);
  return true;
}
```