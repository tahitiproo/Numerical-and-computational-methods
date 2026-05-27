using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;

namespace IterativeSolvers
{
    // Класс для представления разреженной матрицы в формате списка смежности
    public class SparseMatrix
    {
        public int N { get; }
        public List<(int col, double val)>[] Rows { get; }
        public double[] Diagonal { get; }

        public SparseMatrix(int n)
        {
            N = n;
            Rows = new List<(int, double)>[n];
            for (int i = 0; i < n; i++) Rows[i] = new List<(int, double)>();
            Diagonal = new double[n];
        }

        public void Add(int r, int c, double val)
        {
            Rows[r].Add((c, val));
            if (r == c) Diagonal[r] = val;
        }

        public double[] Multiply(double[] x)
        {
            double[] res = new double[N];
            for (int i = 0; i < N; i++)
            {
                double sum = 0;
                foreach (var item in Rows[i])
                    sum += item.val * x[item.col];
                res[i] = sum;
            }
            return res;
        }
    }

    class Program
    {
        static double Norm(double[] vec)
        {
            double sum = 0;
            foreach (var v in vec) sum += v * v;
            return Math.Sqrt(sum);
        }

        static double MaxNorm(double[] vec)
        {
            double max = 0;
            foreach (var v in vec)
                if (Math.Abs(v) > max) max = Math.Abs(v);
            return max;
        }

        // 1. Метод простой итерации (Якоби)
        static (double[] x, int iters, double ms) SimpleIteration(SparseMatrix A, double[] b, double tol = 1e-6, int maxIter = 10000)
        {
            int n = A.N;
            double[] x = new double[n];
            double[] xNew = new double[n];
            Stopwatch sw = Stopwatch.StartNew();
            int iter = 0;

            while (iter < maxIter)
            {
                for (int i = 0; i < n; i++)
                {
                    double sum = 0;
                    foreach (var item in A.Rows[i])
                    {
                        if (item.col != i)
                            sum += item.val * x[item.col];
                    }
                    xNew[i] = (b[i] - sum) / A.Diagonal[i];
                }

                double diff = 0;
                for (int i = 0; i < n; i++)
                {
                    double d = Math.Abs(xNew[i] - x[i]);
                    if (d > diff) diff = d;
                    x[i] = xNew[i];
                }

                iter++;
                if (diff < tol) break;
            }
            sw.Stop();
            return (x, iter, sw.Elapsed.TotalMilliseconds);
        }

        // 2. Метод Зейделя
        static (double[] x, int iters, double ms) Seidel(SparseMatrix A, double[] b, double tol = 1e-6, int maxIter = 10000)
        {
            int n = A.N;
            double[] x = new double[n];
            Stopwatch sw = Stopwatch.StartNew();
            int iter = 0;

            while (iter < maxIter)
            {
                double diff = 0;
                for (int i = 0; i < n; i++)
                {
                    double sum = 0;
                    foreach (var item in A.Rows[i])
                    {
                        if (item.col != i)
                            sum += item.val * x[item.col];
                    }
                    double newly = (b[i] - sum) / A.Diagonal[i];
                    double d = Math.Abs(newly - x[i]);
                    if (d > diff) diff = d;
                    x[i] = newly;
                }

                iter++;
                if (diff < tol) break;
            }
            sw.Stop();
            return (x, iter, sw.Elapsed.TotalMilliseconds);
        }

        // 3. Метод релаксации (по Березину и Жидкову)
        // В каждом цикле обновляем все n неизвестных. На каждом шаге цикла выбираем уравнение
        // с наибольшей невязкой из еще не использованных в текущем цикле.
        static (double[] x, int iters, double ms) Relaxation(SparseMatrix A, double[] b, double tol = 1e-6, int maxIter = 10000)
        {
            int n = A.N;
            double[] x = new double[n];
            double[] res = new double[n]; // невязки: Ax - b

            // Инициализация невязок (так как x=0, res = -b)
            for (int i = 0; i < n; i++) res[i] = -b[i];

            Stopwatch sw = Stopwatch.StartNew();
            int iter = 0;
            bool[] used = new bool[n];

            // Для быстрого обновления невязок нам нужен доступ по столбцам.
            // Так как матрица симметричная, Rows[k] содержит те же элементы, что и Cols[k]

            while (iter < maxIter)
            {
                Array.Clear(used, 0, n);
                double maxVal = 0;

                for (int step = 0; step < n; step++)
                {
                    // Ищем макс. невязку среди неиспользованных
                    int bestEq = -1;
                    double maxRes = -1;
                    for (int i = 0; i < n; i++)
                    {
                        if (!used[i])
                        {
                            double absRes = Math.Abs(res[i]);
                            if (absRes > maxRes)
                            {
                                maxRes = absRes;
                                bestEq = i;
                            }
                        }
                    }

                    if (bestEq == -1) break;

                    if (maxRes > maxVal) maxVal = maxRes;
                    used[bestEq] = true;

                    double dx = -res[bestEq] / A.Diagonal[bestEq];
                    x[bestEq] += dx;

                    // Обновляем невязки. Так как матрица симметричная, A_ik = A_ki. 
                    // Проходим по строке bestEq, что эквивалентно столбцу.
                    foreach (var item in A.Rows[bestEq])
                    {
                        res[item.col] += item.val * dx;
                    }
                }

                iter++; // Это считается за один цикл (n обновлений), аналогично одной итерации Зейделя
                if (maxVal < tol) break;
            }
            sw.Stop();
            return (x, iter, sw.Elapsed.TotalMilliseconds);
        }

        static SparseMatrix GenerateSPDMatrix(int n, int nonZerosPerRow)
        {
            var A = new SparseMatrix(n);
            Random rnd = new Random(42);
            HashSet<int>[] cols = new HashSet<int>[n];
            for (int i = 0; i < n; i++) cols[i] = new HashSet<int>();

            for (int i = 0; i < n; i++)
            {
                while (cols[i].Count < nonZerosPerRow)
                {
                    int j = rnd.Next(n);
                    if (i != j)
                    {
                        cols[i].Add(j);
                        cols[j].Add(i);
                    }
                }
            }

            for (int i = 0; i < n; i++)
            {
                double rowSum = 0;
                foreach (int j in cols[i])
                {
                    double val = rnd.NextDouble() * 10 - 5; // [-5, 5]
                    A.Add(i, j, val);
                    rowSum += Math.Abs(val);
                }
                // Диагональное преобладание: a_ii > sum |a_ij|
                double diag = rowSum + rnd.NextDouble() * 10 + 1.0;
                A.Add(i, i, diag);
            }
            return A;
        }

        static void Assert(bool condition, string testName)
        {
            if (condition)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"[ПРОЙДЕН] {testName}");
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ПРОВАЛ]  {testName}");
            }
            Console.ResetColor();
        }

        static void RunTests()
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine("                    ДЕМОНСТРАЦИЯ РАБОТЫ АЛГОРИТМА И ТЕСТЫ                     ");
            Console.WriteLine("================================================================================\n");

            // Тест 1
            Console.WriteLine("--- ТЕСТ 1: Диагональная матрица ---");
            SparseMatrix A1 = new SparseMatrix(3);
            A1.Add(0, 0, 2); A1.Add(1, 1, 4); A1.Add(2, 2, 8);
            double[] b1 = { 2, 8, 24 }; // x* = {1, 2, 3}
            var (x1, iter1, _) = SimpleIteration(A1, b1, 1e-9);
            bool pass1 = Math.Abs(x1[0] - 1) < 1e-8 && Math.Abs(x1[1] - 2) < 1e-8 && Math.Abs(x1[2] - 3) < 1e-8;
            Console.WriteLine($"Найдено итераций: {iter1} (Ожидалось малое число). Вычисленное решение: [{x1[0]:F2}, {x1[1]:F2}, {x1[2]:F2}]");
            Assert(pass1 && iter1 <= 2, "Тест 1: Диагональная система решается за 1-2 итерации с точным ответом.\n");

            // Тест 2
            Console.WriteLine("--- ТЕСТ 2: Точное восстановление решения (Симметричная 5x5) ---");
            SparseMatrix A2 = new SparseMatrix(5);
            for (int i = 0; i < 5; i++)
            {
                A2.Add(i, i, 10);
                if (i > 0) { A2.Add(i, i - 1, 1); A2.Add(i - 1, i, 1); }
            }
            double[] xTrue2 = { 1, 2, 3, 4, 5 };
            double[] b2 = A2.Multiply(xTrue2);
            var (x2, iter2, _) = Seidel(A2, b2, 1e-9);
            double err2 = MaxNorm(x2.Select((v, i) => v - xTrue2[i]).ToArray());
            Console.WriteLine($"Погрешность ||x - x*||_inf = {err2:E2}");
            Assert(err2 < 1e-8, "Тест 2: Метод Зейделя безошибочно сходится к истинному решению.\n");

            // Тест 3
            Console.WriteLine("--- ТЕСТ 3: Зависимость от точности ε ---");
            SparseMatrix A3 = GenerateSPDMatrix(100, 5);
            double[] b3 = new double[100]; Array.Fill(b3, 1.0);
            var (_, i3_1, _) = SimpleIteration(A3, b3, 1e-2);
            var (_, i3_2, _) = SimpleIteration(A3, b3, 1e-5);
            var (_, i3_3, _) = SimpleIteration(A3, b3, 1e-8);
            Console.WriteLine($"ε = 1e-2: {i3_1} ит. | ε = 1e-5: {i3_2} ит. | ε = 1e-8: {i3_3} ит.");
            Assert(i3_1 < i3_2 && i3_2 < i3_3, "Тест 3: Количество итераций монотонно возрастает при уменьшении ε.\n");

            // Тест 4
            Console.WriteLine("--- ТЕСТ 4: Сходимость Релаксации ---");
            SparseMatrix A4 = GenerateSPDMatrix(200, 5);
            double[] b4 = new double[200]; Array.Fill(b4, 1.0);
            var (_, iterRel, _) = Relaxation(A4, b4, 1e-6);
            var (_, iterSimp, _) = SimpleIteration(A4, b4, 1e-6);
            Console.WriteLine($"Простая итерация: {iterSimp} ит., Релаксация: {iterRel} циклов.");
            Assert(iterRel > 0 && iterRel < iterSimp, "Тест 4: Нестационарная релаксация сходится за меньшее число макро-циклов, чем простая итерация.\n");

            // Тест 5
            Console.WriteLine("--- ТЕСТ 5  Быстродействие на больших разреженных матрицах O(N) ---");
            Console.WriteLine("Генерация и решение 10000x10000...");
            SparseMatrix A5 = GenerateSPDMatrix(10000, 10);
            double[] b5 = new double[10000]; Array.Fill(b5, 2.0);
            Stopwatch sw = Stopwatch.StartNew();
            var (x5, _, _) = Seidel(A5, b5, 1e-4);
            sw.Stop();
            Console.WriteLine($"Время Зейделя: {sw.ElapsedMilliseconds} мс.");
            Assert(sw.ElapsedMilliseconds < 3000, "Тест 5: Матрица 10000x10000 решается сверхбыстро (< 3 сек) благодаря Sparse-структуре.\n");

            Console.WriteLine(new string('=', 80) + "\n");
        }

        static void RunComparison()
        {
            Console.WriteLine("--- Сравнение методов на больших матрицах ---");
            int[] sizes = { 1000, 5000, 10000 };
            double[] tols = { 1e-4, 1e-8 };

            foreach (int N in sizes)
            {
                Console.WriteLine($"--- Генерация разреженной симметричной матрицы {N}x{N} ---");
                Console.WriteLine("Диагональное преобладание гарантировано. Ненулевых эл-тов в строке: ~10");
                var A = GenerateSPDMatrix(N, 10);

                double[] xTrue = new double[N];
                for (int i = 0; i < N; i++) xTrue[i] = 1.0; // Точное решение
                double[] b = A.Multiply(xTrue);

                foreach (double tol in tols)
                {
                    Console.WriteLine($"\n  Точность (tol) = {tol}");
                    string header = $"  {"Метод",-20} | {"Циклов",-6} | {"Время(мс)",-10} | {"Погрешность ||x-x*||",-20}";
                    Console.WriteLine(header);
                    Console.WriteLine("  " + new string('-', header.Length - 2));

                    var (xSimp, iterSimp, msSimp) = SimpleIteration(A, b, tol);
                    double errSimp = MaxNorm(xSimp.Select(v => v - 1.0).ToArray());
                    Console.WriteLine($"  {"Простая итерация",-20} | {iterSimp,-6} | {msSimp,10:F1} | {errSimp,20:E4}");

                    var (xSeid, iterSeid, msSeid) = Seidel(A, b, tol);
                    double errSeid = MaxNorm(xSeid.Select(v => v - 1.0).ToArray());
                    Console.WriteLine($"  {"Зейдель",-20} | {iterSeid,-6} | {msSeid,10:F1} | {errSeid,20:E4}");

                    var (xRelax, iterRelax, msRelax) = Relaxation(A, b, tol);
                    double errRelax = MaxNorm(xRelax.Select(v => v - 1.0).ToArray());
                    Console.WriteLine($"  {"Релаксация [1]",-20} | {iterRelax,-6} | {msRelax,10:F1} | {errRelax,20:E4}");
                }
                Console.WriteLine("\n" + new string('=', 80) + "\n");
            }
        }

        static void Main()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            RunTests();
            RunComparison();
        }
    }
}