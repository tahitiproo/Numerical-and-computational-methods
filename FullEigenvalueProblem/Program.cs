using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;

namespace EigenvaluesJacobi
{
    class Program
    {
        // -------------------------------------------------------------------------
        // Core Matrix Operations
        // -------------------------------------------------------------------------
        static double[,] CopyMatrix(double[,] A)
        {
            int n = A.GetLength(0);
            double[,] C = new double[n, n];
            Array.Copy(A, C, A.Length);
            return C;
        }

        static double OffDiagonalNorm(double[,] A)
        {
            int n = A.GetLength(0);
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    sum += A[i, j] * A[i, j];
                }
            }
            return Math.Sqrt(2.0 * sum);
        }

        static double GetTrace(double[,] A)
        {
            int n = A.GetLength(0);
            double tr = 0;
            for (int i = 0; i < n; i++) tr += A[i, i];
            return tr;
        }

        static double GetFrobeniusNormSq(double[,] A)
        {
            int n = A.GetLength(0);
            double normSq = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    normSq += A[i, j] * A[i, j];
            return normSq;
        }

        // -------------------------------------------------------------------------
        // Jacobi Rotation Logic
        // -------------------------------------------------------------------------
        static void ApplyJacobiRotation(double[,] A, int i, int j)
        {
            int n = A.GetLength(0);
            double a_ii = A[i, i];
            double a_jj = A[j, j];
            double a_ij = A[i, j];

            double x = -2.0 * a_ij;
            double y = a_ii - a_jj;

            double c, s;

            if (Math.Abs(y) < 1e-15)
            {
                c = 1.0 / Math.Sqrt(2);
                s = 1.0 / Math.Sqrt(2);
            }
            else
            {
                // Stable formulas avoiding direct trigonometrics
                double denom = Math.Sqrt(x * x + y * y);
                c = Math.Sqrt(0.5 * (1.0 + Math.Abs(y) / denom));
                s = Math.Sign(x * y) * Math.Abs(x) / (2.0 * c * denom);
            }

            // Update rows and columns i and j
            for (int k = 0; k < n; k++)
            {
                if (k == i || k == j) continue;
                double a_ik = A[i, k];
                double a_jk = A[j, k];
                A[i, k] = c * a_ik - s * a_jk;
                A[k, i] = A[i, k];
                A[j, k] = s * a_ik + c * a_jk;
                A[k, j] = A[j, k];
            }

            // Update diagonal elements
            A[i, i] = c * c * a_ii - 2 * s * c * a_ij + s * s * a_jj;
            A[j, j] = s * s * a_ii + 2 * s * c * a_ij + c * c * a_jj;

            // Force off-diagonal to zero
            A[i, j] = 0.0;
            A[j, i] = 0.0;
        }

        // -------------------------------------------------------------------------
        // Strategies
        // -------------------------------------------------------------------------

        // 1. Max Element Strategy
        static (double[] eigs, int iters, double ms) JacobiMaxElement(double[,] A_in, double tol = 1e-6, int maxIter = 10000)
        {
            double[,] A = CopyMatrix(A_in);
            int n = A.GetLength(0);
            Stopwatch sw = Stopwatch.StartNew();
            int iters = 0;

            while (OffDiagonalNorm(A) > tol && iters < maxIter)
            {
                // Find Max
                double maxVal = -1.0;
                int p = 0, q = 1;
                for (int i = 0; i < n; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        double absVal = Math.Abs(A[i, j]);
                        if (absVal > maxVal)
                        {
                            maxVal = absVal;
                            p = i;
                            q = j;
                        }
                    }
                }

                if (maxVal < tol) break;

                ApplyJacobiRotation(A, p, q);
                iters++;
            }

            sw.Stop();
            double[] eigs = new double[n];
            for (int i = 0; i < n; i++) eigs[i] = A[i, i];
            Array.Sort(eigs);
            return (eigs, iters, sw.Elapsed.TotalMilliseconds);
        }

        // 2. Cyclic Strategy
        static (double[] eigs, int iters, double ms) JacobiCyclic(double[,] A_in, double tol = 1e-6, int maxIter = 10000)
        {
            double[,] A = CopyMatrix(A_in);
            int n = A.GetLength(0);
            Stopwatch sw = Stopwatch.StartNew();
            int iters = 0;

            while (OffDiagonalNorm(A) > tol && iters < maxIter)
            {
                for (int i = 0; i < n - 1; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        if (Math.Abs(A[i, j]) > tol / n) // Small optimization: only rotate if significant
                        {
                            ApplyJacobiRotation(A, i, j);
                            iters++;
                        }
                    }
                }
            }

            sw.Stop();
            double[] eigs = new double[n];
            for (int i = 0; i < n; i++) eigs[i] = A[i, i];
            Array.Sort(eigs);
            return (eigs, iters, sw.Elapsed.TotalMilliseconds);
        }

        // -------------------------------------------------------------------------
        // Gershgorin Theorem
        // -------------------------------------------------------------------------
        struct GershgorinCircle
        {
            public double Center;
            public double Radius;
            public double MinBound => Center - Radius;
            public double MaxBound => Center + Radius;
        }

        static List<GershgorinCircle> GetGershgorinCircles(double[,] A)
        {
            int n = A.GetLength(0);
            var circles = new List<GershgorinCircle>();

            for (int i = 0; i < n; i++)
            {
                double r = 0;
                for (int j = 0; j < n; j++)
                {
                    if (i != j) r += Math.Abs(A[i, j]);
                }
                circles.Add(new GershgorinCircle { Center = A[i, i], Radius = r });
            }
            return circles;
        }

        static bool IsInGershgorinUnion(double lambda, List<GershgorinCircle> circles)
        {
            foreach (var c in circles)
            {
                if (lambda >= c.MinBound && lambda <= c.MaxBound)
                    return true;
            }
            return false;
        }

        // -------------------------------------------------------------------------
        // TESTING FRAMEWORK
        // -------------------------------------------------------------------------
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
            Console.WriteLine("Запуск тестов (Задание 4)...\n");

            // Тест 1: Диагональная матрица (Собственные значения уже найдены)
            double[,] A1 = { { 5, 0 }, { 0, 2 } };
            var (eigs1, iters1, _) = JacobiMaxElement(A1, 1e-9);
            Assert(iters1 == 0 && Math.Abs(eigs1[0] - 2) < 1e-9,
                "Тест 1: Диагональная матрица. Требует 0 итераций и сразу выдает верный ответ.");

            // Тест 2: Равные диагональные элементы (Проверка ветки a_ii == a_jj, y = 0)
            double[,] A2 = { { 3, 4 }, { 4, 3 } };
            // Известные с.ч. для [3 4; 4 3]: 3+4 = 7, 3-4 = -1
            var (eigs2, iters2, _) = JacobiMaxElement(A2, 1e-9);
            bool test2Passed = Math.Abs(eigs2[0] - (-1.0)) < 1e-9 && Math.Abs(eigs2[1] - 7.0) < 1e-9;
            Assert(test2Passed, "Тест 2: Матрица с равными диагональными элементами. Успешное преодоление y = 0.");

            // Тест 3: Теорема Гершгорина 
            double[,] A3 = {
                { 10,  1,  0 },
                {  1, 20,  2 },
                {  0,  2, 30 }
            };
            var circles = GetGershgorinCircles(A3);
            var (eigs3, _, _) = JacobiCyclic(A3);
            bool test3Passed = true;
            foreach (var ev in eigs3)
            {
                if (!IsInGershgorinUnion(ev, circles)) test3Passed = false;
            }
            Assert(test3Passed, "Тест 3: Проверка попадания вычисленных с.ч. в объединение кругов Гершгорина.");

            // Тест 4: Зависимость итераций от Epsilon (Матрица Гильберта 4x4)
            double[,] A4 = new double[4, 4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    A4[i, j] = 1.0 / (i + j + 1);

            var (_, iter1e3, _) = JacobiMaxElement(A4, 1e-3);
            var (_, iter1e6, _) = JacobiMaxElement(A4, 1e-6);
            var (_, iter1e9, _) = JacobiMaxElement(A4, 1e-9);
            Assert(iter1e3 <= iter1e6 && iter1e6 <= iter1e9 && iter1e3 > 0,
                $"Тест 4: Зависимость от ε. Итерации растут при уменьшении ε: {iter1e3} <= {iter1e6} <= {iter1e9}");

            // Тест 5 (НЕТРИВИАЛЬНЫЙ): Проверка сохранения инвариантов преобразований (След и Норма)
            // При многочисленных ортогональных вращениях на плотной матрице след и норма Фробениуса 
            // исходной матрицы A и полученной диагональной матрицы L обязаны совпадать с точностью до ошибки округления.
            int n5 = 6;
            double[,] A5 = new double[n5, n5];
            Random rnd = new Random(42);
            for (int i = 0; i < n5; i++)
                for (int j = i; j < n5; j++)
                {
                    double v = rnd.NextDouble() * 20 - 10;
                    A5[i, j] = v;
                    A5[j, i] = v;
                }

            double traceOriginal = GetTrace(A5);
            double frobNormOriginal = GetFrobeniusNormSq(A5);

            var (eigs5, iters5, _) = JacobiCyclic(A5, 1e-12);
            double traceFinal = eigs5.Sum();
            double frobNormFinal = eigs5.Sum(x => x * x); // На диагональной матрице это просто сумма квадратов с.ч.

            bool test5Passed = Math.Abs(traceOriginal - traceFinal) < 1e-10 &&
                               Math.Abs(frobNormOriginal - frobNormFinal) < 1e-10;

            Assert(test5Passed, "Тест 5: Нетривиальная проверка инвариантов. След и норма Фробениуса сохранены сквозь " + iters5 + " вращений.");

            Console.WriteLine(new string('-', 80));
        }

        // -------------------------------------------------------------------------

        static void RunComparison()
        {
            int n = 8;
            double[,] A = new double[n, n];
            Random rnd = new Random(123);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    A[i, j] = rnd.NextDouble() * 10;
                    A[j, i] = A[i, j];
                }

            Console.WriteLine($"Сравнение стратегий. Случайная матрица {n}x{n}. Точность: 1e-8\\n");
            string header = $"{"Стратегия",-22} | {"Итер.",-6} | {"Время(мс)",-10}";
            Console.WriteLine(header);
            Console.WriteLine(new string('-', header.Length));

            var (eigsM, iterM, msM) = JacobiMaxElement(A, 1e-8);
            Console.WriteLine($"{"Максимальный элемент",-22} | {iterM,-6} | {msM,10:F4}");

            var (eigsC, iterC, msC) = JacobiCyclic(A, 1e-8);
            Console.WriteLine($"{"Циклический выбор",-22} | {iterC,-6} | {msC,10:F4}");
            Console.WriteLine();
        }

        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            RunTests();
            RunComparison();
        }
    }
}