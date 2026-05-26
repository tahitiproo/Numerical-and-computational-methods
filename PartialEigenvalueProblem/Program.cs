using System;
using System.Diagnostics;

namespace PartialEigenvalueProblem
{
    class Program
    {
        // ---------------------------------------------------------------------
        // ВСПОМОГАТЕЛЬНАЯ ЛИНЕЙНАЯ АЛГЕБРА
        // ---------------------------------------------------------------------
        static double[] Multiply(double[,] A, double[] x)
        {
            int n = x.Length;
            double[] y = new double[n];
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++)
                    sum += A[i, j] * x[j];
                y[i] = sum;
            }
            return y;
        }

        static double DotProduct(double[] a, double[] b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
            return sum;
        }

        static double Norm(double[] x)
        {
            return Math.Sqrt(DotProduct(x, x));
        }

        static double[] Normalize(double[] x)
        {
            double norm = Norm(x);
            double[] res = new double[x.Length];
            if (norm > 1e-15)
            {
                for (int i = 0; i < x.Length; i++) res[i] = x[i] / norm;
            }
            return res;
        }

        static int MaxAbsIndex(double[] x)
        {
            int maxIdx = 0;
            double maxVal = Math.Abs(x[0]);
            for (int i = 1; i < x.Length; i++)
            {
                if (Math.Abs(x[i]) > maxVal)
                {
                    maxVal = Math.Abs(x[i]);
                    maxIdx = i;
                }
            }
            return maxIdx;
        }

        static double[,] GenerateHilbertMatrix(int n)
        {
            double[,] H = new double[n, n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    H[i, j] = 1.0 / (i + j + 1);
            return H;
        }

        // ---------------------------------------------------------------------
        // МЕТОДЫ ПОИСКА СОБСТВЕННОГО ЧИСЛА
        // ---------------------------------------------------------------------

        // 1. Степенной метод (вычисление λ по отношению координат)
        static (double lambda, double[] v, int iters) PowerMethod(double[,] A, double[] x0, double eps, int maxIter = 10000)
        {
            double[] x = Normalize(x0);
            double lambdaOld = 0.0;

            for (int k = 1; k <= maxIter; k++)
            {
                double[] y = Multiply(A, x);
                
                // Отношение i-х компонент (берем максимальную по модулю для стабильности)
                int i = MaxAbsIndex(x);
                double lambda = y[i] / x[i];

                x = Normalize(y);

                if (Math.Abs(lambda - lambdaOld) < eps)
                    return (lambda, x, k);

                lambdaOld = lambda;
            }
            return (lambdaOld, x, maxIter);
        }

        // 2. Метод скалярных произведений (отношение Релея)
        static (double lambda, double[] v, int iters) ScalarProductMethod(double[,] A, double[] x0, double eps, int maxIter = 10000)
        {
            double[] x = Normalize(x0);
            double lambdaOld = 0.0;

            for (int k = 1; k <= maxIter; k++)
            {
                double[] y = Multiply(A, x);
                
                // λ = (Ax, x) / (x, x). Так как x нормирован, (x,x) = 1.
                double lambda = DotProduct(y, x);

                x = Normalize(y);

                if (Math.Abs(lambda - lambdaOld) < eps)
                    return (lambda, x, k);

                lambdaOld = lambda;
            }
            return (lambdaOld, x, maxIter);
        }

        // ---------------------------------------------------------------------
        // МИНИ-ФРЕЙМВОРК ДЛЯ ТЕСТИРОВАНИЯ
        // ---------------------------------------------------------------------
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
            Console.WriteLine("=== Запуск нетривиальных тестов ===");

            // Тест 1: Отрицательное доминирующее собственное число
            // Матрица diag(-10, 2). Метод должен найти именно -10, а не 2.
            double[,] A1 = { { -10.0, 0 }, { 0, 2.0 } };
            double[] x0_1 = { 1.0, 1.0 };
            var (lam1, _, _) = ScalarProductMethod(A1, x0_1, 1e-6);
            Assert(Math.Abs(lam1 - (-10.0)) < 1e-5, "Тест 1: Корректное нахождение отрицательного доминирующего с.ч.");

            // Тест 2: Теоретическое ускорение метода скалярных произведений
            // Для симметричной матрицы метод скалярных произведений должен сходиться в ~2 раза быстрее.
            double[,] A2 = { { 3.0, 1.0 }, { 1.0, 2.0 } };
            var (_, _, itersPow) = PowerMethod(A2, x0_1, 1e-8);
            var (_, _, itersScal) = ScalarProductMethod(A2, x0_1, 1e-8);
            Assert(itersScal < itersPow * 0.6, $"Тест 2: Метод скал. произведений быстрее степенного (Скал: {itersScal}, Степ: {itersPow}).");

            // Тест 3: Инвариантность к масштабированию матрицы
            // При умножении матрицы на константу C, количество итераций не должно меняться, так как отношение λ2/λ1 сохраняется.
            double[,] A3_scaled = { { 30.0, 10.0 }, { 10.0, 20.0 } };
            var (_, _, itersScalScaled) = ScalarProductMethod(A3_scaled, x0_1, 1e-8);
            Assert(Math.Abs(itersScal - itersScalScaled) <= 1, "Тест 3: Инвариантность количества итераций к масштабированию матрицы.");

            // Тест 4: Проверка качества собственного вектора (невязка ||Ax - λx||)
            var (lam4, v4, _) = PowerMethod(A2, x0_1, 1e-8);
            double[] Ax = Multiply(A2, v4);
            double residual = Math.Sqrt(Math.Pow(Ax[0] - lam4 * v4[0], 2) + Math.Pow(Ax[1] - lam4 * v4[1], 2));
            Assert(residual < 1e-6, "Тест 4: Вычисленный вектор действительно является собственным (невязка близка к 0).");

            // Тест 5: Матрица Гильберта 3x3
            // Максимальное собственное число известно: ~1.4083189
            double[,] H3 = GenerateHilbertMatrix(3);
            double[] x0_H3 = { 1, 1, 1 };
            var (lam5, _, _) = ScalarProductMethod(H3, x0_H3, 1e-6);
            Assert(Math.Abs(lam5 - 1.4083189) < 1e-5, "Тест 5: Точный поиск с.ч. для плохо обусловленной матрицы Гильберта (n=3).");

            Console.WriteLine();
        }

        // ---------------------------------------------------------------------
        // ЭКСПЕРИМЕНТАЛЬНАЯ ЧАСТЬ (ЗАВИСИМОСТЬ ОТ EPSILON)
        // ---------------------------------------------------------------------
        static void CompareVaryingEpsilon(double[,] A, double[] x0, string matrixName)
        {
            Console.WriteLine($"--- Сравнение методов для матрицы: {matrixName} ---");
            Console.WriteLine($"{"Точность (eps)",-15} | {"Степенной (итер)",-18} | {"Скалярный (итер)",-18} | {"Степенной λ",-15} | {"Скалярный λ",-15}");
            Console.WriteLine(new string('-', 90));

            // Варьируем epsilon от 10^-3 до 10^-10
            for (int p = 3; p <= 10; p++)
            {
                double eps = Math.Pow(10, -p);

                var (lamPow, _, itPow) = PowerMethod(A, x0, eps);
                var (lamScal, _, itScal) = ScalarProductMethod(A, x0, eps);

                Console.WriteLine($"{eps,-15:0.0e+00} | {itPow,-18} | {itScal,-18} | {lamPow,-15:F8} | {lamScal,-15:F8}");
            }
            Console.WriteLine();
        }

        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            // 1. Запуск тестов
            RunTests();

            // 2. Исследование зависимости числа итераций от Epsilon
            // Пример 1: Произвольная симметричная матрица
            double[,] A = { 
                { 4.0, 1.0, 0.0 }, 
                { 1.0, 4.0, 1.0 }, 
                { 0.0, 1.0, 4.0 } 
            };
            double[] x0 = { 1, 1, 1 };
            CompareVaryingEpsilon(A, x0, "Трехдиагональная матрица 3x3");

            // Пример 2: Матрица Гильберта 4x4
            double[,] H4 = GenerateHilbertMatrix(4);
            double[] x0_H4 = { 1, 1, 1, 1 };
            CompareVaryingEpsilon(H4, x0_H4, "Матрица Гильберта 4x4");
        }
    }
}