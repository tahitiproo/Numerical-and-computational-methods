using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace EigenvalueProblemJacobi
{
    class Program
    {
        // -------------------------------------------------------------------------
        // МЕТОД ЯКОБИ
        // -------------------------------------------------------------------------
        
        enum Strategy
        {
            MaxOffDiagonal,
            Cyclic
        }

        // Применение элементарного (плоского) вращения
        // Формулы из слайдов 7 и 8
        static void Rotate(double[,] A, int i, int j, int n)
        {
            if (Math.Abs(A[i, j]) < 1e-15) return;

            double x = -2.0 * A[i, j];
            double y = A[i, i] - A[j, j];

            double c, s;
            if (Math.Abs(y) < 1e-15)
            {
                // По лекции (слайд 8): Если a_ii == a_jj, угол повора = pi/4
                c = 1.0 / Math.Sqrt(2.0);
                s = 1.0 / Math.Sqrt(2.0);
            }
            else
            {
                // По лекции (слайд 8): Расчет тригонометрии
                double norm = Math.Sqrt(x * x + y * y);
                c = Math.Sqrt(0.5 * (1.0 + Math.Abs(y) / norm));
                
                double signXY = (x * y >= 0.0) ? 1.0 : -1.0; 
                s = (signXY * Math.Abs(x)) / (2.0 * c * norm);
            }

            double a_ii = A[i, i];
            double a_jj = A[j, j];
            double a_ij = A[i, j];

            // Обновляем диагональные элементы
            A[i, i] = c * c * a_ii - 2.0 * c * s * a_ij + s * s * a_jj;
            A[j, j] = s * s * a_ii + 2.0 * c * s * a_ij + c * c * a_jj;
            A[i, j] = 0.0;
            A[j, i] = 0.0;

            // Обновляем остальные элементы (крест)
            for (int k = 0; k < n; k++)
            {
                if (k != i && k != j)
                {
                    double a_ik = A[i, k];
                    double a_jk = A[j, k];
                    A[i, k] = c * a_ik - s * a_jk;
                    A[k, i] = A[i, k];      // симметрия
                    A[j, k] = s * a_ik + c * a_jk;
                    A[k, j] = A[j, k];      // симметрия
                }
            }
        }

        // Критерий остановки на основе суммы внедиагональных |a_ij|,
        // что эквивалентно радиусу кругов Гершгорина (R_i < eps - слайд 9)
        static double GetMaxRi(double[,] A, int n)
        {
            double maxRi = 0;
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++)
                    if (i != j) sum += Math.Abs(A[i, j]);
                
                if (sum > maxRi) maxRi = sum;
            }
            return maxRi;
        }

        // Стратегия 1: Поиск максимального по модулю внедиагонального элемента
        static (int, int) FindMaxOffDiagonal(double[,] A, int n)
        {
            double maxVal = -1.0;
            int maxI = 0, maxJ = 1;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double val = Math.Abs(A[i, j]);
                    if (val > maxVal)
                    {
                        maxVal = val;
                        maxI = i; maxJ = j;
                    }
                }
            }
            return (maxI, maxJ);
        }

        // Стратегия 2: Циклический выбор
        static (int, int) GetNextCyclic(ref int i, ref int j, int n)
        {
            int currI = i, currJ = j;
            j++;
            if (j >= n)
            {
                i++;
                if (i >= n - 1) i = 0;
                j = i + 1;
            }
            return (currI, currJ);
        }

        // Основная функция метода Якоби
        static (double[] eigenvalues, int iterations, double ms) JacobiMethod(
            double[,] A_in, Strategy strategy, double tol = 1e-6, int maxIter = 500000)
        {
            int n = A_in.GetLength(0);
            double[,] A = (double[,])A_in.Clone();

            Stopwatch sw = Stopwatch.StartNew();
            int iterations = 0;
            int cycI = 0, cycJ = 1;

            while (iterations < maxIter)
            {
                // Остановка, когда все внедиагональные суммы < epsilon (слайд 9)
                if (GetMaxRi(A, n) < tol) break;

                int i, j;
                if (strategy == Strategy.MaxOffDiagonal)
                {
                    (i, j) = FindMaxOffDiagonal(A, n);
                    if (Math.Abs(A[i, j]) < 1e-15) break; 
                }
                else 
                {
                    (i, j) = GetNextCyclic(ref cycI, ref cycJ, n);
                }

                Rotate(A, i, j, n);
                iterations++;
            }
            sw.Stop();

            double[] eigenvalues = new double[n];
            for (int i = 0; i < n; i++) eigenvalues[i] = A[i, i];

            Array.Sort(eigenvalues);
            return (eigenvalues, iterations, sw.Elapsed.TotalMilliseconds);
        }

        // -------------------------------------------------------------------------
        // ИССЛЕДОВАНИЯ И ТЕОРЕМА ГЕРШГОРИНА
        // -------------------------------------------------------------------------

        class Interval : IComparable<Interval>
        {
            public double Start;
            public double End;
            public int Count; // Количество кругов (с.ч.) в этой структуре

            public int CompareTo(Interval other) => Start.CompareTo(other.Start);
        }

        static List<Interval> GetGershgorinIntervals(double[,] A)
        {
            int n = A.GetLength(0);
            var intervals = new List<Interval>();

            for (int i = 0; i < n; i++)
            {
                double currR = 0;
                for (int j = 0; j < n; j++)
                    if (i != j) currR += Math.Abs(A[i, j]);
                
                intervals.Add(new Interval { Start = A[i, i] - currR, End = A[i, i] + currR, Count = 1 });
            }

            intervals.Sort();
            var merged = new List<Interval>();
            if (intervals.Count == 0) return merged;

            Interval current = intervals[0];
            for (int i = 1; i < intervals.Count; i++)
            {
                // Если интервалы перекрываются или соприкасаются
                if (intervals[i].Start <= current.End + 1e-9) 
                {
                    current.End = Math.Max(current.End, intervals[i].End);
                    current.Count += intervals[i].Count;
                }
                else
                {
                    merged.Add(current);
                    current = intervals[i];
                }
            }
            merged.Add(current);
            return merged;
        }

        static void VerifyGershgorin(double[,] A, double[] eigenvalues)
        {
            var intervals = GetGershgorinIntervals(A);
            bool totalSuccess = true;
            foreach (var iv in intervals)
            {
                int countInside = 0;
                foreach (var ev in eigenvalues)
                {
                    if (ev >= iv.Start - 1e-9 && ev <= iv.End + 1e-9)
                        countInside++;
                }

                bool match = (countInside == iv.Count);
                if (!match) totalSuccess = false;

                Console.ForegroundColor = match ? ConsoleColor.Green : ConsoleColor.Red;
                Console.WriteLine($"  Структура [{iv.Start,7:F4}, {iv.End,7:F4}]: ожидалось с.ч. {iv.Count}, фактически попало {countInside}");
            }
            Console.ResetColor();

            if (totalSuccess)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("  [УСПЕХ] Собственные числа строго подчиняются теореме Гершгорина!");
                Console.ResetColor();
            }
        }

        static void StudyEpsilonDependence(double[,] A)
        {
            Console.WriteLine($"  {"Epsilon",-10} | {"Ит. (Max)",-10} | {"Ит. (Цикл)",-10}");
            double[] epsilons = { 1e-2, 1e-4, 1e-6, 1e-8, 1e-10 };
            foreach (var eps in epsilons)
            {
                var (_, itersMax, _) = JacobiMethod(A, Strategy.MaxOffDiagonal, eps);
                var (_, itersCyc, _) = JacobiMethod(A, Strategy.Cyclic, eps);
                Console.WriteLine($"  {eps,-10:0.e+00} | {itersMax,-10} | {itersCyc,-10}");
            }
        }

        // -------------------------------------------------------------------------
        // СРАВНЕНИЕ (ЭМУЛЯЦИЯ ЗАДАНИЯ 3) 
        // -------------------------------------------------------------------------
        
        static double PowerMethod(double[,] A, double tol = 1e-6, int maxIter = 1000)
        {
            int n = A.GetLength(0);
            double[] x = new double[n];
            x[0] = 1.0; 
            double lambdaOld = 0;

            for (int iter = 0; iter < maxIter; iter++)
            {
                double[] xNew = new double[n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        xNew[i] += A[i, j] * x[j];

                double dotNum = 0, dotDen = 0;
                for (int i = 0; i < n; i++)
                {
                    dotNum += xNew[i] * x[i];
                    dotDen += x[i] * x[i];
                }
                double lambda = dotNum / dotDen;

                if (Math.Abs(lambda - lambdaOld) < tol && iter > 0) return lambda;
                lambdaOld = lambda;

                double norm = Math.Sqrt(xNew.Sum(v => v * v));
                for (int i = 0; i < n; i++) x[i] = xNew[i] / norm;
            }
            return lambdaOld;
        }

        // -------------------------------------------------------------------------
        // МИНИ-ФРЕЙМВОРК ДЛЯ ТЕСТИРОВАНИЯ
        // -------------------------------------------------------------------------

        static void RunValidationsForMatrix(string testName, double[,] A)
        {
            Console.WriteLine($"\n=======================================================");
            Console.WriteLine($"ТЕСТ: {testName}");
            Console.WriteLine($"=======================================================");
            
            int n = A.GetLength(0);
            Console.WriteLine("Матрица A:");
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                    Console.Write($"{A[i, j],7:F2} ");
                Console.WriteLine();
            }

            double eps = 1e-6;
            var (evMax, itersMax, timeMax) = JacobiMethod(A, Strategy.MaxOffDiagonal, eps);
            var (evCyc, itersCyc, timeCyc) = JacobiMethod(A, Strategy.Cyclic, eps);

            Console.WriteLine($"\n[1] Сравнение стратегий выбора при eps = {eps:0.e+00}");
            Console.WriteLine($"  -> По макс. модулю        : Ит. = {itersMax}, Время = {timeMax:F4} мс");
            Console.WriteLine($"  -> Циклический перебор    : Ит. = {itersCyc}, Время = {timeCyc:F4} мс");

            Console.WriteLine("\n[2] Найденные собственные значения:");
            Console.WriteLine("  " + string.Join(", ", evMax.Select(x => x.ToString("F4"))));

            Console.WriteLine("\n[3] Зависимость от точности epsilon:");
            StudyEpsilonDependence(A);

            Console.WriteLine("\n[4] Анализ по Теореме Гершгорина:");
            VerifyGershgorin(A, evMax);

            Console.WriteLine("\n[5] Сравнение максимальных с.ч. с Заданием 3 (Степенной метод)");
            double maxEvalJacobi = evMax.OrderByDescending(Math.Abs).First();
            double maxEvalPower = PowerMethod(A, eps);
            Console.WriteLine($"  С.ч. метод Якоби      : {maxEvalJacobi:F6}");
            Console.WriteLine($"  С.ч. степенной метод  : {maxEvalPower:F6}");
            
            double diff = Math.Abs(Math.Abs(maxEvalJacobi) - Math.Abs(maxEvalPower));
            Console.WriteLine($"  Невязка               : {diff:0.0e+00}");
        }

        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            Console.WriteLine("ПРАКТИКУМ ПО ВЫЧИСЛИТЕЛЬНОЙ МАТЕМАТИКЕ");
            Console.WriteLine("Задание 4. Полная проблема собственных чисел (Метод Якоби)\n");

            // 1. Диагонально преобладающая
            double[,] A1 = {
                {  4.0, -1.0,  0.0,  0.0 },
                { -1.0,  4.0, -1.0,  0.0 },
                {  0.0, -1.0,  4.0, -1.0 },
                {  0.0,  0.0, -1.0,  3.0 }
            };
            RunValidationsForMatrix("1. Разреженная диагонально преобладающая матрица", A1);

            // 2. Тестовая матрица с сильно слипшимися кругами (одинаковые с.ч.)
            double[,] A2 = {
                {  2.0,  1.0,  1.0 },
                {  1.0,  2.0,  1.0 },
                {  1.0,  1.0,  2.0 }
            }; // Теоретические с.ч.: 4, 1, 1.
            RunValidationsForMatrix("2. Матрица с перекрывающимися кругами", A2);

            // 3. Матрица, имеющая отрицательные и положительные собственные значения 
            double[,] A3 = {
                {  0.0,  1.0,  0.5 },
                {  1.0,  0.0, -1.0 },
                {  0.5, -1.0,  0.0 }
            }; // Нулевой след
            RunValidationsForMatrix("3. Матрица с разными знаками спектра (след = 0)", A3);

            // 4. Похожая на матрицу Гильберта сегмент (Плохая обусловленность)
            double[,] A4 = {
                { 1.0,      1.0/2.0,  1.0/3.0 },
                { 1.0/2.0,  1.0/3.0,  1.0/4.0 },
                { 1.0/3.0,  1.0/4.0,  1.0/5.0 }
            };
            RunValidationsForMatrix("4. Сегмент матрицы Гильберта", A4);
        }
    }
}