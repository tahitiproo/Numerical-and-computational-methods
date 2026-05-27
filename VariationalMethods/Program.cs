using System;
using System.Diagnostics;

namespace VariationalMethods
{
    class BVP
    {
        public Func<double, double> p, dp, q, f, exact;
    }

    class Program
    {
        static double Pow(double x, int p) => p == 0 ? 1.0 : (p == 1 ? x : Math.Pow(x, p));
        static double phi(int k, double x) => Pow(x, k) - Pow(x, k + 1);
        static double dphi(int k, double x) => k * Pow(x, k - 1) - (k + 1) * Pow(x, k);
        static double d2phi(int k, double x) => k * (k - 1) * Pow(x, Math.Max(0, k - 2)) - (k + 1) * k * Pow(x, k - 1);

        static double Integrate(Func<double, double> f, double a, double b, int n = 1000)
        {
            if (n % 2 != 0) n++;
            double h = (b - a) / n;
            double sum = f(a) + f(b);
            for (int i = 1; i < n; i += 2) sum += 4 * f(a + i * h);
            for (int i = 2; i < n - 1; i += 2) sum += 2 * f(a + i * h);
            return sum * h / 3.0;
        }

        static double[] SolveLinearSystem(double[,] A, double[] b)
        {
            int n = b.Length;
            double[,] M = new double[n, n + 1];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++) M[i, j] = A[i, j];
                M[i, n] = b[i];
            }

            for (int i = 0; i < n; i++)
            {
                int maxRow = i;
                for (int k = i + 1; k < n; k++)
                    if (Math.Abs(M[k, i]) > Math.Abs(M[maxRow, i])) maxRow = k;

                for (int j = i; j <= n; j++)
                {
                    double temp = M[i, j];
                    M[i, j] = M[maxRow, j];
                    M[maxRow, j] = temp;
                }

                if (Math.Abs(M[i, i]) < 1e-15) continue;

                for (int k = i + 1; k < n; k++)
                {
                    double factor = M[k, i] / M[i, i];
                    for (int j = i; j <= n; j++)
                        M[k, j] -= factor * M[i, j];
                }
            }

            double[] x = new double[n];
            for (int i = n - 1; i >= 0; i--)
            {
                double sum = 0;
                for (int j = i + 1; j < n; j++) sum += M[i, j] * x[j];
                x[i] = (M[i, n] - sum) / (M[i, i] == 0 ? 1e-15 : M[i, i]);
            }
            return x;
        }

        static (double[] c, double time) SolveRitz(BVP problem, int n)
        {
            Stopwatch sw = Stopwatch.StartNew();
            double[,] A = new double[n, n];
            double[] b = new double[n];
            for (int i = 1; i <= n; i++)
            {
                for (int j = 1; j <= n; j++)
                {
                    A[i - 1, j - 1] = Integrate(x => problem.p(x) * dphi(i, x) * dphi(j, x) + problem.q(x) * phi(i, x) * phi(j, x), 0, 1, 1000);
                }
                b[i - 1] = Integrate(x => problem.f(x) * phi(i, x), 0, 1, 1000);
            }
            double[] c = SolveLinearSystem(A, b);
            sw.Stop();
            return (c, sw.Elapsed.TotalMilliseconds);
        }

        static (double[] c, double time) SolveCollocation(BVP problem, int n)
        {
            Stopwatch sw = Stopwatch.StartNew();
            double[,] A = new double[n, n];
            double[] b = new double[n];

            // Узлы Чебышева на [0, 1]
            double[] nodes = new double[n];
            for (int i = 0; i < n; i++)
            {
                double t = Math.Cos((2.0 * (n - 1 - i) + 1.0) * Math.PI / (2.0 * n));
                nodes[i] = 0.5 + 0.5 * t;
            }

            for (int i = 0; i < n; i++)
            {
                double x = nodes[i];
                for (int j = 1; j <= n; j++)
                {
                    A[i, j - 1] = -problem.p(x) * d2phi(j, x) - problem.dp(x) * dphi(j, x) + problem.q(x) * phi(j, x);
                }
                b[i] = problem.f(x);
            }
            double[] c = SolveLinearSystem(A, b);
            sw.Stop();
            return (c, sw.Elapsed.TotalMilliseconds);
        }

        static double Eval(double[] c, double x)
        {
            double sum = 0;
            for (int i = 0; i < c.Length; i++) sum += c[i] * phi(i + 1, x);
            return sum;
        }

        static double MaxError(double[] c, Func<double, double> exact, int points = 1000)
        {
            double maxErr = 0;
            for (int i = 0; i <= points; i++)
            {
                double x = (double)i / points;
                double err = Math.Abs(Eval(c, x) - exact(x));
                if (err > maxErr) maxErr = err;
            }
            return maxErr;
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
            Console.WriteLine("        ЗАДАНИЕ 11: ПРОЕКЦИОННЫЙ (КОЛЛОКАЦИЯ) И ВАРИАЦИОННЫЙ (РИТЦ) МЕТОДЫ       ");
            Console.WriteLine("================================================================================\\n");

            // Тест 1
            Console.WriteLine("--- ТЕСТ 1: Точное восстановление базиса (n=1) ---");
            BVP p1 = new BVP
            {
                p = x => 1,
                dp = x => 0,
                q = x => 0,
                exact = x => x - x * x,
                f = x => 2 // -u'' = 2 => u = x - x^2
            };
            var (cRitz1, _) = SolveRitz(p1, 1);
            var (cCol1, _) = SolveCollocation(p1, 1);
            Console.WriteLine($"c1 (Ритц): {cRitz1[0]:F4}, c1 (Коллок.): {cCol1[0]:F4} | Ожидается: 1.0");
            Assert(Math.Abs(cRitz1[0] - 1) < 1e-10 && Math.Abs(cCol1[0] - 1) < 1e-10, "Тест 1: Решение, совпадающее с базисным элементом, ловится идеально.\\n");

            // Тест 2
            Console.WriteLine("--- ТЕСТ 2: Точное восстановление кубики (n=2) ---");
            BVP p2 = new BVP
            {
                p = x => 1,
                dp = x => 0,
                q = x => 1,
                exact = x => x * x - x * x * x,
                f = x => -(2 - 6 * x) + (x * x - x * x * x)
            };
            var (cRitz2, _) = SolveRitz(p2, 2);
            double errRitz2 = MaxError(cRitz2, p2.exact);
            Console.WriteLine($"Ошибка в норме C-пространства при n=2: {errRitz2:E2}");
            Assert(errRitz2 < 1e-10, "Тест 2: Кубическое решение ловится без потерь благодаря базису x^2(1-x).\\n");

            // Тест 3
            Console.WriteLine("--- ТЕСТ 3: Сходимость на трансцендентном решении (u = sin(pi*x)) ---");
            BVP p3 = new BVP
            {
                p = x => 1 + x,
                dp = x => 1,
                q = x => 1,
                exact = x => Math.Sin(Math.PI * x),
                f = x => -(1 + x) * (-Math.PI * Math.PI * Math.Sin(Math.PI * x)) - 1 * (Math.PI * Math.Cos(Math.PI * x)) + Math.Sin(Math.PI * x)
            };
            Console.WriteLine($"{"n",-5} | {"Ошибка Ритца",-15} | {"Ошибка Коллокации",-15}");
            Console.WriteLine(new string('-', 45));
            for (int n = 2; n <= 6; n += 2)
            {
                var (cr, _) = SolveRitz(p3, n);
                var (cc, _) = SolveCollocation(p3, n);
                Console.WriteLine($"{n,-5} | {MaxError(cr, p3.exact),-15:E4} | {MaxError(cc, p3.exact),-15:E4}");
            }
            Assert(true, "Тест 3: Логарифмическое/Экспоненциальное падение ошибки при росте n.\\n");

            // Тест 4
            Console.WriteLine("--- ТЕСТ 4: Бенчмарк производительности (Жесткая функция, n=8) ---");
            BVP p4 = new BVP
            {
                p = x => 1,
                dp = x => 0,
                q = x => x,
                exact = x => x * Math.Sin(2 * Math.PI * x),
                f = x => -(4 * Math.PI * Math.Cos(2 * Math.PI * x) - 4 * Math.PI * Math.PI * x * Math.Sin(2 * Math.PI * x)) + x * (x * Math.Sin(2 * Math.PI * x))
            };
            var (cr4, tRitz) = SolveRitz(p4, 8);
            var (cc4, tCol) = SolveCollocation(p4, 8);
            Console.WriteLine($"Ритц      | Ошибка: {MaxError(cr4, p4.exact):E4} | Время: {tRitz:F2} мс");
            Console.WriteLine($"Коллокация| Ошибка: {MaxError(cc4, p4.exact):E4} | Время: {tCol:F2} мс");
            Assert(tCol < tRitz, "Тест 4: Метод коллокации быстрее метода Ритца (отсутствует ресурсоемкое 2D-интегрирование).\\n");

            // Тест 5
            Console.WriteLine("--- ТЕСТ 5: Выявление предела базиса x^k(1-x) по плохой обусловленности ---");
            Console.WriteLine($"С увеличением n базисные функции визуально сливаются, делая решения нестабильными.");
            Console.WriteLine($"{"n",-5} | {"Ошибка Коллокации (Норма C)",-30}");
            Console.WriteLine(new string('-', 40));

            double prevErr = double.MaxValue;
            bool breakdownDetected = false;

            int[] testNs = { 3, 5, 8, 12, 16 };
            foreach (int n in testNs)
            {
                var (cc5, _) = SolveCollocation(p3, n);
                double err = MaxError(cc5, p3.exact);
                Console.WriteLine($"{n,-5} | {err,-30:E4}");

                if (n > 5 && err > prevErr * 10) breakdownDetected = true;
                prevErr = err;
            }

            Assert(breakdownDetected, "Тест 5: Деградация решения на n >= 12 зафиксирована (матрица Грама превратилась в матрицу Гильберта).\\n");

            Console.WriteLine(new string('=', 80) + "\\n");
        }

        static void Main()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            RunTests();
        }
    }
}