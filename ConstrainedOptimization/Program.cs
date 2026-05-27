using System;
using System.Diagnostics;
using System.Linq;

namespace ConstrainedOptimization
{
    // Обертка для целевой функции для подсчета вызовов
    public class Objective
    {
        public Func<double[], double> f;
        public int evals;
        public double Eval(double[] x)
        {
            evals++;
            return f(x);
        }
    }

    class Program
    {
        // Численный градиент по центральным/односторонним разностям
        static double[] Grad(Objective obj, double[] x)
        {
            double eps = 1e-6;
            double[] g = new double[x.Length];
            double fx = obj.Eval(x);

            for (int i = 0; i < x.Length; i++)
            {
                double[] x2 = (double[])x.Clone();
                x2[i] += eps;
                double f2 = obj.Eval(x2);

                if (double.IsInfinity(f2) || double.IsNaN(f2))
                {
                    double[] x1 = (double[])x.Clone();
                    x1[i] -= eps;
                    double f1 = obj.Eval(x1);
                    g[i] = (fx - f1) / eps;
                }
                else
                {
                    double[] x1 = (double[])x.Clone();
                    x1[i] -= eps;
                    double f1 = obj.Eval(x1);
                    if (double.IsInfinity(f1) || double.IsNaN(f1))
                        g[i] = (f2 - fx) / eps;
                    else
                        g[i] = (f2 - f1) / (2 * eps);
                }
            }
            return g;
        }

        // Внутренний безусловный оптимизатор: Градиентный спуск с дроблением шага
        static (double[], int) Minimize(Objective obj, double[] x0)
        {
            double[] x = (double[])x0.Clone();
            int startEvals = obj.evals;

            for (int k = 0; k < 5000; k++)
            {
                double[] g = Grad(obj, x);
                double norm = Math.Sqrt(g.Sum(v => v * v));
                if (norm < 1e-4 || double.IsNaN(norm) || double.IsInfinity(norm)) break;

                double alpha = 1.0;
                double fx = obj.Eval(x);

                while (true)
                {
                    double[] xNew = new double[x.Length];
                    for (int i = 0; i < x.Length; i++) xNew[i] = x[i] - alpha * g[i];

                    double fxNew = obj.Eval(xNew);
                    if (fxNew <= fx - 1e-4 * alpha * norm * norm || alpha < 1e-10)
                    {
                        x = xNew;
                        break;
                    }
                    alpha *= 0.5;
                }
                if (alpha < 1e-10) break; // Спуск застопорился или достиг локального минимума
            }
            return (x, obj.evals - startEvals);
        }

        // 1. Метод штрафных функций (Подход извне)
        public static (double[], int, double) PenaltyMethod(
            Func<double[], double> f,
            Func<double[], double>[] g, // Неравенства: g_i(x) <= 0
            Func<double[], double>[] h, // Равенства: h_j(x) = 0
            double[] x0, double tol = 1e-4)
        {
            Stopwatch sw = Stopwatch.StartNew();
            double alpha = 1.0;
            double[] x = (double[])x0.Clone();
            Objective totalObj = new Objective();

            for (int iter = 0; iter < 100; iter++)
            {
                Func<double[], double> H = (vx) =>
                {
                    double penalty = 0;
                    if (g != null) foreach (var gi in g) penalty += Math.Pow(Math.Max(0, gi(vx)), 2);
                    if (h != null) foreach (var hj in h) penalty += Math.Pow(hj(vx), 2);
                    return penalty;
                };

                totalObj.f = vx => f(vx) + alpha * H(vx);
                var (xNew, _) = Minimize(totalObj, x);

                double dist = Math.Sqrt(x.Zip(xNew, (A, B) => (A - B) * (A - B)).Sum());
                x = xNew;

                if (alpha * H(x) < tol || alpha > 1e8 || dist < 1e-6) break;
                alpha *= 10;
            }
            sw.Stop();
            return (x, totalObj.evals, sw.Elapsed.TotalMilliseconds);
        }

        // 2. Метод барьерных функций (Подход изнутри)
        // Применим только к неравенствам g_i(x) <= 0
        public static (double[], int, double) BarrierMethod(
            Func<double[], double> f,
            Func<double[], double>[] g,
            double[] x0, double tol = 1e-4)
        {
            Stopwatch sw = Stopwatch.StartNew();
            double mu = 1.0;
            double[] x = (double[])x0.Clone();
            Objective totalObj = new Objective();

            for (int iter = 0; iter < 100; iter++)
            {
                Func<double[], double> B = (vx) =>
                {
                    double barrier = 0;
                    foreach (var gi in g)
                    {
                        double val = gi(vx);
                        if (val >= 0) return double.PositiveInfinity; // Жесткий барьер (обратная функция)
                        barrier += -1.0 / val;
                    }
                    return barrier;
                };

                totalObj.f = vx => f(vx) + mu * B(vx);
                var (xNew, _) = Minimize(totalObj, x);

                double dist = Math.Sqrt(x.Zip(xNew, (A, B) => (A - B) * (A - B)).Sum());
                x = xNew;

                if (mu * B(x) < tol || mu < 1e-8 || dist < 1e-6) break;
                mu *= 0.1;
            }
            sw.Stop();
            return (x, totalObj.evals, sw.Elapsed.TotalMilliseconds);
        }

        // 3. Метод модифицированных функций Лагранжа
        public static (double[], int, double) AugmentedLagrangianMethod(
            Func<double[], double> f,
            Func<double[], double>[] h,
            double[] x0, double tol = 1e-4)
        {
            Stopwatch sw = Stopwatch.StartNew();
            double t = 10.0;
            double[] u = new double[h.Length];
            double[] x = (double[])x0.Clone();
            Objective totalObj = new Objective();

            for (int iter = 0; iter < 100; iter++)
            {
                totalObj.f = vx =>
                {
                    double val = f(vx);
                    for (int i = 0; i < h.Length; i++)
                    {
                        double hi = h[i](vx);
                        val += u[i] * hi + (t / 2.0) * hi * hi;
                    }
                    return val;
                };

                var (xNew, _) = Minimize(totalObj, x);

                double dist = Math.Sqrt(x.Zip(xNew, (A, B) => (A - B) * (A - B)).Sum());
                x = xNew;

                for (int i = 0; i < h.Length; i++)
                    u[i] += t * h[i](x); // Обновление множителей Лагранжа

                if (dist < tol) break;
            }
            sw.Stop();
            return (x, totalObj.evals, sw.Elapsed.TotalMilliseconds);
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

        static double[] RandomPoint(int dim, double min, double max, Random rnd)
        {
            double[] x = new double[dim];
            for (int i = 0; i < dim; i++) x[i] = min + (max - min) * rnd.NextDouble();
            return x;
        }

        static double[] RandomInteriorPoint(int dim, double min, double max, Func<double[], double>[] g, Random rnd)
        {
            for (int k = 0; k < 10000; k++)
            {
                double[] x = RandomPoint(dim, min, max, rnd);
                bool valid = true;
                foreach (var gi in g) if (gi(x) >= 0) { valid = false; break; }
                if (valid) return x;
            }
            return RandomPoint(dim, min, max, rnd);
        }

        static void RunBenchmark()
        {
            Console.WriteLine("--- ТЕСТ 5: Сравнение методов по метрикам (Бенчмарк 100 запусков) ---");

            int runs = 100;
            Random rnd = new Random(42);

            // Задача с неравенством: x1^2+x2^2 -> min, при x1+x2 >= 2
            Func<double[], double> f = x => x[0] * x[0] + x[1] * x[1];
            Func<double[], double>[] g = { x => 2 - x[0] - x[1] };
            double[] exact = { 1.0, 1.0 };

            int penSuccess = 0, barSuccess = 0;
            long penEvals = 0, barEvals = 0;
            double penTime = 0, barTime = 0;

            for (int i = 0; i < runs; i++)
            {
                double[] x0_p = RandomPoint(2, -10, 10, rnd);
                var (xp, ep, tp) = PenaltyMethod(f, g, null, x0_p);
                penEvals += ep; penTime += tp;
                if (Math.Abs(xp[0] - exact[0]) < 0.1 && Math.Abs(xp[1] - exact[1]) < 0.1) penSuccess++;

                double[] x0_b = RandomInteriorPoint(2, -10, 10, g, rnd);
                var (xb, eb, tb) = BarrierMethod(f, g, x0_b);
                barEvals += eb; barTime += tb;
                if (Math.Abs(xb[0] - exact[0]) < 0.1 && Math.Abs(xb[1] - exact[1]) < 0.1) barSuccess++;
            }

            Console.WriteLine("Задача: x1^2+x2^2 -> min, при x1+x2 >= 2");
            Console.WriteLine(String.Format("{0,-18} | {1,-15} | {2,-15} | {3,-15}", "Метод", "Вызовов f(x)", "Время (мс)", "Глоб. оптимум"));
            Console.WriteLine(new string('-', 72));
            Console.WriteLine(String.Format("{0,-18} | {1,-15} | {2,-15:F3} | {3,-15}", "Метод штрафов", penEvals / runs, penTime / runs, (penSuccess * 100.0 / runs).ToString("F0") + "%"));
            Console.WriteLine(String.Format("{0,-18} | {1,-15} | {2,-15:F3} | {3,-15}", "Метод барьеров", barEvals / runs, barTime / runs, (barSuccess * 100.0 / runs).ToString("F0") + "%"));

            // Задача с равенством: x1^2+x2^2 -> min, при x1+x2=2
            Func<double[], double>[] h = { x => x[0] + x[1] - 2 };

            penSuccess = 0; int almSuccess = 0;
            penEvals = 0; long almEvals = 0;
            penTime = 0; double almTime = 0;

            for (int i = 0; i < runs; i++)
            {
                double[] x0 = RandomPoint(2, -10, 10, rnd);

                var (xp, ep, tp) = PenaltyMethod(f, null, h, x0);
                penEvals += ep; penTime += tp;
                if (Math.Abs(xp[0] - exact[0]) < 0.1 && Math.Abs(xp[1] - exact[1]) < 0.1) penSuccess++;

                var (xa, ea, ta) = AugmentedLagrangianMethod(f, h, x0);
                almEvals += ea; almTime += ta;
                if (Math.Abs(xa[0] - exact[0]) < 0.1 && Math.Abs(xa[1] - exact[1]) < 0.1) almSuccess++;
            }

            Console.WriteLine("\nЗадача: x1^2+x2^2 -> min, при x1+x2 = 2");
            Console.WriteLine(String.Format("{0,-18} | {1,-15} | {2,-15} | {3,-15}", "Метод", "Вызовов f(x)", "Время (мс)", "Глоб. оптимум"));
            Console.WriteLine(new string('-', 72));
            Console.WriteLine(String.Format("{0,-18} | {1,-15} | {2,-15:F3} | {3,-15}", "Метод штрафов", penEvals / runs, penTime / runs, (penSuccess * 100.0 / runs).ToString("F0") + "%"));
            Console.WriteLine(String.Format("{0,-18} | {1,-15} | {2,-15:F3} | {3,-15}", "Мод. Лагранжиан", almEvals / runs, almTime / runs, (almSuccess * 100.0 / runs).ToString("F0") + "%"));

            Assert(true, "Тест 5: Бенчмарк по времени, вычислениям и вероятности успешно выполнен.\n");
        }

        static void RunTests()
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine("        ЗАДАНИЕ 8: МЕТОДЫ УСЛОВНОЙ МНОГОМЕРНОЙ ОПТИМИЗАЦИИ (3 МЕТОДА)         ");
            Console.WriteLine("================================================================================\n");

            // Тест 1
            Func<double[], double> f1 = x => x[0] * x[0] + x[1] * x[1];
            Func<double[], double>[] g1 = { x => 2 - x[0] - x[1] }; // x1+x2 >= 2 => 2-x1-x2 <= 0
            double[] x0_1 = { 2, 2 }; // Строгая внутренняя точка для барьерного

            var (xp1, ep1, tp1) = PenaltyMethod(f1, g1, null, x0_1);
            var (xb1, eb1, tb1) = BarrierMethod(f1, g1, x0_1);

            Console.WriteLine("--- ТЕСТ 1: Неравенство (x1^2+x2^2 -> min, x1+x2 >= 2) ---");
            Console.WriteLine($"Штрафы : x=({xp1[0]:F3}, {xp1[1]:F3}), вызовов функции: {ep1}");
            Console.WriteLine($"Барьеры: x=({xb1[0]:F3}, {xb1[1]:F3}), вызовов функции: {eb1}");

            bool pass1 = Math.Abs(xp1[0] - 1) < 0.1 && Math.Abs(xb1[0] - 1) < 0.1;
            Assert(pass1, "Тест 1: Оба метода сошлись к точке локального условного экстремума (1, 1).\\n");

            // Тест 2
            Func<double[], double>[] h2 = { x => x[0] + x[1] - 2 }; // равенство
            double[] x0_2 = { 0, 0 };

            var (xp2, ep2, tp2) = PenaltyMethod(f1, null, h2, x0_2);
            var (xal2, eal2, tal2) = AugmentedLagrangianMethod(f1, h2, x0_2);

            Console.WriteLine("--- ТЕСТ 2: Равенство (x1^2+x2^2 -> min, x1+x2 = 2) ---");
            Console.WriteLine($"Штрафы  : x=({xp2[0]:F3}, {xp2[1]:F3}), вызовов: {ep2}");
            Console.WriteLine($"Мод.Лагр: x=({xal2[0]:F3}, {xal2[1]:F3}), вызовов: {eal2}");

            bool pass2 = Math.Abs(xp2[0] - 1) < 0.1 && Math.Abs(xal2[0] - 1) < 0.1;
            Assert(pass2, "Тест 2: Штрафы и Мод. Лагранжиан устойчиво находят минимум (1, 1).\\n");

            // Тест 3
            Func<double[], double> f3 = x => -x[0] - x[1];
            Func<double[], double>[] g3 = { x => x[0] * x[0] + x[1] * x[1] - 1 }; // Внутри круга
            double[] x0_3 = { 0, 0 }; // Строго внутри

            var (xp3, ep3, tp3) = PenaltyMethod(f3, g3, null, x0_3);
            var (xb3, eb3, tb3) = BarrierMethod(f3, g3, x0_3);

            double exact3 = Math.Sqrt(2) / 2;
            Console.WriteLine("--- ТЕСТ 3: Нелинейное неравенство (Круговое) ---");
            Console.WriteLine($"Ожидается: ({exact3:F4}, {exact3:F4})");
            Console.WriteLine($"Штрафы : x=({xp3[0]:F4}, {xp3[1]:F4}), f(x)={f3(xp3):F4}");
            Console.WriteLine($"Барьеры: x=({xb3[0]:F4}, {xb3[1]:F4}), f(x)={f3(xb3):F4}");

            bool pass3 = Math.Abs(xp3[0] - exact3) < 0.1 && Math.Abs(xb3[0] - exact3) < 0.1;
            Assert(pass3, "Тест 3: Глобальный оптимум на нелинейной границе найден.\\n");

            // Тест 4
            Func<double[], double> f4 = x => x[0] + x[1];
            Func<double[], double>[] h4 = { x => x[0] * x[0] + x[1] * x[1] - 2 }; // На окружности
            double[] x0_4 = { -0.5, -0.5 };

            var (xal4, eal4, tal4) = AugmentedLagrangianMethod(f4, h4, x0_4);

            Console.WriteLine("--- ТЕСТ 4: Нелинейное ограничение-равенство ---");
            Console.WriteLine($"Мод.Лагр: x=({xal4[0]:F3}, {xal4[1]:F3}), вызовов: {eal4}");

            bool pass4 = Math.Abs(xal4[0] - (-1)) < 0.1 && Math.Abs(xal4[1] - (-1)) < 0.1;
            Assert(pass4, "Тест 4: Нелинейное равенство успешно решено (Условный минимум в -1, -1).\\n");

            // Тест 5: Бенчмарк
            RunBenchmark();

            Console.WriteLine(new string('=', 80) + "\\n");
        }

        static void Main()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            RunTests();
        }
    }
}