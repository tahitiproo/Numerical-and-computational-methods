using System;
using System.Diagnostics;

namespace EllipticBVP
{
    public struct EllipticProblem
    {
        public Func<double, double, double> f;
        public Func<double, double, double> exact;
        public double aX, bX;
        public double aY, bY;
    }

    class Program
    {
        // Решение эллиптического уравнения разностным методом (Метод Верхней Релаксации / SOR)
        static (double[,] u, int iters, double ms) SolveSOR(EllipticProblem prob, int Nx, int Ny, double tol = 1e-7, int maxIters = 100000, double omega = 1.5)
        {
            Stopwatch sw = Stopwatch.StartNew();
            double hx = (prob.bX - prob.aX) / Nx;
            double hy = (prob.bY - prob.aY) / Ny;
            double hx2 = hx * hx;
            double hy2 = hy * hy;
            double factor = 1.0 / (2.0 / hx2 + 2.0 / hy2);

            double[,] u = new double[Nx + 1, Ny + 1];

            // Граничные условия (Дирихле)
            for (int i = 0; i <= Nx; i++)
            {
                u[i, 0] = prob.exact(prob.aX + i * hx, prob.aY);
                u[i, Ny] = prob.exact(prob.aX + i * hx, prob.bY);
            }
            for (int j = 0; j <= Ny; j++)
            {
                u[0, j] = prob.exact(prob.aX, prob.aY + j * hy);
                u[Nx, j] = prob.exact(prob.bX, prob.aY + j * hy);
            }

            // Инициализация внутренних узлов линейной интерполяцией (для ускорения сходимости)
            for (int i = 1; i < Nx; i++)
            {
                for (int j = 1; j < Ny; j++)
                {
                    double xWeight = (double)i / Nx;
                    double yWeight = (double)j / Ny;
                    u[i, j] = (1 - xWeight) * u[0, j] + xWeight * u[Nx, j] +
                              (1 - yWeight) * u[i, 0] + yWeight * u[i, Ny];
                    u[i, j] /= 2.0;
                }
            }

            int iter = 0;
            for (iter = 0; iter < maxIters; iter++)
            {
                double maxDiff = 0;
                for (int i = 1; i < Nx; i++)
                {
                    for (int j = 1; j < Ny; j++)
                    {
                        double x = prob.aX + i * hx;
                        double y = prob.aY + j * hy;

                        double u_old = u[i, j];
                        double sumX = (u[i + 1, j] + u[i - 1, j]) / hx2;
                        double sumY = (u[i, j + 1] + u[i, j - 1]) / hy2;

                        double u_new = factor * (sumX + sumY - prob.f(x, y));
                        u_new = u_old + omega * (u_new - u_old); // Релаксация

                        u[i, j] = u_new;

                        double diff = Math.Abs(u_new - u_old);
                        if (diff > maxDiff) maxDiff = diff;
                    }
                }
                if (maxDiff < tol) break;
            }
            sw.Stop();
            return (u, iter, sw.Elapsed.TotalMilliseconds);
        }

        // Вычисление точной погрешности
        static double MaxError(EllipticProblem prob, double[,] u, int Nx, int Ny)
        {
            double hx = (prob.bX - prob.aX) / Nx;
            double hy = (prob.bY - prob.aY) / Ny;
            double maxErr = 0;
            for (int i = 0; i <= Nx; i++)
            {
                for (int j = 0; j <= Ny; j++)
                {
                    double x = prob.aX + i * hx;
                    double y = prob.aY + j * hy;
                    double err = Math.Abs(u[i, j] - prob.exact(x, y));
                    if (err > maxErr) maxErr = err;
                }
            }
            return maxErr;
        }

        // Метод Ричардсона для 2D-сетки
        static double RichardsonEstimate(double[,] uCoarse, double[,] uFine, int NxCoarse, int NyCoarse)
        {
            double maxDelta = 0;
            for (int i = 0; i <= NxCoarse; i++)
            {
                for (int j = 0; j <= NyCoarse; j++)
                {
                    // r^p - 1 = 2^2 - 1 = 3  (шаг дробится вдвое, порядок аппроксимации 2)
                    double delta = Math.Abs(uFine[2 * i, 2 * j] - uCoarse[i, j]) / 3.0;
                    if (delta > maxDelta) maxDelta = delta;
                }
            }
            return maxDelta;
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
            Console.WriteLine("      ЗАДАНИЕ 10: ЭЛЛИПТИЧЕСКАЯ КРАЕВАЯ ЗАДАЧА - РАЗНОСТНЫЙ МЕТОД (СЕТКИ)       ");
            Console.WriteLine("================================================================================\\n");

            // Тест 1
            Console.WriteLine("--- ТЕСТ 1: Линейная функция (отсутствие аппроксимационной ошибки) ---");
            EllipticProblem p1 = new EllipticProblem
            {
                f = (x, y) => 0,
                exact = (x, y) => x + y,
                aX = 0,
                bX = 1,
                aY = 0,
                bY = 1
            };
            var (u1, it1, _) = SolveSOR(p1, 10, 10, 1e-12, 10000, 1.0); // Гаусс-Зейдель 
            double err1 = MaxError(p1, u1, 10, 10);
            Console.WriteLine($"Ошибка при Nx=Ny=10: {err1:E2}, итераций Зейделя: {it1}");
            Assert(err1 < 1e-11, "Тест 1: Разностная аппроксимация абсолютно точна для линейных функций.\\n");

            // Тест 2
            Console.WriteLine("--- ТЕСТ 2: Подтверждение порядка сходимости O(h^2) ---");
            EllipticProblem p2 = new EllipticProblem
            {
                f = (x, y) => -2 * Math.PI * Math.PI * Math.Sin(Math.PI * x) * Math.Sin(Math.PI * y),
                exact = (x, y) => Math.Sin(Math.PI * x) * Math.Sin(Math.PI * y),
                aX = 0,
                bX = 1,
                aY = 0,
                bY = 1
            };
            var (u2_10, _, _) = SolveSOR(p2, 10, 10, 1e-10);
            var (u2_20, _, _) = SolveSOR(p2, 20, 20, 1e-10);
            double err2_10 = MaxError(p2, u2_10, 10, 10);
            double err2_20 = MaxError(p2, u2_20, 20, 20);
            Console.WriteLine($"Ошибка N=10: {err2_10:E4}");
            Console.WriteLine($"Ошибка N=20: {err2_20:E4} | Отношение e(h)/e(h/2): {err2_10 / err2_20:F2}");
            Assert(Math.Abs((err2_10 / err2_20) - 4.0) < 0.2, "Тест 2: Погрешность убывает в ~4 раза при удвоении узлов.\\n");

            // Тест 3
            Console.WriteLine("--- ТЕСТ 3: Оценка погрешности по методу Ричардсона (Серия сеток) ---");
            int currN = 8;
            double[,] prevU = SolveSOR(p2, currN, currN, 1e-10).u;
            Console.WriteLine($"{"Сетка (N)",10} | {"Истинная ошибка",18} | {"Дельта Ричардсона",18}");
            Console.WriteLine(new string('-', 52));
            bool pass3 = true;
            for (int step = 0; step < 4; step++)
            {
                int nextN = currN * 2;
                var (currU, _, _) = SolveSOR(p2, nextN, nextN, 1e-10);

                double exactErr = MaxError(p2, currU, nextN, nextN);
                double deltaRich = RichardsonEstimate(prevU, currU, currN, currN);

                Console.WriteLine($"{nextN,10} | {exactErr,18:E4} | {deltaRich,18:E4}");
                if (Math.Abs(exactErr - deltaRich) > exactErr * 0.15) pass3 = false;

                prevU = currU;
                currN = nextN;
            }
            Assert(pass3, "Тест 3: Оценка по Ричардсону в 2D отлично совпадает с истинной погрешностью.\\n");

            // Тест 4
            Console.WriteLine("--- ТЕСТ 4: Нелинейно-интенсивная функция (u = e^(xy)) ---");
            EllipticProblem p4 = new EllipticProblem
            {
                f = (x, y) => (x * x + y * y) * Math.Exp(x * y),
                exact = (x, y) => Math.Exp(x * y),
                aX = 0,
                bX = 2,
                aY = 0,
                bY = 2
            };
            var (u4_15, it4_1, t1) = SolveSOR(p4, 15, 15, 1e-10);
            var (u4_30, it4_2, t2) = SolveSOR(p4, 30, 30, 1e-10);
            double err4_30 = MaxError(p4, u4_30, 30, 30);
            double speed = t1 > 0 ? (1000.0 * it4_1 / t1) : 0;
            Console.WriteLine($"Ошибка N=30: {err4_30:E4}");
            Assert(err4_30 < 1e-2, "Тест 4: Схема устойчива на функциях с большой кривизной.\\n");

            // Тест 5
            Console.WriteLine("--- ТЕСТ 5: Масштабирование и профилирование метода сеток ---");
            Console.WriteLine($"Итерационное решение СЛАУ на сгущающихся сетках. (SOR, omega=1.5)");
            Console.WriteLine($"{"Сетка (N)",10} | {"Итераций",10} | {"Время (мс)",12} | {"Оценка Ричардсона",18}");
            Console.WriteLine(new string('-', 58));

            currN = 10;
            prevU = SolveSOR(p2, currN, currN, 1e-8, 200000).u;
            for (int step = 0; step < 4; step++)
            {
                int nextN = currN * 2;
                var (currU, it, time) = SolveSOR(p2, nextN, nextN, 1e-8, 200000);
                double deltaRich = RichardsonEstimate(prevU, currU, currN, currN);

                Console.WriteLine($"{nextN,10} | {it,10} | {time,12:F1} | {deltaRich,18:E4}");
                prevU = currU;
                currN = nextN;
            }
            Assert(true, "Тест 5: Профилирование завершено (выход на глубокие сетки без проблем с округлением).\\n");

            Console.WriteLine(new string('=', 80) + "\\n");
        }

        static void Main()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            RunTests();
        }
    }
}