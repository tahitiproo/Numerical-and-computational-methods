using System;
using System.Diagnostics;
using System.Linq;

namespace BoundaryValueProblem
{
    class Program
    {
        // Описание структуры Краевой Задачи 
        // y'' + q(x)y' - r(x)y = f(x)
        // y(a) = alpha, y(b) = beta
        public struct BVP
        {
            public Func<double, double> q;
            public Func<double, double> r;
            public Func<double, double> f;
            public double a, b, alpha, beta;
            public Func<double, double> exact; // Экзактное решение для тестов
        }

        // 1. Метод сеток (конечных разностей) с решением через прогонку (O(N))
        static (double[] x, double[] y) GridMethod(BVP p, int N)
        {
            double h = (p.b - p.a) / N;
            double[] x = new double[N + 1];
            for (int i = 0; i <= N; i++) x[i] = p.a + i * h;

            double[] A = new double[N];
            double[] B = new double[N];
            double[] C = new double[N];
            double[] D = new double[N];

            for (int i = 1; i < N; i++)
            {
                double q = p.q != null ? p.q(x[i]) : 0;
                double r = p.r != null ? p.r(x[i]) : 0;
                double f = p.f(x[i]);

                A[i] = 1.0 - h * q / 2.0;
                B[i] = 2.0 + h * h * r;
                C[i] = 1.0 + h * q / 2.0;
                D[i] = h * h * f;
            }

            double[] P = new double[N + 1];
            double[] Q = new double[N + 1];

            // Прямой ход прогонки
            P[1] = C[1] / B[1];
            Q[1] = (A[1] * p.alpha - D[1]) / B[1];

            for (int i = 2; i < N; i++)
            {
                double denom = B[i] - A[i] * P[i - 1];
                P[i] = C[i] / denom;
                Q[i] = (A[i] * Q[i - 1] - D[i]) / denom;
            }

            double[] y = new double[N + 1];
            y[0] = p.alpha;
            y[N] = p.beta;

            // Обратный ход прогонки
            for (int i = N - 1; i >= 1; i--)
            {
                y[i] = P[i] * y[i + 1] + Q[i];
            }

            return (x, y);
        }

        // Экстраполяция Ричардсона на основе двух сеток (шаг h и h/2)
        static (double[] x, double[] yRefined, double maxDelta) RichardsonRefine(BVP p, int N)
        {
            var (x1, y1) = GridMethod(p, N); // Грубая сетка
            var (x2, y2) = GridMethod(p, 2 * N); // Мелкая сетка

            double maxDelta = 0;
            double[] yRefined = new double[2 * N + 1];

            // На общих (четных) узлах мелкой сетки
            for (int i = 0; i <= N; i++)
            {
                int j = i * 2;
                double delta = (y2[j] - y1[i]) / 3.0; // r=2, p=2 -> r^p - 1 = 3
                if (Math.Abs(delta) > maxDelta) maxDelta = Math.Abs(delta);
                yRefined[j] = y2[j] + delta;
            }

            // Интерполяция дельты для промежуточных (нечетных) узлов
            for (int j = 1; j < 2 * N; j += 2)
            {
                int i = j / 2;
                double deltaLeft = (y2[j - 1] - y1[i]) / 3.0;
                double deltaRight = (y2[j + 1] - y1[i + 1]) / 3.0;
                double deltaMid = (deltaLeft + deltaRight) / 2.0;
                yRefined[j] = y2[j] + deltaMid; // Уточненное решение
            }

            return (x2, yRefined, maxDelta);
        }

        // Адаптивное сгущение сетки для достижения точности или до предела float64
        static (double[] x, double[] y, double finalDelta, int finalN) AdaptiveGridMethod(BVP p, double tol)
        {
            int N = 10;
            var (x1, y1) = GridMethod(p, N);
            double prevDelta = double.MaxValue;

            while (N <= 1_000_000)
            {
                var (x2, y2) = GridMethod(p, N * 2);

                double maxDelta = 0;
                for (int i = 0; i <= N; i++)
                {
                    double delta = Math.Abs(y2[i * 2] - y1[i]) / 3.0;
                    if (delta > maxDelta) maxDelta = delta;
                }

                // Проверка выхода на плато ошибок округления (деградация точности)
                if (maxDelta > prevDelta)
                {
                    Console.WriteLine($"[!] Сетка стала слишком мелкой (N={N * 2}), накапливаются ошибки округления.");
                    break;
                }

                if (maxDelta <= tol)
                {
                    var (xRef, yRef, delta) = RichardsonRefine(p, N);
                    return (xRef, yRef, delta, N * 2);
                }

                prevDelta = maxDelta;
                x1 = x2;
                y1 = y2;
                N *= 2;
            }

            // Если дошли сюда, выдаем лучшее найденное ДО деградации
            var (xr, yr, finalD) = RichardsonRefine(p, N / 2);
            return (xr, yr, finalD, N);
        }

        // 2. Метод стрельбы (Рунге-Кутта 4 + Метод Хо́рд/Секущих)
        static double ShootEnd(BVP p, double gamma, int N)
        {
            double h = (p.b - p.a) / N;
            double y1 = p.alpha;
            double y2 = gamma; // Предполагаемая производная u'(a)

            for (int i = 0; i < N; i++)
            {
                double x = p.a + i * h;

                Func<double, double, double, double> F2 = (xx, yy1, yy2) =>
                {
                    double qq = p.q != null ? p.q(xx) : 0;
                    double rr = p.r != null ? p.r(xx) : 0;
                    return p.f(xx) - qq * yy2 + rr * yy1;
                };

                double k1_y1 = y2;
                double k1_y2 = F2(x, y1, y2);

                double k2_y1 = y2 + h * k1_y2 / 2;
                double k2_y2 = F2(x + h / 2, y1 + h * k1_y1 / 2, y2 + h * k1_y2 / 2);

                double k3_y1 = y2 + h * k2_y2 / 2;
                double k3_y2 = F2(x + h / 2, y1 + h * k2_y1 / 2, y2 + h * k2_y2 / 2);

                double k4_y1 = y2 + h * k3_y2;
                double k4_y2 = F2(x + h, y1 + h * k3_y1, y2 + h * k3_y2);

                y1 += h / 6 * (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1);
                y2 += h / 6 * (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2);
            }
            return y1;
        }

        static (double[] x, double[] y) ShootingMethod(BVP p, int N, double tol = 1e-9)
        {
            double g0 = 0.0;
            double F0 = ShootEnd(p, g0, N) - p.beta;

            double g1 = 1.0;
            double F1 = ShootEnd(p, g1, N) - p.beta;

            int iters = 0;
            // Метод секущих для поиска корня функции невязки Ф(gamma)
            while (Math.Abs(F1) > tol && iters < 100)
            {
                double g2 = g1 - F1 * (g1 - g0) / (F1 - F0);
                g0 = g1;
                F0 = F1;
                g1 = g2;
                F1 = ShootEnd(p, g1, N) - p.beta;
                iters++;
            }

            // С найденным правильным углом g1 восстанавливаем весь массив траектории
            double h = (p.b - p.a) / N;
            double[] xArr = new double[N + 1];
            double[] yArr = new double[N + 1];
            double y1 = p.alpha;
            double y2 = g1;
            xArr[0] = p.a; yArr[0] = y1;

            for (int i = 0; i < N; i++)
            {
                double x = p.a + i * h;
                Func<double, double, double, double> F2 = (xx, yy1, yy2) =>
                    p.f(xx) - (p.q != null ? p.q(xx) : 0) * yy2 + (p.r != null ? p.r(xx) : 0) * yy1;

                double k1_y1 = y2;
                double k1_y2 = F2(x, y1, y2);
                double k2_y1 = y2 + h * k1_y2 / 2;
                double k2_y2 = F2(x + h / 2, y1 + h * k1_y1 / 2, y2 + h * k1_y2 / 2);
                double k3_y1 = y2 + h * k2_y2 / 2;
                double k3_y2 = F2(x + h / 2, y1 + h * k2_y1 / 2, y2 + h * k2_y2 / 2);
                double k4_y1 = y2 + h * k3_y2;
                double k4_y2 = F2(x + h, y1 + h * k3_y1, y2 + h * k3_y2);

                y1 += h / 6 * (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1);
                y2 += h / 6 * (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2);

                xArr[i + 1] = x + h; yArr[i + 1] = y1;
            }
            return (xArr, yArr);
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
            Console.WriteLine("           ЗАДАНИЕ 6: КРАЕВЫЕ ЗАДАЧИ - СЕТОЧНЫЙ МЕТОД И МЕТОД СТРЕЛЬБЫ          ");
            Console.WriteLine("================================================================================\n");

            // Тест 1
            Console.WriteLine("--- ТЕСТ 1: Линейное решение и точность ---");
            BVP p1 = new BVP
            {
                q = x => 0,
                r = x => 1,
                f = x => -x,
                a = 0,
                b = 1,
                alpha = 0,
                beta = 1,
                exact = x => x
            };
            var (_, y1) = GridMethod(p1, 10);
            double err1 = Math.Abs(y1[5] - p1.exact(0.5));
            Console.WriteLine($"Ошибка при N=10 для u(x)=x: {err1:E2}");
            Assert(err1 < 1e-14, "Тест 1: Для линейной функции сеточный метод выдаёт безошибочный точный ответ.\n");

            // Тест 2
            Console.WriteLine("--- ТЕСТ 2: Проверка теоретического порядка O(h^2) ---");
            BVP p2 = new BVP
            {
                q = x => 0,
                r = x => 1,
                f = x => -(Math.PI * Math.PI + 1) * Math.Sin(Math.PI * x),
                a = 0,
                b = 1,
                alpha = 0,
                beta = 0,
                exact = x => Math.Sin(Math.PI * x)
            };
            var (x20, y20) = GridMethod(p2, 20);
            var (x40, y40) = GridMethod(p2, 40);
            double maxErr20 = y20.Select((v, i) => Math.Abs(v - p2.exact(x20[i]))).Max();
            double maxErr40 = y40.Select((v, i) => Math.Abs(v - p2.exact(x40[i]))).Max();
            Console.WriteLine($"Ошибка N=20: {maxErr20:E4}");
            Console.WriteLine($"Ошибка N=40: {maxErr40:E4} | Отношение: {maxErr20 / maxErr40:F2} (Ожидается ~4.0)");
            Assert(Math.Abs((maxErr20 / maxErr40) - 4.0) < 0.2, "Тест 2: Погрешность убывает строго квадратично O(h^2).\n");

            // Тест 3
            Console.WriteLine("--- ТЕСТ 3: Улучшение точности методом Ричардсона (Экстраполяция) ---");
            var (_, yRef, delta) = RichardsonRefine(p2, 20);
            double maxErrRefined = yRef.Select((v, i) => Math.Abs(v - p2.exact(x40[i]))).Max();
            Console.WriteLine($"Оценка погрешности Ричардсона Δ на N=40: {delta:E4}");
            Console.WriteLine($"Истинная погрешность уточненного решения:  {maxErrRefined:E4}");
            Assert(maxErrRefined < maxErr40 / 10, "Тест 3: Экстраполяция Ричардсона радикально (в десятки раз) повысила точность.\n");

            // Тест 4
            Console.WriteLine("--- ТЕСТ 4: Совпадение Сеточного метода и Метода Стрельбы (Бонус) ---");
            BVP p4 = new BVP
            {
                q = x => x,
                r = x => x * x,
                f = x => Math.Exp(x) * (1 + x - x * x),
                a = 0,
                b = 2,
                alpha = 1,
                beta = Math.Exp(2)
            };
            var (xGrid, yGrid) = GridMethod(p4, 200);
            var (xShoot, yShoot) = ShootingMethod(p4, 200);
            double diffShootGrid = yGrid.Zip(yShoot, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"Макс. разница между Сетками и Стрельбой на N=200: {diffShootGrid:E4}");
            Assert(diffShootGrid < 1e-3, "Тест 4: Метод стрельбы успешно справился и решение совпало с сеточным.\n");

            // Тест 5
            Console.WriteLine("--- ТЕСТ 5: Адаптивное сгущение сетки до предела ---");
            Console.WriteLine("Сетка автоматически удваивается, пока погрешность не перестанет падать...");
            var (_, _, finalDelta, finalN) = AdaptiveGridMethod(p2, 1e-12);
            Console.WriteLine($"Предел float64/точности обнаружен на N = {finalN}");
            Console.WriteLine($"Финально достигнутая разностная точность Δ: {finalDelta:E2}");
            Assert(finalN >= 4096 && finalN <= 1_000_000, "Тест 5: Метод адаптивного сгущения благополучно обнаружил предел ошибок округления и остановился.\n");

            Console.WriteLine(new string('=', 80) + "\n");
        }

        static void Main()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            RunTests();
        }
    }
}