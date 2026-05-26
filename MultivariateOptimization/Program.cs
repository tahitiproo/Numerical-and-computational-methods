using System;
using System.Diagnostics;

namespace MultivariateOptimization
{
    class Program
    {
        // Rosenbrock function and derivatives
        static double Rosenbrock(double[] x)
        {
            double a = 1 - x[0];
            double b = x[1] - x[0] * x[0];
            return a * a + 100 * b * b;
        }

        static double[] GradRosenbrock(double[] x)
        {
            double df_dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] * x[0]);
            double df_dy = 200 * (x[1] - x[0] * x[0]);
            return new double[] { df_dx, df_dy };
        }

        static double[][] HessianRosenbrock(double[] x)
        {
            double d2f_dx2 = 2 - 400 * (x[1] - x[0] * x[0]) + 800 * x[0] * x[0];
            double d2f_dxdy = -400 * x[0];
            double d2f_dy2 = 200;
            return new double[][] {
                new double[] { d2f_dx2, d2f_dxdy },
                new double[] { d2f_dxdy, d2f_dy2 }
            };
        }

        // 2x2 linear system solver H * step = g
        static double[] Solve2x2(double[][] H, double[] g)
        {
            double a = H[0][0], b = H[0][1];
            double c = H[1][0], d = H[1][1];
            double det = a * d - b * c;
            if (Math.Abs(det) < 1e-12)
                throw new InvalidOperationException("Singular Hessian.");
            double step0 = (d * g[0] - b * g[1]) / det;
            double step1 = (-c * g[0] + a * g[1]) / det;
            return new double[] { step0, step1 };
        }

        static double Norm(double[] x)
        {
            return Math.Sqrt(x[0] * x[0] + x[1] * x[1]);
        }

        // Golden‑section search for minimizer of f on [a, b]
        static double MinimizeGoldenSection(Func<double, double> f, double a, double b,
                                            double tol = 1e-6, int maxIter = 1000)
        {
            double phi = (1 + Math.Sqrt(5)) / 2.0;      // ≈ 1.618
            double resphi = 2.0 - phi;                  // ≈ 0.382
            double x1 = a + resphi * (b - a);
            double x2 = b - resphi * (b - a);
            double f1 = f(x1);
            double f2 = f(x2);

            for (int i = 0; i < maxIter && (b - a) > tol; i++)
            {
                if (f1 < f2)
                {
                    b = x2; x2 = x1; f2 = f1;
                    x1 = a + resphi * (b - a);
                    f1 = f(x1);
                }
                else
                {
                    a = x1; x1 = x2; f1 = f2;
                    x2 = b - resphi * (b - a);
                    f2 = f(x2);
                }
            }
            return (a + b) / 2.0;
        }

        // Oracle that tracks calls
        class Oracle
        {
            public int FCount { get; private set; }
            public int GradCount { get; private set; }
            public int HessCount { get; private set; }

            public double F(double[] x)
            {
                FCount++;
                return Rosenbrock(x);
            }

            public double[] Grad(double[] x)
            {
                GradCount++;
                return GradRosenbrock(x);
            }

            public double[][] Hess(double[] x)
            {
                HessCount++;
                return HessianRosenbrock(x);
            }

            public void Reset()
            {
                FCount = 0;
                GradCount = 0;
                HessCount = 0;
            }
        }

        // Градиентный спуск
        static (double[] x, int iters, double ms) GradientDescent(double[] start, Oracle oracle,
            double lr = 0.0015, double tol = 1e-6, int maxIter = 10000)
        {
            double[] x = (double[])start.Clone();
            oracle.Reset();
            Stopwatch sw = Stopwatch.StartNew();
            int i;
            for (i = 0; i < maxIter; i++)
            {
                double[] g = oracle.Grad(x);
                if (Norm(g) < tol) break;
                for (int j = 0; j < x.Length; j++)
                    x[j] -= lr * g[j];
            }
            sw.Stop();
            return (x, i + 1, sw.Elapsed.TotalMilliseconds);
        }

        // Наискорейший спуск
        static (double[] x, int iters, double ms) SteepestDescent(double[] start, Oracle oracle,
            double tol = 1e-6, int maxIter = 10000)
        {
            double[] x = (double[])start.Clone();
            oracle.Reset();
            Stopwatch sw = Stopwatch.StartNew();
            int i;
            for (i = 0; i < maxIter; i++)
            {
                double[] g = oracle.Grad(x);
                if (Norm(g) < tol) break;

                double alpha = MinimizeGoldenSection(alphaVal =>
                {
                    double[] tmp = new double[2];
                    tmp[0] = x[0] - alphaVal * g[0];
                    tmp[1] = x[1] - alphaVal * g[1];
                    return oracle.F(tmp);
                }, 0.0, 1.0, 1e-6);

                for (int j = 0; j < x.Length; j++)
                    x[j] -= alpha * g[j];
            }
            sw.Stop();
            return (x, i + 1, sw.Elapsed.TotalMilliseconds);
        }

        // Тяжелый шарик
        static (double[] x, int iters, double ms) HeavyBall(double[] start, Oracle oracle,
            double lr = 0.001, double beta = 0.9, double tol = 1e-6, int maxIter = 10000)
        {
            double[] x = (double[])start.Clone();
            double[] v = new double[2];
            oracle.Reset();
            Stopwatch sw = Stopwatch.StartNew();
            int i;
            for (i = 0; i < maxIter; i++)
            {
                double[] g = oracle.Grad(x);
                if (Norm(g) < tol) break;
                for (int j = 0; j < x.Length; j++)
                {
                    v[j] = beta * v[j] - lr * g[j];
                    x[j] += v[j];
                }
            }
            sw.Stop();
            return (x, i + 1, sw.Elapsed.TotalMilliseconds);
        }

        // Метод Ньютона
        static (double[] x, int iters, double ms) NewtonMethod(double[] start, Oracle oracle,
            double tol = 1e-6, int maxIter = 1000)
        {
            double[] x = (double[])start.Clone();
            oracle.Reset();
            Stopwatch sw = Stopwatch.StartNew();
            int i;
            for (i = 0; i < maxIter; i++)
            {
                double[] g = oracle.Grad(x);
                if (Norm(g) < tol) break;
                double[][] H = oracle.Hess(x);
                try
                {
                    double[] step = Solve2x2(H, g);
                    for (int j = 0; j < x.Length; j++)
                        x[j] -= step[j];
                }
                catch (InvalidOperationException)
                {
                    break; // singular Hessian – abort
                }
            }
            sw.Stop();
            return (x, i + 1, sw.Elapsed.TotalMilliseconds);
        }

        // -------------------------------------------------------------------------
        // МИНИ-ФРЕЙМВОРК ДЛЯ ТЕСТИРОВАНИЯ
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
            Console.WriteLine("Запуск нетривиальных тестов...\n");

            // Тест 1
            var oracle1 = new Oracle();
            NewtonMethod(new double[] { -1.2, 1.2 }, oracle1, tol: 0, maxIter: 5);
            bool test1Passed = oracle1.GradCount == 5 && oracle1.HessCount == 5 && oracle1.FCount == 0;
            
            oracle1.Reset();
            SteepestDescent(new double[] { -1.2, 1.2 }, oracle1, tol: 0, maxIter: 5);
            test1Passed &= oracle1.GradCount == 5 && oracle1.HessCount == 0 && oracle1.FCount > 0;
            Assert(test1Passed, "Тест 1: Оракул корректно считает вызовы F, Grad и Hess.");

            // Тест 2
            var oracle2 = new Oracle();
            var (_, iters2, _) = NewtonMethod(new double[] { 0, 0.005 }, oracle2);
            Assert(iters2 == 1, "Тест 2: Метод Ньютона безопасно прерывается при встрече с сингулярным гессианом.");

            // Тест 3
            var oracle3 = new Oracle();
            double[] start3 = { -1.5, 2.5 };
            var (xGD, itersGD, _) = GradientDescent(start3, oracle3, lr: 0.001, tol: 1e-6, maxIter: 50);
            var (xHB, itersHB, _) = HeavyBall(start3, oracle3, lr: 0.001, beta: 0.0, tol: 1e-6, maxIter: 50);
            bool test3Passed = Math.Abs(xGD[0] - xHB[0]) < 1e-12 && Math.Abs(xGD[1] - xHB[1]) < 1e-12 && itersGD == itersHB;
            Assert(test3Passed, "Тест 3: Метод Тяжелого шарика с инерцией 0 побитово совпадает с классическим GD.");

            // Тест 4
            double min1D = MinimizeGoldenSection(x => (x - 0.333) * (x - 0.333), 0, 1, 1e-6);
            Assert(Math.Abs(min1D - 0.333) < 1e-5, "Тест 4: Золотое сечение корректно локализует минимум 1D функции.");

            // Тест 5 
            // Стартуем из точки (0.99, 0.98), где Гессиан положительно определен (det(H) > 0).
            var oracle5 = new Oracle();
            var (xNewt, itersNewt, _) = NewtonMethod(new double[] { 0.99, 0.98 }, oracle5, tol: 1e-7, maxIter: 100);
            double errX = Math.Abs(xNewt[0] - 1.0);
            double errY = Math.Abs(xNewt[1] - 1.0);
            // Если находимся в зоне выпуклости, метод сойдется менее чем за 5 шагов
            Assert(itersNewt <= 5 && errX < 1e-6 && errY < 1e-6, "Тест 5: Локальная квадратичная сходимость метода Ньютона.");

            Console.WriteLine(new string('-', 80));
        }

        // -------------------------------------------------------------------------
        static void RunSingleComparison()
        {
            double[] startPoint = { -1.2, 1.2 };
            double[] trueMin = { 1.0, 1.0 };

            var oracle = new Oracle();

            var methods = new (string name, Func<double[], (double[] x, int iters, double ms)> func)[]
            {
                ("Градиентный спуск",   s => GradientDescent(s, oracle)),
                ("Наискорейший спуск",  s => SteepestDescent(s, oracle)),
                ("Тяжелый шарик",       s => HeavyBall(s, oracle)),
                ("Метод Ньютона",       s => NewtonMethod(s, oracle))
            };

            string header = $"{"Метод",-22} | {"Ит.",-5} | {"Время(мс)",-10} | {"Точность",-10} | {"#f ",-5} | {"#grad",-5} | {"#hess",-5}";
            Console.WriteLine(header);
            Console.WriteLine(new string('-', header.Length));

            foreach (var (name, func) in methods)
            {
                oracle.Reset();
                var (resX, iters, duration) = func(startPoint);
                double accuracy = Norm(new double[] { resX[0] - trueMin[0], resX[1] - trueMin[1] });
                Console.WriteLine($"{name,-22} | {iters,-5} | {duration,10:F4} | {accuracy,10:0.0e+00} | {oracle.FCount,-5} | {oracle.GradCount,-5} | {oracle.HessCount,-5}");
            }
        }

        static void RunProbabilisticComparison(int numTrials, double boxHalfWidth, int maxIter = 10000)
        {
            double[] trueMin = { 1.0, 1.0 };
            double successTol = 1e-3;
            var rng = new Random(42);

            var methods = new (string name, Func<double[], (double[] x, int iters, double ms)> func)[]
            {
                ("Градиентный спуск",   s => GradientDescent(s, new Oracle())),
                ("Наискорейший спуск",  s => SteepestDescent(s, new Oracle())),
                ("Тяжелый шарик",       s => HeavyBall(s, new Oracle())),
                ("Метод Ньютона",       s => NewtonMethod(s, new Oracle()))
            };

            Console.WriteLine($"\nВероятностное тестирование: {numTrials} запусков из [-{boxHalfWidth},{boxHalfWidth}]^2");
            Console.WriteLine($"Критерий успеха: ||x - (1,1)|| < {successTol}");
            Console.WriteLine(new string('-', 80));

            foreach (var (name, func) in methods)
            {
                int successCount = 0;
                int totalIters = 0;
                double totalTime = 0;
                double totalAccuracy = 0;

                for (int t = 0; t < numTrials; t++)
                {
                    double[] start = new double[]
                    {
                        rng.NextDouble() * 2 * boxHalfWidth - boxHalfWidth,
                        rng.NextDouble() * 2 * boxHalfWidth - boxHalfWidth
                    };

                    var (resX, iters, duration) = func(start);
                    double acc = Norm(new double[] { resX[0] - trueMin[0], resX[1] - trueMin[1] });

                    if (acc < successTol)
                    {
                        successCount++;
                        totalIters += iters;
                        totalTime += duration;
                        totalAccuracy += acc;
                    }
                }

                double prob = (double)successCount / numTrials * 100.0;
                if (successCount > 0)
                {
                    double avgIters = (double)totalIters / successCount;
                    double avgTime = totalTime / successCount;
                    double avgAcc = totalAccuracy / successCount;
                    Console.WriteLine($"{name,-22} | успех {prob,5:F1}% | ср.ит. {avgIters,6:F1} | ср.время(мс) {avgTime,8:F3} | ср.точность {avgAcc,10:0.0e+00}");
                }
                else
                {
                    Console.WriteLine($"{name,-22} | успех {prob,5:F1}% | (нет успешных запусков)");
                }
            }
        }

        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            
            // Сначала запустим модульные тесты
            RunTests();

            Console.WriteLine("Сравнение методов для функции Розенброка\n");
            RunSingleComparison();
            RunProbabilisticComparison(100, 2.0);
        }
    }
}