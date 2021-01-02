package algo;

import lombok.extern.slf4j.Slf4j;

import java.util.Random;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

@Slf4j
public class GradientBF {
    ToDoubleFunction<double[]> cf;

    public GradientBF(ToDoubleFunction<double[]> cf) {
        this.cf = cf;
    }

    public double[] gradient(double[] theta) {
        double fix = 1e-4;
        double[] ans = new double[theta.length];
        for (int i = 0; i < theta.length; i++) {
            double old = theta[i];
            theta[i] = old + fix;
            double a = cf.applyAsDouble(theta);
            theta[i] = old - fix;
            double b = cf.applyAsDouble(theta);
            theta[i] = old;
            ans[i] = (a - b) / (2 * fix);
        }
        return ans;
    }

    public static void test(GradientBF bf, Function<double[], double[]> g, int round, double prec, int n) {
        for (int i = 1; i <= round; i++) {
            log.info("{}-th gradient test start", i);
            double[] theta = new double[n];
            for (int j = 0; j < n; j++) {
                theta[j] = Math.random() * 2 - 1;
            }
            double[] bfRes = bf.gradient(theta);
            double[] smartRes = g.apply(theta);
            for (int j = 0; j < theta.length; j++) {
                if (Math.abs(bfRes[j] - smartRes[j]) >= prec) {
                    log.error("fail gradient test, bf is {}, and smart is {}", bfRes[j], smartRes[j]);
                    throw new AssertionError();
                }
            }
        }
    }
}
