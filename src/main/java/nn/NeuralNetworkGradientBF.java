package nn;

import lombok.extern.slf4j.Slf4j;

import java.util.function.Function;
import java.util.function.ToDoubleFunction;

import static numeric.Vector.*;

@Slf4j
public class NeuralNetworkGradientBF {
    ToDoubleFunction<double[][][]> cf;

    public NeuralNetworkGradientBF(ToDoubleFunction<double[][][]> cf) {
        this.cf = cf;
    }

    public double[][][] gradient(double[][][] theta) {
        double fix = 1e-4;
        double[][][] der = deepClone(theta);
        for (int i = 0; i < theta.length; i++) {
            for (int j = 0; j < theta[i].length; j++) {
                for (int k = 0; k < theta[i][j].length; k++) {
                    double old = theta[i][j][k];
                    theta[i][j][k] = old + fix;
                    double a = cf.applyAsDouble(theta);
                    theta[i][j][k] = old - fix;
                    double b = cf.applyAsDouble(theta);
                    theta[i][j][k] = old;
                    der[i][j][k] = (a - b) / (2 * fix);
                }
            }
        }
        return der;
    }

    public static void test(NeuralNetworkGradientBF bf, Function<double[][][], double[][][]> g, int round, double prec,
                            int... levels) {
        for (int r = 1; r <= round; r++) {
            log.info("{}-th gradient test start", r);
            double[][][] theta = NeuralNetworkUtils.createTheta(levels);
            for (int i = 0; i < theta.length; i++) {
                for (int j = 0; j < theta[i].length; j++) {
                    for (int k = 0; k < theta[i][j].length; k++) {
                        theta[i][j][k] = Math.random() * 2 - 1;
                    }
                }
            }
            double[][][] bfRes = bf.gradient(theta);
            double[][][] smartRes = g.apply(theta);
            for (int i = 0; i < theta.length; i++) {
                for (int j = 0; j < theta[i].length; j++) {
                    for (int k = 0; k < theta[i][j].length; k++) {
                        if (Math.abs(bfRes[i][j][k] - smartRes[i][j][k]) >= prec) {
                            log.error("fail gradient test, bf is {}, and smart is {}", bfRes[i][j][k], smartRes[i][j][k]);
                            throw new AssertionError();
                        }
                    }
                }
            }
        }
    }
}
