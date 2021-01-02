package nn;

import numeric.KahamSummation;
import utils.MathUtils;
import utils.Pair;

import java.util.function.ToDoubleFunction;

public class NeuralNetworkCostFunction implements ToDoubleFunction<double[][][]> {
    double[][] xs;
    double[][] ys;
    double regularLambda;

    public NeuralNetworkCostFunction(double[][] xs, double[][] ys, double regularLambda) {
        this.xs = xs;
        this.ys = ys;
        this.regularLambda = regularLambda;
    }

    private double cost(double[][] expect, double[][] actual) {
        int m = expect.length;
        int k = expect[0].length;
        KahamSummation sum = new KahamSummation();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                sum.add(expect[i][j] * MathUtils.log(actual[i][j]));
                sum.add((1 - expect[i][j]) * MathUtils.log(1 - actual[i][j]));
            }
        }
        return -sum.sum() / expect.length;
    }

    @Override
    public double applyAsDouble(double[][][] theta) {
        KahamSummation sum = new KahamSummation();
        int m = xs.length;
        double[][] actual = new double[m][];
        for (int i = 0; i < xs.length; i++) {
            Pair<double[][], double[][]> res = NeuralNetworkUtils.forwardPropagation(xs[i], theta);
            actual[i] = res.a[theta.length];
        }
        sum.add(cost(ys, actual));
        for (int i = 0; i < theta.length; i++) {
            for (int j = 0; j < theta[i].length; j++) {
                for (int k = 0; k < theta[i][j].length; k++) {
                    if (k != 0) {
                        sum.add(regularLambda / (2 * m) * theta[i][j][k] * theta[i][j][k]);
                    }
                }
            }
        }
        return sum.sum();
    }
}
