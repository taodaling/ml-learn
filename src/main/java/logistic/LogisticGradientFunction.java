package logistic;

import algo.Sigmoid;
import numeric.KahamSummation;
import numeric.Vector;

import java.util.Arrays;
import java.util.function.Function;

public class LogisticGradientFunction implements Function<double[], double[]> {
    double[][] xs;
    double[] ys;
    double regularLambda;

    public LogisticGradientFunction(double[][] xs, double[] ys, double regularLambda) {
        this.xs = xs;
        this.ys = ys;
        this.regularLambda = regularLambda;
    }

    protected double computeActual(double[] theta, double[] x) {
        return Sigmoid.apply(Vector.dotmul(theta, x));
    }

    @Override
    public double[] apply(double[] theta) {
        int m = xs.length;
        int n = xs[0].length;
        KahamSummation[] sum = new KahamSummation[n];
        for (int i = 0; i < n; i++) {
            sum[i] = new KahamSummation();
        }
        for (int i = 0; i < m; i++) {
            double hx = computeActual(theta, xs[i]);
            double delta = hx - ys[i];
            for (int j = 0; j < n; j++) {
                sum[j].add(delta * xs[i][j]);
            }
        }
        for (int i = 1; i < n; i++) {
            sum[i].add(regularLambda / m * theta[i]);
        }
        return Arrays.stream(sum).mapToDouble(x -> x.sum() / m).toArray();
    }
}
