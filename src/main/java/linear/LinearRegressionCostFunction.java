package linear;

import numeric.KahamSummation;

import java.util.function.ToDoubleFunction;

public class LinearRegressionCostFunction implements ToDoubleFunction<double[]> {
    private double[][] xs;
    private double[] ys;
    private double regularLambda;

    public LinearRegressionCostFunction(double[][] xs, double[] ys, double regularLambda) {
        this.xs = xs;
        this.ys = ys;
        this.regularLambda = regularLambda;
    }

    @Override
    public double applyAsDouble(double[] theta) {
        LinearRegression lr = new LinearRegression(theta);
        double[] actual = new double[ys.length];
        for (int i = 0; i < ys.length; i++) {
            actual[i] = lr.apply(xs[i]);
        }
        KahamSummation dist = new KahamSummation();
        for (int i = 0; i < ys.length; i++) {
            double delta = actual[i] - ys[i];
            dist.add(delta * delta);
        }
        KahamSummation reg = new KahamSummation();
        for (int i = 1; i < theta.length; i++) {
            reg.add(theta[i] * theta[i]);
        }
        return (dist.sum() + regularLambda * reg.sum()) / (2 * xs.length);
    }
}
