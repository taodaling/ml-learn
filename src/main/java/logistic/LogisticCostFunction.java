package logistic;

import algo.Sigmoid;
import numeric.KahamSummation;
import numeric.Vector;
import utils.MathUtils;

import java.util.function.ToDoubleFunction;

public class LogisticCostFunction implements ToDoubleFunction<double[]> {
    double[][] xs;
    double[] ys;
    double regularLambda;

    public LogisticCostFunction(double[][] xs, double[] ys, double regularLambda) {
        this.xs = xs;
        this.ys = ys;
        this.regularLambda = regularLambda;
    }

    @Override
    public double applyAsDouble(double[] theta) {
        KahamSummation sum = new KahamSummation();
        int m = xs.length;
        for (int i = 0; i < m; i++) {
            double hx = Sigmoid.apply(Vector.dotmul(theta, xs[i]));
            sum.add(ys[i] * MathUtils.log(hx));
            sum.add((1 - ys[i]) * MathUtils.log(1 - hx));
        }
        double cost = sum.sum() / -m;
        return cost + (Vector.dotmul(theta, theta) - theta[0] * theta[0]) / (2 * m);
    }
}
