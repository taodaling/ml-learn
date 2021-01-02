package linear;

import logistic.LogisticGradientFunction;
import numeric.Vector;

public class LinearRegressionGradientFunction extends LogisticGradientFunction {

    public LinearRegressionGradientFunction(double[][] xs, double[] ys, double regularLambda) {
        super(xs, ys, regularLambda);
    }

    @Override
    protected double computeActual(double[] theta, double[] x) {
        return Vector.dotmul(theta, x);
    }
}
