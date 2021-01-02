package linear;

import algo.Hypothesis;
import numeric.Vector;

import java.util.Arrays;

public class LinearRegression implements Hypothesis<Double> {
    double[] theta;

    public LinearRegression(double[] theta) {
        this.theta = theta;
    }

    @Override
    public Double apply(double[] x) {
        return Vector.dotmul(theta, x);
    }

    @Override
    public String toString() {
        return Arrays.toString(theta);
    }
}
