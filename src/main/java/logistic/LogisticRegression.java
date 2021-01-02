package logistic;

import algo.Hypothesis;
import algo.Sigmoid;
import numeric.Vector;

import java.util.Arrays;

public class LogisticRegression implements Hypothesis<Double> {
    private double[] theta;

    public LogisticRegression(double[] theta) {
        this.theta = theta;
    }

    @Override
    public Double apply(double[] x) {
        double hx = Sigmoid.apply(Vector.dotmul(theta, x));
        return hx;
    }

    @Override
    public String toString() {
        return "Theta: \n" + Arrays.toString(theta);
    }
}
