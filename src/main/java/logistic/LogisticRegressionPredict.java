package logistic;

import algo.Hypothesis;

public class LogisticRegressionPredict {
    Hypothesis<Double> h;

    public LogisticRegressionPredict(Hypothesis<Double> h) {
        this.h = h;
    }

    public boolean predicate(double[] x) {
        double p = h.apply(x);
        return p >= 0.5;
    }
}
