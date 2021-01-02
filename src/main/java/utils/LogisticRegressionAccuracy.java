package utils;

import logistic.LogisticRegressionPredict;
import logistic.MultiLogisticRegressionPredicate;

public class LogisticRegressionAccuracy {
    public static double testMulti(MultiLogisticRegressionPredicate predicate,
                                   double[][] xs, double[][] ys) {
        int succ = 0;
        for (int i = 0; i < xs.length; i++) {
            if (ys[i][predicate.predicate(xs[i])] == 1) {
                succ++;
            }
        }
        return (double) succ / xs.length;
    }

    public static double testBinary(LogisticRegressionPredict predicate,
                                    double[][] xs, double[] ys) {
        int succ = 0;
        for (int i = 0; i < xs.length; i++) {
            if (predicate.predicate(xs[i]) == (ys[i] == 1)) {
                succ++;
            }
        }
        return (double) succ / xs.length;
    }
}
