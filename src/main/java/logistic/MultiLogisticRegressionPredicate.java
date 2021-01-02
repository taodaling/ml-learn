package logistic;

import algo.Hypothesis;

public class MultiLogisticRegressionPredicate {
    private Hypothesis<double[]> h;

    public MultiLogisticRegressionPredicate(Hypothesis<double[]> h) {
        this.h = h;
    }

    public int predicate(double[] x) {
        double[] p = h.apply(x);
        int ans = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[ans]) {
                ans = i;
            }
        }
        return ans;
    }
}
